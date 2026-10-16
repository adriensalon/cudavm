#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <list>
#include <memory>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include "command_domain.hpp"

namespace cudavm {
// Public stream ancestry is independent of native transport serialization.
// Vector clocks retain ordering evidence without retaining command histories.
using command_order = std::map<uint64_t, uint64_t>;
// Completion data is immutable after the release publication of ready.
struct command_completion {
    command_order order; // Immutable before the command is published to a queue.
    std::atomic<bool> ready{false};
    std::mutex mutex;
    std::condition_variable changed;
    std::exception_ptr error;
    std::chrono::steady_clock::time_point time;
    void wait(unsigned schedule = CU_CTX_SCHED_BLOCKING_SYNC) {
        if (schedule == CU_CTX_SCHED_SPIN || schedule == CU_CTX_SCHED_YIELD) {
            while (!ready.load(std::memory_order_acquire))
                if (schedule == CU_CTX_SCHED_YIELD) std::this_thread::yield();
            if (error) std::rethrow_exception(error);
            return;
        }
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return ready.load(std::memory_order_acquire); });
        if (error) std::rethrow_exception(error);
    }
    void query() {
        require(ready.load(std::memory_order_acquire), CUDA_ERROR_NOT_READY, "stream or event work is incomplete");
        if (error) std::rethrow_exception(error);
    }
};
using completion = std::shared_ptr<command_completion>;
inline void inherit_order(command_order& order, const completion& dependency) {
    if (dependency) for (auto [stream, sequence] : dependency->order)
        order[stream] = std::max(order[stream], sequence);
}
inline thread_local bool executing_host_function = false;

class command_queue {
    struct command {
        completion done;
        std::vector<completion> dependencies;
        std::function<void()> action;
        int priority = 0;
        std::function<device_work()> start;
        bool exclusive = false;
        std::shared_ptr<command_domain> domain{};
        std::function<void(std::exception_ptr)> notify;
    };
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<command> pending;
    std::vector<std::weak_ptr<command_completion>> issued;
    std::exception_ptr first_error;
    std::shared_ptr<command_domain> domain;
    bool stopping = false;
    std::thread worker;

    void run() noexcept {
        struct flight { command work; device_work task; bool domain_locked = false; };
        std::list<flight> active;
        auto unlock_domain = [&](const command& work) {
            if (work.exclusive) work.domain->unlock(); else work.domain->unlock_shared();
        };
        auto lock_domain = [&](const command& work) {
            return !work.domain || (work.exclusive ? work.domain->try_lock() : work.domain->try_lock_shared());
        };
        auto finish = [&](command& work) {
            work.done->time = std::chrono::steady_clock::now();
            work.action = {}; work.start = {}; work.notify = {}; work.dependencies.clear();
            if (work.done->error) {
                std::lock_guard lock(mutex);
                if (!first_error) first_error = work.done->error;
            }
            {
                std::lock_guard lock(work.done->mutex);
                work.done->ready.store(true, std::memory_order_release);
            }
            work.done->changed.notify_all();
        };
        for (;;) {
            bool exclusive_active = std::any_of(active.begin(), active.end(), [](const auto& item) { return item.work.exclusive; });
            for (auto item = active.begin(); item != active.end();) {
                if (item->work.domain && !item->work.exclusive && item->task->yields_to_exclusive() && item->domain_locked) {
                    unlock_domain(item->work); item->domain_locked = false;
                }
                if (exclusive_active && item->task->yields_to_exclusive()) { ++item; continue; }
                if (item->work.domain && !item->domain_locked) {
                    if (!lock_domain(item->work)) { ++item; continue; }
                    item->domain_locked = true;
                }
                bool ready = false;
                try { command_domain_scope scope(item->work.domain.get()); ready = item->task->query(); }
                catch (...) { item->work.done->error = std::current_exception(); ready = true; }
                if (item->domain_locked && (ready || (!item->work.exclusive && item->task->yields_to_exclusive()))) {
                    unlock_domain(item->work); item->domain_locked = false;
                }
                if (ready) { item->task.reset(); finish(item->work); item = active.erase(item); }
                else ++item;
            }
            command work;
            bool domain_locked = false;
            {
                std::unique_lock lock(mutex);
                for (;;) {
                    if (stopping && pending.empty() && active.empty()) return;
                    auto candidate = pending.end();
                    bool exclusive = std::any_of(active.begin(), active.end(), [](const auto& item) { return item.work.exclusive; });
                    for (auto item = pending.begin(); item != pending.end(); ++item) {
                        if (exclusive || (item->exclusive && std::any_of(active.begin(), active.end(), [](const auto& flight) {
                            return !flight.task->yields_to_exclusive();
                        }))) continue;
                        bool ready = std::all_of(item->dependencies.begin(), item->dependencies.end(), [](const completion& dependency) {
                            return !dependency || dependency->ready.load(std::memory_order_acquire);
                        });
                        if (ready && (candidate == pending.end() || item->priority < candidate->priority)) candidate = item;
                    }
                    if (candidate != pending.end()) {
                        if (candidate->domain && (candidate->start || candidate->exclusive)) {
                            if (!lock_domain(*candidate)) { changed.wait_for(lock, std::chrono::milliseconds(1)); break; }
                            domain_locked = true;
                        }
                        work = std::move(*candidate); pending.erase(candidate); break;
                    }
                    // Dependencies may complete on another context's worker.
                    if (pending.empty() && active.empty()) changed.wait(lock);
                    else { changed.wait_for(lock, std::chrono::milliseconds(1)); break; }
                }
            }
            if (!work.done) continue;
            // Allocate the in-flight slot before submission. Even allocation
            // failure must not discard a command that has already reached a GPU.
            auto slot = active.end();
            try {
                command_domain_scope scope(domain_locked ? work.domain.get() : nullptr);
                std::exception_ptr error;
                for (const auto& dependency : work.dependencies)
                    if (dependency && dependency->error) { error = dependency->error; break; }
                if (work.notify) {
                    { std::lock_guard lock(mutex); if (!error) error = first_error; }
                    // Callbacks run exactly once, including on a failed stream.
                    // Their completion retains the fault for subsequent users.
                    work.done->error = error; work.notify(error);
                } else if (error) std::rethrow_exception(error);
                else if (work.start) {
                    active.emplace_back(); slot = std::prev(active.end());
                    auto task = work.start();
                    if (task) { slot->task = std::move(task); slot->work = std::move(work); slot->domain_locked = domain_locked; continue; }
                } else work.action();
            } catch (...) { work.done->error = std::current_exception(); }
            if (slot != active.end()) active.erase(slot);
            if (domain_locked) unlock_domain(work);
            finish(work);
        }
    }
public:
    struct submission {
        completion done;
        std::vector<completion> dependencies;
        std::function<void()> action;
        int priority = 0;
        std::function<device_work()> start;
        bool exclusive = false;
        std::shared_ptr<command_domain> domain{};
        std::function<void(std::exception_ptr)> notify;
    };
    explicit command_queue(std::shared_ptr<command_domain> value = {}) : domain(std::move(value)), worker([this] { run(); }) {}
    ~command_queue() {
        { std::lock_guard lock(mutex); stopping = true; }
        changed.notify_all(); worker.join();
    }
    completion submit(std::vector<completion> dependencies, std::function<void()> action, int priority = 0, std::function<device_work()> start = {}, bool exclusive = false, command_order order = {}, std::function<void(std::exception_ptr)> notify = {}) {
        auto done = std::make_shared<command_completion>();
        done->order = std::move(order);
        {
            std::lock_guard lock(mutex);
            std::erase_if(issued, [](const auto& item) { return item.expired(); });
            issued.push_back(done);
            try { pending.push_back({done, std::move(dependencies), std::move(action), priority, std::move(start), exclusive, domain, std::move(notify)}); }
            catch (...) { issued.pop_back(); throw; }
        }
        changed.notify_one(); return done;
    }
    // Publish a fully preflighted replay atomically. Allocation failure rolls
    // back the queue before the worker can observe any of the batch.
    void submit_batch(std::vector<submission> work) {
        {
            std::lock_guard lock(mutex);
            auto pending_size = pending.size(), issued_size = issued.size();
            try {
                issued.reserve(issued.size() + work.size());
                for (auto& item : work) {
                    issued.push_back(item.done);
                    pending.push_back({std::move(item.done), std::move(item.dependencies), std::move(item.action), item.priority, std::move(item.start), item.exclusive, item.domain ? std::move(item.domain) : domain, std::move(item.notify)});
                }
            } catch (...) {
                while (pending.size() > pending_size) pending.pop_back();
                issued.resize(issued_size); throw;
            }
        }
        changed.notify_one();
    }
    void drain(bool report_error = true, unsigned schedule = CU_CTX_SCHED_BLOCKING_SYNC) {
        std::vector<completion> work;
        { std::lock_guard lock(mutex); for (auto& item : issued) if (auto done = item.lock()) work.push_back(std::move(done)); }
        std::exception_ptr error;
        for (auto& done : work) {
            try { done->wait(schedule); } catch (...) { if (!error) error = std::current_exception(); }
        }
        { std::lock_guard lock(mutex); if (!error) error = first_error; }
        if (report_error && error) std::rethrow_exception(error);
    }
};
}
