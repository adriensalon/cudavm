#pragma once
// User destructors run independently of stream workers and API locks. Allocate
// the callback record at creation so releasing the final reference cannot fail.
struct user_destructor {
    CUhostFn function = nullptr;
    void* data = nullptr;
    std::unique_ptr<user_destructor> next;
};
class user_destructor_queue {
    std::mutex mutex;
    std::condition_variable changed;
    std::unique_ptr<user_destructor> first;
    user_destructor* last = nullptr;
    bool stopping = false;
    std::thread worker;
    void run() noexcept {
        for (;;) {
            std::unique_ptr<user_destructor> task;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return stopping || first; });
                if (!first) return;
                task = std::move(first); first = std::move(task->next);
                if (!first) last = nullptr;
            }
            executing_host_function = true;
            try { task->function(task->data); } catch (...) {}
            executing_host_function = false;
        }
    }
public:
    user_destructor_queue() : worker([this] { run(); }) {}
    ~user_destructor_queue() {
        { std::lock_guard lock(mutex); stopping = true; }
        changed.notify_one(); worker.join();
    }
    void post(std::unique_ptr<user_destructor> task) noexcept {
        {
            std::lock_guard lock(mutex);
            auto* tail = task.get();
            if (last) last->next = std::move(task); else first = std::move(task);
            last = tail;
        }
        changed.notify_one();
    }
};
struct user_object {
    std::shared_ptr<user_destructor_queue> queue;
    std::unique_ptr<user_destructor> task;
    bool armed = false;
    ~user_object() { if (armed) queue->post(std::move(task)); }
};
struct user_reference {
    std::shared_ptr<user_object> object;
    uint64_t count = 0;
};
using user_references = std::unordered_map<CUuserObject, user_reference>;
struct user_object_entry {
    std::weak_ptr<user_object> object;
    user_reference caller;
};
