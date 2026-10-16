#pragma once
#include <memory>
#include <shared_mutex>
#include <condition_variable>
#include <mutex>

namespace cudavm {
// Shader kernels snapshot and write back every visible allocation. Contexts on
// one device share this exclusion domain so their transports cannot overwrite
// another context's transfers. Cooperative waits release it between polls.
class command_domain {
    std::mutex mutex;
    std::condition_variable changed;
    size_t readers = 0;
    bool writer = false;
public:
    bool try_lock() { std::lock_guard lock(mutex); if (writer || readers) return false; writer = true; return true; }
    bool try_lock_shared() { std::lock_guard lock(mutex); if (writer) return false; ++readers; return true; }
    void lock_shared() { std::unique_lock lock(mutex); changed.wait(lock, [&] { return !writer; }); ++readers; }
    void unlock() { { std::lock_guard lock(mutex); writer = false; } changed.notify_all(); }
    void unlock_shared() { { std::lock_guard lock(mutex); --readers; } changed.notify_all(); }
};
inline thread_local const command_domain* executing_command_domain = nullptr;
struct command_domain_scope {
    const command_domain* previous;
    explicit command_domain_scope(const command_domain* value)
        : previous(executing_command_domain) { executing_command_domain = value; }
    ~command_domain_scope() { executing_command_domain = previous; }
};
inline std::shared_lock<command_domain> command_domain_read(const std::shared_ptr<command_domain>& domain) {
    if (domain && executing_command_domain != domain.get()) return std::shared_lock(*domain);
    return {};
}
}
