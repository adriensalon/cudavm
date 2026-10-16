#pragma once
#include <memory>
#include <mutex>
#include <vector>

namespace cudavm {
// Logical CUDA contexts keep independent registries and schedulers. Native
// resource owners are shared on each device so pool buffers can be bound by
// another context without importing or copying their backing. Weak entries
// release the native device when its last session, buffer and task retire.
template<class T> class device_resources {
    std::mutex mutex;
    std::vector<std::weak_ptr<T>> devices;
public:
    template<class F> std::shared_ptr<T> get(size_t ordinal, F create) {
        std::lock_guard lock(mutex);
        if (devices.size() <= ordinal) devices.resize(ordinal + 1);
        auto value = devices[ordinal].lock();
        if (!value) { value = create(); devices[ordinal] = value; }
        return value;
    }
};
}
