#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <utility>

namespace cudavm {
void check_ordered_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define ORDERED_RUNTIME(call) ::cudavm::check_ordered_runtime((call), #call)
struct ordered_stream {
    CUstream value{};
    explicit ordered_stream(bool runtime) {
        if (runtime) {
            cudaStream_t stream{};
            ORDERED_RUNTIME(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); value = reinterpret_cast<CUstream>(stream);
        } else CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~ordered_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct ordered_pool {
    CUmemoryPool value{};
    bool runtime;
    ordered_pool(bool enabled, bool use_runtime, int device) : runtime(use_runtime) {
        if (!enabled) return;
        if (runtime) {
            cudaMemPoolProps props{};
            props.allocType = cudaMemAllocationTypePinned;
            props.location.type = cudaMemLocationTypeDevice; props.location.id = device;
            cudaMemPool_t pool{};
            ORDERED_RUNTIME(cudaMemPoolCreate(&pool, &props)); value = reinterpret_cast<CUmemoryPool>(pool);
        } else {
            CUmemPoolProps props{};
            props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
            props.location.type = CU_MEM_LOCATION_TYPE_DEVICE; props.location.id = device;
            CONFORMANCE_CUDA(cuMemPoolCreate(&value, &props));
        }
    }
    ~ordered_pool() { if (value) cuMemPoolDestroy(value); }
    void destroy() {
        if (!value) return;
        if (runtime) CONFORMANCE_CUDA(cuMemPoolDestroy(value));
        else ORDERED_RUNTIME(cudaMemPoolDestroy(reinterpret_cast<cudaMemPool_t>(value)));
        value = nullptr;
    }
};
struct ordered_storage {
    CUdeviceptr value{};
    ~ordered_storage() { if (value) cuMemFree(value); }
    void allocate(size_t size, CUstream stream, CUmemoryPool pool, bool runtime) {
        if (runtime) {
            void* pointer{};
            if (pool) ORDERED_RUNTIME(cudaMallocFromPoolAsync(&pointer, size, reinterpret_cast<cudaMemPool_t>(pool), reinterpret_cast<cudaStream_t>(stream)));
            else ORDERED_RUNTIME(cudaMallocAsync(&pointer, size, reinterpret_cast<cudaStream_t>(stream)));
            value = reinterpret_cast<uintptr_t>(pointer);
        } else if (pool) CONFORMANCE_CUDA(cuMemAllocFromPoolAsync(&value, size, pool, stream));
        else CONFORMANCE_CUDA(cuMemAllocAsync(&value, size, stream));
        require(value != 0, "nonempty stream-ordered allocation publishes a pointer");
    }
    void retire(CUstream stream, bool runtime) {
        if (runtime) ORDERED_RUNTIME(cudaFreeAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(value)), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuMemFreeAsync(value, stream));
        value = 0;
    }
};
struct ordered_readback {
    void* value{};
    size_t size;
    explicit ordered_readback(size_t bytes) : size(bytes) { CONFORMANCE_CUDA(cuMemAllocHost(&value, size)); std::memset(value, 0xe7, size); }
    ~ordered_readback() { if (value) cuMemFreeHost(value); }
    void expect_pending() const {
        const auto* bytes = static_cast<unsigned char*>(value);
        require(std::all_of(bytes, bytes + size, [](auto byte) { return byte == 0xe7; }), "pending async allocation readback remains initialized");
    }
    void verify(size_t payload) const {
        std::vector<unsigned char> expected(size, 0xe7);
        std::fill_n(expected.begin() + 16, payload + 32, 0xa9);
        std::fill_n(expected.begin() + 32, payload, 0x5b);
        require(std::equal(expected.begin(), expected.end(), static_cast<unsigned char*>(value)), "stream-ordered payload, device guards and host guards");
    }
};
struct ordered_event {
    CUevent value{};
    ordered_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~ordered_event() { if (value) cuEventDestroy(value); }
};
struct ordered_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* data) {
        auto& gate = *static_cast<ordered_gate*>(data);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "ordered allocation gate entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
int ordered_device() {
    CUdevice device{}; int supported = 0;
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
    require(supported != 0, "stream-ordered allocation profile requires advertised memory-pool support");
    return device;
}
void verify_ordered_survivor(CUdeviceptr pointer) {
    std::array<unsigned char, 257> actual{};
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), pointer, actual.size()));
    require(std::all_of(actual.begin(), actual.end(), [](auto byte) { return byte == 0x39; }), "ordinary live allocation survives async free and pool retirement");
}
void ordered_allocation_roundtrip(bool cross_stream, bool explicit_pool) {
    const auto device = ordered_device();
    for (bool runtime : {false, true}) for (size_t payload : {size_t(1), size_t(257), size_t(4093)}) {
        device_buffer survivor(257);
        CONFORMANCE_CUDA(cuMemsetD8(survivor.value, 0x39, 257));
        CUmemoryPool original_pool{};
        CONFORMANCE_CUDA(cuDeviceGetMemPool(&original_pool, device));
        ordered_pool pool(explicit_pool, runtime, device);
        ordered_stream producer(runtime), consumer(!runtime);
        ordered_readback readback(payload + 64);
        ordered_storage allocation;
        ordered_event ready, used;
        ordered_gate gate;
        struct cleanup_pending {
            ordered_gate& gate;
            ~cleanup_pending() { gate.release(); cuCtxSynchronize(); }
        } cleanup{gate};
        CONFORMANCE_CUDA(cuLaunchHostFunc(producer.value, ordered_gate::call, &gate)); gate.await();
        allocation.allocate(payload + 32, producer.value, pool.value, runtime);
        CONFORMANCE_CUDA(cuMemsetD8Async(allocation.value, 0xa9, payload + 32, producer.value));
        ORDERED_RUNTIME(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(allocation.value + 16)),
            0x5b, payload, reinterpret_cast<cudaStream_t>(producer.value)));
        CUstream target = producer.value;
        if (cross_stream) {
            CONFORMANCE_CUDA(cuEventRecord(ready.value, producer.value));
            ORDERED_RUNTIME(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(consumer.value), reinterpret_cast<cudaEvent_t>(ready.value), 0));
            target = consumer.value;
        }
        CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(readback.value) + 16, allocation.value, payload + 32, target));
        if (cross_stream) {
            ORDERED_RUNTIME(cudaEventRecord(reinterpret_cast<cudaEvent_t>(used.value), reinterpret_cast<cudaStream_t>(consumer.value)));
            CONFORMANCE_CUDA(cuStreamWaitEvent(producer.value, used.value, 0));
        }
        allocation.retire(producer.value, !runtime);
        require(cuStreamQuery(producer.value) == CUDA_ERROR_NOT_READY, "allocation/use/free chain remains pending at gate");
        if (cross_stream) require(cuStreamQuery(consumer.value) == CUDA_ERROR_NOT_READY, "consumer waits for allocation readiness");
        readback.expect_pending();
        gate.release();
        ORDERED_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(producer.value)));
        CONFORMANCE_CUDA(cuStreamSynchronize(consumer.value));
        require(!gate.expired, "ordered allocation gate deadline");
        readback.verify(payload);
        pool.destroy();
        CUmemoryPool current_pool{};
        cudaMemPool_t runtime_pool{};
        CONFORMANCE_CUDA(cuDeviceGetMemPool(&current_pool, device));
        ORDERED_RUNTIME(cudaDeviceGetMemPool(&runtime_pool, device));
        require(current_pool == original_pool && reinterpret_cast<CUmemoryPool>(runtime_pool) == original_pool,
            "explicit-pool operations preserve the configured device pool");
        verify_ordered_survivor(survivor.value);
    }
}
void stream_ordered_same_stream() { ordered_allocation_roundtrip(false, false); }
void stream_ordered_cross_stream() { ordered_allocation_roundtrip(true, false); }
void stream_ordered_pool() { ordered_allocation_roundtrip(false, true); ordered_allocation_roundtrip(true, true); }
void stream_ordered_validation() {
    const auto device = ordered_device();
    ordered_pool pool(true, false, device);
    ordered_stream stream(false);
    require(cuMemAllocAsync(nullptr, 257, stream.value) == CUDA_ERROR_INVALID_VALUE, "null Driver async allocation output");
    require(cudaMallocAsync(nullptr, 257, reinterpret_cast<cudaStream_t>(stream.value)) == cudaErrorInvalidValue, "null Runtime async allocation output");
    require(cuMemAllocFromPoolAsync(nullptr, 257, pool.value, stream.value) == CUDA_ERROR_INVALID_VALUE, "null Driver explicit-pool allocation output");
    require(cudaMallocFromPoolAsync(nullptr, 257, reinterpret_cast<cudaMemPool_t>(pool.value), reinterpret_cast<cudaStream_t>(stream.value))
        == cudaErrorInvalidValue, "null Runtime explicit-pool allocation output");
    ordered_storage allocation;
    ordered_readback readback(257 + 64);
    struct drain { ~drain() { cuCtxSynchronize(); } } cleanup;
    allocation.allocate(257 + 32, stream.value, pool.value, true);
    CONFORMANCE_CUDA(cuMemsetD8Async(allocation.value, 0xa9, 257 + 32, stream.value));
    CONFORMANCE_CUDA(cuMemsetD8Async(allocation.value + 16, 0x5b, 257, stream.value));
    CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(readback.value) + 16, allocation.value, 257 + 32, stream.value));
    allocation.retire(stream.value, false);
    CONFORMANCE_CUDA(cuStreamSynchronize(stream.value)); readback.verify(257);
    pool.destroy();
}
std::string_view stream_ordered_allocation_case = "all";
void execute_stream_ordered_allocation() {
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"same_stream", stream_ordered_same_stream}, {"cross_stream", stream_ordered_cross_stream},
        {"pool", stream_ordered_pool}, {"validation", stream_ordered_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (stream_ordered_allocation_case == "all" || stream_ordered_allocation_case == name) {
        found = true; std::cout << "stream_ordered_allocation=" << name << std::endl; execute();
    }
    require(found, "unknown stream_ordered_allocation case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::stream_ordered_allocation_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_stream_ordered_allocation);
}
