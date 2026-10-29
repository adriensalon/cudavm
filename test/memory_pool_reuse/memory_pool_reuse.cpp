#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>

namespace cudavm {
void check_reuse_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define REUSE_RUNTIME(call) ::cudavm::check_reuse_runtime((call), #call)
struct reuse_pool {
    CUmemoryPool value{};
    bool runtime;
    reuse_pool(bool use_runtime, size_t maximum = 0) : runtime(use_runtime) {
        CUdevice device{}; int supported = 0;
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
        require(supported != 0, "pool reuse profile requires advertised memory-pool support");
        if (runtime) {
            cudaMemPoolProps props{}; props.allocType = cudaMemAllocationTypePinned;
            props.location.type = cudaMemLocationTypeDevice; props.location.id = device; props.maxSize = maximum;
            cudaMemPool_t pool{}; REUSE_RUNTIME(cudaMemPoolCreate(&pool, &props)); value = reinterpret_cast<CUmemoryPool>(pool);
        } else {
            CUmemPoolProps props{}; props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
            props.location.type = CU_MEM_LOCATION_TYPE_DEVICE; props.location.id = device; props.maxSize = maximum;
            CONFORMANCE_CUDA(cuMemPoolCreate(&value, &props));
        }
        uint64_t threshold = UINT64_MAX;
        CONFORMANCE_CUDA(cuMemPoolSetAttribute(value, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold));
    }
    ~reuse_pool() { if (value) cuMemPoolDestroy(value); }
    cudaMemPool_t runtime_value() const { return reinterpret_cast<cudaMemPool_t>(value); }
    int get_policy(CUmemPool_attribute attribute) const {
        alignas(int) std::array<unsigned char, sizeof(int) + 32> driver, runtime;
        driver.fill(0xe7); runtime.fill(0xe7);
        CONFORMANCE_CUDA(cuMemPoolGetAttribute(value, attribute, driver.data() + 16));
        REUSE_RUNTIME(cudaMemPoolGetAttribute(runtime_value(), static_cast<cudaMemPoolAttr>(attribute), runtime.data() + 16));
        for (size_t i = 0; i < driver.size(); ++i)
            if (i < 16 || i >= 16 + sizeof(int)) require(driver[i] == 0xe7 && runtime[i] == 0xe7, "reuse-policy output width guards");
        int d = -1, r = -1;
        std::memcpy(&d, driver.data() + 16, sizeof(int)); std::memcpy(&r, runtime.data() + 16, sizeof(int));
        require(d == r && (d == 0 || d == 1), "Driver/Runtime boolean reuse-policy agreement"); return d;
    }
    void policy(CUmemPool_attribute attribute, int setting, bool use_runtime) const {
        if (use_runtime) REUSE_RUNTIME(cudaMemPoolSetAttribute(runtime_value(), static_cast<cudaMemPoolAttr>(attribute), &setting));
        else CONFORMANCE_CUDA(cuMemPoolSetAttribute(value, attribute, &setting));
        require(get_policy(attribute) == setting, "reuse-policy roundtrip");
    }
    uint64_t get(CUmemPool_attribute attribute) const {
        uint64_t driver = 0, runtime = 0;
        CONFORMANCE_CUDA(cuMemPoolGetAttribute(value, attribute, &driver));
        REUSE_RUNTIME(cudaMemPoolGetAttribute(runtime_value(), static_cast<cudaMemPoolAttr>(attribute), &runtime));
        require(driver == runtime, "pool reuse reservation/usage agreement"); return driver;
    }
    void destroy() {
        if (runtime) CONFORMANCE_CUDA(cuMemPoolDestroy(value));
        else REUSE_RUNTIME(cudaMemPoolDestroy(runtime_value()));
        value = nullptr;
    }
};
constexpr std::array<CUmemPool_attribute, 3> reuse_policies{CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES};
struct reuse_stream {
    CUstream value{};
    reuse_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~reuse_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct reused_allocation {
    CUdeviceptr value{};
    ~reused_allocation() { if (value) cuMemFree(value); }
    void allocate(const reuse_pool& pool, CUstream stream, size_t size, bool runtime) {
        if (runtime) {
            void* pointer{}; REUSE_RUNTIME(cudaMallocFromPoolAsync(&pointer, size, pool.runtime_value(), reinterpret_cast<cudaStream_t>(stream)));
            value = reinterpret_cast<uintptr_t>(pointer);
        } else CONFORMANCE_CUDA(cuMemAllocFromPoolAsync(&value, size, pool.value, stream));
        require(value != 0, "reuse allocation publishes a nonempty pointer");
    }
    void retire(CUstream stream, bool runtime) {
        if (runtime) REUSE_RUNTIME(cudaFreeAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(value)), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuMemFreeAsync(value, stream));
        value = 0;
    }
};
struct reuse_snapshot {
    void* value{};
    size_t chunk, stride, size;
    explicit reuse_snapshot(size_t capacity) : chunk(std::min<size_t>(64, capacity / 3)), stride(chunk + 32), size(stride * 3) {
        require(chunk != 0, "positive snapshot chunk");
        CONFORMANCE_CUDA(cuMemAllocHost(&value, size)); std::memset(value, 0xd3, size);
    }
    ~reuse_snapshot() { if (value) cuMemFreeHost(value); }
    void capture(CUdeviceptr pointer, size_t capacity, CUstream stream) const {
        const std::array<size_t, 3> offsets{0, (capacity - chunk) / 2, capacity - chunk};
        for (size_t row = 0; row < offsets.size(); ++row)
            CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(value) + row * stride + 16, pointer + offsets[row], chunk, stream));
    }
    void verify(unsigned char pattern) const {
        const auto* bytes = static_cast<unsigned char*>(value);
        for (size_t i = 0; i < size; ++i) {
            const auto at = i % stride;
            require(bytes[i] == (at >= 16 && at < 16 + chunk ? pattern : 0xd3), "reuse lifetime snapshot and host guards");
        }
    }
};
size_t reuse_capacity(bool runtime) {
    reuse_pool pool(runtime);
    reuse_stream stream;
    reused_allocation allocation;
    struct drain { ~drain() { cuCtxSynchronize(); } } cleanup;
    allocation.allocate(pool, stream.value, 7, runtime);
    CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
    const auto reserved = pool.get(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT);
    require(reserved >= 7 && reserved <= std::numeric_limits<size_t>::max(), "measured pool backing capacity");
    allocation.retire(stream.value, !runtime); CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
    pool.destroy();
    return size_t(reserved);
}
void memory_pool_reuse_policies() {
    for (bool runtime : {false, true}) {
        reuse_pool pool(runtime), independent(!runtime);
        for (auto attribute : reuse_policies) require(pool.get_policy(attribute) == 1 && independent.get_policy(attribute) == 1, "reuse policies default enabled");
        for (auto attribute : reuse_policies) for (int setting : {0, 1, 0}) {
            pool.policy(attribute, setting, runtime);
            for (auto other : reuse_policies)
                if (other != attribute) require(pool.get_policy(other) == (other < attribute ? 0 : 1), "policy setters preserve other policy values");
            require(independent.get_policy(attribute) == 1, "reuse policies belong to their pool");
        }
        pool.destroy(); independent.destroy();
    }
}
enum class reuse_route { same_stream, events, opportunistic, internal };
void memory_pool_reuse_route(reuse_route route) {
    for (bool runtime : {false, true}) {
        const auto capacity = reuse_capacity(runtime);
        reuse_pool pool(runtime, capacity);
        for (size_t i = 0; i < reuse_policies.size(); ++i) {
            const bool enabled = (i == 0 && route == reuse_route::events) || (i == 1 && route == reuse_route::opportunistic)
                || (i == 2 && route == reuse_route::internal);
            pool.policy(reuse_policies[i], int(enabled), (i % 2) ? runtime : !runtime);
        }
        require(pool.get(CU_MEMPOOL_ATTR_MAX_POOL_SIZE) == capacity, "configured finite reuse budget");
        reuse_stream producer, consumer;
        reuse_snapshot first(capacity), second(capacity);
        reused_allocation old, next;
        struct retained_event {
            CUevent value{};
            retained_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
            ~retained_event() { if (value) cuEventDestroy(value); }
        } freed;
        struct drain { ~drain() { cuCtxSynchronize(); } } cleanup;
        old.allocate(pool, producer.value, capacity, runtime);
        for (unsigned repeat = 0; repeat < 4; ++repeat) CONFORMANCE_CUDA(cuMemsetD8Async(old.value, 0x39, capacity, producer.value));
        first.capture(old.value, capacity, producer.value);
        old.retire(producer.value, !runtime);
        if (route == reuse_route::events) {
            CONFORMANCE_CUDA(cuEventRecord(freed.value, producer.value));
            REUSE_RUNTIME(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(consumer.value), reinterpret_cast<cudaEvent_t>(freed.value), 0));
        } else if (route == reuse_route::opportunistic) {
            CONFORMANCE_CUDA(cuStreamSynchronize(producer.value)); first.verify(0x39);
        }
        const auto target = route == reuse_route::same_stream ? producer.value : consumer.value;
        next.allocate(pool, target, capacity, !runtime);
        REUSE_RUNTIME(cudaMemsetAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(next.value)), 0x5b, capacity, reinterpret_cast<cudaStream_t>(target)));
        second.capture(next.value, capacity, target);
        next.retire(target, runtime);
        CONFORMANCE_CUDA(cuStreamSynchronize(producer.value));
        REUSE_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(target)));
        first.verify(0x39); second.verify(0x5b);
        require(pool.get(CU_MEMPOOL_ATTR_USED_MEM_CURRENT) == 0, "completed reused lifetimes retire application usage");
        require(pool.get(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT) <= capacity && pool.get(CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH) <= capacity,
            "reuse route stays within one measured backing budget");
        pool.destroy();
    }
}
void memory_pool_reuse_same_stream() { memory_pool_reuse_route(reuse_route::same_stream); }
void memory_pool_reuse_events() { memory_pool_reuse_route(reuse_route::events); }
void memory_pool_reuse_opportunistic() { memory_pool_reuse_route(reuse_route::opportunistic); }
void memory_pool_reuse_internal() { memory_pool_reuse_route(reuse_route::internal); }
std::string_view memory_pool_reuse_case = "all";
void execute_memory_pool_reuse() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"policies", memory_pool_reuse_policies}, {"same_stream", memory_pool_reuse_same_stream}, {"events", memory_pool_reuse_events},
        {"opportunistic", memory_pool_reuse_opportunistic}, {"internal", memory_pool_reuse_internal}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (memory_pool_reuse_case == "all" || memory_pool_reuse_case == name) {
        found = true; std::cout << "memory_pool_reuse=" << name << std::endl; execute();
    }
    require(found, "unknown memory_pool_reuse case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::memory_pool_reuse_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_memory_pool_reuse);
}
