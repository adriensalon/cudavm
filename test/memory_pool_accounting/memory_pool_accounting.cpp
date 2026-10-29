#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>

namespace cudavm {
void check_accounting_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define ACCOUNTING_RUNTIME(call) ::cudavm::check_accounting_runtime((call), #call)
template<class T> struct accounting_output {
    alignas(T) std::array<unsigned char, sizeof(T) + 32> bytes;
    accounting_output() { bytes.fill(0xe7); }
    void* data() { return bytes.data() + 16; }
    T get() const {
        for (size_t i = 0; i < bytes.size(); ++i)
            if (i < 16 || i >= 16 + sizeof(T)) require(bytes[i] == 0xe7, "pool attribute/access output guards");
        T value{}; std::memcpy(&value, bytes.data() + 16, sizeof(T)); return value;
    }
};
struct accounting_pool {
    CUmemoryPool value{};
    CUdevice device{};
    bool runtime;
    explicit accounting_pool(bool use_runtime) : runtime(use_runtime) {
        CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        int supported = 0;
        CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
        require(supported != 0, "pool accounting requires advertised memory-pool support");
        if (runtime) {
            cudaMemPoolProps props{}; props.allocType = cudaMemAllocationTypePinned;
            props.location.type = cudaMemLocationTypeDevice; props.location.id = device;
            cudaMemPool_t pool{}; ACCOUNTING_RUNTIME(cudaMemPoolCreate(&pool, &props)); value = reinterpret_cast<CUmemoryPool>(pool);
        } else {
            CUmemPoolProps props{}; props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
            props.location.type = CU_MEM_LOCATION_TYPE_DEVICE; props.location.id = device;
            CONFORMANCE_CUDA(cuMemPoolCreate(&value, &props));
        }
    }
    ~accounting_pool() { if (value) cuMemPoolDestroy(value); }
    cudaMemPool_t runtime_value() const { return reinterpret_cast<cudaMemPool_t>(value); }
    template<class T> T get(CUmemPool_attribute attribute) const {
        accounting_output<T> driver, runtime;
        CONFORMANCE_CUDA(cuMemPoolGetAttribute(value, attribute, driver.data()));
        ACCOUNTING_RUNTIME(cudaMemPoolGetAttribute(runtime_value(), static_cast<cudaMemPoolAttr>(attribute), runtime.data()));
        const auto result = driver.get(); require(runtime.get() == result, "Driver/Runtime pool attribute agreement"); return result;
    }
    void set(CUmemPool_attribute attribute, uint64_t setting, bool use_runtime) const {
        if (use_runtime) ACCOUNTING_RUNTIME(cudaMemPoolSetAttribute(runtime_value(), static_cast<cudaMemPoolAttr>(attribute), &setting));
        else CONFORMANCE_CUDA(cuMemPoolSetAttribute(value, attribute, &setting));
    }
    void trim(size_t minimum, bool use_runtime) const {
        if (use_runtime) ACCOUNTING_RUNTIME(cudaMemPoolTrimTo(runtime_value(), minimum));
        else CONFORMANCE_CUDA(cuMemPoolTrimTo(value, minimum));
    }
    void destroy() {
        if (runtime) CONFORMANCE_CUDA(cuMemPoolDestroy(value));
        else ACCOUNTING_RUNTIME(cudaMemPoolDestroy(runtime_value()));
        value = nullptr;
    }
};
struct accounting_stream {
    CUstream value{};
    accounting_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~accounting_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct accounted_allocation {
    CUdeviceptr value{};
    size_t size;
    unsigned char pattern;
    accounted_allocation(const accounting_pool& pool, CUstream stream, size_t bytes, unsigned char fill, bool runtime)
        : size(bytes), pattern(fill) {
        if (runtime) {
            void* pointer{}; ACCOUNTING_RUNTIME(cudaMallocFromPoolAsync(&pointer, bytes, pool.runtime_value(), reinterpret_cast<cudaStream_t>(stream)));
            value = reinterpret_cast<uintptr_t>(pointer);
        } else CONFORMANCE_CUDA(cuMemAllocFromPoolAsync(&value, bytes, pool.value, stream));
        CONFORMANCE_CUDA(cuMemsetD8Async(value, pattern, bytes, stream));
    }
    ~accounted_allocation() { if (value) cuMemFree(value); }
    void retire(CUstream stream, bool runtime) {
        if (runtime) ACCOUNTING_RUNTIME(cudaFreeAsync(reinterpret_cast<void*>(static_cast<uintptr_t>(value)), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuMemFreeAsync(value, stream));
        value = 0;
    }
    void verify() const {
        std::vector<unsigned char> actual(size + 32, 0xe7);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data() + 16, value, size));
        for (size_t i = 0; i < actual.size(); ++i)
            require(actual[i] == (i >= 16 && i < size + 16 ? pattern : 0xe7), "live pooled bytes and readback guards");
    }
};
struct accounting_sample {
    uint64_t used, used_high, reserved, reserved_high;
};
accounting_sample sample_accounting(const accounting_pool& pool, uint64_t minimum_used) {
    const accounting_sample result{
        pool.get<uint64_t>(CU_MEMPOOL_ATTR_USED_MEM_CURRENT), pool.get<uint64_t>(CU_MEMPOOL_ATTR_USED_MEM_HIGH),
        pool.get<uint64_t>(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT), pool.get<uint64_t>(CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)};
    require(result.used >= minimum_used && result.reserved >= result.used && result.used_high >= result.used
        && result.reserved_high >= result.reserved, "pool usage/reservation/current/high invariants");
    return result;
}
void memory_pool_attributes() {
    for (bool runtime : {false, true}) {
        accounting_pool pool(runtime);
        const auto empty = sample_accounting(pool, 0);
        require(!empty.used && !empty.used_high && !empty.reserved && !empty.reserved_high, "fresh pool accounting is empty");
        require(pool.get<uint64_t>(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) == 0, "default pool release threshold");
        require(pool.get<int>(CU_MEMPOOL_ATTR_ALLOCATION_TYPE) == CU_MEM_ALLOCATION_TYPE_PINNED
            && pool.get<int>(CU_MEMPOOL_ATTR_LOCATION_TYPE) == CU_MEM_LOCATION_TYPE_DEVICE
            && pool.get<int>(CU_MEMPOOL_ATTR_LOCATION_ID) == pool.device
            && pool.get<int>(CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES) == CU_MEM_HANDLE_TYPE_NONE, "pool creation metadata");
        for (uint64_t threshold : {uint64_t(0), uint64_t(4096), std::numeric_limits<uint64_t>::max()}) {
            pool.set(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, threshold, !runtime);
            require(pool.get<uint64_t>(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) == threshold, "full-width release-threshold roundtrip");
        }
        CUmemLocation location{}; location.type = CU_MEM_LOCATION_TYPE_DEVICE; location.id = pool.device;
        cudaMemLocation runtime_location{}; runtime_location.type = cudaMemLocationTypeDevice; runtime_location.id = pool.device;
        accounting_output<CUmemAccess_flags> driver_access;
        accounting_output<cudaMemAccessFlags> runtime_access;
        CONFORMANCE_CUDA(cuMemPoolGetAccess(static_cast<CUmemAccess_flags*>(driver_access.data()), pool.value, &location));
        ACCOUNTING_RUNTIME(cudaMemPoolGetAccess(static_cast<cudaMemAccessFlags*>(runtime_access.data()), pool.runtime_value(), &runtime_location));
        require(driver_access.get() == CU_MEM_ACCESS_FLAGS_PROT_READWRITE && runtime_access.get() == cudaMemAccessFlagsProtReadWrite,
            "owning device has read/write pool access");
        pool.destroy();
    }
}
void memory_pool_accounting_live() {
    for (bool runtime : {false, true}) {
        device_buffer independent(257); CONFORMANCE_CUDA(cuMemsetD8(independent.value, 0x91, 257));
        accounting_pool pool(runtime); pool.set(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, UINT64_MAX, runtime);
        accounting_stream stream;
        accounted_allocation first(pool, stream.value, 257, 0x39, runtime), second(pool, stream.value, 4093, 0x5b, !runtime);
        struct drain { ~drain() { cuCtxSynchronize(); } } cleanup;
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value)); first.verify(); second.verify();
        const auto live = sample_accounting(pool, first.size + second.size);
        first.retire(stream.value, !runtime); CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        const auto one = sample_accounting(pool, second.size);
        require(one.used < live.used && one.used_high >= live.used_high && one.reserved_high >= live.reserved_high,
            "retired range reduces current usage while peaks remain historical");
        pool.trim(0, runtime);
        const auto trimmed = sample_accounting(pool, second.size);
        require(trimmed.used == one.used && trimmed.reserved <= one.reserved, "trimming preserves live usage and cannot increase reservation");
        second.verify();
        second.retire(stream.value, runtime); CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        const auto idle = sample_accounting(pool, 0);
        require(idle.used == 0 && idle.used_high >= live.used_high, "completed frees retire application usage while retaining peak");
        pool.set(CU_MEMPOOL_ATTR_USED_MEM_HIGH, 0, !runtime);
        require(pool.get<uint64_t>(CU_MEMPOOL_ATTR_USED_MEM_HIGH) == 0, "reset used high-water mark after complete retirement");
        pool.trim(0, !runtime);
        const auto final = sample_accounting(pool, 0);
        require(!final.used && final.reserved <= idle.reserved, "idle trim cannot resurrect usage or increase reservation");
        pool.destroy();
        std::array<unsigned char, 257> actual{};
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), independent.value, actual.size()));
        require(std::all_of(actual.begin(), actual.end(), [](auto value) { return value == 0x91; }), "pool retirement preserves ordinary live storage");
    }
}
void memory_pool_trim() {
    for (bool runtime : {false, true}) {
        accounting_pool pool(runtime); pool.set(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, UINT64_MAX, runtime);
        accounting_stream stream;
        accounted_allocation first(pool, stream.value, 1024 * 1024, 0x39, runtime), second(pool, stream.value, 4093, 0x5b, !runtime);
        struct drain { ~drain() { cuCtxSynchronize(); } } cleanup;
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value)); first.verify(); second.verify();
        first.retire(stream.value, !runtime); second.retire(stream.value, runtime);
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        const auto before = sample_accounting(pool, 0);
        require(!before.used && before.reserved > 0 && before.reserved < std::numeric_limits<size_t>::max(), "observed idle reservation retained by threshold");
        pool.trim(size_t(before.reserved + 1), runtime);
        require(pool.get<uint64_t>(CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT) == before.reserved, "trim above current reservation is a no-op");
        const auto minimum = size_t(before.reserved / 2);
        pool.trim(minimum, !runtime);
        const auto partial = sample_accounting(pool, 0);
        require(!partial.used && partial.reserved >= minimum && partial.reserved <= before.reserved,
            "trim respects requested minimum and backing granularity");
        pool.trim(0, runtime);
        const auto final = sample_accounting(pool, 0);
        require(!final.used && final.reserved <= partial.reserved, "trim-to-zero permits only safe reservation reduction");
        pool.destroy();
    }
}
void memory_pool_accounting_validation() {
    accounting_pool pool(false);
    pool.set(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, 4096, false);
    uint64_t value = 0;
    require(cuMemPoolGetAttribute(pool.value, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver pool attribute output");
    require(cudaMemPoolGetAttribute(pool.runtime_value(), cudaMemPoolAttrReleaseThreshold, nullptr) == cudaErrorInvalidValue, "null Runtime pool attribute output");
    require(cuMemPoolSetAttribute(pool.value, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver pool attribute input");
    require(cudaMemPoolSetAttribute(pool.runtime_value(), cudaMemPoolAttrReleaseThreshold, nullptr) == cudaErrorInvalidValue, "null Runtime pool attribute input");
    CUmemLocation location{}; location.type = CU_MEM_LOCATION_TYPE_DEVICE; location.id = pool.device;
    cudaMemLocation runtime_location{}; runtime_location.type = cudaMemLocationTypeDevice; runtime_location.id = pool.device;
    require(cuMemPoolGetAccess(nullptr, pool.value, &location) == CUDA_ERROR_INVALID_VALUE, "null Driver pool access output");
    require(cudaMemPoolGetAccess(nullptr, pool.runtime_value(), &runtime_location) == cudaErrorInvalidValue, "null Runtime pool access output");
    require(cuMemPoolGetAttribute(pool.value, static_cast<CUmemPool_attribute>(0x7fffffff), &value) == CUDA_ERROR_INVALID_VALUE, "invalid Driver pool attribute");
    require(cudaMemPoolGetAttribute(pool.runtime_value(), static_cast<cudaMemPoolAttr>(0x7fffffff), &value) == cudaErrorInvalidValue, "invalid Runtime pool attribute");
    for (auto attribute : {CU_MEMPOOL_ATTR_USED_MEM_CURRENT, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT}) {
        require(cuMemPoolSetAttribute(pool.value, attribute, &value) == CUDA_ERROR_INVALID_VALUE, "read-only Driver pool counter");
        require(cudaMemPoolSetAttribute(pool.runtime_value(), static_cast<cudaMemPoolAttr>(attribute), &value) == cudaErrorInvalidValue, "read-only Runtime pool counter");
    }
    value = 1;
    for (auto attribute : {CU_MEMPOOL_ATTR_USED_MEM_HIGH, CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH}) {
        require(cuMemPoolSetAttribute(pool.value, attribute, &value) == CUDA_ERROR_INVALID_VALUE, "nonzero Driver high-water reset");
        require(cudaMemPoolSetAttribute(pool.runtime_value(), static_cast<cudaMemPoolAttr>(attribute), &value) == cudaErrorInvalidValue, "nonzero Runtime high-water reset");
    }
    require(pool.get<uint64_t>(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) == 4096, "rejected setters preserve pool threshold");
    const auto empty = sample_accounting(pool, 0);
    require(!empty.used && !empty.used_high && !empty.reserved && !empty.reserved_high, "rejected operations preserve empty accounting");
    pool.destroy();
}
std::string_view memory_pool_accounting_case = "all";
void execute_memory_pool_accounting() {
    const std::array<std::pair<std::string_view, void (*)()>, 4> cases{{
        {"attributes", memory_pool_attributes}, {"accounting", memory_pool_accounting_live},
        {"trim", memory_pool_trim}, {"validation", memory_pool_accounting_validation}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (memory_pool_accounting_case == "all" || memory_pool_accounting_case == name) {
        found = true; std::cout << "memory_pool_accounting=" << name << std::endl; execute();
    }
    require(found, "unknown memory_pool_accounting case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::memory_pool_accounting_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_memory_pool_accounting);
}
