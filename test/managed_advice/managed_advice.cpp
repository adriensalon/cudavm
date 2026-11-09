#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace cudavm {
std::string_view advice_case;
void advice_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("managed advice Runtime: ") + cudaGetErrorString(status));
}
struct advice_storage {
    CUdeviceptr allocation{}, page{};
    size_t page_size, bytes;
    std::vector<unsigned char> expected;
    advice_storage() {
#ifdef _WIN32
        SYSTEM_INFO info{}; GetSystemInfo(&info); page_size = info.dwPageSize;
#else
        page_size = size_t(sysconf(_SC_PAGESIZE));
#endif
        require(page_size && !(page_size & (page_size - 1)), "CPU page size");
        bytes = page_size * 6;
        CONFORMANCE_CUDA(cuMemAllocManaged(&allocation, bytes, CU_MEM_ATTACH_GLOBAL));
        page = (allocation + page_size - 1) & ~(CUdeviceptr(page_size) - 1);
        expected.resize(bytes);
        for (size_t i = 0; i < bytes; ++i) expected[i] = static_cast<unsigned char>(i * 37 + 29);
        std::copy(expected.begin(), expected.end(), host());
    }
    unsigned char* host() const { return reinterpret_cast<unsigned char*>(uintptr_t(allocation)); }
    ~advice_storage() { if (allocation) cuMemFree(allocation); }
    void verify() const {
        CONFORMANCE_CUDA(cuCtxSynchronize());
        require(std::equal(expected.begin(), expected.end(), host()), "advice/query preserves all managed payload bytes");
    }
};
CUmemLocation advice_location(CUmemLocationType type = CU_MEM_LOCATION_TYPE_HOST, int id = 0) {
    CUmemLocation result{}; result.type = type; result.id = id; return result;
}
void advice_apply(bool runtime, CUdeviceptr pointer, size_t bytes, CUmem_advise advice, CUmemLocation location = advice_location()) {
    if (runtime) {
        cudaMemLocation value{}; value.type = static_cast<cudaMemLocationType>(location.type); value.id = location.id;
        advice_runtime(cudaMemAdvise(reinterpret_cast<void*>(uintptr_t(pointer)), bytes, static_cast<cudaMemoryAdvise>(advice), value));
    } else CONFORMANCE_CUDA(cuMemAdvise(pointer, bytes, advice, location));
}
std::vector<int> advice_query(bool runtime, CUdeviceptr pointer, size_t bytes, CUmem_range_attribute attribute, size_t count = 1) {
    std::vector<int> guarded(count + 4, 0x31715926);
    if (runtime) advice_runtime(cudaMemRangeGetAttribute(guarded.data() + 2, count * sizeof(int), static_cast<cudaMemRangeAttribute>(attribute), reinterpret_cast<void*>(uintptr_t(pointer)), bytes));
    else CONFORMANCE_CUDA(cuMemRangeGetAttribute(guarded.data() + 2, count * sizeof(int), attribute, pointer, bytes));
    require(guarded[0] == 0x31715926 && guarded[1] == 0x31715926 && guarded[count + 2] == 0x31715926 && guarded[count + 3] == 0x31715926, "range query exact output guards");
    return {guarded.begin() + 2, guarded.end() - 2};
}
void advice_expect(bool runtime, CUdeviceptr pointer, size_t bytes, CUmem_range_attribute attribute, int expected) {
    require(advice_query(runtime, pointer, bytes, attribute)[0] == expected, "managed attribute " + std::to_string(attribute) + " expected " + std::to_string(expected));
}
void advice_read_mostly() {
    for (bool runtime : {false, true}) {
        advice_storage storage;
        for (unsigned repetition = 0; repetition < 2; ++repetition) {
            advice_apply(runtime, storage.page + storage.page_size / 2, 1, CU_MEM_ADVISE_SET_READ_MOSTLY, advice_location(CU_MEM_LOCATION_TYPE_INVALID, -123));
            advice_expect(!runtime, storage.page, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 1);
            advice_expect(runtime, storage.page + 2 * storage.page_size, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 0);
            advice_expect(!runtime, storage.page, 3 * storage.page_size, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 0);
            storage.verify();
        }
        advice_apply(!runtime, storage.page, storage.page_size, CU_MEM_ADVISE_UNSET_READ_MOSTLY);
        advice_expect(runtime, storage.page, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 0);
        storage.verify();
    }
}
void advice_preferred(bool device) {
    int ordinal = 0; CONFORMANCE_CUDA(cuCtxGetDevice(&ordinal));
    const auto location = device ? advice_location(CU_MEM_LOCATION_TYPE_DEVICE, ordinal) : advice_location(CU_MEM_LOCATION_TYPE_HOST, 12345);
    for (bool runtime : {false, true}) {
        advice_storage storage;
        advice_apply(runtime, storage.page + storage.page_size / 2, 1, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, location);
        advice_expect(!runtime, storage.page, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, device ? ordinal : CU_DEVICE_CPU);
        advice_expect(runtime, storage.page, 3 * storage.page_size, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, CU_DEVICE_INVALID);
        advice_expect(!runtime, storage.page + 2 * storage.page_size, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, CU_DEVICE_INVALID);
        advice_apply(!runtime, storage.page, storage.page_size, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, advice_location(CU_MEM_LOCATION_TYPE_INVALID));
        advice_expect(runtime, storage.page, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, CU_DEVICE_INVALID);
        storage.verify();
    }
}
void advice_accessed_by(bool device) {
    int ordinal = 0; CONFORMANCE_CUDA(cuCtxGetDevice(&ordinal));
    const auto location = device ? advice_location(CU_MEM_LOCATION_TYPE_DEVICE, ordinal) : advice_location();
    for (bool runtime : {false, true}) {
        advice_storage storage;
        advice_apply(runtime, storage.page, storage.page_size, CU_MEM_ADVISE_SET_ACCESSED_BY, location);
        const auto values = advice_query(!runtime, storage.page, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, 4);
        require(std::count(values.begin(), values.end(), device ? ordinal : CU_DEVICE_CPU) == 1 &&
                std::count(values.begin(), values.end(), CU_DEVICE_INVALID) == 3, "accessed-by complete location list and fillers");
        advice_apply(!runtime, storage.page, storage.page_size, CU_MEM_ADVISE_UNSET_ACCESSED_BY, location);
        require(advice_query(runtime, storage.page, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, 4) == std::vector<int>(4, CU_DEVICE_INVALID), "unset accessed-by removes location");
        storage.verify();
    }
}
void advice_queries() {
    advice_storage storage, independent;
    const std::array<CUmem_range_attribute, 4> attributes{CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
        CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION};
    for (bool runtime : {false, true}) for (auto pointer : {storage.page, independent.page}) {
        std::array<std::array<int, 3>, 4> outputs{};
        std::array<void*, 4> data{}; std::array<size_t, 4> sizes{4, 4, 4, 4};
        for (size_t i = 0; i < outputs.size(); ++i) { outputs[i] = {0x19375928, 99, 0x29183745}; data[i] = &outputs[i][1]; }
        auto kinds = attributes;
        if (runtime) {
            std::array<cudaMemRangeAttribute, 4> types{};
            std::transform(kinds.begin(), kinds.end(), types.begin(), [](auto value) { return static_cast<cudaMemRangeAttribute>(value); });
            advice_runtime(cudaMemRangeGetAttributes(data.data(), sizes.data(), types.data(), 4, reinterpret_cast<void*>(uintptr_t(pointer)), storage.page_size));
        } else CONFORMANCE_CUDA(cuMemRangeGetAttributes(data.data(), sizes.data(), kinds.data(), 4, pointer, storage.page_size));
        for (size_t i = 0; i < outputs.size(); ++i) {
            require(outputs[i][0] == 0x19375928 && outputs[i][2] == 0x29183745, "multiple range query output guards");
            const int expected = i == 0 ? 0 : CU_DEVICE_INVALID;
            require(outputs[i][1] == expected, "fresh allocation range metadata");
            advice_expect(!runtime, pointer + 7, storage.page_size - 17, kinds[i], expected);
        }
    }
    storage.verify(); independent.verify();
}
void advice_validation() {
    advice_storage storage;
    int output = 0;
    const auto pointer = reinterpret_cast<void*>(uintptr_t(storage.page));
    for (bool runtime : {false, true}) {
        for (size_t size : {size_t(0), size_t(1), size_t(3), size_t(8)}) {
            if (runtime) require(cudaMemRangeGetAttribute(&output, size, cudaMemRangeAttributeReadMostly, pointer, storage.page_size) == cudaErrorInvalidValue, "Runtime invalid scalar query size");
            else require(cuMemRangeGetAttribute(&output, size, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, storage.page, storage.page_size) == CUDA_ERROR_INVALID_VALUE, "Driver invalid scalar query size");
        }
        if (runtime) {
            require(cudaMemRangeGetAttribute(nullptr, 4, cudaMemRangeAttributeReadMostly, pointer, storage.page_size) == cudaErrorInvalidValue, "Runtime missing query output");
            require(cudaMemRangeGetAttribute(&output, 4, static_cast<cudaMemRangeAttribute>(0), pointer, storage.page_size) == cudaErrorInvalidValue, "Runtime invalid query attribute");
            require(cudaMemRangeGetAttribute(&output, 4, cudaMemRangeAttributeReadMostly, pointer, 0) == cudaErrorInvalidValue, "Runtime empty managed range");
            require(cudaMemAdvise(pointer, storage.page_size, static_cast<cudaMemoryAdvise>(0), {}) == cudaErrorInvalidValue, "Runtime invalid advice");
            require(cudaMemRangeGetAttributes(nullptr, nullptr, nullptr, 1, pointer, storage.page_size) == cudaErrorInvalidValue, "Runtime missing multiple query arrays");
            require(cudaGetLastError() == cudaErrorInvalidValue && cudaGetLastError() == cudaSuccess, "advice Runtime last error clears");
        } else {
            require(cuMemRangeGetAttribute(nullptr, 4, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, storage.page, storage.page_size) == CUDA_ERROR_INVALID_VALUE, "Driver missing query output");
            require(cuMemRangeGetAttribute(&output, 4, static_cast<CUmem_range_attribute>(0), storage.page, storage.page_size) == CUDA_ERROR_INVALID_VALUE, "Driver invalid query attribute");
            require(cuMemRangeGetAttribute(&output, 4, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, storage.page, 0) == CUDA_ERROR_INVALID_VALUE, "Driver empty managed range");
            require(cuMemAdvise(storage.page, storage.page_size, static_cast<CUmem_advise>(0), {}) == CUDA_ERROR_INVALID_VALUE, "Driver invalid advice");
            require(cuMemRangeGetAttributes(nullptr, nullptr, nullptr, 1, storage.page, storage.page_size) == CUDA_ERROR_INVALID_VALUE, "Driver missing multiple query arrays");
        }
        advice_apply(runtime, storage.page, storage.page_size, CU_MEM_ADVISE_UNSET_READ_MOSTLY);
        advice_expect(!runtime, storage.page, storage.page_size, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 0);
    }
    storage.verify();
}
void execute_managed_advice() {
    CUdevice device{}; int managed = 0, concurrent = 0;
    CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device));
    require_capability(managed != 0, "managed advice requires managed memory");
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&concurrent, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device));
    if (advice_case != "queries" && advice_case != "validation")
        require_capability(concurrent != 0, "advisory state-change profile requires concurrent managed access; basic managed behavior remains open");
    if (advice_case == "queries") advice_queries();
    else if (advice_case == "validation") advice_validation();
    else if (advice_case == "read_mostly") advice_read_mostly();
    else if (advice_case == "preferred") advice_preferred(false);
    else if (advice_case == "accessed_by") advice_accessed_by(false);
    else if (advice_case == "device") { advice_preferred(true); advice_accessed_by(true); }
    else throw std::runtime_error("unknown managed_advice case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::advice_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_managed_advice);
}
