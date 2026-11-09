#include "test_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace cudavm {
void resource_runtime(cudaError_t status, const char* operation) {
    require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}
#define RESOURCE_RUNTIME(call) ::cudavm::resource_runtime((call), #call)
int resource_attribute(CUfunction function, CUfunction_attribute which) {
    struct { uint32_t before = 0x12345678; int value = -999; uint32_t after = 0x87654321; } result;
    CONFORMANCE_CUDA(cuFuncGetAttribute(&result.value, which, function));
    require(result.before == 0x12345678 && result.after == 0x87654321, "function attribute output width");
    return result.value;
}
struct resource_settings {
    CUfunction function;
    int dynamic, carveout;
    explicit resource_settings(CUfunction value) : function(value),
        dynamic(resource_attribute(value, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)),
        carveout(resource_attribute(value, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT)) {}
    ~resource_settings() {
        cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic);
        cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, carveout);
        cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_NONE);
    }
};
void resource_metadata(CUfunction function, CUkernel kernel) {
    const char* name{};
    CONFORMANCE_CUDA(cuFuncGetName(&name, function));
    require(name && std::strcmp(name, "mixed_exchange") == 0, "bound function name");
    size_t count = 999;
    CONFORMANCE_CUDA(cuFuncGetParamCount(function, &count));
    require(count == 3, "exact bound function parameter count");
    for (size_t index = 0; index < count; ++index) {
        size_t offset = 999, size = 999;
        CONFORMANCE_CUDA(cuFuncGetParamInfo(function, index, &offset, &size));
        require(offset == index * 8 && size == (index == 2 ? 4 : 8), "exact function parameter ABI");
        if (kernel) {
            size_t kernel_offset = 999, kernel_size = 999, runtime_offset = 999, runtime_size = 999;
            CONFORMANCE_CUDA(cuKernelGetParamInfo(kernel, index, &kernel_offset, &kernel_size));
            RESOURCE_RUNTIME(cudaFuncGetParamInfo(reinterpret_cast<const void*>(kernel), index, &runtime_offset, &runtime_size));
            require(kernel_offset == offset && runtime_offset == offset && kernel_size == size && runtime_size == size,
                    "contextless and Runtime parameter records agree");
        }
    }
    size_t offset = 123, size = 456;
    require(cuFuncGetParamInfo(function, count, &offset, &size) == CUDA_ERROR_INVALID_VALUE
        && offset == 123 && size == 456, "out-of-range parameter query preserves outputs");
    require(cuFuncGetParamCount(function, nullptr) == CUDA_ERROR_INVALID_VALUE
        && cuFuncGetName(nullptr, function) == CUDA_ERROR_INVALID_VALUE, "null metadata output validation");
    require(resource_attribute(function, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) >= 16
        && resource_attribute(function, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) >= 4
        && resource_attribute(function, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES) >= 0, "executable function resource bounds");
    if (!kernel) return;
    const void* token = reinterpret_cast<const void*>(kernel);
    const char *kernel_name{}, *runtime_name{};
    CONFORMANCE_CUDA(cuKernelGetName(&kernel_name, kernel)); RESOURCE_RUNTIME(cudaFuncGetName(&runtime_name, token));
    require(std::strcmp(name, kernel_name) == 0 && std::strcmp(name, runtime_name) == 0, "library/Runtime names agree");
    size_t kernel_count = 999, runtime_count = 999;
    CONFORMANCE_CUDA(cuKernelGetParamCount(kernel, &kernel_count)); RESOURCE_RUNTIME(cudaFuncGetParamCount(token, &runtime_count));
    require(kernel_count == count && runtime_count == count, "library/Runtime parameter counts agree");
    struct { uint64_t before = 0x123456789abcdef0; cudaFuncAttributes value{}; uint64_t after = 0xfedcba9876543210; } aggregate;
    RESOURCE_RUNTIME(cudaFuncGetAttributes(&aggregate.value, token));
    require(aggregate.before == 0x123456789abcdef0 && aggregate.after == 0xfedcba9876543210, "Runtime attribute aggregate bounds");
    const auto& a = aggregate.value;
    const std::pair<CUfunction_attribute, size_t> fields[]{
        {CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, a.sharedSizeBytes}, {CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, a.constSizeBytes},
        {CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, a.localSizeBytes}, {CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, size_t(a.maxThreadsPerBlock)},
        {CU_FUNC_ATTRIBUTE_NUM_REGS, size_t(a.numRegs)}, {CU_FUNC_ATTRIBUTE_PTX_VERSION, size_t(a.ptxVersion)},
        {CU_FUNC_ATTRIBUTE_BINARY_VERSION, size_t(a.binaryVersion)}, {CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, size_t(a.cacheModeCA)},
        {CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size_t(a.maxDynamicSharedSizeBytes)}};
    for (auto [attribute, expected] : fields) require(size_t(resource_attribute(function, attribute)) == expected, "Runtime aggregate matches Driver resources");
    require(a.preferredShmemCarveout == resource_attribute(function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT), "Runtime carveout matches Driver");
}
void resource_exchange(CUfunction function, CUkernel kernel, unsigned shared, bool reject) {
    constexpr unsigned lanes = 16, blocks = 3, total = lanes * blocks, guard_words = 4, words = total + guard_words * 2 + 5;
    constexpr uint32_t guard = 0x9abcdef0;
    std::vector<uint32_t> source(words, guard), expected(words, guard), actual(words);
    for (unsigned i = 0; i < total; ++i) source[guard_words + i] = 0xfffffff0u + i * 0x1234567u;
    device_buffer input(words * 4), output(words * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(input.value, source.data(), words * 4));
    for (bool runtime : {false, true}) {
        if (runtime && !kernel) continue;
        expected.assign(words, guard);
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), words * 4));
        unsigned count = total - 1;
        auto from = input.value + guard_words * 4, to = output.value + guard_words * 4;
        void* args[]{&to, &from, &count};
        if (runtime) {
            const auto status = cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(blocks), dim3(lanes), args, shared, nullptr);
            if (reject) {
                require(status == cudaErrorInvalidConfiguration, "Runtime launch above configured dynamic limit is rejected");
                require(cudaGetLastError() == status, "clear rejected Runtime launch status");
            } else RESOURCE_RUNTIME(status);
        } else {
            const auto status = cuLaunchKernel(function, blocks, 1, 1, lanes, 1, 1, shared, nullptr, args, nullptr);
            if (reject) require(status == CUDA_ERROR_INVALID_VALUE, "Driver launch above configured dynamic limit is rejected");
            else CONFORMANCE_CUDA(status);
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, words * 4));
        if (!reject) for (unsigned i = 0; i < count; ++i) {
            const unsigned block = i / lanes, peer = block * lanes + lanes - 1 - i % lanes;
            expected[guard_words + i] = (peer < count ? source[guard_words + peer] : 0u) + block + 7;
        }
        expect_words(actual, expected, reject ? "rejected resource launch preserves all output" : "resource boundary shared exchange");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), input.value, words * 4));
        expect_words(actual, source, "resource launch source and guard preservation");
    }
}
void execute_function_resources(const loaded_program& program) {
    const auto function = program.function("mixed_exchange"), other = program.function("dynamic_reduce");
    const auto kernel = program.library_kernel("mixed_exchange");
    resource_settings restore(function);
    require(restore.dynamic >= 128, "fixture has a usable dynamic shared-memory budget");
    resource_metadata(function, kernel);
    const int other_initial = resource_attribute(other, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
    for (int limit : {64, 128, 64}) {
        if (kernel && limit == 128) RESOURCE_RUNTIME(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, limit));
        else CONFORMANCE_CUDA(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, limit));
        require(resource_attribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) == limit
            && resource_attribute(other, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) == other_initial, "function settings are independent");
        resource_metadata(function, kernel);
        resource_exchange(function, kernel, unsigned(limit), false);
        resource_exchange(function, kernel, unsigned(limit) + 1, true);
        resource_exchange(function, kernel, unsigned(limit), false);
    }
    for (int hint : {-1, 0, 50, 100}) {
        CONFORMANCE_CUDA(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, hint));
        require(resource_attribute(function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) == hint, "carveout preference roundtrip");
        require(resource_attribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) == 64, "carveout preserves configured dynamic limit");
    }
    for (auto cache : {CU_FUNC_CACHE_PREFER_NONE, CU_FUNC_CACHE_PREFER_SHARED, CU_FUNC_CACHE_PREFER_L1, CU_FUNC_CACHE_PREFER_EQUAL}) {
        if (kernel) RESOURCE_RUNTIME(cudaFuncSetCacheConfig(reinterpret_cast<const void*>(kernel), static_cast<cudaFuncCache>(cache)));
        else CONFORMANCE_CUDA(cuFuncSetCacheConfig(function, cache));
        resource_exchange(function, kernel, 64, false);
    }
    int unchanged = 123;
    require(cuFuncGetAttribute(nullptr, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function) == CUDA_ERROR_INVALID_VALUE
        && cuFuncGetAttribute(&unchanged, CU_FUNC_ATTRIBUTE_MAX, function) == CUDA_ERROR_INVALID_VALUE && unchanged == 123,
        "invalid resource queries preserve outputs");
    for (int invalid : {-1, std::numeric_limits<int>::max()})
        require(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, invalid) == CUDA_ERROR_INVALID_VALUE, "invalid dynamic limit rejected");
    for (int invalid : {-2, 101})
        require(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, invalid) == CUDA_ERROR_INVALID_VALUE, "invalid carveout rejected");
    require(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 16) == CUDA_ERROR_INVALID_VALUE
        && cuFuncSetCacheConfig(function, static_cast<CUfunc_cache>(4)) == CUDA_ERROR_INVALID_VALUE, "read-only attributes and invalid cache rejected");
    require(resource_attribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) == 64
        && resource_attribute(function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) == 100, "rejected configuration preserves settings");
    CUcontext original{}, separate{}, observed{};
    CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    CONFORMANCE_CUDA(cuCtxCreate(&separate, nullptr, 0, device));
    try {
        require(resource_attribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) == 64, "bound resources use their owning context");
        CONFORMANCE_CUDA(cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 128));
        CONFORMANCE_CUDA(cuCtxGetCurrent(&observed)); require(observed == separate, "bound configuration preserves the current context");
    } catch (...) { cuCtxSetCurrent(original); cuCtxDestroy(separate); throw; }
    CONFORMANCE_CUDA(cuCtxSetCurrent(original)); CONFORMANCE_CUDA(cuCtxDestroy(separate));
    require(resource_attribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) == 128, "foreign-current setter changes the owning function");
    resource_exchange(function, kernel, 128, false);
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_function_resources);
}
