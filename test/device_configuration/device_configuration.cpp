#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <utility>

namespace cudavm {
void configuration_runtime(cudaError_t result, const char* operation) {
    require(result == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CONFIGURATION_RUNTIME(call) ::cudavm::configuration_runtime((call), #call)
template<class T> struct configuration_output {
    uint64_t before = 0x4b8e13d96a02f7c5ull;
    T value{};
    std::array<unsigned char, 8> after;
    configuration_output() { after.fill(0xb9); }
    T get() const {
        require(before == 0x4b8e13d96a02f7c5ull
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0xb9; }), "configuration query output guards");
        return value;
    }
};
CUcontext configuration_current() {
    CUcontext value{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&value)); return value;
}
struct configuration_context {
    CUcontext parent{}, value{};
    explicit configuration_context(unsigned flags = 0) {
        parent = configuration_current();
        CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
        CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, flags, device));
    }
    ~configuration_context() { if (value) cuCtxDestroy(value); cuCtxSetCurrent(parent); }
    configuration_context(const configuration_context&) = delete;
    configuration_context& operator=(const configuration_context&) = delete;
    void bind() const { CONFORMANCE_CUDA(cuCtxSetCurrent(value)); }
};
unsigned configuration_flags() {
    configuration_output<unsigned> driver, runtime;
    CONFORMANCE_CUDA(cuCtxGetFlags(&driver.value));
    CONFIGURATION_RUNTIME(cudaGetDeviceFlags(&runtime.value));
    require(driver.get() == runtime.get(), "Driver/Runtime context flags agree");
    return driver.get();
}
size_t configuration_limit(CUlimit driver_limit, cudaLimit runtime_limit) {
    configuration_output<size_t> driver, runtime, repeated;
    CONFORMANCE_CUDA(cuCtxGetLimit(&driver.value, driver_limit));
    CONFIGURATION_RUNTIME(cudaDeviceGetLimit(&runtime.value, runtime_limit));
    CONFORMANCE_CUDA(cuCtxGetLimit(&repeated.value, driver_limit));
    require(driver.get() == runtime.get() && repeated.get() == driver.get(), "effective limit is stable and agrees across facades");
    return driver.get();
}
CUfunc_cache configuration_cache() {
    configuration_output<CUfunc_cache> driver;
    configuration_output<cudaFuncCache> runtime;
    CONFORMANCE_CUDA(cuCtxGetCacheConfig(&driver.value));
    CONFIGURATION_RUNTIME(cudaDeviceGetCacheConfig(&runtime.value));
    require(driver.get() >= CU_FUNC_CACHE_PREFER_NONE && driver.get() <= CU_FUNC_CACHE_PREFER_EQUAL
        && int(driver.get()) == int(runtime.get()), "valid effective cache preference agrees across facades");
    return driver.get();
}
struct configuration_storage {
    std::vector<unsigned char> expected;
    device_buffer memory{4099};
    configuration_storage() : expected(4099) {
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = static_cast<unsigned char>((i * 47 + (i >> 3) * 19 + 101) & 255);
        CONFORMANCE_CUDA(cuMemcpyHtoD(memory.value, expected.data(), expected.size()));
    }
    void check() const {
        std::vector<unsigned char> bytes(expected.size() + 32, 0xa7);
        CONFORMANCE_CUDA(cuMemcpyDtoH(bytes.data() + 16, memory.value, expected.size()));
        for (size_t i = 0; i < bytes.size(); ++i)
            require(bytes[i] == (i >= 16 && i < 16 + expected.size() ? expected[i - 16] : 0xa7), "configuration transitions preserve complete allocation and readback guards");
    }
};
void configuration_scheduling() {
    configuration_context owner;
    configuration_storage storage;
    const auto fixed = configuration_flags() & unsigned(CU_CTX_MAP_HOST);
    for (const unsigned schedule : {CU_CTX_SCHED_AUTO, CU_CTX_SCHED_SPIN, CU_CTX_SCHED_YIELD, CU_CTX_SCHED_BLOCKING_SYNC}) {
        for (bool runtime : {false, true}) {
            if (runtime) CONFIGURATION_RUNTIME(cudaSetDeviceFlags(schedule));
            else CONFORMANCE_CUDA(cuCtxSetFlags(schedule));
            require(configuration_flags() == (schedule | fixed), "scheduling updates preserve creation-time map-host state");
            CONFORMANCE_CUDA(cuCtxSynchronize()); storage.check();
            require(configuration_current() == owner.value, "configuration leaves explicit context bound");
        }
    }
}
void configuration_limits() {
    configuration_context owner;
    configuration_storage storage;
    for (const auto& [driver_limit, runtime_limit] : std::array<std::pair<CUlimit, cudaLimit>, 2>{{
        {CU_LIMIT_MALLOC_HEAP_SIZE, cudaLimitMallocHeapSize}, {CU_LIMIT_PRINTF_FIFO_SIZE, cudaLimitPrintfFifoSize}}}) {
        require(configuration_limit(driver_limit, runtime_limit) > 0, "nonzero initial heap/FIFO capacity");
        for (const size_t request : {size_t{65537}, size_t{131075}, size_t{262151}}) {
            CONFORMANCE_CUDA(cuCtxSetLimit(driver_limit, request));
            require(configuration_limit(driver_limit, runtime_limit) > 0, "Driver request yields queryable effective capacity");
            CONFIGURATION_RUNTIME(cudaDeviceSetLimit(runtime_limit, request));
            require(configuration_limit(driver_limit, runtime_limit) > 0, "Runtime request yields queryable effective capacity");
            storage.check();
        }
    }
}
void configuration_preferences() {
    configuration_context owner;
    configuration_storage storage;
    for (const auto preference : {CU_FUNC_CACHE_PREFER_NONE, CU_FUNC_CACHE_PREFER_SHARED, CU_FUNC_CACHE_PREFER_L1, CU_FUNC_CACHE_PREFER_EQUAL}) {
        CONFORMANCE_CUDA(cuCtxSetCacheConfig(preference));
        const auto effective = configuration_cache();
        CONFIGURATION_RUNTIME(cudaDeviceSetCacheConfig(static_cast<cudaFuncCache>(preference)));
        require(configuration_cache() == effective, "same cache request through either facade has same effective preference");
        storage.check();
    }
}
void configuration_isolation() {
    configuration_context first;
    CONFORMANCE_CUDA(cuCtxSetFlags(CU_CTX_SCHED_YIELD));
    CONFORMANCE_CUDA(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, 131075));
    CONFORMANCE_CUDA(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 65537));
    const auto flags = configuration_flags();
    const auto heap = configuration_limit(CU_LIMIT_MALLOC_HEAP_SIZE, cudaLimitMallocHeapSize);
    const auto fifo = configuration_limit(CU_LIMIT_PRINTF_FIFO_SIZE, cudaLimitPrintfFifoSize);
    const auto cache = configuration_cache();
    configuration_storage first_storage;
    {
        configuration_context second;
        configuration_storage second_storage;
        CONFIGURATION_RUNTIME(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        CONFIGURATION_RUNTIME(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 262151));
        CONFIGURATION_RUNTIME(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 131075));
        CONFIGURATION_RUNTIME(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
        const auto second_flags = configuration_flags();
        const auto second_heap = configuration_limit(CU_LIMIT_MALLOC_HEAP_SIZE, cudaLimitMallocHeapSize);
        const auto second_fifo = configuration_limit(CU_LIMIT_PRINTF_FIFO_SIZE, cudaLimitPrintfFifoSize);
        const auto second_cache = configuration_cache();
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            first.bind();
            require(configuration_flags() == flags && configuration_limit(CU_LIMIT_MALLOC_HEAP_SIZE, cudaLimitMallocHeapSize) == heap
                && configuration_limit(CU_LIMIT_PRINTF_FIFO_SIZE, cudaLimitPrintfFifoSize) == fifo && configuration_cache() == cache, "first context retains independent configuration");
            first_storage.check();
            second.bind();
            require(configuration_flags() == second_flags && configuration_limit(CU_LIMIT_MALLOC_HEAP_SIZE, cudaLimitMallocHeapSize) == second_heap
                && configuration_limit(CU_LIMIT_PRINTF_FIFO_SIZE, cudaLimitPrintfFifoSize) == second_fifo && configuration_cache() == second_cache, "second context retains independent configuration");
            second_storage.check();
        }
    }
    require(configuration_current() == first.value && configuration_flags() == flags, "retiring another context preserves first configuration");
    first_storage.check();
}
void configuration_validation() {
    configuration_context owner;
    configuration_storage storage;
    const auto flags = configuration_flags();
    const auto heap = configuration_limit(CU_LIMIT_MALLOC_HEAP_SIZE, cudaLimitMallocHeapSize);
    const auto cache = configuration_cache();
    for (const unsigned invalid : {3u, 5u, 0x80000000u}) {
        require(cuCtxSetFlags(invalid) == CUDA_ERROR_INVALID_VALUE && cudaSetDeviceFlags(invalid) == cudaErrorInvalidValue, "invalid scheduling flags rejected");
        require(configuration_flags() == flags, "invalid flags preserve configured state");
    }
    require(cuCtxGetFlags(nullptr) == CUDA_ERROR_INVALID_VALUE && cudaGetDeviceFlags(nullptr) == cudaErrorInvalidValue, "null flags outputs");
    require(cuCtxGetLimit(nullptr, CU_LIMIT_MALLOC_HEAP_SIZE) == CUDA_ERROR_INVALID_VALUE
        && cudaDeviceGetLimit(nullptr, cudaLimitMallocHeapSize) == cudaErrorInvalidValue, "null limit outputs");
    size_t limit{};
    for (int invalid : {-1, 0x7fffffff}) {
        require(cuCtxGetLimit(&limit, static_cast<CUlimit>(invalid)) == CUDA_ERROR_INVALID_VALUE
            && cuCtxSetLimit(static_cast<CUlimit>(invalid), 65536) == CUDA_ERROR_INVALID_VALUE, "invalid Driver limit enum");
        require(cudaDeviceGetLimit(&limit, static_cast<cudaLimit>(invalid)) == cudaErrorInvalidValue
            && cudaDeviceSetLimit(static_cast<cudaLimit>(invalid), 65536) == cudaErrorInvalidValue, "invalid Runtime limit enum");
        require(cuCtxSetCacheConfig(static_cast<CUfunc_cache>(invalid)) == CUDA_ERROR_INVALID_VALUE
            && cudaDeviceSetCacheConfig(static_cast<cudaFuncCache>(invalid)) == cudaErrorInvalidValue, "invalid cache preference enum");
        require(cuCtxSetSharedMemConfig(static_cast<CUsharedconfig>(invalid)) == CUDA_ERROR_INVALID_VALUE
            && cudaDeviceSetSharedMemConfig(static_cast<cudaSharedMemConfig>(invalid)) == cudaErrorInvalidValue, "invalid bank preference enum");
    }
    require(cuCtxGetCacheConfig(nullptr) == CUDA_ERROR_INVALID_VALUE && cudaDeviceGetCacheConfig(nullptr) == cudaErrorInvalidValue, "null cache preference outputs");
    require(cuCtxGetSharedMemConfig(nullptr) == CUDA_ERROR_INVALID_VALUE && cudaDeviceGetSharedMemConfig(nullptr) == cudaErrorInvalidValue, "null bank preference outputs");
    require(configuration_flags() == flags && configuration_limit(CU_LIMIT_MALLOC_HEAP_SIZE, cudaLimitMallocHeapSize) == heap
        && configuration_cache() == cache, "rejected configuration changes preserve valid state");
    storage.check();
    require(cudaGetLastError() == cudaErrorInvalidValue, "clear expected configuration validation error");
}
std::string_view device_configuration_case = "all";
void execute_device_configuration() {
    const auto parent = configuration_current();
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"scheduling", configuration_scheduling}, {"limits", configuration_limits}, {"preferences", configuration_preferences},
        {"isolation", configuration_isolation}, {"validation", configuration_validation}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (device_configuration_case == "all" || device_configuration_case == name) {
        selected = true; execute(); require(configuration_current() == parent, "configuration case restores parent context");
    }
    require(selected, "unknown device_configuration case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::device_configuration_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_device_configuration);
}
