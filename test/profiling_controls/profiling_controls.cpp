#include "test_framework.hpp"
#include <cudaProfiler.h>
#include <cudaProfilerTypedefs.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <algorithm>

namespace cudavm {
std::string_view profiling_case;
void profiling_runtime(cudaError_t status) {
    require(status == cudaSuccess, std::string("Runtime profiler control: ") + cudaGetErrorString(status));
}
void profiling_transitions() {
    CONFORMANCE_CUDA(cuProfilerStop()); CONFORMANCE_CUDA(cuProfilerStop());
    profiling_runtime(cudaProfilerStart()); CONFORMANCE_CUDA(cuProfilerStart());
    profiling_runtime(cudaProfilerStart()); CONFORMANCE_CUDA(cuProfilerStop());
    profiling_runtime(cudaProfilerStop()); profiling_runtime(cudaProfilerStop());
    CONFORMANCE_CUDA(cuProfilerStart()); profiling_runtime(cudaProfilerStop());
}
void profiling_ordering() {
    constexpr size_t bytes = 257;
    std::vector<unsigned char> expected(bytes, 0xa5), actual(bytes);
    device_buffer input(bytes), output(bytes);
    CONFORMANCE_CUDA(cuMemcpyHtoD(input.value, expected.data(), bytes));
    CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), bytes));
    CUstream stream{};
    CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    try {
        for (unsigned phase = 0; phase < 3; ++phase) {
            if (phase & 1) profiling_runtime(cudaProfilerStart()); else CONFORMANCE_CUDA(cuProfilerStart());
            const unsigned char pattern = static_cast<unsigned char>(0x31 + phase * 0x23);
            CONFORMANCE_CUDA(cuMemsetD8Async(input.value + 13, pattern, bytes - 29, stream));
            CONFORMANCE_CUDA(cuMemcpyDtoDAsync(output.value + 7, input.value + 7, bytes - 19, stream));
            if (phase & 1) CONFORMANCE_CUDA(cuProfilerStop()); else profiling_runtime(cudaProfilerStop());
            CONFORMANCE_CUDA(cuStreamSynchronize(stream));
            std::fill(expected.begin() + 13, expected.end() - 16, pattern);
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), input.value, bytes)); require(actual == expected, "profiling preserves full source payload and guards");
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, bytes)); require(actual == expected, "profiling preserves ordered copied payload and guards");
        }
    } catch (...) { cuProfilerStop(); cuStreamSynchronize(stream); cuStreamDestroy(stream); throw; }
    CONFORMANCE_CUDA(cuStreamDestroy(stream));
}
void profiling_contexts() {
    CUcontext original{}, separate{}, current{}; CUdevice device{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&original)); CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    device_buffer first(4); CONFORMANCE_CUDA(cuMemsetD32(first.value, 0x12345678, 1));
    CONFORMANCE_CUDA(cuProfilerStart());
    CONFORMANCE_CUDA(cuCtxCreate(&separate, nullptr, 0, device));
    try {
        {
            device_buffer second(4); CONFORMANCE_CUDA(cuMemsetD32(second.value, 0x87654321, 1));
            profiling_transitions();
            CONFORMANCE_CUDA(cuCtxGetCurrent(&current)); require(current == separate, "controls preserve second current context");
            uint32_t actual = 0; CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, second.value, 4)); require(actual == 0x87654321, "second context allocation retained");
        }
        CONFORMANCE_CUDA(cuCtxSetCurrent(original)); profiling_runtime(cudaProfilerStop());
        CONFORMANCE_CUDA(cuCtxGetCurrent(&current)); require(current == original, "controls preserve restored current context");
        uint32_t actual = 0; CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, first.value, 4)); require(actual == 0x12345678, "first context allocation retained");
    } catch (...) { cuCtxSetCurrent(original); cuProfilerStop(); cuCtxDestroy(separate); throw; }
    CONFORMANCE_CUDA(cuCtxDestroy(separate));
}
void profiling_validation() {
    CUcontext original{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&original));
    CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr));
    try {
        require(cuProfilerStart() == CUDA_ERROR_INVALID_CONTEXT && cuProfilerStop() == CUDA_ERROR_INVALID_CONTEXT, "initialized controls require a current Driver context");
    } catch (...) { cuCtxSetCurrent(original); throw; }
    CONFORMANCE_CUDA(cuCtxSetCurrent(original));
    profiling_transitions();
}
void profiling_cold() {
    require(cuProfilerStart() == CUDA_ERROR_NOT_INITIALIZED && cuProfilerStop() == CUDA_ERROR_NOT_INITIALIZED, "cold Driver controls require initialization");
    for (const char* name : {"cuProfilerStart", "cuProfilerStop"}) {
        void* address{}; CUdriverProcAddressQueryResult status{};
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, 4000, 0, &status));
        require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "profiler control discovered at its pinned ABI version");
        require(reinterpret_cast<PFN_cuProfilerStart_v4000>(address)() == CUDA_ERROR_NOT_INITIALIZED, "discovered profiler control preserves cold behavior");
        CONFORMANCE_CUDA(cuGetProcAddress(name, &address, 3999, 0, &status));
        require(!address && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "profiler discovery rejects an older requested ABI");
    }
    require(cuProfilerStart() == CUDA_ERROR_NOT_INITIALIZED, "discovery does not initialize the Driver");
}
void execute_profiling_controls() {
    if (profiling_case == "transitions") profiling_transitions();
    else if (profiling_case == "ordering") profiling_ordering();
    else if (profiling_case == "contexts") profiling_contexts();
    else if (profiling_case == "validation") profiling_validation();
    else throw std::runtime_error("unknown profiling_controls case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "cold") return cudavm::run_host_conformance(argc, argv, cudavm::profiling_cold);
    if (argc == 3) { cudavm::profiling_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_profiling_controls);
}
