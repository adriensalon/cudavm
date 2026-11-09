#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

namespace cudavm {
std::string_view asynchronous_error_case;
void asynchronous_runtime(cudaError_t status, const char* operation) {
    require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}
#define ASYNC_RUNTIME(call) ::cudavm::asynchronous_runtime((call), #call)
void asynchronous_compile(nvrtcResult status, nvrtcProgram program) {
    if (status == NVRTC_SUCCESS) return;
    size_t size = 0;
    nvrtcGetProgramLogSize(program, &size);
    std::string log(size, '\0');
    if (size) nvrtcGetProgramLog(program, log.data());
    throw std::runtime_error(std::string(nvrtcGetErrorString(status)) + ": " + log);
}
// This probe is independent of assertion compilation and fault observation.
void asynchronous_probe() {
    constexpr char ptx[] = R"(.version 6.5
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 output) {
 .reg .u64 p; ld.param.u64 p,[output]; st.global.u32 [p],0x43564d31; ret;
})";
    CUmodule module{};
    CONFORMANCE_CUDA(cuModuleLoadData(&module, ptx));
    CUfunction function{};
    CONFORMANCE_CUDA(cuModuleGetFunction(&function, module, "probe"));
    {
        device_buffer output(4);
        void* args[]{&output.value};
        CONFORMANCE_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
        uint32_t actual = 0;
        CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, output.value, 4));
        require(actual == 0x43564d31, "independent execution probe result");
    }
    CONFORMANCE_CUDA(cuModuleUnload(module));
}
struct asynchronous_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB hold(void* pointer) {
        auto& gate = *static_cast<asynchronous_gate*>(pointer);
        std::unique_lock lock(gate.mutex);
        gate.entered = true;
        gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "fault stream reaches host gate");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    ~asynchronous_gate() { release(); }
};
CUresult asynchronous_poll(auto query) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    CUresult status;
    do {
        status = query();
        if (status != CUDA_ERROR_NOT_READY) return status;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    throw std::runtime_error("fault query did not complete before its deadline");
}
void execute_asynchronous_errors() {
    require(asynchronous_error_case == "stream" || asynchronous_error_case == "event"
        || asynchronous_error_case == "context" || asynchronous_error_case == "query"
        || asynchronous_error_case == "dependencies" || asynchronous_error_case == "library", "unknown asynchronous_errors case");
    constexpr char source[] = R"(
extern "C" __device__ void __assertfail(const char*, const char*, unsigned, const char*, __SIZE_TYPE__);
extern "C" __global__ void deferred_assert(unsigned* output, unsigned fail) {
    if (fail) { __assertfail("asynchronous_errors witness", "asynchronous_errors.cu", 4, "deferred_assert", 1); }
    else output[0] = 0x1234abcd;
}
)";
    nvrtcProgram compiler{};
    asynchronous_compile(nvrtcCreateProgram(&compiler, source, "asynchronous_errors.cu", 0, nullptr, nullptr), compiler);
    try {
        const char* options[]{"--std=c++17", "-arch=sm_75"};
        asynchronous_compile(nvrtcCompileProgram(compiler, 2, options), compiler);
        size_t size = 0;
        asynchronous_compile(nvrtcGetCUBINSize(compiler, &size), compiler);
        require(size > 0, "assertion compilation produces an owned image");
        std::vector<char> image(size);
        asynchronous_compile(nvrtcGetCUBIN(compiler, image.data()), compiler);
        asynchronous_compile(nvrtcDestroyProgram(&compiler), nullptr);
        CUmodule module{};
        CUlibrary library{};
        CUfunction function{};
        if (asynchronous_error_case == "library") {
            CONFORMANCE_CUDA(cuLibraryLoadData(&library, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
            CUkernel kernel{};
            CONFORMANCE_CUDA(cuLibraryGetKernel(&kernel, library, "deferred_assert"));
            CONFORMANCE_CUDA(cuKernelGetFunction(&function, kernel));
        } else {
            CONFORMANCE_CUDA(cuModuleLoadData(&module, image.data()));
            CONFORMANCE_CUDA(cuModuleGetFunction(&function, module, "deferred_assert"));
        }
        CUstream producer{}, consumer{};
        CUevent ready{};
        CONFORMANCE_CUDA(cuStreamCreate(&producer, CU_STREAM_NON_BLOCKING));
        CONFORMANCE_CUDA(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING));
        CONFORMANCE_CUDA(cuEventCreate(&ready, CU_EVENT_DISABLE_TIMING));
        CUdeviceptr output = 0;
        CONFORMANCE_CUDA(cuMemAlloc(&output, 4));
        unsigned fail = 0;
        void* args[]{&output, &fail};
        CONFORMANCE_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, producer, args, nullptr));
        CONFORMANCE_CUDA(cuStreamSynchronize(producer));
        uint32_t actual = 0;
        CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, output, 4));
        require(actual == 0x1234abcd, "same kernel completes normally before fault submission");
        ASYNC_RUNTIME(cudaGetLastError());
        // Callback userdata must survive even a broken completion implementation
        // that reports an error before the callback returns. This disposable
        // process owns the state until immediate exit, including failure paths.
        auto& gate = *new asynchronous_gate;
        CONFORMANCE_CUDA(cuLaunchHostFunc(producer, asynchronous_gate::hold, &gate));
        gate.await();
        fail = 1;
        auto& downstream_calls = *new std::atomic<unsigned>{0};
        try {
            CONFORMANCE_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, producer, args, nullptr));
            CONFORMANCE_CUDA(cuEventRecord(ready, producer));
            CONFORMANCE_CUDA(cuStreamWaitEvent(consumer, ready, 0));
            CONFORMANCE_CUDA(cuLaunchHostFunc(consumer, [](void* value) { ++*static_cast<std::atomic<unsigned>*>(value); }, &downstream_calls));
            require(cuStreamQuery(producer) == CUDA_ERROR_NOT_READY, "fault remains pending behind gate");
            require(cuEventQuery(ready) == CUDA_ERROR_NOT_READY, "fault event remains pending behind gate");
            require(cudaStreamQuery(reinterpret_cast<cudaStream_t>(consumer)) == cudaErrorNotReady, "dependent stream remains pending");
            require(cudaGetLastError() == cudaErrorNotReady, "clear pending Runtime query status");
        } catch (...) { gate.release(); throw; }
        gate.release();
        CUresult observed = CUDA_SUCCESS;
        if (asynchronous_error_case == "event") observed = cuEventSynchronize(ready);
        else if (asynchronous_error_case == "context") observed = cuCtxSynchronize();
        else if (asynchronous_error_case == "query") observed = asynchronous_poll([&] { return cuStreamQuery(producer); });
        else if (asynchronous_error_case == "dependencies") observed = cuStreamSynchronize(consumer);
        else observed = cuStreamSynchronize(producer);
        require(observed == CUDA_ERROR_ASSERT, "selected completion point reports the exact assertion status");
        { std::lock_guard lock(gate.mutex); require(!gate.expired, "host gate released before its deadline"); }
        require(cuStreamSynchronize(producer) == CUDA_ERROR_ASSERT, "producer completion retains assertion");
        require(cuEventSynchronize(ready) == CUDA_ERROR_ASSERT, "event completion retains assertion");
        require(cuStreamSynchronize(consumer) == CUDA_ERROR_ASSERT, "dependent completion propagates assertion");
        require(cuEventQuery(ready) == CUDA_ERROR_ASSERT && cuStreamQuery(producer) == CUDA_ERROR_ASSERT,
                "completed queries report assertion instead of success or pending");
        require(downstream_calls == 0, "host function after failed event dependency is suppressed");
        require(cudaPeekAtLastError() == cudaSuccess, "Driver observation does not modify Runtime last error");
        require(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(ready)) == cudaErrorAssert, "Runtime event translates assertion");
        require(cudaPeekAtLastError() == cudaErrorAssert && cudaGetLastError() == cudaErrorAssert
            && cudaPeekAtLastError() == cudaSuccess, "Runtime inspection clears only the host error slot");
        require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(consumer)) == cudaErrorAssert, "clearing last error does not recover device completion");
        require(cudaGetLastError() == cudaErrorAssert, "clear second Runtime assertion observation");
        require(cudaDeviceSynchronize() == cudaErrorAssert && cuCtxSynchronize() == CUDA_ERROR_ASSERT,
                "both context synchronization surfaces retain the device fault");
        // Native assertions can require a new process. The isolated runner
        // flushes diagnostics and exits without attempting CUDA destruction.
    } catch (...) { if (compiler) nvrtcDestroyProgram(&compiler); throw; }
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::asynchronous_error_case = argv[2]; --argc; }
    cudavm::run_isolated_fault_conformance(argc, argv, cudavm::execute_asynchronous_errors, cudavm::asynchronous_probe);
}
