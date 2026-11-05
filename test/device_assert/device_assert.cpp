#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace cudavm {
bool assert_library = false;
struct assert_gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB hold(void* pointer) {
        auto& gate = *static_cast<assert_gate*>(pointer);
        std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "assertion stream reaches host gate");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
[[noreturn]] void finish_assert_worker(int status) {
    std::cout.flush(); std::cerr.flush(); std::fflush(nullptr);
    std::_Exit(status);
}
void execute_device_assert(const loaded_program& program) {
    // A library worker passes the clean module load before executing its fault.
    // Once the selected fault starts, every path exits without CUDA destruction.
    if (bool(program.library_kernel("assert_checked")) != assert_library) return;
    // Callback storage lives until immediate process exit, including failures.
    auto& gate = *new assert_gate;
    try {
        constexpr uint32_t guard = 0xa1b2c3d4;
        constexpr unsigned guards = 4, capacity = 40;
        CUdeviceptr allocation = 0;
        CONFORMANCE_CUDA(cuMemAlloc(&allocation, (capacity + guards * 2) * 4));
        auto output = allocation + guards * 4;
        auto function = program.function("assert_checked");
        CUstream stream{};
        CUevent done{};
        CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CONFORMANCE_CUDA(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING));
        std::vector<uint32_t> expected(capacity + guards * 2, guard), actual(expected.size());
        unsigned fail = 0;
        for (unsigned count : {0u, 1u, 17u, 39u}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(allocation, expected.data(), expected.size() * 4));
            void* args[]{&output, &count, &fail};
            if (assert_library) {
                const auto kernel = program.library_kernel("assert_checked");
                require(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(3), dim3(13), args, 0,
                    reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess, "Runtime launches clean assertion kernel");
            } else CONFORMANCE_CUDA(cuLaunchKernel(function, 3, 1, 1, 13, 1, 1, 0, stream, args, nullptr));
            CONFORMANCE_CUDA(cuStreamSynchronize(stream));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocation, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) expected[guards + i] = i * 37 + count;
            expect_words(actual, expected, "nonfailing assertion branch, inactive rows and allocation guards");
        }
        require(cudaGetLastError() == cudaSuccess, "clean Runtime status before assertion");
        unsigned count = 17; fail = 1;
        void* args[]{&output, &count, &fail};
        CONFORMANCE_CUDA(cuLaunchHostFunc(stream, assert_gate::hold, &gate));
        gate.await();
        CONFORMANCE_CUDA(cuLaunchKernel(function, 3, 1, 1, 13, 1, 1, 0, stream, args, nullptr));
        CONFORMANCE_CUDA(cuEventRecord(done, stream));
        require(cuEventQuery(done) == CUDA_ERROR_NOT_READY, "assertion event remains pending behind the gate");
        gate.release();
        require(cuStreamSynchronize(stream) == CUDA_ERROR_ASSERT, "assertion faults the producing stream");
        { std::lock_guard lock(gate.mutex); require(!gate.expired, "assertion host gate released before its deadline"); }
        require(cuCtxSynchronize() == CUDA_ERROR_ASSERT && cuEventSynchronize(done) == CUDA_ERROR_ASSERT,
            "assertion faults context and recorded event completion");
        require(cuStreamQuery(stream) == CUDA_ERROR_ASSERT && cuEventQuery(done) == CUDA_ERROR_ASSERT,
            "completed assertion queries report the fault");
        require(cudaPeekAtLastError() == cudaSuccess, "Driver assertion observations leave Runtime last error alone");
        require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaErrorAssert,
            "Runtime observes the same stream assertion");
        require(cudaPeekAtLastError() == cudaErrorAssert && cudaGetLastError() == cudaErrorAssert
            && cudaPeekAtLastError() == cudaSuccess, "Runtime can inspect and clear the host error slot");
        require(cudaDeviceSynchronize() == cudaErrorAssert && cuCtxSynchronize() == CUDA_ERROR_ASSERT,
            "clearing host last error does not recover the assertion context");
        std::cout << "PASS: isolated assertion " << (assert_library ? "library" : "module") << std::endl;
        finish_assert_worker(0);
    } catch (const std::exception& error) {
        gate.release();
        std::cerr << "FAIL: " << error.what() << '\n';
        finish_assert_worker(1);
    }
}
}
int main(int argc, char** argv) {
    if (argc == 4 && std::string_view(argv[3]) == "--worker") {
        cudavm::assert_library = std::string_view(argv[2]) == "library";
        char source_importer[] = "cu";
        char* arguments[]{argv[0], argv[1], source_importer};
        return cudavm::run_conformance(3, arguments, cudavm::fixtures, cudavm::execute_device_assert);
    }
    try {
        cudavm::require(argc == 3 && std::string_view(argv[2]) == "cu", "usage: <executable> <backend> cu");
        const std::array<std::string_view, 5> backends{"cuda", "vulkan", "directx", "opencl", "opengl"};
        cudavm::require(std::find(backends.begin(), backends.end(), argv[1]) != backends.end(), "known backend");
        for (const char* loader : {"module", "library"}) {
            const auto child = cudavm::run_conformance_child(argv[0], argv[1], loader, true);
            std::cout << child.output;
            if (child.status != 0) return child.status;
            for (const char* token : {"device_assert exact witness", "device_assert.cu:75", "assert_checked"})
                cudavm::require(child.output.find(token) != std::string::npos, std::string("assertion diagnostic contains ") + token);
            cudavm::require(child.output.find(std::string("PASS: isolated assertion ") + loader) != std::string::npos,
                "selected loader completed the fault oracle without SDK teardown");
        }
        std::cout << "PASS: assertions observed in disposable workers\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
