#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

namespace cudavm {
void check_last_error_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define LAST_ERROR_RUNTIME(call) ::cudavm::check_last_error_runtime((call), #call)
void expect_runtime_error(cudaError_t expected, const std::string& operation) {
    for (unsigned repeat = 0; repeat < 3; ++repeat)
        require(cudaPeekAtLastError() == expected, operation + ": nondestructive peek");
}
void produce_runtime_error(cudaError_t expected) {
    const auto status = expected == cudaErrorInvalidValue ? cudaGetDeviceCount(nullptr) : cudaSetDevice(-1);
    require(status == expected, "defined validation failure status");
    expect_runtime_error(expected, "failed Runtime call records error");
}
void clear_runtime_error(cudaError_t expected) {
    require(cudaGetLastError() == expected, "get returns the preceding last error");
    expect_runtime_error(cudaSuccess, "get clears pending error");
    require(cudaGetLastError() == cudaSuccess, "repeated clear returns success");
}
void last_error_peek_clear() {
    expect_runtime_error(cudaSuccess, "initial last error");
    clear_runtime_error(cudaSuccess);
    for (unsigned repeat = 0; repeat < 4; ++repeat) {
        produce_runtime_error(cudaErrorInvalidValue);
        expect_runtime_error(cudaErrorInvalidValue, "repeated inspection preserves failure");
        clear_runtime_error(cudaErrorInvalidValue);
    }
}
void last_error_replacement() {
    clear_runtime_error(cudaSuccess);
    for (const auto error : {cudaErrorInvalidValue, cudaErrorInvalidDevice, cudaErrorInvalidValue,
                            cudaErrorInvalidDevice, cudaErrorInvalidDevice}) {
        produce_runtime_error(error);
        for (auto inspected : {cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice}) {
            const auto* name = cudaGetErrorName(inspected);
            const auto* description = cudaGetErrorString(inspected);
            require(name && *name && description && *description, "known Runtime error inspection provides strings");
            expect_runtime_error(error, "error inspection preserves pending failure");
        }
    }
    clear_runtime_error(cudaErrorInvalidDevice);
}
struct last_error_storage {
    void* value{};
    last_error_storage() { LAST_ERROR_RUNTIME(cudaMalloc(&value, 257)); }
    ~last_error_storage() { if (value) cudaFree(value); }
};
struct last_error_stream {
    cudaStream_t value{};
    last_error_stream() { LAST_ERROR_RUNTIME(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~last_error_stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};
void last_error_success() {
    clear_runtime_error(cudaSuccess);
    produce_runtime_error(cudaErrorInvalidValue);
    int count = 0, device = -1;
    LAST_ERROR_RUNTIME(cudaGetDeviceCount(&count));
    require(count > 0, "selected backend remains discoverable");
    expect_runtime_error(cudaErrorInvalidValue, "successful inventory query");
    LAST_ERROR_RUNTIME(cudaGetDevice(&device));
    require(device >= 0 && device < count, "successful current-device query");
    expect_runtime_error(cudaErrorInvalidValue, "successful current-device query");
    {
        last_error_storage storage;
        expect_runtime_error(cudaErrorInvalidValue, "successful allocation");
        last_error_stream stream;
        expect_runtime_error(cudaErrorInvalidValue, "successful stream creation");
        LAST_ERROR_RUNTIME(cudaMemsetAsync(storage.value, 0x5b, 257, stream.value));
        expect_runtime_error(cudaErrorInvalidValue, "successful asynchronous fill submission");
        LAST_ERROR_RUNTIME(cudaStreamSynchronize(stream.value));
        expect_runtime_error(cudaErrorInvalidValue, "successful stream completion");
        std::array<unsigned char, 289> actual;
        actual.fill(0xd3);
        LAST_ERROR_RUNTIME(cudaMemcpy(actual.data() + 16, storage.value, 257, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i)
            require(actual[i] == (i >= 16 && i < 273 ? 0x5b : 0xd3), "successful readback payload and guards");
        expect_runtime_error(cudaErrorInvalidValue, "successful readback");
        LAST_ERROR_RUNTIME(cudaStreamDestroy(stream.value)); stream.value = nullptr;
        expect_runtime_error(cudaErrorInvalidValue, "successful stream destruction");
        LAST_ERROR_RUNTIME(cudaFree(storage.value)); storage.value = nullptr;
        expect_runtime_error(cudaErrorInvalidValue, "successful allocation free");
    }
    expect_runtime_error(cudaErrorInvalidValue, "successful resource retirement");
    clear_runtime_error(cudaErrorInvalidValue);
}
void last_error_driver() {
    clear_runtime_error(cudaSuccess);
    produce_runtime_error(cudaErrorInvalidDevice);
    CUcontext context{};
    CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
    require(context != nullptr, "Driver context remains current");
    expect_runtime_error(cudaErrorInvalidDevice, "successful Driver query");
    require(cuMemAlloc(nullptr, 64) == CUDA_ERROR_INVALID_VALUE, "defined Driver validation failure");
    expect_runtime_error(cudaErrorInvalidDevice, "Driver failure preserves Runtime last error");
    CONFORMANCE_CUDA(cuCtxSynchronize());
    expect_runtime_error(cudaErrorInvalidDevice, "Driver synchronization preserves Runtime last error");
    clear_runtime_error(cudaErrorInvalidDevice);
    require(cuMemAlloc(nullptr, 64) == CUDA_ERROR_INVALID_VALUE, "Driver failure after Runtime clear");
    expect_runtime_error(cudaSuccess, "Driver failure does not create Runtime error");
}
struct last_error_gate {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned ready = 0;
    bool released = false;
    void enter() {
        std::unique_lock lock(mutex);
        ++ready; changed.notify_all();
        require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return released; }), "last-error worker release deadline");
    }
    void await() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return ready == 2; }), "last-error worker entry deadline");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
void last_error_threading() {
    clear_runtime_error(cudaSuccess);
    produce_runtime_error(cudaErrorInvalidValue);
    last_error_gate gate;
    std::array<std::exception_ptr, 2> errors;
    auto work = [&](unsigned index, cudaError_t expected) {
        try {
            expect_runtime_error(cudaSuccess, "fresh worker starts with success");
            produce_runtime_error(expected);
            gate.enter();
            expect_runtime_error(expected, "parent operations preserve worker failure");
            for (unsigned repeat = 0; repeat < 32; ++repeat) {
                clear_runtime_error(expected);
                produce_runtime_error(expected);
                int count = 0;
                LAST_ERROR_RUNTIME(cudaGetDeviceCount(&count));
                expect_runtime_error(expected, "worker successful API preserves its error");
            }
            clear_runtime_error(expected);
        } catch (...) { errors[index] = std::current_exception(); }
    };
    std::jthread first([&] { work(0, cudaErrorInvalidValue); }), second([&] { work(1, cudaErrorInvalidDevice); });
    struct release_workers { last_error_gate& gate; ~release_workers() { gate.release(); } } cleanup{gate};
    gate.await();
    expect_runtime_error(cudaErrorInvalidValue, "worker failures preserve parent error");
    clear_runtime_error(cudaErrorInvalidValue);
    produce_runtime_error(cudaErrorInvalidDevice);
    gate.release();
    first.join(); second.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    expect_runtime_error(cudaErrorInvalidDevice, "worker clears preserve parent pending failure");
    clear_runtime_error(cudaErrorInvalidDevice);
    std::exception_ptr fresh_error;
    std::jthread fresh([&] {
        try { expect_runtime_error(cudaSuccess, "later fresh thread does not inherit retired thread errors"); }
        catch (...) { fresh_error = std::current_exception(); }
    });
    fresh.join();
    if (fresh_error) std::rethrow_exception(fresh_error);
    expect_runtime_error(cudaSuccess, "later thread preserves parent success");
}
std::string_view last_error_case = "all";
void execute_last_error() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"peek_clear", last_error_peek_clear}, {"replacement", last_error_replacement}, {"success", last_error_success},
        {"driver", last_error_driver}, {"threading", last_error_threading}}};
    bool found = false;
    for (const auto& [name, execute] : cases) if (last_error_case == "all" || last_error_case == name) {
        found = true;
        std::cout << "last_error=" << name << std::endl;
        execute();
    }
    require(found, "unknown last_error case");
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::last_error_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_last_error);
}
