#include "test_framework.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <exception>
#include <thread>
#include <utility>

namespace cudavm {
template<class T> struct error_query_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    error_query_output() { before.fill(0xe9); after.fill(0x43); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0xe9; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x43; }), "error query pointer output guards");
        return value;
    }
};
void errors_uninitialized() {
    int count = -1;
    require(cuDeviceGetCount(&count) == CUDA_ERROR_NOT_INITIALIZED, "host error-query profile does not initialize Driver");
}
void errors_driver() {
#define DRIVER_ERROR_CASE(code) std::pair<CUresult, const char*>{code, #code}
    const std::array cases{DRIVER_ERROR_CASE(CUDA_SUCCESS), DRIVER_ERROR_CASE(CUDA_ERROR_INVALID_VALUE), DRIVER_ERROR_CASE(CUDA_ERROR_OUT_OF_MEMORY),
        DRIVER_ERROR_CASE(CUDA_ERROR_NOT_INITIALIZED), DRIVER_ERROR_CASE(CUDA_ERROR_INVALID_DEVICE), DRIVER_ERROR_CASE(CUDA_ERROR_INVALID_CONTEXT),
        DRIVER_ERROR_CASE(CUDA_ERROR_INVALID_HANDLE), DRIVER_ERROR_CASE(CUDA_ERROR_NOT_READY), DRIVER_ERROR_CASE(CUDA_ERROR_NOT_SUPPORTED)};
#undef DRIVER_ERROR_CASE
    std::vector<std::pair<const char*, std::string>> retained;
    for (const auto& [code, expected] : cases) {
        error_query_output<const char*> name, description;
        require(cuGetErrorName(code, &name.value) == CUDA_SUCCESS && cuGetErrorString(code, &description.value) == CUDA_SUCCESS, "recognized Driver error queries succeed before initialization");
        require(name.get() && std::string(name.get()) == expected, "Driver error name is exact enum spelling");
        require(description.get() && *description.get(), "recognized Driver error has nonempty description");
        retained.emplace_back(name.get(), name.get()); retained.emplace_back(description.get(), description.get());
    }
    for (const auto& [pointer, copy] : retained) require(std::string(pointer) == copy, "borrowed Driver error strings survive other error queries");
}
void errors_runtime() {
#define RUNTIME_ERROR_CASE(code) std::pair<cudaError_t, const char*>{code, #code}
    const std::array cases{RUNTIME_ERROR_CASE(cudaSuccess), RUNTIME_ERROR_CASE(cudaErrorInvalidValue), RUNTIME_ERROR_CASE(cudaErrorMemoryAllocation),
        RUNTIME_ERROR_CASE(cudaErrorInitializationError), RUNTIME_ERROR_CASE(cudaErrorInvalidDevice), RUNTIME_ERROR_CASE(cudaErrorInvalidResourceHandle),
        RUNTIME_ERROR_CASE(cudaErrorNotReady), RUNTIME_ERROR_CASE(cudaErrorNotSupported)};
#undef RUNTIME_ERROR_CASE
    std::vector<std::pair<const char*, std::string>> retained;
    for (const auto& [code, expected] : cases) {
        const auto* name = cudaGetErrorName(code); const auto* description = cudaGetErrorString(code);
        require(name && std::string(name) == expected && description && *description, "recognized Runtime error has exact enum name and nonempty description");
        retained.emplace_back(name, name); retained.emplace_back(description, description);
    }
    for (const auto& [pointer, copy] : retained) require(std::string(pointer) == copy, "borrowed Runtime error strings survive other queries");
}
void errors_validation() {
    require(cuGetErrorName(CUDA_SUCCESS, nullptr) == CUDA_ERROR_INVALID_VALUE
        && cuGetErrorString(CUDA_SUCCESS, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver error-query outputs have invalid-value status");
    for (int code : {-1, 123456}) {
        error_query_output<const char*> name, description;
        name.value = description.value = reinterpret_cast<const char*>(uintptr_t{1});
        require(cuGetErrorName(static_cast<CUresult>(code), &name.value) == CUDA_ERROR_INVALID_VALUE
            && cuGetErrorString(static_cast<CUresult>(code), &description.value) == CUDA_ERROR_INVALID_VALUE, "unknown Driver errors have invalid-value status");
        require(name.get() == nullptr && description.get() == nullptr, "unknown Driver error queries clear only caller pointer slots");
        const auto* runtime_name = cudaGetErrorName(static_cast<cudaError_t>(code));
        const auto* runtime_description = cudaGetErrorString(static_cast<cudaError_t>(code));
        require(runtime_name && runtime_description && std::string(runtime_name) == "unrecognized error code"
            && std::string(runtime_description) == "unrecognized error code", "unknown Runtime errors have specified fallback text");
    }
    errors_driver(); errors_runtime();
}
void errors_state() {
    require(cudaGetLastError() == cudaSuccess, "host diagnostic state starts clear");
    require(cudaDriverGetVersion(nullptr) == cudaErrorInvalidValue, "host-only invalid version output supplies pending Runtime error");
    errors_driver(); errors_runtime(); errors_validation();
    require(cudaPeekAtLastError() == cudaErrorInvalidValue && cudaGetLastError() == cudaErrorInvalidValue,
        "valid/rejected diagnostic queries preserve pending Runtime last error");
    errors_driver(); errors_runtime();
    require(cudaPeekAtLastError() == cudaSuccess, "successful diagnostic queries leave clear Runtime error state clear");
}
void errors_threading() {
    require(cudaGetLastError() == cudaSuccess, "parent diagnostic thread starts clear");
    require(cudaDriverGetVersion(nullptr) == cudaErrorInvalidValue, "parent owns distinct pending Runtime error");
    std::array<std::exception_ptr, 4> errors{};
    std::array<std::jthread, 4> workers;
    for (unsigned index = 0; index < workers.size(); ++index) workers[index] = std::jthread([&, index] {
        try {
            require(cudaGetLastError() == cudaSuccess, "fresh diagnostic worker starts with independent clear error");
            const auto expected = index & 1 ? cudaErrorInvalidValue : cudaSuccess;
            if (index & 1) require(cudaDriverGetVersion(nullptr) == expected, "odd diagnostic worker owns pending error");
            for (unsigned repeat = 0; repeat < 32; ++repeat) {
                errors_driver(); errors_runtime();
                require(cudaPeekAtLastError() == expected, "concurrent diagnostic queries preserve each worker's error state");
            }
            require(cudaGetLastError() == expected && cudaPeekAtLastError() == cudaSuccess, "worker diagnostic cleanup affects only its own error state");
        } catch (...) { errors[index] = std::current_exception(); }
    });
    for (auto& worker : workers) worker.join();
    const auto parent = cudaGetLastError();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    require(parent == cudaErrorInvalidValue && cudaPeekAtLastError() == cudaSuccess, "worker queries preserve parent's pending error until parent cleanup");
}
std::string_view api_errors_case = "all";
void execute_api_errors() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"driver", errors_driver}, {"runtime", errors_runtime}, {"validation", errors_validation}, {"state", errors_state}, {"threading", errors_threading}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (api_errors_case == "all" || api_errors_case == name) {
        selected = true; errors_uninitialized(); execute(); errors_uninitialized();
    }
    require(selected, "unknown api_errors case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::api_errors_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_api_errors);
}
