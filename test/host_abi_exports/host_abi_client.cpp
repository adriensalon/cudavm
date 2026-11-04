#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <cstring>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define ABI_CLIENT_EXPORT __declspec(dllexport)
#else
#include <dlfcn.h>
#define ABI_CLIENT_EXPORT __attribute__((visibility("default")))
#endif

namespace cudavm {
struct abi_client_module {
#if defined(_WIN32)
    HMODULE value{};
    explicit abi_client_module(const char* path) : value(LoadLibraryA(path)) {}
    ~abi_client_module() { if (value) FreeLibrary(value); }
    void* symbol(const char* name) const { return value ? reinterpret_cast<void*>(GetProcAddress(value, name)) : nullptr; }
#else
    void* value{};
    explicit abi_client_module(const char* path) : value(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {}
    ~abi_client_module() { if (value) dlclose(value); }
    void* symbol(const char* name) const { return value ? dlsym(value, name) : nullptr; }
#endif
    abi_client_module(const abi_client_module&) = delete;
    abi_client_module& operator=(const abi_client_module&) = delete;
    template<class Signature> Signature get(const char* name) const { return reinterpret_cast<Signature>(symbol(name)); }
};
template<class T> struct abi_client_output {
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    abi_client_output() { before.fill(0xc3); after.fill(0x5d); }
    bool guarded() const {
        for (auto byte : before) if (byte != 0xc3) return false;
        for (auto byte : after) if (byte != 0x5d) return false;
        return true;
    }
};
int abi_client_driver(const abi_client_module& driver) {
    const auto version = driver.get<decltype(&cuDriverGetVersion)>("cuDriverGetVersion");
    const auto name = driver.get<decltype(&cuGetErrorName)>("cuGetErrorName");
    const auto count = driver.get<decltype(&cuDeviceGetCount)>("cuDeviceGetCount");
    const auto legacy_lookup = driver.get<PFN_cuGetProcAddress_v11030>("cuGetProcAddress");
    const auto lookup = driver.get<PFN_cuGetProcAddress_v12000>("cuGetProcAddress_v2");
    if (!version || !name || !count || !legacy_lookup || !lookup) return 101;
    abi_client_output<int> number;
    number.value = -1;
    if (version(&number.value) != CUDA_SUCCESS || !number.guarded() || number.value != 13040) return 102;
    abi_client_output<const char*> text;
    if (name(CUDA_ERROR_INVALID_VALUE, &text.value) != CUDA_SUCCESS || !text.guarded()
        || !text.value || std::strcmp(text.value, "CUDA_ERROR_INVALID_VALUE")) return 103;
    abi_client_output<void*> address;
    abi_client_output<CUdriverProcAddressQueryResult> status;
    if (legacy_lookup("cuDriverGetVersion", &address.value, 2020, 0) != CUDA_SUCCESS || !address.guarded() || !address.value) return 104;
    number.value = -1;
    if (reinterpret_cast<PFN_cuDriverGetVersion_v2020>(address.value)(&number.value) != CUDA_SUCCESS
        || !number.guarded() || number.value != 13040) return 105;
    address.value = nullptr;
    status.value = static_cast<CUdriverProcAddressQueryResult>(12345);
    if (lookup("cuDriverGetVersion", &address.value, 2020, 0, &status.value) != CUDA_SUCCESS || !address.guarded()
        || !status.guarded() || !address.value || status.value != CU_GET_PROC_ADDRESS_SUCCESS) return 106;
    number.value = -1;
    if (reinterpret_cast<PFN_cuDriverGetVersion_v2020>(address.value)(&number.value) != CUDA_SUCCESS
        || !number.guarded() || number.value != 13040) return 107;
    if (count(&number.value) != CUDA_ERROR_NOT_INITIALIZED || !number.guarded()) return 108;
    return 0;
}
int abi_client_runtime(const abi_client_module& runtime, const abi_client_module& driver) {
    const auto version = runtime.get<decltype(&cudaRuntimeGetVersion)>("cudaRuntimeGetVersion");
    const auto name = runtime.get<decltype(&cudaGetErrorName)>("cudaGetErrorName");
    const auto last = runtime.get<decltype(&cudaGetLastError)>("cudaGetLastError");
    const auto lookup = runtime.get<decltype(&cudaGetDriverEntryPointByVersion)>("cudaGetDriverEntryPointByVersion");
    const auto count = driver.get<decltype(&cuDeviceGetCount)>("cuDeviceGetCount");
    if (!version || !name || !last || !lookup || !count) return 201;
    abi_client_output<int> number;
    if (version(&number.value) != cudaSuccess || !number.guarded() || number.value != 13040) return 202;
    if (last() != cudaSuccess || version(nullptr) != cudaErrorInvalidValue) return 203;
    const auto* text = name(cudaErrorInvalidValue);
    if (!text || std::strcmp(text, "cudaErrorInvalidValue") || last() != cudaErrorInvalidValue || last() != cudaSuccess) return 204;
    abi_client_output<void*> address;
    abi_client_output<cudaDriverEntryPointQueryResult> status;
    status.value = static_cast<cudaDriverEntryPointQueryResult>(12345);
    if (lookup("cuDriverGetVersion", &address.value, 2020, 0, &status.value) != cudaSuccess || !address.guarded()
        || !status.guarded() || !address.value || status.value != cudaDriverEntryPointSuccess) return 205;
    number.value = -1;
    if (reinterpret_cast<PFN_cuDriverGetVersion_v2020>(address.value)(&number.value) != CUDA_SUCCESS
        || !number.guarded() || number.value != 13040) return 206;
    if (count(&number.value) != CUDA_ERROR_NOT_INITIALIZED || !number.guarded()) return 207;
    return 0;
}
}

// This library has no cudavm link dependency. Only this small C boundary is
// exported; all SDK calls resolve dynamically through exact public signatures.
extern "C" ABI_CLIENT_EXPORT int cudavm_host_abi_client(const char* driver_path, const char* runtime_path, unsigned group) {
    if (!driver_path || !runtime_path || group > 2) return 1;
    cudavm::abi_client_module driver(driver_path);
    if (!driver.value) return 2;
    if (group != 1) { const auto result = cudavm::abi_client_driver(driver); if (result) return result; }
    if (group != 0) {
        cudavm::abi_client_module runtime(runtime_path);
        if (!runtime.value) return 3;
        const auto result = cudavm::abi_client_runtime(runtime, driver); if (result) return result;
    }
    return 0;
}
