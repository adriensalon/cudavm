// Produced separately against NVIDIA's headers and cuda.lib. Candidate tests
// execute the pinned binary; they never compile or relink this source.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <cuda.h>
#include "driver_images.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cudavm {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "unknown error"));
}
void verify_driver_origin(const std::filesystem::path& expected) {
    auto driver = GetModuleHandleW(L"nvcuda.dll");
    require(driver != nullptr, "NVIDIA Driver import was loaded");
    std::wstring path(32768, L'\0');
    auto size = GetModuleFileNameW(driver, path.data(), DWORD(path.size()));
    require(size && size < path.size(), "Driver module path is available");
    path.resize(size);
    require(std::filesystem::equivalent(path, expected), "loaded Driver origin matches the selected deployment");
    std::printf("driver=%s\n", std::filesystem::path(path).string().c_str());
}
void verify_dependencies(const std::filesystem::path& driver, std::string_view backend) {
    auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    require(snapshot != INVALID_HANDLE_VALUE, "enumerate actual process library origins");
    MODULEENTRY32W entry{}; entry.dwSize = sizeof(entry);
    bool compiler = false, vendor = false;
    if (Module32FirstW(snapshot, &entry)) do {
        std::filesystem::path path(entry.szExePath);
        auto name = path.filename().wstring();
        if (_wcsicmp(name.c_str(), L"cudavm_core.dll") == 0) {
            compiler = true;
            require(std::filesystem::equivalent(path, driver.parent_path() / L"cudavm_core.dll"),
                "shader compiler comes from the relocated deployment");
            std::printf("compiler=%s\n", path.string().c_str());
        }
        if (_wcsicmp(name.c_str(), L"nvcuda.dll") == 0 && !std::filesystem::equivalent(path, driver)) {
            std::wstring system(32768, L'\0');
            auto size = GetSystemDirectoryW(system.data(), unsigned(system.size()));
            require(size && size < system.size(), "native system Driver directory is available");
            system.resize(size);
            require(std::filesystem::equivalent(path, std::filesystem::path(system) / L"nvcuda.dll"),
                "coexisting native CUDA Driver comes from the system directory");
            vendor = true;
            std::printf("vendor_driver=%s\n", path.string().c_str());
        }
    } while (Module32NextW(snapshot, &entry));
    CloseHandle(snapshot);
    if (backend == "vulkan" || backend == "directx") require(compiler, "shader input loads its deployed compiler");
    if (backend == "cuda") require(vendor, "native backend loads a distinct vendor Driver");
}
void discovery() {
    int version = 0;
    check(cuDriverGetVersion(&version), "cuDriverGetVersion");
    require(version >= 12000, "Driver supports the client's CUDA 12.0 procedure-discovery ABI");
    void* address = nullptr; CUdriverProcAddressQueryResult query{};
    check(cuGetProcAddress("cuDriverGetVersion", &address, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &query), "cuGetProcAddress");
    require(address && query == CU_GET_PROC_ADDRESS_SUCCESS, "versioned procedure discovery succeeds");
    int indirect = 0;
    check(reinterpret_cast<decltype(&cuDriverGetVersion)>(address)(&indirect), "discovered version query");
    require(indirect == version, "imported and discovered ABI calls agree");
    int count = 0x345678;
    require(cuDeviceGetCount(&count) == CUDA_ERROR_NOT_INITIALIZED && count == 0x345678,
        "discovery leaves the Driver uninitialized and preserves output on failure");
}
int execute(std::string_view mode, const std::filesystem::path& driver, std::string_view backend) {
    verify_driver_origin(driver);
    discovery();
    if (mode == "discovery") return 0;
    if (mode == "blocked") {
        auto status = cuInit(0);
        require(status == CUDA_ERROR_NO_DEVICE || status == CUDA_SUCCESS,
            "facade selected as native Driver is rejected without recursion");
        int count = 0x123456;
        if (status == CUDA_SUCCESS) {
            check(cuDeviceGetCount(&count), "blocked native device count");
            require(count == 0, "facade cannot publish itself as a native CUDA device");
        } else require(cuDeviceGetCount(&count) == CUDA_ERROR_NOT_INITIALIZED && count == 0x123456,
            "failed native discovery publishes no initialized state");
        return 0;
    }
    require(mode == "ptx" || mode == "cubin", "known client execution mode");
    auto status = cuInit(0);
    if (status == CUDA_ERROR_NO_DEVICE || status == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) {
        std::puts("SKIP: selected backend has no available device"); return 77;
    }
    check(status, "cuInit");
    int count = 0; check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (!count) { std::puts("SKIP: selected backend has no available device"); return 77; }
    CUdevice device{}; check(cuDeviceGet(&device, 0), "cuDeviceGet");
    char name[256]{}; check(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName");
    if (backend != "native") {
        const auto suffix = backend == "vulkan" ? " (Vulkan)" : backend == "directx" ? " (DirectX)" : " (CUDA)";
        auto text = std::string_view(name), tail = std::string_view(suffix);
        require(text.size() >= tail.size() && text.substr(text.size() - tail.size()) == tail,
            "selected backend is the executable's actual device");
    }
    if (mode == "cubin" && (backend == "cuda" || backend == "native")) {
        int major = 0; check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "device architecture");
        if (major != 7) { std::puts("SKIP: original SM75 cubin requires a compatible native architecture"); return 77; }
    }
    std::printf("device=%s\n", name);
    CUcontext context{}; check(cuCtxCreate(&context, 0, device), "cuCtxCreate_v2");
    CUstream stream{}; CUevent event{}; CUmodule module{}; CUdeviceptr input{}, output{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    check(cuModuleLoadData(&module, mode == "ptx" ? static_cast<const void*>(driver_ptx) : driver_cubin), "cuModuleLoadData");
    CUfunction kernel{}; check(cuModuleGetFunction(&kernel, module, "cubin_indexed"), "cuModuleGetFunction");
    constexpr size_t extent = 80, offset = 3;
    constexpr uint32_t guard = 0xa95b371d;
    std::array<uint32_t, extent> source{}, actual{}, expected{};
    for (size_t i = 0; i < extent; ++i) source[i] = uint32_t(0x10203040u + 17u * i);
    actual.fill(guard); expected = actual;
    check(cuMemAlloc(&input, sizeof(source)), "cuMemAlloc_v2 input");
    check(cuMemAlloc(&output, sizeof(actual)), "cuMemAlloc_v2 output");
    check(cuMemcpyHtoD(input, source.data(), sizeof(source)), "cuMemcpyHtoD_v2 input");
    check(cuMemcpyHtoD(output, actual.data(), sizeof(actual)), "cuMemcpyHtoD_v2 output");
    CUdeviceptr in = input + offset * sizeof(uint32_t), out = output + offset * sizeof(uint32_t);
    uint32_t bound = 53; void* arguments[]{&out, &in, &bound};
    check(cuLaunchKernel(kernel, 2, 2, 1, 4, 2, 2, 0, stream, arguments, nullptr), "cuLaunchKernel");
    check(cuEventRecord(event, stream), "cuEventRecord");
    check(cuEventSynchronize(event), "cuEventSynchronize");
    check(cuMemcpyDtoH(actual.data(), output, sizeof(actual)), "cuMemcpyDtoH_v2 output");
    for (uint32_t i = 0; i < bound; ++i) expected[offset + i] = source[offset + i] + i + 4u;
    require(actual == expected, "kernel results, inactive lanes and allocation guards agree with the independent integer oracle");
    std::array<uint32_t, extent> preserved{};
    check(cuMemcpyDtoH(preserved.data(), input, sizeof(preserved)), "cuMemcpyDtoH_v2 input");
    require(preserved == source, "kernel preserves the complete input allocation");
    verify_dependencies(driver, backend);
    check(cuEventDestroy(event), "cuEventDestroy_v2");
    check(cuStreamDestroy(stream), "cuStreamDestroy_v2");
    check(cuMemFree(output), "cuMemFree_v2 output"); check(cuMemFree(input), "cuMemFree_v2 input");
    check(cuModuleUnload(module), "cuModuleUnload"); check(cuCtxDestroy(context), "cuCtxDestroy_v2");
    return 0;
}
}
int main(int argc, char** argv) {
    try {
        cudavm::require(argc == 4, "usage: driver_client <discovery|blocked|ptx|cubin> <expected-driver-path> <backend|native>");
        auto result = cudavm::execute(argv[1], std::filesystem::path(argv[2]), argv[3]);
        if (!result) std::puts("PASS: unchanged NVIDIA-linked Driver client");
        return result;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
