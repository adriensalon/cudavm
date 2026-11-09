#include "device_inventory.hpp"
#include "../cuda/driver_api.hpp"
#include <cuda/cudavm_cuda.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <filesystem>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cudavm {
namespace {
std::filesystem::path inventory_driver_path() {
#if defined(_WIN32)
    std::wstring path(32768, L'\0');
    auto size = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (!size || size >= path.size()) throw std::runtime_error("cannot locate nvcc for device discovery");
    path.resize(size);
    auto directory = std::filesystem::path(path).parent_path();
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&print_device_inventory), &info) || !info.dli_fname)
        throw std::runtime_error("cannot locate nvcc for device discovery");
    auto directory = std::filesystem::absolute(info.dli_fname).parent_path();
#endif
    std::error_code error;
    if (std::filesystem::equivalent(directory, CUDAVM_NVCC_BUILD_DIRECTORY, error))
        return CUDAVM_NVCC_BUILD_DRIVER;
    return (directory / CUDAVM_NVCC_DRIVER_DIRECTORY / CUDAVM_NVCC_DRIVER_FILENAME).lexically_normal();
}

class inventory_library {
    void* handle_ = nullptr;
public:
    inventory_library() {
        auto path = inventory_driver_path();
#if defined(_WIN32)
        handle_ = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle_) throw std::runtime_error("device inventory Driver unavailable: " + path.string());
    }
    ~inventory_library() {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
    }
    inventory_library(const inventory_library&) = delete;
    inventory_library& operator=(const inventory_library&) = delete;
    template<class T> T get(const char* name) const {
#if defined(_WIN32)
        auto address = GetProcAddress(static_cast<HMODULE>(handle_), name);
#else
        auto address = dlsym(handle_, name);
#endif
        if (!address) throw std::runtime_error(std::string("device inventory Driver entry point unavailable: ") + name);
        return reinterpret_cast<T>(address);
    }
};

struct inventory_api {
    inventory_library library;
    decltype(&cuInit) init = library.get<decltype(&::cuInit)>("cuInit");
    decltype(&cuDeviceGetCount) count = library.get<decltype(&::cuDeviceGetCount)>("cuDeviceGetCount");
    decltype(&cuDeviceGetName) name = library.get<decltype(&::cuDeviceGetName)>("cuDeviceGetName");
    decltype(&cuDeviceGetPCIBusId) pci = library.get<decltype(&::cuDeviceGetPCIBusId)>("cuDeviceGetPCIBusId");
    decltype(&cudavmDriverPhysicalDevice) physical = library.get<decltype(&::cudavmDriverPhysicalDevice)>("cudavmDriverPhysicalDevice");
    decltype(&cudavmGetDiagnostic) diagnostic = library.get<decltype(&::cudavmGetDiagnostic)>("cudavmGetDiagnostic");
    decltype(&cudavmShutdown) shutdown = library.get<decltype(&::cudavmShutdown)>("cudavmShutdown");
};
}

int print_device_inventory() {
    inventory_api api;
    auto result = [&] {
        auto status = api.init(0);
        if (status != CUDA_SUCCESS) {
            std::fprintf(stderr, "initialization failed (%d): %s\n", int(status), api.diagnostic());
            return status == CUDA_ERROR_NO_DEVICE ? 77 : 1;
        }
        int count = 0;
        if (api.count(&count) != CUDA_SUCCESS) return 1;
        std::printf("%d logical CUDA device(s)\n", count);
        for (int i = 0; i < count; ++i) {
            char name[256]{}, pci[32]{};
            if (api.name(name, sizeof(name), i) != CUDA_SUCCESS) return 1;
            std::printf("  device %d, physical GPU %d: %s", i, api.physical(i), name);
            if (api.pci(pci, sizeof(pci), i) == CUDA_SUCCESS) std::printf(" [PCI %s]", pci);
            std::puts("");
        }
        return 0;
    }();
    auto shutdown = api.shutdown();
    return result ? result : shutdown != 0;
}
}
