#pragma once
#include "dynamic_library.hpp"

namespace cudavm {
inline std::filesystem::path native_cuda_driver_path() {
    if (auto selected = backend_environment("CUDAVM_CUDA_DRIVER")) {
        auto path = std::filesystem::u8path(*selected);
        require(path.is_absolute(), CUDA_ERROR_INVALID_VALUE,
            "CUDAVM_CUDA_DRIVER must name an absolute vendor Driver library path");
        return path;
    }
#if defined(_WIN32)
    std::wstring directory(32768, L'\0');
    auto size = GetSystemDirectoryW(directory.data(), unsigned(directory.size()));
    require(size && size < directory.size(), CUDA_ERROR_NO_DEVICE, "cannot locate the native CUDA system driver directory");
    directory.resize(size);
    return std::filesystem::path(directory) / L"nvcuda.dll";
#else
    // When our SONAME masks the vendor library, the explicit absolute selector
    // above supplies its installation-specific path. Never call back into a facade.
    return "libcuda.so.1";
#endif
}
class native_driver_library : public dynamic_library {
public:
    native_driver_library() : dynamic_library(native_cuda_driver_path(), CUDA_ERROR_NO_DEVICE) {
        require(!find("cudavmGetDiagnostic"), CUDA_ERROR_NO_DEVICE,
            "native CUDA driver resolved to a cudavm facade; select the vendor library with CUDAVM_CUDA_DRIVER");
    }
};
}
