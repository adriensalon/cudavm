#include "backend.hpp"
#include "device_inventory.hpp"
#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
namespace cudavm {
std::optional<std::string> backend_environment(const char* name) {
#if defined(_WIN32)
    // DLLs built with the static CRT have separate getenv snapshots. Read the
    // process environment so a failed initialization can be retried after the
    // host fixes its policy, independently of linkage.
    for (;;) {
        SetLastError(ERROR_SUCCESS);
        auto size = GetEnvironmentVariableA(name, nullptr, 0);
        if (!size) {
            // GetEnvironmentVariable cannot distinguish an empty entry from
            // an absent one. The environment block preserves that distinction.
            std::optional<std::string> result;
            auto* block = GetEnvironmentStringsA();
            if (!block) return {};
            auto length = std::strlen(name);
            for (auto* entry = block; *entry; entry += std::strlen(entry) + 1)
                if (_strnicmp(entry, name, length) == 0 && entry[length] == '=') { result = entry + length + 1; break; }
            FreeEnvironmentStringsA(block); return result;
        }
        std::string value(size, '\0');
        SetLastError(ERROR_SUCCESS);
        auto count = GetEnvironmentVariableA(name, value.data(), size);
        if (!count) {
            if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) continue;
            return std::string{};
        }
        if (count >= size) continue;
        value.resize(count); return value;
    }
#else
    auto* value = std::getenv(name); return value ? std::optional<std::string>(value) : std::nullopt;
#endif
}
int shader_device_attribute(const shader_resource_limits& limits, CUdevice_attribute which) {
    switch (which) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: return int(std::min(1024u, limits.threads));
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
        return int(std::min(1024u, limits.block[which - CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X]));
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
        return int(std::min(unsigned(std::numeric_limits<int>::max()), limits.grid[which - CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X]));
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
        return int(std::min(32768u, limits.shared_bytes));
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: return 32; // Logical CUDA warp used by compiler collective lowering.
    case CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: return 1;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: return 0;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH: return std::numeric_limits<int>::max();
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: case CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: return 256;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH:
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH:
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT:
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH: return (1 << 30) - 1;
    case CU_DEVICE_ATTRIBUTE_NUMA_ID: case CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID: return -1;
    default: break;
    }
    // The transport has no CUDA texture/constant/register file, SM residency or
    // NVIDIA architecture version. Zero reports no such exposed facility; it
    // must never be used to synthesize an occupancy result.
    switch (which) {
#define DEVICE_PROPERTY(field, attribute) case CU_DEVICE_ATTRIBUTE_##attribute:
#define DEVICE_OPTIONAL_PROPERTY(field, attribute, fallback) case CU_DEVICE_ATTRIBUTE_##attribute:
#include "device_properties.inl"
#undef DEVICE_PROPERTY
#undef DEVICE_OPTIONAL_PROPERTY
        return 0;
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "shader device resource attribute is unavailable");
    }
}
std::unique_ptr<backend> make_backend(cudavmBackend kind) {
    switch (kind) {
#if defined(CUDAVM_ENABLE_CUDA)
    case cudavmBackendCuda: return make_cuda_backend();
#endif
#if defined(CUDAVM_ENABLE_OPENCL)
    case cudavmBackendOpenCL: return make_opencl_backend();
#endif
#if defined(CUDAVM_ENABLE_OPENGL)
    case cudavmBackendOpenGL: return make_opengl_backend();
#endif
#if defined(CUDAVM_ENABLE_VULKAN)
    case cudavmBackendVulkan: return make_vulkan_backend();
#endif
#if defined(CUDAVM_ENABLE_DIRECTX)
    case cudavmBackendDirectX: return make_directx_backend();
#endif
    default:
        require(kind >= cudavmBackendCuda && kind <= cudavmBackendDirectX,
                CUDA_ERROR_INVALID_VALUE, "unknown cudavm backend");
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "requested backend is disabled in this cudavm build");
    }
}
std::span<const cudavmBackend> enabled_backends() {
    static constexpr cudavmBackend kinds[] = {
#if defined(CUDAVM_ENABLE_CUDA)
        cudavmBackendCuda,
#endif
#if defined(CUDAVM_ENABLE_VULKAN)
        cudavmBackendVulkan,
#endif
#if defined(CUDAVM_ENABLE_DIRECTX)
        cudavmBackendDirectX,
#endif
#if defined(CUDAVM_ENABLE_OPENCL)
        cudavmBackendOpenCL,
#endif
#if defined(CUDAVM_ENABLE_OPENGL)
        cudavmBackendOpenGL,
#endif
    };
    return kinds;
}
std::unique_ptr<backend> discover_devices() {
    return discover_devices(enabled_backends(), backend_environment("CUDAVM_BACKENDS"),
                            backend_environment("CUDAVM_DEVICES"), make_backend);
}
}
