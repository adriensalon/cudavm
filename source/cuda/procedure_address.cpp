// Internal mode exposes the exact versioned declarations without header aliases.
#define __CUDA_API_VERSION_INTERNAL
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/cudavm_cuda.h>
#include <string_view>

// cudaGLTypedefs.h's two registration signatures use GLuint/GLenum, both
// unsigned int. Keep discovery usable without an OpenGL SDK or enabled adapter.
extern "C" CUresult CUDAAPI cuGraphicsGLRegisterBuffer(CUgraphicsResource*, unsigned int, unsigned int);
extern "C" CUresult CUDAAPI cuGraphicsGLRegisterImage(CUgraphicsResource*, unsigned int, unsigned int, unsigned int);

#if defined(__clang__) && defined(_WIN32)
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#endif

extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuGetProcAddress_v2(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);

namespace cudavm {
namespace {
using graphics_buffer_procedure = CUresult (CUDAAPI *)(CUgraphicsResource*, unsigned int, unsigned int);
using graphics_image_procedure = CUresult (CUDAAPI *)(CUgraphicsResource*, unsigned int, unsigned int, unsigned int);
struct driver_procedure {
    std::string_view name;
    int version;
    bool per_thread;
    void* address;
};
// The pinned PFN types check calling convention, argument widths and every ABI
// revision at compile time. Never discover symbols in the vendor CUDA library:
// its handles and ownership belong to a different registry.
template<class Signature> void* procedure_address(Signature function) {
    return reinterpret_cast<void*>(function);
}
const driver_procedure driver_procedures[]{
#define DRIVER_PROCEDURE(name, version, symbol) {#name, version, false, procedure_address<PFN_##name##_v##version>(&symbol)},
#define DRIVER_PROCEDURE_PTDS(name, version, suffix, symbol) {#name, version, true, procedure_address<PFN_##name##_v##version##_##suffix>(&symbol)},
#define DRIVER_GRAPHICS_PROCEDURE(name, version, signature) {#name, version, false, procedure_address<signature>(&name)},
#include "driver_procedures.inl"
#undef DRIVER_PROCEDURE
#undef DRIVER_PROCEDURE_PTDS
#undef DRIVER_GRAPHICS_PROCEDURE
};
}
}

extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuGetProcAddress_v2(
    const char* symbol, void** output, int version, cuuint64_t flags, CUdriverProcAddressQueryResult* status) {
    if (!symbol || !output || version < 0 || version > CUDA_VERSION || (flags & ~cuuint64_t(3)))
        return CUDA_ERROR_INVALID_VALUE;
    const cudavm::driver_procedure* legacy = nullptr;
    const cudavm::driver_procedure* per_thread = nullptr;
    bool known = false;
    for (const auto& entry : cudavm::driver_procedures) {
        if (entry.name != symbol) continue;
        known = true;
        if (entry.version > version) continue;
        auto*& selected = entry.per_thread ? per_thread : legacy;
        if (!selected || entry.version > selected->version) selected = &entry;
    }
    auto* selected = legacy;
    if ((flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) && per_thread &&
        (!legacy || per_thread->version >= legacy->version)) selected = per_thread;
    *output = selected ? selected->address : nullptr;
    if (status) *status = selected ? CU_GET_PROC_ADDRESS_SUCCESS : known ?
        CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_SUCCESS;
}

extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuGetProcAddress(
    const char* symbol, void** output, int version, cuuint64_t flags) {
    auto result = cuGetProcAddress_v2(symbol, output, version, flags, nullptr);
    return result == CUDA_SUCCESS && !*output ? CUDA_ERROR_NOT_FOUND : result;
}
