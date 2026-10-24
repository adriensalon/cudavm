#pragma once
#include <cudart/export.h>
#include <cuda/cudavm_cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Registration extension used by generated application host code.
   These are not NVIDIA CUDA declarations.
   Binding copies metadata and image bytes; it never initializes a GPU. */
typedef struct cudavmKernelParameter {
    size_t size, alignment, offset;
} cudavmKernelParameter;
typedef struct cudavmKernelDescriptor {
    const char* symbol;
    size_t parameter_count;
    const cudavmKernelParameter* parameters;
} cudavmKernelDescriptor;
typedef struct cudavmProgram {
    uint32_t abi_version;
    cudavmBackend backend;
    const char* identity;
    const void* image;
    size_t image_size;
    size_t kernel_count;
    const cudavmKernelDescriptor* kernels;
} cudavmProgram;
CUDAVM_CUDART_API cudaError_t cudavmBindKernel(const void* host_function, const cudavmProgram* program, const char* symbol);
// A host variable address is an identity token, never its device storage.
CUDAVM_CUDART_API cudaError_t cudavmBindSymbol(const void* host_variable, const cudavmProgram* program, const char* symbol);
#ifdef __cplusplus
}
#endif
