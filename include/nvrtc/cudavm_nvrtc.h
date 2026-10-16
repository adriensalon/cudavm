#pragma once
#include <cudavm/export.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Cudavm extension ABI for mixed-source builds and portable IR.
   This does not claim NVIDIA LTO/NVVM format compatibility. */
typedef struct cudavmCompileResult cudavmCompileResult;
typedef struct cudavmCompileOptions {
    uint32_t abi_version; /* 1 */
    const char *input, *host, *bitcode, *manifest, *depfile;
    const char *standard, *architecture;
    const char *resources_root; /* Optional SDK prefix for its public include directory. */
    size_t include_count;
    const char* const* includes;
    size_t define_count;
    const char* const* defines;
} cudavmCompileOptions;
CUDAVM_NVRTC_API int cudavmCompileTranslationUnit(const cudavmCompileOptions*, cudavmCompileResult**);
CUDAVM_NVRTC_API int cudavmCompileGetResources(const char* resources_root, cudavmCompileResult**);
CUDAVM_NVRTC_API const char* cudavmCompileOutput(const cudavmCompileResult*);
CUDAVM_NVRTC_API const char* cudavmCompileDiagnostics(const cudavmCompileResult*);
CUDAVM_NVRTC_API void cudavmCompileRelease(cudavmCompileResult*);
#ifdef __cplusplus
}
#endif
