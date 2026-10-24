#pragma once
#include <nvjitlink/export.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Cudavm extension ABI for mixed-source builds and portable IR.
   This does not claim NVIDIA LTO/NVVM format compatibility. */
typedef struct cudavmLinkResult cudavmLinkResult;
typedef struct cudavmLinkOptions {
    uint32_t abi_version; /* 1 */
    size_t input_count;
    const char* const* inputs;
    const char *portable, *ir, *manifest, *binary, *target, *module, *architecture;
} cudavmLinkOptions;
/* ABI 2 extends ABI 1 without changing its layout. library_inputs are raw
   compile-tu candidates selected by device references or host registration
   providers found in native host objects/self-contained static archives. */
typedef struct cudavmLinkOptionsV2 {
    cudavmLinkOptions base; /* base.abi_version = 2 */
    size_t library_input_count;
    const char* const* library_inputs;
    size_t host_input_count;
    const char* const* host_inputs;
} cudavmLinkOptionsV2;
CUDAVM_NVJITLINK_API int cudavmLinkDevice(const cudavmLinkOptions*, cudavmLinkResult**);
CUDAVM_NVJITLINK_API const char* cudavmLinkDiagnostics(const cudavmLinkResult*);
CUDAVM_NVJITLINK_API void cudavmLinkRelease(cudavmLinkResult*);
#ifdef __cplusplus
}
#endif
