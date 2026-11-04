#define NVJITLINK_NO_INLINE
#include <nvjitlink/nvJitLink.h>
#include <nvptxcompiler/nvPTXCompiler.h>
#include <nvfatbin/nvFatbin.h>
#include <nvvm/nvvm.h>
#include <nvrtc/nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc/nvrtc.h>
#include <nvvm/nvvm.h>
#include <nvfatbin/nvFatbin.h>
#include <nvptxcompiler/nvPTXCompiler.h>
#include <nvjitlink/nvJitLink.h>

#if defined(__cplusplus) || defined(__CUDACC__) || defined(__CUDACC_RTC__)
#error The public C header probe must compile as ordinary C.
#endif
_Static_assert(CUDA_VERSION == 13040 && CUDART_VERSION == 13040, "pinned C header baseline");
_Static_assert(sizeof(CUdeviceptr) == 8, "C Driver device-pointer width");

unsigned cudavm_public_headers_c_profile(void) {
    return (CUDA_VERSION == 13040 ? 1u : 0u) | (CUDART_VERSION == 13040 ? 2u : 0u)
        | (sizeof(CUuuid) == 16 ? 4u : 0u) | (cudaMemcpyDefault == 4 ? 8u : 0u)
        | (sizeof(nvrtcProgram) == sizeof(void*) && sizeof(nvvmProgram) == sizeof(void*) ? 16u : 0u)
        | (sizeof(nvFatbinHandle) == sizeof(void*) && sizeof(nvPTXCompilerHandle) == sizeof(void*) ? 32u : 0u)
        | (sizeof(nvJitLinkHandle) == sizeof(void*) && sizeof(struct cudaExtent) == 3 * sizeof(size_t) ? 64u : 0u);
}
