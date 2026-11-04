#define NVJITLINK_NO_INLINE
#include <nvjitlink/nvJitLink.h>
#include <nvptxcompiler/nvPTXCompiler.h>
#include <nvfatbin/nvFatbin.h>
#include <nvvm/nvvm.h>
#include <nvrtc/nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc/nvrtc.h>
#include <nvvm/nvvm.h>
#include <nvfatbin/nvFatbin.h>
#include <nvptxcompiler/nvPTXCompiler.h>
#include <nvjitlink/nvJitLink.h>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#error Public header inclusion must preserve ordinary host compilation mode.
#endif

namespace cudavm {
unsigned public_headers_reverse_profile() {
    return (CUDA_VERSION == 13040 ? 1u : 0u) | (CUDART_VERSION == 13040 ? 2u : 0u)
        | (sizeof(CUuuid) == 16 ? 4u : 0u) | (cudaMemcpyDefault == 4 ? 8u : 0u)
        | (sizeof(nvrtcProgram) == sizeof(void*) && sizeof(nvvmProgram) == sizeof(void*) ? 16u : 0u)
        | (sizeof(nvFatbinHandle) == sizeof(void*) && sizeof(nvPTXCompilerHandle) == sizeof(void*) ? 32u : 0u)
        | (sizeof(nvJitLinkHandle) == sizeof(void*) && sizeof(cudaExtent) == 3 * sizeof(size_t) ? 64u : 0u);
}
}
