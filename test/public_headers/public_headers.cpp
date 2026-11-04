#include "conformance.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc/nvrtc.h>
#include <nvvm/nvvm.h>
#include <nvfatbin/nvFatbin.h>
#include <nvptxcompiler/nvPTXCompiler.h>
#define NVJITLINK_NO_INLINE
#include <nvjitlink/nvJitLink.h>
// Repeated public includes must coexist with their normal include guards.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <nvrtc/nvrtc.h>
#include <nvvm/nvvm.h>
#include <nvfatbin/nvFatbin.h>
#include <nvptxcompiler/nvPTXCompiler.h>
#include <nvjitlink/nvJitLink.h>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#error Public header inclusion must not turn ordinary host compilation into CUDA compilation.
#endif

namespace cudavm {
unsigned public_headers_reverse_profile();
#ifdef CUDAVM_CONFORMANCE_HAVE_C_HEADERS
extern "C" unsigned cudavm_public_headers_c_profile(void);
#endif
unsigned public_headers_profile() {
    return (CUDA_VERSION == 13040 ? 1u : 0u) | (CUDART_VERSION == 13040 ? 2u : 0u)
        | (sizeof(CUuuid) == 16 ? 4u : 0u) | (cudaMemcpyDefault == 4 ? 8u : 0u)
        | (sizeof(nvrtcProgram) == sizeof(void*) && sizeof(nvvmProgram) == sizeof(void*) ? 16u : 0u)
        | (sizeof(nvFatbinHandle) == sizeof(void*) && sizeof(nvPTXCompilerHandle) == sizeof(void*) ? 32u : 0u)
        | (sizeof(nvJitLinkHandle) == sizeof(void*) && sizeof(cudaExtent) == 3 * sizeof(size_t) ? 64u : 0u);
}
std::string_view public_headers_case = "all";
void execute_public_headers() {
    bool selected = false;
    if (public_headers_case == "all" || public_headers_case == "cxx") {
        selected = true;
        require(public_headers_profile() == 127, "C++ combined/repeated includes expose pinned declarations and types");
        dim3 dimensions;
        require(dimensions.x == 1 && dimensions.y == 1 && dimensions.z == 1, "high-level Runtime header exposes default host dimension constructor");
        dim3 converted(uint3{3, 5, 7});
        require(converted.x == 3 && converted.y == 5 && converted.z == 7, "high-level Runtime header exposes vector conversion constructor");
    }
    if (public_headers_case == "all" || public_headers_case == "reverse") {
        selected = true;
        require(public_headers_reverse_profile() == 127, "independent C++ translation unit accepts reversed/repeated include order");
    }
#ifdef CUDAVM_CONFORMANCE_HAVE_C_HEADERS
    if (public_headers_case == "all" || public_headers_case == "c") {
        selected = true;
        require(cudavm_public_headers_c_profile() == 127, "independent C11 translation unit exposes compatible public declarations and types");
    }
#endif
    require(selected, "unknown or unavailable public_headers case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::public_headers_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_public_headers);
}
