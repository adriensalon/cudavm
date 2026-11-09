#include "test_framework.hpp"
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>
#include <nvrtc/nvrtc.h>
#include <nvvm/nvvm.h>
#include <nvfatbin/nvFatbin.h>
#include <nvptxcompiler/nvPTXCompiler.h>
#define NVJITLINK_NO_INLINE
#include <nvjitlink/nvJitLink.h>
#include <type_traits>
#include <utility>

namespace cudavm {
template<class Actual, class Expected> void declaration_matches(const char* name) {
    require(std::is_same_v<Actual, Expected>, std::string(name) + ": pinned public declaration");
}
#define DECLARATION(name, expected) ::cudavm::declaration_matches<decltype(&name), expected>(#name)
void declarations_driver() {
    using initialize = CUresult (CUDAAPI *)(unsigned);
    using count = CUresult (CUDAAPI *)(int*);
    using device = CUresult (CUDAAPI *)(CUdevice*, int);
    using create = CUresult (CUDAAPI *)(CUcontext*, CUctxCreateParams*, unsigned, CUdevice);
    using allocation = CUresult (CUDAAPI *)(CUdeviceptr*, size_t);
    using upload = CUresult (CUDAAPI *)(CUdeviceptr, const void*, size_t);
    using download = CUresult (CUDAAPI *)(void*, CUdeviceptr, size_t);
    using launch = CUresult (CUDAAPI *)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
    using lookup = CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
    DECLARATION(cuInit, initialize); DECLARATION(cuDeviceGetCount, count); DECLARATION(cuDeviceGet, device);
    DECLARATION(cuCtxCreate, create); DECLARATION(cuMemAlloc, allocation); DECLARATION(cuMemcpyHtoD, upload); DECLARATION(cuMemcpyDtoH, download);
    DECLARATION(cuLaunchKernel, launch); DECLARATION(cuGetProcAddress, lookup);
    require(std::is_invocable_r_v<CUresult, upload, CUdeviceptr, const unsigned char*, size_t>
        && !std::is_invocable_v<upload, const void*, CUdeviceptr, size_t>, "Driver upload accepts const host input and rejects reversed pointer roles");
}
template<class Output> concept declaration_runtime_allocate = requires(Output output) { cudaMalloc(output, size_t{1}); };
template<class Name> concept declaration_runtime_lookup_default = requires(Name name, void** output) {
    cudaGetDriverEntryPointByVersion(name, output, 13040u, 0ull);
};
void declarations_runtime() {
    using allocation = cudaError_t (CUDARTAPI *)(void**, size_t);
    using copy = cudaError_t (CUDARTAPI *)(void*, const void*, size_t, cudaMemcpyKind);
    using async_copy = cudaError_t (CUDARTAPI *)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    using properties = cudaError_t (CUDARTAPI *)(cudaDeviceProp*, int);
    using launch = cudaError_t (CUDARTAPI *)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    using lookup = cudaError_t (CUDARTAPI *)(const char*, void**, unsigned, unsigned long long, cudaDriverEntryPointQueryResult*);
    DECLARATION(cudaMalloc, allocation); DECLARATION(cudaMemcpy, copy); DECLARATION(cudaMemcpyAsync, async_copy);
    DECLARATION(cudaGetDeviceProperties, properties); DECLARATION(cudaLaunchKernel, launch); DECLARATION(cudaGetDriverEntryPointByVersion, lookup);
    require(declaration_runtime_allocate<void**> && !declaration_runtime_allocate<void*>
        && !declaration_runtime_allocate<const void**>, "Runtime C API allocation output requires writable void pointer slot");
    require(declaration_runtime_lookup_default<const char*> && !declaration_runtime_lookup_default<int>, "Runtime lookup has optional status and requires string symbol");
    require(std::is_invocable_r_v<cudaError_t, copy, void*, const void*, size_t, cudaMemcpyKind>
        && !std::is_invocable_v<copy, const void*, const void*, size_t, cudaMemcpyKind>, "Runtime copy destination is writable and source may be const");
}
void declarations_compilers() {
    using rtc_create = nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
    using rtc_compile = nvrtcResult (*)(nvrtcProgram, int, const char* const*);
    using rtc_name = nvrtcResult (*)(nvrtcProgram, const char*, const char**);
    using vm_create = nvvmResult (*)(nvvmProgram*);
    using vm_compile = nvvmResult (*)(nvvmProgram, int, const char**);
    using fat_create = nvFatbinResult (*)(nvFatbinHandle*, const char**, size_t);
    using fat_add = nvFatbinResult (*)(nvFatbinHandle, const char*, size_t, const char*, const char*, const char*);
    using ptx_create = nvPTXCompileResult (*)(nvPTXCompilerHandle*, size_t, const char*);
    using ptx_compile = nvPTXCompileResult (*)(nvPTXCompilerHandle, int, const char* const*);
    using jit_create = nvJitLinkResult (*)(nvJitLinkHandle*, uint32_t, const char**);
    using jit_pinned_create = nvJitLinkResult (*)(nvJitLinkHandle*, uint32_t, const char* const*);
    using jit_add = nvJitLinkResult (*)(nvJitLinkHandle, nvJitLinkInputType, const void*, size_t, const char*);
    DECLARATION(nvrtcCreateProgram, rtc_create); DECLARATION(nvrtcCompileProgram, rtc_compile); DECLARATION(nvrtcGetLoweredName, rtc_name);
    DECLARATION(nvvmCreateProgram, vm_create); DECLARATION(nvvmCompileProgram, vm_compile); DECLARATION(nvvmVerifyProgram, vm_compile);
    DECLARATION(nvFatbinCreate, fat_create); DECLARATION(nvFatbinAddPTX, fat_add);
    DECLARATION(nvPTXCompilerCreate, ptx_create); DECLARATION(nvPTXCompilerCompile, ptx_compile);
    DECLARATION(nvJitLinkCreate, jit_create); DECLARATION(__nvJitLinkCreate_13_4, jit_pinned_create);
    DECLARATION(nvJitLinkAddData, jit_add); DECLARATION(__nvJitLinkAddData_13_4, jit_add);
    require(!std::is_same_v<rtc_compile, nvrtcResult (*)(nvrtcProgram, int, const char**)>, "NVRTC option pointer array has its pinned const qualification");
}
void declarations_callbacks() {
    using driver_host = void (CUDA_CB *)(void*);
    using runtime_host = void (CUDART_CB *)(void*);
    using driver_stream = void (CUDA_CB *)(CUstream, CUresult, void*);
    using runtime_stream = void (CUDART_CB *)(cudaStream_t, cudaError_t, void*);
    using driver_enqueue = CUresult (CUDAAPI *)(CUstream, driver_host, void*);
    using runtime_enqueue = cudaError_t (CUDARTAPI *)(cudaStream_t, runtime_host, void*);
    using flow = int (*)(void*, void*);
    using flow_set = nvPTXCompileResult (*)(nvPTXCompilerHandle, flow, void*);
    declaration_matches<CUhostFn, driver_host>("CUhostFn"); declaration_matches<cudaHostFn_t, runtime_host>("cudaHostFn_t");
    declaration_matches<CUstreamCallback, driver_stream>("CUstreamCallback"); declaration_matches<cudaStreamCallback_t, runtime_stream>("cudaStreamCallback_t");
    DECLARATION(cuLaunchHostFunc, driver_enqueue); DECLARATION(cudaLaunchHostFunc, runtime_enqueue); DECLARATION(nvPTXCompilerSetFlowCallback, flow_set);
    require(!std::is_same_v<driver_host, void (*)(const void*)> && !std::is_same_v<flow, void (*)(void*, void*)>, "callback payload mutability and cancellation result are part of signature");
}
void declarations_versions() {
    using create_old = CUresult (CUDAAPI *)(CUcontext*, unsigned, CUdevice);
    using create_affinity = CUresult (CUDAAPI *)(CUcontext*, CUexecAffinityParam*, int, unsigned, CUdevice);
    using create_current = CUresult (CUDAAPI *)(CUcontext*, CUctxCreateParams*, unsigned, CUdevice);
    using current_device = CUresult (CUDAAPI *)(CUdevice*);
    using explicit_device = CUresult (CUDAAPI *)(CUdevice*, CUcontext);
    using old_lookup = CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t);
    using current_lookup = CUresult (CUDAAPI *)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
    declaration_matches<PFN_cuCtxCreate_v3020, create_old>("PFN_cuCtxCreate_v3020");
    declaration_matches<PFN_cuCtxCreate_v11040, create_affinity>("PFN_cuCtxCreate_v11040");
    declaration_matches<PFN_cuCtxCreate_v12050, create_current>("PFN_cuCtxCreate_v12050");
    declaration_matches<PFN_cuCtxGetDevice_v2000, current_device>("PFN_cuCtxGetDevice_v2000");
    declaration_matches<PFN_cuCtxGetDevice_v13000, explicit_device>("PFN_cuCtxGetDevice_v13000");
    declaration_matches<PFN_cuGetProcAddress_v11030, old_lookup>("PFN_cuGetProcAddress_v11030");
    declaration_matches<PFN_cuGetProcAddress_v12000, current_lookup>("PFN_cuGetProcAddress_v12000");
    require(!std::is_same_v<create_old, create_affinity> && !std::is_same_v<create_affinity, create_current>
        && !std::is_same_v<current_device, explicit_device> && !std::is_same_v<old_lookup, current_lookup>, "historical ABI revisions retain distinct signatures");
}
std::string_view public_declarations_case = "all";
void execute_public_declarations() {
    require(CUDA_VERSION == 13040 && CUDART_VERSION == 13040, "pinned declaration header baseline");
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"driver", declarations_driver}, {"runtime", declarations_runtime}, {"compilers", declarations_compilers},
        {"callbacks", declarations_callbacks}, {"versions", declarations_versions}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (public_declarations_case == "all" || public_declarations_case == name) { selected = true; execute(); }
    require(selected, "unknown public_declarations case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::public_declarations_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_public_declarations);
}
