#include <cuda.h>
#include <cuda_runtime_api.h>
#ifdef HOST_WITH_NVRTC
#include <nvrtc.h>
#endif
#ifdef HOST_WITH_NVVM
#include <nvvm.h>
#endif
#ifdef HOST_WITH_NVJITLINK
#include <nvJitLink.h>
#endif
#ifdef HOST_WITH_NVFATBIN
#include <nvFatbin.h>
#endif
#ifdef HOST_WITH_NVPTXCOMPILER
#include <nvPTXCompiler.h>
#endif
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int host_c_contract(void);
void check(bool value, const char* operation) { if (!value) throw std::runtime_error(operation); }
int main() {
    try {
        check(host_c_contract() == 0, "independent C header/link contract");
        CUcontext context{};
        check(cuCtxGetCurrent(&context) == CUDA_ERROR_NOT_INITIALIZED, "Host calls begin without Driver initialization");
        CUmodule module{};
        check(cuModuleLoadFatBinary(&module, nullptr) == CUDA_ERROR_NOT_INITIALIZED && !module,
            "fatbinary loader export preserves uninitialized Driver state");
        struct { unsigned before = 0x12345678; int value = 0; unsigned after = 0x87654321; } driver, runtime;
        check(cuDriverGetVersion(&driver.value) == CUDA_SUCCESS && driver.value == CUDA_VERSION, "Driver header/library version");
        check(cudaRuntimeGetVersion(&runtime.value) == cudaSuccess && runtime.value == CUDART_VERSION, "Runtime header/library version");
        check(driver.before == 0x12345678 && driver.after == 0x87654321 && runtime.before == 0x12345678 && runtime.after == 0x87654321, "version output guards");
        check(std::strcmp(cudaGetErrorName(cudaErrorInvalidValue), "cudaErrorInvalidValue") == 0, "Runtime export call");
#ifdef HOST_WITH_NVRTC
        int rtc_major = 0, rtc_minor = -1;
        check(nvrtcVersion(&rtc_major, &rtc_minor) == NVRTC_SUCCESS && rtc_major > 0 && rtc_minor >= 0, "NVRTC version");
        nvrtcProgram program{};
        check(nvrtcCreateProgram(&program, "extern \"C\" __global__ void sdk_owned_kernel(unsigned* out) { out[threadIdx.x] = threadIdx.x ^ 0x13579bdfu; }", "sdk.cu", 0, nullptr, nullptr) == NVRTC_SUCCESS, "NVRTC CreateProgram");
        struct retire { nvrtcProgram& value; ~retire() { if (value) nvrtcDestroyProgram(&value); } } owner{program};
        const char* options[]{"--gpu-architecture=compute_75", "--std=c++17"};
        check(nvrtcCompileProgram(program, 2, options) == NVRTC_SUCCESS, "NVRTC compiler dependency closure");
        size_t size = 0; check(nvrtcGetPTXSize(program, &size) == NVRTC_SUCCESS && size > 1, "PTX extent");
        std::vector<char> bytes(size + 32, '!');
        check(nvrtcGetPTX(program, bytes.data() + 16) == NVRTC_SUCCESS, "PTX readback");
        check(nvrtcDestroyProgram(&program) == NVRTC_SUCCESS && !program, "compiler retirement");
        check(bytes[16 + size - 1] == 0 && std::strstr(bytes.data() + 16, ".entry sdk_owned_kernel("), "owned PTX outlives compiler handle");
        for (size_t i = 0; i < 16; ++i) check(bytes[i] == '!' && bytes[size + 16 + i] == '!', "PTX byte guards");
#endif
#ifdef HOST_WITH_NVVM
        int nvvm_major = 0, nvvm_minor = -1;
        check(nvvmVersion(&nvvm_major, &nvvm_minor) == NVVM_SUCCESS && nvvm_major > 0 && nvvm_minor >= 0, "libNVVM export");
#endif
#ifdef HOST_WITH_NVJITLINK
        unsigned link_major = 0, link_minor = 0;
        check(nvJitLinkVersion(&link_major, &link_minor) == NVJITLINK_SUCCESS && link_major > 0, "nvJitLink export");
#endif
#ifdef HOST_WITH_NVFATBIN
        unsigned fatbin_major = 0, fatbin_minor = 0;
        check(nvFatbinVersion(&fatbin_major, &fatbin_minor) == NVFATBIN_SUCCESS && fatbin_major > 0, "nvFatbin export");
#endif
#ifdef HOST_WITH_NVPTXCOMPILER
        unsigned ptx_major = 0, ptx_minor = 0;
        check(nvPTXCompilerGetVersion(&ptx_major, &ptx_minor) == NVPTXCOMPILE_SUCCESS && ptx_major > 0, "PTX Compiler export");
#endif
        check(cuCtxGetCurrent(&context) == CUDA_ERROR_NOT_INITIALIZED, "Host/compiler calls preserve uninitialized Driver");
        std::puts("PASS configured shared/static host linkage and API calls"); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
