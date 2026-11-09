#include <nvrtc.h>
#include <nvPTXCompiler.h>
#include <nvJitLink.h>
#include <nvFatbin.h>
#include <nvvm.h>
#include <cuda.h>
#include <cstdio>
#include <string>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
namespace cudavm {
static bool nvvm_sdk() {
    nvvmProgram program = nullptr;
    if (nvvmCreateProgram(&program) != NVVM_SUCCESS) return false;
    const char* input = "target triple = \"nvptx64-nvidia-cuda\"\n"
        "target datalayout = \"e-p:64:64-i64:64-i128:128-v16:16-v32:32-n16:32:64\"\n"
        "define void @sdk_nvvm(){ret void}\n!nvvm.annotations=!{!0}\n"
        "!0=!{void ()* @sdk_nvvm,!\"kernel\",i32 1}\n!nvvmir.version=!{!1}\n!1=!{i32 2,i32 0,i32 3,i32 1}\n";
    size_t size = 0;
    bool valid = nvvmAddModuleToProgram(program, input, std::char_traits<char>::length(input), "sdk.ll") == NVVM_SUCCESS
        && nvvmVerifyProgram(program, 0, nullptr) == NVVM_SUCCESS && nvvmCompileProgram(program, 0, nullptr) == NVVM_SUCCESS
        && nvvmGetCompiledResultSize(program, &size) == NVVM_SUCCESS && size > 1;
    if (valid) { std::vector<char> bytes(size); valid = nvvmGetCompiledResult(program, bytes.data()) == NVVM_SUCCESS && std::string(bytes.data()).find(".entry sdk_nvvm") != std::string::npos; }
    return nvvmDestroyProgram(&program) == NVVM_SUCCESS && valid;
}
static bool nvfatbin_sdk() {
    nvFatbinHandle handle = nullptr;
    const char* source = ".version 6.3\n.target sm_75\n.address_size 64\n.visible .entry sdk_fatbin(){ret;}";
    if (nvFatbinCreate(&handle, nullptr, 0) != NVFATBIN_SUCCESS) return false;
    size_t size = 0;
    bool valid = nvFatbinAddPTX(handle, source, std::char_traits<char>::length(source), "75", "sdk", "") == NVFATBIN_SUCCESS
        && nvFatbinSize(handle, &size) == NVFATBIN_SUCCESS && size;
    if (valid) { std::vector<char> bytes(size); valid = nvFatbinGet(handle, bytes.data()) == NVFATBIN_SUCCESS; }
    return nvFatbinDestroy(&handle) == NVFATBIN_SUCCESS && valid;
}
static bool ptx_compiler_sdk() {
    const char* source = ".version 6.3\n.target sm_75\n.address_size 64\n.visible .entry sdk_ptx() {ret;}";
    nvPTXCompilerHandle handle = nullptr;
    if (nvPTXCompilerCreate(&handle, std::char_traits<char>::length(source), source) != NVPTXCOMPILE_SUCCESS) return false;
    const char* option = "--gpu-name=sm_75";
    auto status = nvPTXCompilerCompile(handle, 1, &option); size_t size = 0;
    bool valid = status == NVPTXCOMPILE_SUCCESS && nvPTXCompilerGetCompiledProgramSize(handle, &size) == NVPTXCOMPILE_SUCCESS && size;
    if (valid) { std::vector<char> image(size); valid = nvPTXCompilerGetCompiledProgram(handle, image.data()) == NVPTXCOMPILE_SUCCESS; }
    return nvPTXCompilerDestroy(&handle) == NVPTXCOMPILE_SUCCESS && valid;
}
static bool nvjitlink_sdk() {
    nvJitLinkHandle handle = nullptr; const char* options[]{"-arch=sm_75", "-lto"};
    if (nvJitLinkCreate(&handle, 2, options) != NVJITLINK_SUCCESS) return false;
    const char* sources[]{"extern \"C\" __device__ int helper(int); extern \"C\" __global__ void sdk_link(int* out){*out=helper(35);}",
                          "extern \"C\" __device__ int helper(int x){return x+7;}"};
    bool valid = true;
    for (const auto* source : sources) {
        nvrtcProgram program = nullptr; const char* option = "-dlto"; size_t size = 0;
        if (nvrtcCreateProgram(&program, source, "same-name.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) { valid = false; break; }
        valid = nvrtcCompileProgram(program, 1, &option) == NVRTC_SUCCESS && nvrtcGetLTOIRSize(program, &size) == NVRTC_SUCCESS && size;
        std::vector<char> bytes(size);
        if (valid) valid = nvrtcGetLTOIR(program, bytes.data()) == NVRTC_SUCCESS;
        nvrtcDestroyProgram(&program);
        if (valid) valid = nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, bytes.data(), bytes.size(), nullptr) == NVJITLINK_SUCCESS;
        if (!valid) break;
    }
    size_t size = 0;
    if (valid) valid = nvJitLinkComplete(handle) == NVJITLINK_SUCCESS && nvJitLinkGetLinkedCubinSize(handle, &size) == NVJITLINK_SUCCESS && size;
    if (valid) { std::vector<char> bytes(size); valid = nvJitLinkGetLinkedCubin(handle, bytes.data()) == NVJITLINK_SUCCESS; }
    return nvJitLinkDestroy(&handle) == NVJITLINK_SUCCESS && valid;
}
}

int main() {
    using namespace cudavm;
#if defined(_WIN32)
    // An attempted compiler helper process must fail, even if a maintainer's
    // clang.exe remains installed. The frontend has to run in this process.
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
    limits.BasicLimitInformation.ActiveProcessLimit = 1;
    if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) ||
        !AssignProcessToJobObject(job, GetCurrentProcess())) {
        std::fputs("Could not prohibit child processes for in-process compiler verification\n", stderr);
        return 1;
    }
#endif
    auto cold = [] { int count = 0; return cuDeviceGetCount(&count) == CUDA_ERROR_NOT_INITIALIZED; };
    if (!cold() || !ptx_compiler_sdk() || !cold() || !nvjitlink_sdk() || !cold() || !nvfatbin_sdk() || !cold() || !nvvm_sdk() || !cold()) return 1;
    std::puts("PASS public compiler APIs without GPU initialization or compiler helper processes");
#if defined(_WIN32)
    CloseHandle(job);
#endif
}
