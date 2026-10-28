#include "conformance.hpp"
#include <cuda/cudavm_cuda.h>
#include <nvrtc.h>
#include <nvvm.h>
#include <nvJitLink.h>
#include <algorithm>
#include <iostream>

namespace cudavm {
struct conformance_driver_error : std::runtime_error {
    CUresult result;
    conformance_driver_error(CUresult status, const std::string& message)
        : std::runtime_error(message), result(status) {}
};
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void check_driver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    const char* diagnostic = cudavmGetDiagnostic();
    throw conformance_driver_error(result, std::string(operation) + ": " + (name ? name : "unknown Driver error")
        + (diagnostic && *diagnostic ? std::string("; ") + diagnostic : ""));
}
void expect_words(std::span<const uint32_t> actual, std::span<const uint32_t> expected,
                  const std::string& context) {
    require(actual.size() == expected.size(), context + ": output extent");
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i])
            throw std::runtime_error(context + ": word " + std::to_string(i)
                + " got " + std::to_string(actual[i]) + ", expected " + std::to_string(expected[i]));
}

struct source_program {
    nvrtcProgram value{};
    ~source_program() { if (value) nvrtcDestroyProgram(&value); }
    void check(nvrtcResult result) const {
        if (result == NVRTC_SUCCESS) return;
        size_t size = 0;
        nvrtcGetProgramLogSize(value, &size);
        std::string log(size, '\0');
        if (size) nvrtcGetProgramLog(value, log.data());
        throw std::runtime_error(std::string(nvrtcGetErrorString(result)) + ": " + log);
    }
};
struct nvvm_program {
    nvvmProgram value{};
    ~nvvm_program() { if (value) nvvmDestroyProgram(&value); }
    void check(nvvmResult result) const {
        if (result == NVVM_SUCCESS) return;
        size_t size = 0;
        nvvmGetProgramLogSize(value, &size);
        std::string log(size, '\0');
        if (size) nvvmGetProgramLog(value, log.data());
        throw std::runtime_error(std::string(nvvmGetErrorString(result)) + ": " + log);
    }
};
struct link_program {
    nvJitLinkHandle value{};
    ~link_program() { if (value) nvJitLinkDestroy(&value); }
    void check(nvJitLinkResult result) const {
        if (result == NVJITLINK_SUCCESS) return;
        size_t size = 0;
        nvJitLinkGetErrorLogSize(value, &size);
        std::string log(size + 1, '\0');
        nvJitLinkGetErrorLog(value, log.data());
        throw std::runtime_error("nvJitLink error " + std::to_string(result) + ": " + log);
    }
};

std::vector<char> import_fixture(const fixture_data& fixtures, std::string_view importer) {
    if (importer == "ptx") {
        std::vector<char> bytes(fixtures.ptx.begin(), fixtures.ptx.end());
        bytes.push_back('\0');
        return bytes;
    }
    if (importer == "cubin") return {fixtures.cubin.begin(), fixtures.cubin.end()};
    if (importer == "cu") {
        source_program program;
        program.check(nvrtcCreateProgram(&program.value, fixtures.source.data(), "fixture.cu", 0, nullptr, nullptr));
        const char* options[]{"-arch=sm_75", "--std=c++17"};
        program.check(nvrtcCompileProgram(program.value, 2, options));
        size_t size = 0;
        program.check(nvrtcGetCUBINSize(program.value, &size));
        require(size > 0, "NVRTC returned an empty program");
        std::vector<char> bytes(size);
        program.check(nvrtcGetCUBIN(program.value, bytes.data()));
        return bytes;
    }
    require(importer == "nvvm", "unknown importer: " + std::string(importer));
    nvvm_program program;
    program.check(nvvmCreateProgram(&program.value));
    program.check(nvvmAddModuleToProgram(program.value, fixtures.nvvm.data(), fixtures.nvvm.size(), "fixture.ll"));
    const char* options[]{"-arch=compute_75", "-gen-lto"};
    program.check(nvvmCompileProgram(program.value, 2, options));
    size_t size = 0;
    program.check(nvvmGetCompiledResultSize(program.value, &size));
    require(size > 0, "libNVVM returned an empty program");
    std::vector<char> ir(size);
    program.check(nvvmGetCompiledResult(program.value, ir.data()));
    link_program link;
    const char* link_options[]{"-arch=sm_75", "-lto"};
    link.check(nvJitLinkCreate(&link.value, 2, link_options));
    link.check(nvJitLinkAddData(link.value, NVJITLINK_INPUT_LTOIR, ir.data(), ir.size(), "fixture"));
    link.check(nvJitLinkComplete(link.value));
    link.check(nvJitLinkGetLinkedCubinSize(link.value, &size));
    require(size > 0, "nvJitLink returned an empty program");
    std::vector<char> bytes(size);
    link.check(nvJitLinkGetLinkedCubin(link.value, bytes.data()));
    return bytes;
}

loaded_program::loaded_program(const void* image, bool use_library) {
    if (use_library) CONFORMANCE_CUDA(cuLibraryLoadData(&library, image, nullptr, nullptr, 0, nullptr, nullptr, 0));
    else CONFORMANCE_CUDA(cuModuleLoadData(&module, image));
}
loaded_program::~loaded_program() {
    if (library) cuLibraryUnload(library);
    if (module) cuModuleUnload(module);
}
CUfunction loaded_program::function(const char* name) const {
    CUfunction result{};
    if (library) {
        CUkernel kernel{};
        CONFORMANCE_CUDA(cuLibraryGetKernel(&kernel, library, name));
        CONFORMANCE_CUDA(cuKernelGetFunction(&result, kernel));
    } else CONFORMANCE_CUDA(cuModuleGetFunction(&result, module, name));
    return result;
}
CUdeviceptr loaded_program::global(const char* name, size_t expected_size) const {
    CUdeviceptr result = 0;
    size_t size = 0;
    if (library) CONFORMANCE_CUDA(cuLibraryGetGlobal(&result, &size, library, name));
    else CONFORMANCE_CUDA(cuModuleGetGlobal(&result, &size, module, name));
    require(result && size == expected_size, std::string("global symbol extent: ") + name);
    return result;
}

struct device_context {
    CUcontext value{};
    explicit device_context(CUdevice device) { CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, 0, device)); }
    ~device_context() { if (value) cuCtxDestroy(value); }
};

// Availability is checked independently of the slice's input and features.
// Enumerating an OpenCL endpoint does not imply SPIR-V ingestion.
void probe_execution() {
    static constexpr char probe_ptx[] = R"(
.version 6.5
.target sm_75
.address_size 64
.visible .entry conformance_probe(.param .u64 out) {
 .reg .b64 %out;
 .reg .b32 %value;
 ld.param.u64 %out,[out];
 mov.u32 %value,0x43564d31;
 st.global.u32 [%out],%value;
 ret;
}
)";
    loaded_program probe(probe_ptx, false);
    const auto kernel = probe.function("conformance_probe");
    device_buffer output(sizeof(uint32_t));
    CONFORMANCE_CUDA(cuMemsetD32(output.value, 0, 1));
    void* args[]{&output.value};
    CONFORMANCE_CUDA(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    uint32_t actual = 0;
    CONFORMANCE_CUDA(cuMemcpyDtoH(&actual, output.value, sizeof(actual)));
    require(actual == 0x43564d31, "backend availability probe produced an incorrect result");
}

int run_conformance(int argc, char** argv, const fixture_data& fixtures,
                    void (*execute)(const loaded_program&), unsigned load_repetitions) {
    try {
        require(argc == 3, "usage: <executable> <backend> <cu|ptx|cubin|nvvm>");
        require(load_repetitions > 0, "at least one loader execution is required");
        const std::string backend = argv[1];
        const std::array<std::string_view, 5> backends{"cuda", "vulkan", "directx", "opencl", "opengl"};
        const std::array<std::string_view, 5> labels{"CUDA", "Vulkan", "DirectX", "OpenCL", "OpenGL"};
        auto found = std::find(backends.begin(), backends.end(), backend);
        require(found != backends.end(), "unknown backend: " + backend);
        auto status = cuInit(0);
        if (status == CUDA_ERROR_NO_DEVICE || status == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) {
            std::cout << "SKIP: " << backend << " device/driver unavailable\n";
            return 77;
        }
        CONFORMANCE_CUDA(status);
        int count = 0;
        CONFORMANCE_CUDA(cuDeviceGetCount(&count));
        CUdevice device = -1;
        char name[256]{};
        const auto suffix = " (" + std::string(labels[found - backends.begin()]) + ")";
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            CUdevice candidate{};
            CONFORMANCE_CUDA(cuDeviceGet(&candidate, ordinal));
            CONFORMANCE_CUDA(cuDeviceGetName(name, sizeof(name), candidate));
            if (!std::string_view(name).ends_with(suffix)) continue;
            try {
                device_context context(candidate);
                probe_execution();
                device = candidate;
                break;
            } catch (const conformance_driver_error& error) {
                if (error.result != CUDA_ERROR_NOT_SUPPORTED
                    && error.result != CUDA_ERROR_NO_BINARY_FOR_GPU
                    && error.result != CUDA_ERROR_UNSUPPORTED_PTX_VERSION) throw;
                std::cout << "unavailable device=" << name << "; " << error.what() << std::endl;
            }
        }
        if (device == -1) {
            std::cout << "SKIP: no executable " << backend << " device\n";
            return 77;
        }
        std::cout << "backend=" << backend << " importer=" << argv[2] << " device=" << name << std::endl;
        device_context context(device);
        // Compiler handles are destroyed before either loader sees the copy.
        const auto image = import_fixture(fixtures, argv[2]);
        for (unsigned repeat = 0; repeat < load_repetitions; ++repeat)
        for (bool library : {false, true}) {
            std::cout << "loader=" << (library ? "library" : "module") << std::endl;
            loaded_program program(image.data(), library);
            execute(program);
            CONFORMANCE_CUDA(cuCtxSynchronize());
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
}
