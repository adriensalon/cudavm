#include "conformance.hpp"
#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
#include <cuda/cudavm_cuda.h>
#ifdef CUDAVM_CONFORMANCE_EXECUTION
#include <nvrtc.h>
#ifndef CUDAVM_CONFORMANCE_SOURCE_ONLY
#include <nvvm.h>
#include <nvJitLink.h>
#endif
#endif
#endif
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace cudavm {
#ifdef CUDAVM_CONFORMANCE_REGISTRATION
template<class T> void registration_word(std::vector<unsigned char>& bytes, size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
std::vector<unsigned char> registration_ptx_image(std::string_view source, unsigned major, unsigned minor, unsigned architecture) {
    constexpr size_t header_bytes = 72;
    const size_t payload_bytes = (source.size() + 1 + 7) & ~size_t(7);
    std::vector<unsigned char> image(16 + header_bytes + payload_bytes);
    registration_word<uint32_t>(image, 0, 0xba55ed50);
    registration_word<uint16_t>(image, 4, 1); registration_word<uint16_t>(image, 6, 16);
    registration_word<uint64_t>(image, 8, header_bytes + payload_bytes);
    registration_word<uint16_t>(image, 16, 1); registration_word<uint16_t>(image, 18, 0x101);
    registration_word<uint32_t>(image, 20, header_bytes); registration_word<uint64_t>(image, 24, payload_bytes);
    registration_word<uint32_t>(image, 36, 64); registration_word<uint32_t>(image, 40, (major << 16) | minor);
    registration_word<uint32_t>(image, 44, architecture);
#ifdef _WIN32
    registration_word<uint64_t>(image, 56, 0x41);
#else
    registration_word<uint64_t>(image, 56, 0x11);
#endif
    std::memcpy(image.data() + 16 + header_bytes, source.data(), source.size());
    return image;
}
#endif
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
// Start before GPU initialization. DLLs with separate CRTs and native drivers
// must inherit the redirected output at process creation, not after startup.
conformance_child_result run_conformance_child(const char* executable, const char* backend, const char* case_name,
                                               bool capture_stderr) {
    FILE* file = nullptr;
#ifdef _WIN32
    tmpfile_s(&file);
#else
    file = std::tmpfile();
#endif
    require(file != nullptr, "create child output file");
    struct file_guard { FILE* value; ~file_guard() { std::fclose(value); } } cleanup{file};
    int status = 1;
#ifdef _WIN32
    std::array<wchar_t, 32768> path{};
    auto length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    require(length && length < path.size(), "locate conformance executable");
    auto widen = [](const char* value) { std::wstring result; for (; *value; ++value) result += wchar_t(*value); return result; };
    std::wstring command = L"\"" + std::wstring(path.data(), length) + L"\" " + widen(backend) + L" " + widen(case_name) + L" --worker";
    HANDLE output = INVALID_HANDLE_VALUE;
    require(DuplicateHandle(GetCurrentProcess(), reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file))),
        GetCurrentProcess(), &output, 0, TRUE, DUPLICATE_SAME_ACCESS), "create inheritable child output handle");
    STARTUPINFOW startup{}; startup.cb = sizeof(startup); startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE); startup.hStdOutput = output;
    startup.hStdError = capture_stderr ? output : GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    const bool created = CreateProcessW(path.data(), command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(output);
    require(created, "start conformance worker");
    CloseHandle(process.hThread);
    const auto waited = WaitForSingleObject(process.hProcess, 270000);
    if (waited != WAIT_OBJECT_0) {
        TerminateProcess(process.hProcess, 1); WaitForSingleObject(process.hProcess, 5000);
        CloseHandle(process.hProcess);
        throw std::runtime_error("conformance worker exceeded its deadline");
    }
    DWORD exit_status = 1;
    const bool queried = GetExitCodeProcess(process.hProcess, &exit_status);
    CloseHandle(process.hProcess);
    require(queried, "read conformance worker status");
    status = int(exit_status);
#else
    const auto child = fork();
    require(child >= 0, "start conformance worker");
    if (child == 0) {
        if (dup2(fileno(file), STDOUT_FILENO) < 0) _exit(126);
        if (capture_stderr && dup2(fileno(file), STDERR_FILENO) < 0) _exit(126);
        execlp(executable, executable, backend, case_name, "--worker", static_cast<char*>(nullptr));
        _exit(127);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(270);
    int raw_status = 0;
    for (;;) {
        auto observed = waitpid(child, &raw_status, WNOHANG);
        if (observed == child) break;
        if (observed < 0 && errno == EINTR) continue;
        if (observed < 0 || std::chrono::steady_clock::now() >= deadline) {
            kill(child, SIGKILL); waitpid(child, &raw_status, 0);
            throw std::runtime_error("conformance worker failed to complete before its deadline");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    status = WIFEXITED(raw_status) ? WEXITSTATUS(raw_status) : 1;
#endif
    require(std::fseek(file, 0, SEEK_END) == 0, "seek child output end");
    const auto bytes = std::ftell(file);
    require(bytes >= 0 && std::fseek(file, 0, SEEK_SET) == 0, "seek child output start");
    std::string output_text(static_cast<size_t>(bytes), '\0');
    require(std::fread(output_text.data(), 1, output_text.size(), file) == output_text.size(), "read complete child output");
    return {status, std::move(output_text)};
}

int run_host_conformance(int argc, char** argv, void (*execute)()) {
    try {
        require(argc == 1 || argc == 2, "usage: <executable> [case]");
        std::cout << "host-only case=" << (argc == 2 ? argv[1] : "all") << std::endl;
        execute();
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

#ifndef CUDAVM_CONFORMANCE_HOST_ONLY
struct conformance_unavailable : std::runtime_error { using std::runtime_error::runtime_error; };
void require_capability(bool available, const std::string& reason) {
    if (!available) throw conformance_unavailable(reason);
}
struct conformance_driver_error : std::runtime_error {
    CUresult result;
    conformance_driver_error(CUresult status, const std::string& message)
        : std::runtime_error(message), result(status) {}
};
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

struct device_context {
    CUcontext value{};
    explicit device_context(CUdevice device, unsigned flags = 0) { CONFORMANCE_CUDA(cuCtxCreate(&value, nullptr, flags, device)); }
    ~device_context() { if (value) cuCtxDestroy(value); }
};

struct conformance_device {
    CUdevice value{-1};
    std::string name;
};

conformance_device select_conformance_device(std::string_view backend, void (*probe)() = nullptr, unsigned architecture = 0) {
    const std::array<std::string_view, 5> backends{"cuda", "vulkan", "directx", "opencl", "opengl"};
    const std::array<std::string_view, 5> labels{"CUDA", "Vulkan", "DirectX", "OpenCL", "OpenGL"};
    const auto found = std::find(backends.begin(), backends.end(), backend);
    require(found != backends.end(), "unknown backend: " + std::string(backend));
    const auto status = cuInit(0);
    if (status == CUDA_ERROR_NO_DEVICE || status == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) {
        std::cout << "SKIP: " << backend << " device/driver unavailable\n";
        return {};
    }
    CONFORMANCE_CUDA(status);
    int count = 0;
    CONFORMANCE_CUDA(cuDeviceGetCount(&count));
    const auto suffix = " (" + std::string(labels[found - backends.begin()]) + ")";
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        CUdevice candidate{};
        char name[256]{};
        CONFORMANCE_CUDA(cuDeviceGet(&candidate, ordinal));
        CONFORMANCE_CUDA(cuDeviceGetName(name, sizeof(name), candidate));
        if (!std::string_view(name).ends_with(suffix)) continue;
        if (backend == "cuda" && architecture) {
            int major = 0, minor = 0;
            CONFORMANCE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, candidate));
            CONFORMANCE_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, candidate));
            if (unsigned(major * 10 + minor) < architecture) {
                std::cout << "unavailable device=" << name << "; requires sm_" << architecture
                    << ", reports sm_" << major << minor << std::endl;
                continue;
            }
        }
        if (probe) {
            try {
                device_context context(candidate);
                probe();
            } catch (const conformance_driver_error& error) {
                if (error.result != CUDA_ERROR_NOT_SUPPORTED
                    && error.result != CUDA_ERROR_NO_BINARY_FOR_GPU
                    && error.result != CUDA_ERROR_UNSUPPORTED_PTX_VERSION) throw;
                std::cout << "unavailable device=" << name << "; " << error.what() << std::endl;
                continue;
            }
        }
        return {candidate, name};
    }
    std::cout << "SKIP: no " << (probe ? "executable " : "") << backend << " device\n";
    return {};
}

int run_context_conformance(int argc, char** argv, void (*execute)(), unsigned context_flags,
                            void (*probe)(), bool isolated_fault) {
    try {
        require(argc == 2, "usage: <executable> <backend>");
        const auto device = select_conformance_device(argv[1], probe);
        if (device.value == -1) return 77;
        std::cout << "backend=" << argv[1] << " device=" << device.name << std::endl;
        device_context context(device.value, context_flags);
        // The process, rather than CUDA teardown, retires this context. Fatal
        // native faults can make even destruction unsafe or non-progressing.
        if (isolated_fault) context.value = nullptr;
        execute();
        if (!isolated_fault) CONFORMANCE_CUDA(cuCtxSynchronize());
        std::cout << "PASS\n";
        return 0;
    } catch (const conformance_unavailable& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

int run_conformance(int argc, char** argv, void (*execute)(), unsigned context_flags, void (*probe)()) {
    return run_context_conformance(argc, argv, execute, context_flags, probe, false);
}

[[noreturn]] void run_isolated_fault_conformance(int argc, char** argv, void (*execute)(), void (*probe)()) {
    const int status = run_context_conformance(argc, argv, execute, 0, probe, true);
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(status);
}

#ifdef CUDAVM_CONFORMANCE_EXECUTION
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
#ifndef CUDAVM_CONFORMANCE_SOURCE_ONLY
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

#endif
std::vector<char> import_fixture(const fixture_data& fixtures, std::string_view importer) {
#ifdef CUDAVM_CONFORMANCE_SOURCE_ONLY
    require(importer == "cu", "this execution profile declares only CUDA source input");
#endif
    if (importer == "ptx") {
        std::vector<char> bytes(fixtures.ptx.begin(), fixtures.ptx.end());
        bytes.push_back('\0');
        return bytes;
    }
    if (importer == "cubin") return {fixtures.cubin.begin(), fixtures.cubin.end()};
    if (importer == "cu") {
        source_program program;
        program.check(nvrtcCreateProgram(&program.value, fixtures.source.data(), "fixture.cu", 0, nullptr, nullptr));
        const auto architecture = "-arch=sm_" + std::to_string(fixtures.architecture);
        const char* options[]{architecture.c_str(), "--std=c++17"};
        program.check(nvrtcCompileProgram(program.value, 2, options));
        size_t size = 0;
        program.check(nvrtcGetCUBINSize(program.value, &size));
        require(size > 0, "NVRTC returned an empty program");
        std::vector<char> bytes(size);
        program.check(nvrtcGetCUBIN(program.value, bytes.data()));
        return bytes;
    }
#ifndef CUDAVM_CONFORMANCE_SOURCE_ONLY
    require(importer == "nvvm", "unknown importer: " + std::string(importer));
    nvvm_program program;
    program.check(nvvmCreateProgram(&program.value));
    program.check(nvvmAddModuleToProgram(program.value, fixtures.nvvm.data(), fixtures.nvvm.size(), "fixture.ll"));
    const auto architecture = "-arch=compute_" + std::to_string(fixtures.architecture);
    const char* options[]{architecture.c_str(), "-gen-lto"};
    program.check(nvvmCompileProgram(program.value, 2, options));
    size_t size = 0;
    program.check(nvvmGetCompiledResultSize(program.value, &size));
    require(size > 0, "libNVVM returned an empty program");
    std::vector<char> ir(size);
    program.check(nvvmGetCompiledResult(program.value, ir.data()));
    link_program link;
    const auto link_architecture = "-arch=sm_" + std::to_string(fixtures.architecture);
    const char* link_options[]{link_architecture.c_str(), "-lto"};
    link.check(nvJitLinkCreate(&link.value, 2, link_options));
    link.check(nvJitLinkAddData(link.value, NVJITLINK_INPUT_LTOIR, ir.data(), ir.size(), "fixture"));
    link.check(nvJitLinkComplete(link.value));
    link.check(nvJitLinkGetLinkedCubinSize(link.value, &size));
    require(size > 0, "nvJitLink returned an empty program");
    std::vector<char> bytes(size);
    link.check(nvJitLinkGetLinkedCubin(link.value, bytes.data()));
    return bytes;
#else
    throw std::runtime_error("unknown source execution importer");
#endif
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
CUkernel loaded_program::library_kernel(const char* name) const {
    CUkernel result{};
    if (library) CONFORMANCE_CUDA(cuLibraryGetKernel(&result, library, name));
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
                    void (*execute)(const loaded_program&), unsigned load_repetitions, unsigned context_flags) {
    try {
        require(argc == 3, "usage: <executable> <backend> <cu|ptx|cubin|nvvm>");
        require(load_repetitions > 0, "at least one loader execution is required");
        const std::string backend = argv[1];
        const auto device = select_conformance_device(backend, probe_execution, fixtures.architecture);
        if (device.value == -1) return 77;
        std::cout << "backend=" << backend << " importer=" << argv[2] << " device=" << device.name << std::endl;
        device_context context(device.value, context_flags);
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
    } catch (const conformance_unavailable& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
#endif
#endif
}
