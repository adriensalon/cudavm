#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "perf_framework.hpp"
#ifdef CUDAVM_PERF_CUDA
#include <cuda/cudavm_cuda.h>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace cudavm {

static size_t parse_perf_count(std::string_view text) {
    size_t value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    perf_require(error == std::errc{} && end == text.data() + text.size(),
                 "invalid count: " + std::string(text));
    return value;
}

static void write_perf_json_string(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::cout << '"';
    for (unsigned char character : value) {
        if (character == '"' || character == '\\') std::cout << '\\' << char(character);
        else if (character < 0x20)
            std::cout << "\\u00" << digits[character >> 4] << digits[character & 15];
        else std::cout << char(character);
    }
    std::cout << '"';
}

static void report_perf(const perf_options& options, const perf_result& result) {
    perf_require(result.milliseconds.size() == options.samples, "incorrect performance sample count");
    perf_require(options.backend == "host" || !result.device.empty(), "GPU results require a device name");
    perf_require(options.backend == "host" || result.device_ordinal >= 0, "GPU results require the selected logical ordinal");
    perf_require(result.device_uuid.empty() || result.device_uuid.size() == 32, "invalid native UUID extent");
    perf_require(result.device_luid.empty() || result.device_luid.size() == 16, "invalid native LUID extent");
    for (double sample : result.milliseconds)
        perf_require(std::isfinite(sample) && sample >= 0, "invalid performance sample");
    auto sorted = result.milliseconds;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    const double median = sorted.size() % 2 ? sorted[middle] : sorted[middle - 1] / 2 + sorted[middle] / 2;
    std::cout << "{\"slice\":";
    write_perf_json_string(options.slice);
    std::cout << ",\"backend\":";
    write_perf_json_string(options.backend);
    std::cout << ",\"case\":";
    write_perf_json_string(options.case_name);
    std::cout << ",\"device\":";
    write_perf_json_string(result.device);
    std::cout << ",\"device_uuid\":";
    write_perf_json_string(result.device_uuid);
    std::cout << ",\"device_luid\":";
    write_perf_json_string(result.device_luid);
    std::cout << ",\"device_node_mask\":" << result.device_node_mask << ",\"device_pci_address\":";
    write_perf_json_string(result.device_pci_address);
    std::cout << ",\"device_ordinal\":" << result.device_ordinal << ",\"build_type\":";
    write_perf_json_string(CUDAVM_PERF_BUILD_TYPE);
    std::cout << ",\"host_system\":";
    write_perf_json_string(CUDAVM_PERF_HOST_SYSTEM);
    std::cout << ",\"host_processor\":";
    write_perf_json_string(CUDAVM_PERF_HOST_PROCESSOR);
    std::cout << ",\"compiler\":";
    write_perf_json_string(CUDAVM_PERF_COMPILER);
    std::cout << ",\"device_filter\":";
    const auto* filter = std::getenv("CUDAVM_DEVICES");
    write_perf_json_string(filter ? filter : "unset");
    std::cout << ",\"parameters\":{";
    for (size_t index = 0; index < result.parameters.size(); ++index) {
        if (index) std::cout << ',';
        write_perf_json_string(result.parameters[index].first);
        std::cout << ':';
        write_perf_json_string(result.parameters[index].second);
    }
    std::cout << '}';
    std::cout << ",\"api_linkage\":\"shared\",\"clock\":\"steady_clock\",\"unit\":\"ms/iteration\",\"warmup\":" << options.warmup
              << ",\"iterations\":" << options.iterations << ",\"samples\":[" << std::setprecision(17);
    for (size_t index = 0; index < result.milliseconds.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << result.milliseconds[index];
    }
    std::cout << "],\"minimum\":" << sorted.front() << ",\"median\":" << median << "}\n";
}

void perf_require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void perf_expect_bytes(std::span<const unsigned char> actual, std::span<const unsigned char> expected,
                       std::string_view context) {
    perf_require(actual.size() == expected.size(), std::string(context) + ": byte extent");
    for (size_t index = 0; index < actual.size(); ++index)
        if (actual[index] != expected[index])
            throw std::runtime_error(std::string(context) + ": byte " + std::to_string(index)
                + " got " + std::to_string(actual[index]) + ", expected " + std::to_string(expected[index]));
}

std::vector<unsigned char> perf_input_bytes(size_t bytes, uint32_t seed) {
    std::vector<unsigned char> result(bytes);
    for (size_t index = 0; index < bytes; ++index) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        result[index] = static_cast<unsigned char>(seed >> 24);
    }
    return result;
}

#ifdef CUDAVM_PERF_CUDA
perf_driver_error::perf_driver_error(CUresult status, const std::string& message)
    : std::runtime_error(message), result(status) {}

void perf_check_driver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    const auto* diagnostic = cudavmGetDiagnostic();
    throw perf_driver_error(result, std::string(operation) + ": " + (name ? name : "unknown Driver error")
        + (diagnostic && *diagnostic ? std::string("; ") + diagnostic : ""));
}

static std::string perf_identity_hex(std::span<const char> bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (unsigned char byte : bytes) {
        result += digits[byte >> 4];
        result += digits[byte & 15];
    }
    return result;
}

perf_device select_perf_device(std::string_view backend, void (*probe)()) {
    constexpr std::array<std::string_view, 5> backends{"cuda", "vulkan", "directx", "opencl", "opengl"};
    constexpr std::array<std::string_view, 5> labels{"CUDA", "Vulkan", "DirectX", "OpenCL", "OpenGL"};
    const auto selected = std::find(backends.begin(), backends.end(), backend);
    perf_require(selected != backends.end(), "GPU slice requires a GPU backend");
    const auto status = cuInit(0);
    if (status == CUDA_ERROR_NO_DEVICE || status == CUDA_ERROR_SYSTEM_DRIVER_MISMATCH)
        throw perf_unavailable(std::string(backend) + " device/driver unavailable");
    PERF_CUDA(status);
    int count{};
    PERF_CUDA(cuDeviceGetCount(&count));
    const auto suffix = " (" + std::string(labels[selected - backends.begin()]) + ")";
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        CUdevice device{};
        char name[256]{};
        PERF_CUDA(cuDeviceGet(&device, ordinal));
        PERF_CUDA(cuDeviceGetName(name, sizeof(name), device));
        if (!std::string_view(name).ends_with(suffix)) continue;
        perf_device result{device, ordinal, name};
        CUuuid uuid{};
        auto identity_status = cuDeviceGetUuid(&uuid, device);
        if (identity_status == CUDA_SUCCESS) result.uuid = perf_identity_hex(uuid.bytes);
        else if (identity_status != CUDA_ERROR_NOT_SUPPORTED) PERF_CUDA(identity_status);
        std::array<char, 8> luid{};
        identity_status = cuDeviceGetLuid(luid.data(), &result.node_mask, device);
        if (identity_status == CUDA_SUCCESS) result.luid = perf_identity_hex(luid);
        else if (identity_status != CUDA_ERROR_NOT_SUPPORTED) PERF_CUDA(identity_status);
        char pci[32]{};
        identity_status = cuDeviceGetPCIBusId(pci, sizeof(pci), device);
        if (identity_status == CUDA_SUCCESS) result.pci_address = pci;
        else if (identity_status != CUDA_ERROR_NOT_SUPPORTED) PERF_CUDA(identity_status);
        if (probe) {
            try {
                perf_context context(device);
                probe();
                context.close();
            } catch (const perf_unavailable& error) {
                std::cerr << "unavailable endpoint=" << name << " ordinal=" << ordinal << "; " << error.what() << '\n';
                continue;
            }
        }
        return result;
    }
    throw perf_unavailable("no " + std::string(backend) + " device matches the caller's filters"
        + (probe ? " and the declared execution prerequisite" : ""));
}

void perf_device::describe(perf_result& result) const {
    result.device = name;
    result.device_uuid = uuid;
    result.device_luid = luid;
    result.device_node_mask = node_mask;
    result.device_pci_address = pci_address;
    result.device_ordinal = ordinal;
    int version{};
    PERF_CUDA(cuDriverGetVersion(&version));
    result.parameters.emplace_back("driver_api_version", std::to_string(version));
}

perf_context::perf_context(CUdevice device) { PERF_CUDA(cuCtxCreate(&value, nullptr, 0, device)); }
perf_context::~perf_context() { if (value) cuCtxDestroy(value); }
void perf_context::close() {
    if (value) { PERF_CUDA(cuCtxDestroy(value)); value = nullptr; }
}

perf_primary_context::perf_primary_context(CUdevice selected) : device(selected) {
    PERF_CUDA(cuDevicePrimaryCtxRetain(&value, device));
}
perf_primary_context::~perf_primary_context() { if (value) cuDevicePrimaryCtxRelease(device); }
void perf_primary_context::close() {
    if (value) { PERF_CUDA(cuDevicePrimaryCtxRelease(device)); value = nullptr; }
}

perf_device_buffer::perf_device_buffer(size_t bytes) { PERF_CUDA(cuMemAlloc(&value, bytes)); }
perf_device_buffer::~perf_device_buffer() { if (value) cuMemFree(value); }
void perf_device_buffer::close() {
    if (value) { PERF_CUDA(cuMemFree(value)); value = 0; }
}

perf_stream::perf_stream() { PERF_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
perf_stream::~perf_stream() { if (value) cuStreamDestroy(value); }
void perf_stream::synchronize() const { PERF_CUDA(cuStreamSynchronize(value)); }
void perf_stream::close() {
    if (value) { PERF_CUDA(cuStreamDestroy(value)); value = nullptr; }
}

perf_module::perf_module(const void* image) { PERF_CUDA(cuModuleLoadData(&value, image)); }
perf_module::~perf_module() { if (value) cuModuleUnload(value); }
CUfunction perf_module::function(const char* name) const {
    CUfunction result{};
    PERF_CUDA(cuModuleGetFunction(&result, value, name));
    return result;
}
void perf_module::close() {
    if (value) { PERF_CUDA(cuModuleUnload(value)); value = nullptr; }
}

void require_perf_execution() {
    static constexpr char input[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64
.visible .entry perf_execution_probe(.param .u64 output) {
    .reg .b64 %pointer;
    .reg .b32 %value;
    ld.param.u64 %pointer, [output];
    mov.u32 %value, 0x1943b6ef;
    st.global.u32 [%pointer], %value;
    ret;
}
)ptx";
    try {
        perf_module module(input);
        perf_device_buffer output(sizeof(uint32_t));
        perf_stream stream;
        PERF_CUDA(cuMemsetD32(output.value, 0, 1));
        auto pointer = output.value;
        void* arguments[]{&pointer};
        PERF_CUDA(cuLaunchKernel(module.function("perf_execution_probe"), 1, 1, 1, 1, 1, 1, 0,
                                 stream.value, arguments, nullptr));
        stream.synchronize();
        uint32_t actual{};
        PERF_CUDA(cuMemcpyDtoH(&actual, output.value, sizeof(actual)));
        perf_require(actual == 0x1943b6ef, "independent execution prerequisite produced the expected word");
        stream.close();
        output.close();
        module.close();
    } catch (const perf_driver_error& error) {
        if (error.result != CUDA_ERROR_NOT_SUPPORTED && error.result != CUDA_ERROR_NO_BINARY_FOR_GPU
            && error.result != CUDA_ERROR_UNSUPPORTED_PTX_VERSION) throw;
        throw perf_unavailable(std::string("selected endpoint cannot execute the independent PTX 6.0/sm_50 prerequisite: ")
            + error.what());
    }
}
#endif

#ifdef CUDAVM_PERF_NVRTC
void perf_check_nvrtc(nvrtcResult result, const char* operation) {
    if (result != NVRTC_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + nvrtcGetErrorString(result));
}

perf_source_program::perf_source_program(const char* source, const char* name) {
    perf_check_nvrtc(nvrtcCreateProgram(&value, source, name, 0, nullptr, nullptr), "nvrtcCreateProgram");
}
perf_source_program::~perf_source_program() { if (value) nvrtcDestroyProgram(&value); }
void perf_source_program::compile(std::span<const char* const> options) const {
    const auto status = nvrtcCompileProgram(value, static_cast<int>(options.size()), options.data());
    if (status != NVRTC_SUCCESS) {
        size_t size{};
        perf_check_nvrtc(nvrtcGetProgramLogSize(value, &size), "nvrtcGetProgramLogSize");
        std::vector<char> log(size + 1);
        perf_check_nvrtc(nvrtcGetProgramLog(value, log.data()), "nvrtcGetProgramLog");
        throw std::runtime_error(std::string("nvrtcCompileProgram: ") + nvrtcGetErrorString(status) + "; " + log.data());
    }
}
std::vector<char> perf_source_program::ptx() const {
    size_t size{};
    perf_check_nvrtc(nvrtcGetPTXSize(value, &size), "nvrtcGetPTXSize");
    perf_require(size > 1 && size < 64 * 1024 * 1024, "bounded nonempty PTX output");
    std::vector<char> result(size + 16, '\x59');
    perf_check_nvrtc(nvrtcGetPTX(value, result.data()), "nvrtcGetPTX");
    for (size_t index = size; index < result.size(); ++index)
        perf_require(result[index] == '\x59', "NVRTC PTX output guard");
    perf_require(result[size - 1] == '\0', "NVRTC PTX terminal NUL");
    result.resize(size);
    return result;
}
void perf_source_program::close() {
    if (value) perf_check_nvrtc(nvrtcDestroyProgram(&value), "nvrtcDestroyProgram");
}
#endif

perf_result measure_perf(const perf_options& options, const std::function<void()>& operation,
                         const std::function<void()>& synchronize) {
    perf_require(options.samples > 0 && options.iterations > 0, "samples and iterations must be positive");
    perf_require(bool(operation), "missing performance operation");
    perf_result result;
    result.milliseconds.reserve(options.samples);
    const auto batch = [&] {
        for (size_t iteration = 0; iteration < options.iterations; ++iteration) operation();
        if (synchronize) synchronize();
    };
    for (size_t warmup = 0; warmup < options.warmup; ++warmup) {
        if (synchronize) synchronize();
        batch();
    }
    for (size_t sample = 0; sample < options.samples; ++sample) {
        if (synchronize) synchronize();
        const auto start = std::chrono::steady_clock::now();
        batch();
        const auto end = std::chrono::steady_clock::now();
        result.milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count() / options.iterations);
    }
    return result;
}

int run_perf(int argc, char** argv, std::string_view slice,
             perf_result (*execute)(const perf_options&)) {
    try {
        perf_require(argc >= 3, "usage: <executable> <backend|host> <case> [--warmup N] [--samples N] [--iterations N]");
        perf_options options{slice, argv[1], argv[2]};
        constexpr std::array<std::string_view, 6> backends{"host", "cuda", "vulkan", "directx", "opencl", "opengl"};
        perf_require(std::find(backends.begin(), backends.end(), options.backend) != backends.end(), "unknown backend");
        for (int argument = 3; argument < argc; argument += 2) {
            perf_require(argument + 1 < argc, "performance option requires a count");
            const std::string_view name = argv[argument];
            const auto value = parse_perf_count(argv[argument + 1]);
            if (name == "--warmup") options.warmup = value;
            else if (name == "--samples") options.samples = value;
            else if (name == "--iterations") options.iterations = value;
            else throw std::runtime_error("unknown performance option: " + std::string(name));
        }
        perf_require(options.samples > 0 && options.iterations > 0, "samples and iterations must be positive");
        perf_require(execute != nullptr, "missing performance slice");
        report_perf(options, execute(options));
        return 0;
    } catch (const perf_unavailable& error) {
        std::cerr << "SKIP: " << error.what() << '\n';
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

}
