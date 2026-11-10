// Regenerated only by produce.py with NVIDIA's SDK. Acceptance executes the
// pinned executable without recompiling its host code, Runtime or GPU image.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <array>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

__global__ void transform(unsigned* output, const unsigned* input, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i] * 3u + i + 7u;
}

namespace cudavm {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorName(status));
}
void origins(const std::filesystem::path& expected, bool shader) {
    auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    require(snapshot != INVALID_HANDLE_VALUE, "enumerate loaded module origins");
    MODULEENTRY32W entry{}; entry.dwSize = sizeof(entry);
    bool driver = false, compiler = false;
    if (Module32FirstW(snapshot, &entry)) do {
        std::filesystem::path path(entry.szExePath);
        std::wstring name = path.filename().wstring();
        for (auto& c : name) c = wchar_t(towlower(c));
        if (name == L"nvcuda.dll") {
            require(!driver && std::filesystem::equivalent(path, expected), "Driver loaded from the selected deployment");
            driver = true; std::printf("driver=%s\n", path.string().c_str());
        }
        if (name == L"cudavm_core.dll") {
            require(shader && std::filesystem::equivalent(path, expected.parent_path() / name), "compiler loaded from the selected deployment");
            compiler = true; std::printf("compiler=%s\n", path.string().c_str());
        }
        require(name.find(L"cudart") == std::wstring::npos, "no dynamic Runtime DLL is involved");
    } while (Module32NextW(snapshot, &entry));
    CloseHandle(snapshot);
    require(driver && (!shader || compiler), "required deployed modules were actually used");
}
void round(unsigned iteration) {
    constexpr unsigned capacity = 80, offset = 3, count = 53, guard = 0xa95b371d;
    std::array<unsigned, capacity> source{}, output{}, read_input{};
    for (unsigned i = 0; i < capacity; ++i) source[i] = i * 17 + 1 + iteration * 31;
    output.fill(guard);
    unsigned *input = nullptr, *result = nullptr;
    cudaStream_t stream{}; cudaEvent_t event{};
    check(cudaMalloc(&input, sizeof(source)), "allocate input");
    check(cudaMalloc(&result, sizeof(output)), "allocate output");
    check(cudaMemcpy(input, source.data(), sizeof(source), cudaMemcpyHostToDevice), "upload input");
    check(cudaMemcpy(result, output.data(), sizeof(output), cudaMemcpyHostToDevice), "upload guards");
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create event");
    transform<<<3, 32, 0, stream>>>(result + offset, input + offset, count);
    check(cudaGetLastError(), "launch original NVIDIA cubin");
    check(cudaEventRecord(event, stream), "record event");
    check(cudaEventSynchronize(event), "wait for kernel event");
    check(cudaStreamSynchronize(stream), "synchronize stream");
    check(cudaMemcpy(output.data(), result, sizeof(output), cudaMemcpyDeviceToHost), "read output");
    check(cudaMemcpy(read_input.data(), input, sizeof(read_input), cudaMemcpyDeviceToHost), "read input");
    require(source == read_input, "kernel preserves the complete input allocation");
    for (unsigned i = 0; i < capacity; ++i)
        require(output[i] == (i >= offset && i < offset + count ? source[i] * 3u + i - offset + 7u : guard),
            "kernel output and both allocation guards match the independent integer oracle");
    check(cudaEventDestroy(event), "destroy event");
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(input), "free input"); check(cudaFree(result), "free output");
    check(cudaDeviceReset(), "reset primary context");
    std::printf("PASS: round=%u allocation/copy/kernel/stream/event/reset\n", iteration);
}
}
int main(int argc, char** argv) {
    try {
        cudavm::require(argc == 3, "usage: runtime_client.exe expected-driver backend");
        int version = 0; cudavm::check(cudaRuntimeGetVersion(&version), "Runtime version");
        cudavm::require(version == 12040, "genuine pinned CUDA 12.4 Runtime");
        std::printf("runtime_version=%d\n", version);
        int count = 0; auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && !count)) return 77;
        cudavm::check(status, "device discovery");
        int max_threads = 0;
        cudavm::check(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, 0), "device attribute");
        cudavm::require(max_threads >= 32, "device supports the fixture's block size");
        // Validate backend identity without requiring unrelated physical clock
        // and memory-bus telemetry from cudaGetDeviceProperties.
        auto name_query = reinterpret_cast<CUresult (CUDAAPI*)(char*, int, CUdevice)>(
            GetProcAddress(GetModuleHandleW(L"nvcuda.dll"), "cuDeviceGetName"));
        char device_name[256]{};
        cudavm::require(name_query && name_query(device_name, sizeof(device_name), 0) == CUDA_SUCCESS, "device name");
        std::string_view backend(argv[2]);
        if (backend != "native") {
            std::string suffix = backend == "vulkan" ? " (Vulkan)" : " (DirectX)";
            std::string name(device_name);
            cudavm::require(name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0,
                "the selected shader backend executes the client");
        }
        std::printf("device=%s\n", device_name);
        cudavm::check(cudaSetDevice(0), "select device");
        for (unsigned iteration = 0; iteration < 2; ++iteration) cudavm::round(iteration);
        cudavm::origins(argv[1], backend != "native");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "static Runtime client: %s\n", error.what()); return 1; }
}
