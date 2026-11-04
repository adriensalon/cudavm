#include "conformance.hpp"
#include <nvrtc.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

namespace cudavm {
template<class T> struct nvrtc_guarded {
    alignas(T) std::array<unsigned char, sizeof(T) + 32> bytes;
    nvrtc_guarded() { bytes.fill(0xe7); }
    T* data() { return reinterpret_cast<T*>(bytes.data() + 16); }
    T get() const {
        for (size_t i = 0; i < bytes.size(); ++i) if (i < 16 || i >= sizeof(T) + 16) require(bytes[i] == 0xe7, "NVRTC scalar output byte guards");
        T result{}; std::memcpy(&result, bytes.data() + 16, sizeof(T)); return result;
    }
};
using nvrtc_size_query = nvrtcResult (*)(nvrtcProgram, size_t*);
using nvrtc_data_query = nvrtcResult (*)(nvrtcProgram, char*);
std::vector<char> nvrtc_result_bytes(nvrtcProgram program, nvrtc_size_query size_query, nvrtc_data_query data_query, bool text) {
    nvrtc_guarded<size_t> size, repeated;
    require(size_query(program, size.data()) == NVRTC_SUCCESS && size_query(program, repeated.data()) == NVRTC_SUCCESS, "NVRTC result size query");
    const auto count = size.get(); require(repeated.get() == count, "NVRTC repeated result size agrees");
    if (text) require(count > 0, "text result size includes terminal NUL");
    if (!count) return {};
    std::vector<char> first(count + 64, '\x27'), second(count + 64, '\x53');
    require(data_query(program, first.data() + 32) == NVRTC_SUCCESS && data_query(program, second.data() + 32) == NVRTC_SUCCESS, "NVRTC result retrieval");
    for (size_t i = 0; i < first.size(); ++i) {
        if (i < 32 || i >= count + 32) require(first[i] == '\x27' && second[i] == '\x53', "NVRTC output extent and guards");
        else require(first[i] == second[i], "NVRTC writes every result byte and repeat retrieval agrees");
    }
    std::vector<char> result(first.begin() + 32, first.end() - 32);
    if (text) require(result.back() == 0 && std::find(result.begin(), result.end() - 1, '\0') == result.end() - 1, "NVRTC text has one exact terminal NUL");
    return result;
}
struct nvrtc_lifecycle_program {
    nvrtcProgram value{}; std::string kernel;
    explicit nvrtc_lifecycle_program(unsigned seed) : kernel("nvrtc_lifecycle_" + std::to_string(seed)) {
        std::string source = "#include \"lifecycle_input.hpp\"\n#ifdef FORCE_FAILURE\n#error lifecycle_requested_failure\n#endif\nextern \"C\" __global__ void " + kernel + "(unsigned* output) { output[threadIdx.x] = lifecycle_bias + threadIdx.x; }\n";
        std::string header = "static constexpr unsigned lifecycle_bias = " + std::to_string(seed) + "u;\n", header_name = "lifecycle_input.hpp", name = "lifecycle.cu";
        const char* headers[]{header.c_str()}; const char* names[]{header_name.c_str()};
        nvrtc_guarded<nvrtcProgram> output;
        require(nvrtcCreateProgram(output.data(), source.c_str(), name.c_str(), 1, headers, names) == NVRTC_SUCCESS, "NVRTC creates program with temporary input/header arrays");
        value = output.get(); require(value != nullptr, "successful NVRTC create returns a program");
        // Make retaining caller strings observably wrong before their storage dies.
        std::fill(source.begin(), source.end(), '!'); std::fill(header.begin(), header.end(), '!'); std::fill(header_name.begin(), header_name.end(), '!'); std::fill(name.begin(), name.end(), '!');
        headers[0] = nullptr; names[0] = nullptr;
    }
    ~nvrtc_lifecycle_program() { if (value) nvrtcDestroyProgram(&value); }
    nvrtc_lifecycle_program(const nvrtc_lifecycle_program&) = delete;
    nvrtc_lifecycle_program& operator=(const nvrtc_lifecycle_program&) = delete;
    nvrtcResult compile(std::vector<std::string> options = {"-arch=compute_75", "--std=c++17"}) const {
        std::vector<const char*> pointers; for (const auto& option : options) pointers.push_back(option.c_str());
        return nvrtcCompileProgram(value, static_cast<int>(pointers.size()), pointers.data());
    }
    void compile_success(std::vector<std::string> options = {"-arch=compute_75", "--std=c++17"}) const {
        const auto result = compile(std::move(options));
        if (result != NVRTC_SUCCESS) {
            const auto log = nvrtc_result_bytes(value, nvrtcGetProgramLogSize, nvrtcGetProgramLog, true);
            throw std::runtime_error(std::string(nvrtcGetErrorString(result)) + ": " + log.data());
        }
    }
    std::vector<char> ptx() const {
        auto result = nvrtc_result_bytes(value, nvrtcGetPTXSize, nvrtcGetPTX, true);
        require(std::string(result.data()).find(kernel) != std::string::npos, "PTX contains kernel from retained program source"); return result;
    }
    void destroy() {
        nvrtc_guarded<nvrtcProgram> output; std::memcpy(output.data(), &value, sizeof(value));
        require(nvrtcDestroyProgram(output.data()) == NVRTC_SUCCESS, "NVRTC destroys live program"); value = nullptr;
        require(output.get() == nullptr, "NVRTC destroy nulls only caller-owned handle slot");
    }
};
void nvrtc_lifecycle_discovery() {
    nvrtc_guarded<int> major, minor, count;
    require(nvrtcVersion(major.data(), minor.data()) == NVRTC_SUCCESS && major.get() == 13 && minor.get() == 4, "NVRTC pinned public version");
    require(nvrtcGetNumSupportedArchs(count.data()) == NVRTC_SUCCESS && count.get() > 0, "NVRTC supports at least one architecture");
    const auto size = static_cast<size_t>(count.get()); std::vector<int> architectures(size + 16, -197);
    require(nvrtcGetSupportedArchs(architectures.data() + 8) == NVRTC_SUCCESS, "NVRTC architecture list query");
    for (size_t i = 0; i < architectures.size(); ++i) if (i < 8 || i >= size + 8) require(architectures[i] == -197, "NVRTC architecture output array guards");
    std::vector<int> seen;
    for (size_t i = 8; i < 8 + size; ++i) {
        const auto architecture = architectures[i]; require(architecture > 0 && std::find(seen.begin(), seen.end(), architecture) == seen.end(), "NVRTC architecture list contains unique positive entries"); seen.push_back(architecture);
        nvrtc_lifecycle_program program(static_cast<unsigned>(architecture)); program.compile_success({"-arch=compute_" + std::to_string(architecture), "--std=c++17"}); program.ptx();
    }
    for (int error = NVRTC_SUCCESS; error <= NVRTC_ERROR_BUSY; ++error) { const char* text = nvrtcGetErrorString(static_cast<nvrtcResult>(error)); require(text && *text, "defined NVRTC result has a description"); }
}
void nvrtc_lifecycle_ownership() {
    for (unsigned seed : {0u, 79u, 0xffffffffu}) { nvrtc_lifecycle_program program(seed); program.compile_success(); program.ptx(); program.destroy(); }
}
void nvrtc_lifecycle_outputs() {
    for (unsigned mode = 0; mode < 3; ++mode) {
        nvrtc_lifecycle_program program(31 + mode);
        std::vector<std::string> options{mode == 1 ? "-arch=sm_75" : "-arch=compute_75", "--std=c++17"}; if (mode == 2) options.push_back("-dlto");
        program.compile_success(options); nvrtc_result_bytes(program.value, nvrtcGetProgramLogSize, nvrtcGetProgramLog, true);
        if (mode != 2) program.ptx();
        auto cubin = nvrtc_result_bytes(program.value, nvrtcGetCUBINSize, nvrtcGetCUBIN, false), lto = nvrtc_result_bytes(program.value, nvrtcGetLTOIRSize, nvrtcGetLTOIR, false);
        require((mode == 1) == !cubin.empty() && (mode == 2) == !lto.empty(), "NVRTC binary output availability matches physical/LTO options");
        const auto cubin_copy = cubin, lto_copy = lto; program.destroy(); require(cubin == cubin_copy && lto == lto_copy, "binary caller-owned outputs survive producer destruction");
    }
}
void nvrtc_lifecycle_retry() {
    nvrtc_lifecycle_program program(137);
    require(program.compile({"--lifecycle-invalid-option"}) == NVRTC_ERROR_INVALID_OPTION, "invalid option failure");
    require(nvrtc_result_bytes(program.value, nvrtcGetProgramLogSize, nvrtcGetProgramLog, true).size() > 1, "invalid option has a retrievable diagnostic");
    program.compile_success(); const auto retained = program.ptx(), before = retained;
    require(program.compile({"-arch=compute_75", "--std=c++17", "-DFORCE_FAILURE=1"}) == NVRTC_ERROR_COMPILATION, "source-level compile failure");
    require(nvrtc_result_bytes(program.value, nvrtcGetProgramLogSize, nvrtcGetProgramLog, true).size() > 1, "compile failure has a retrievable diagnostic");
    program.compile_success(); program.ptx(); require(retained == before, "recompile/failure leaves caller-owned PTX intact"); program.destroy(); require(retained == before, "caller-owned PTX survives program destruction");
}
void nvrtc_lifecycle_validation() {
    int integer{}; require(nvrtcVersion(nullptr, &integer) == NVRTC_ERROR_INVALID_INPUT && nvrtcVersion(&integer, nullptr) == NVRTC_ERROR_INVALID_INPUT, "null NVRTC version outputs");
    require(nvrtcGetNumSupportedArchs(nullptr) == NVRTC_ERROR_INVALID_INPUT && nvrtcGetSupportedArchs(nullptr) == NVRTC_ERROR_INVALID_INPUT, "null NVRTC architecture outputs");
    nvrtcProgram rejected{};
    require(nvrtcCreateProgram(nullptr, "", nullptr, 0, nullptr, nullptr) == NVRTC_ERROR_INVALID_INPUT, "null create output");
    require(nvrtcCreateProgram(&rejected, nullptr, nullptr, 0, nullptr, nullptr) == NVRTC_ERROR_INVALID_INPUT, "null program source");
    require(nvrtcCreateProgram(&rejected, "", nullptr, -1, nullptr, nullptr) == NVRTC_ERROR_INVALID_INPUT && nvrtcCreateProgram(&rejected, "", nullptr, 1, nullptr, nullptr) == NVRTC_ERROR_INVALID_INPUT, "invalid header count/arrays");
    require(nvrtcDestroyProgram(nullptr) == NVRTC_ERROR_INVALID_PROGRAM && nvrtcCompileProgram(nullptr, 0, nullptr) == NVRTC_ERROR_INVALID_PROGRAM, "null program arguments");
    nvrtc_lifecycle_program program(193);
    require(nvrtcCompileProgram(program.value, -1, nullptr) == NVRTC_ERROR_INVALID_INPUT && nvrtcCompileProgram(program.value, 1, nullptr) == NVRTC_ERROR_INVALID_INPUT, "invalid option count/arrays");
    const std::array<nvrtc_size_query, 4> size_queries{nvrtcGetPTXSize, nvrtcGetCUBINSize, nvrtcGetLTOIRSize, nvrtcGetProgramLogSize};
    const std::array<nvrtc_data_query, 4> data_queries{nvrtcGetPTX, nvrtcGetCUBIN, nvrtcGetLTOIR, nvrtcGetProgramLog};
    for (auto query : size_queries) { size_t size{}; require(query(nullptr, &size) == NVRTC_ERROR_INVALID_PROGRAM && query(program.value, nullptr) == NVRTC_ERROR_INVALID_INPUT, "invalid result size arguments"); }
    for (auto query : data_queries) require(query(program.value, nullptr) == NVRTC_ERROR_INVALID_INPUT, "null result storage");
    program.compile_success(); program.ptx(); program.destroy();
}
void nvrtc_lifecycle_threading() {
    std::array<std::thread, 4> workers; std::array<std::exception_ptr, 4> failures{};
    const auto join = [&] { for (auto& worker : workers) if (worker.joinable()) worker.join(); };
    try {
        for (unsigned worker = 0; worker < workers.size(); ++worker) workers[worker] = std::thread([&, worker] {
            try { for (unsigned round = 0; round < 2; ++round) { nvrtc_lifecycle_program program(211 + worker * 19 + round); program.compile_success(); program.ptx(); program.destroy(); } }
            catch (...) { failures[worker] = std::current_exception(); }
        });
    } catch (...) { join(); throw; }
    join(); for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
}
std::string nvrtc_lifecycle_case = "all";
void execute_nvrtc_lifecycle() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (nvrtc_lifecycle_case == "all" || nvrtc_lifecycle_case == name) { execute(); selected = true; } };
    run("discovery", nvrtc_lifecycle_discovery); run("ownership", nvrtc_lifecycle_ownership); run("outputs", nvrtc_lifecycle_outputs);
    run("retry", nvrtc_lifecycle_retry); run("validation", nvrtc_lifecycle_validation); run("threading", nvrtc_lifecycle_threading);
    require(selected, "unknown nvrtc_lifecycle case: " + nvrtc_lifecycle_case);
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvrtc_lifecycle_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvrtc_lifecycle);
}
