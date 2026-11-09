#include "test_framework.hpp"
#include <nvPTXCompiler.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

namespace cudavm {
template<class T> struct ptx_lifecycle_guarded {
    alignas(T) std::array<unsigned char, sizeof(T) + 32> bytes;
    ptx_lifecycle_guarded() { bytes.fill(0xe7); }
    T* data() { return reinterpret_cast<T*>(bytes.data() + 16); }
    T get() const {
        for (size_t i = 0; i < bytes.size(); ++i) if (i < 16 || i >= sizeof(T) + 16) require(bytes[i] == 0xe7, "PTX Compiler scalar output byte guards");
        T result{}; std::memcpy(&result, bytes.data() + 16, sizeof(T)); return result;
    }
};
using ptx_lifecycle_size_query = nvPTXCompileResult (*)(nvPTXCompilerHandle, size_t*);
template<class Getter> std::vector<unsigned char> ptx_lifecycle_bytes(nvPTXCompilerHandle program, ptx_lifecycle_size_query size_query, Getter get, bool text) {
    ptx_lifecycle_guarded<size_t> size, repeated;
    require(size_query(program, size.data()) == NVPTXCOMPILE_SUCCESS && size_query(program, repeated.data()) == NVPTXCOMPILE_SUCCESS, "PTX Compiler size queries");
    const auto count = size.get(); require(repeated.get() == count, "PTX Compiler stable result size"); const auto written = text && count == 0 ? size_t{1} : count;
    std::vector<unsigned char> first(written + 64, 0x27), second(written + 64, 0x53);
    require(get(first.data() + 32) == NVPTXCOMPILE_SUCCESS && get(second.data() + 32) == NVPTXCOMPILE_SUCCESS, "PTX Compiler result retrieval");
    for (size_t i = 0; i < first.size(); ++i) {
        if (i < 32 || i >= written + 32) require(first[i] == 0x27 && second[i] == 0x53, "PTX Compiler exact write extent and guards");
        else require(first[i] == second[i], "PTX Compiler complete writes and repeated retrieval");
    }
    std::vector<unsigned char> result(first.begin() + 32, first.end() - 32);
    if (text) require(result.back() == 0 && std::find(result.begin(), result.end() - 1, 0) == result.end() - 1, "PTX Compiler exact log termination, including empty log");
    return result;
}
struct ptx_lifecycle_program {
    nvPTXCompilerHandle value{};
    explicit ptx_lifecycle_program(unsigned seed = 0, bool include_nul = false) {
        std::string source = ".version 6.5\n.target sm_75\n.address_size 64\n.visible .entry lifecycle_" + std::to_string(seed) + "() { ret; }\n";
        const auto length = source.size() + size_t(include_nul); source.push_back('\0'); source += "deliberately invalid bytes outside explicit PTX extent";
        ptx_lifecycle_guarded<nvPTXCompilerHandle> output;
        require(nvPTXCompilerCreate(output.data(), length, source.data()) == NVPTXCOMPILE_SUCCESS, "PTX Compiler creates explicitly sized input"); value = output.get(); require(value != nullptr, "PTX Compiler create publishes a handle");
        std::fill(source.begin(), source.end(), '!');
    }
    ~ptx_lifecycle_program() { if (value) nvPTXCompilerDestroy(&value); }
    ptx_lifecycle_program(const ptx_lifecycle_program&) = delete;
    ptx_lifecycle_program& operator=(const ptx_lifecycle_program&) = delete;
    nvPTXCompileResult compile(bool parse = false) const { const char* options[]{parse ? "--gpu-name=compute_75" : "--gpu-name=sm_75", "-O0", "--verbose"}; return nvPTXCompilerCompile(value, 3, options); }
    std::vector<unsigned char> binary() const { return ptx_lifecycle_bytes(value, nvPTXCompilerGetCompiledProgramSize, [&](void* output) { return nvPTXCompilerGetCompiledProgram(value, output); }, false); }
    std::vector<unsigned char> log(bool error) const {
        return ptx_lifecycle_bytes(value, error ? nvPTXCompilerGetErrorLogSize : nvPTXCompilerGetInfoLogSize,
            [&](void* output) { return error ? nvPTXCompilerGetErrorLog(value, static_cast<char*>(output)) : nvPTXCompilerGetInfoLog(value, static_cast<char*>(output)); }, true);
    }
    void destroy() {
        ptx_lifecycle_guarded<nvPTXCompilerHandle> output; std::memcpy(output.data(), &value, sizeof(value)); require(nvPTXCompilerDestroy(output.data()) == NVPTXCOMPILE_SUCCESS, "PTX Compiler destroy"); value = nullptr;
        require(output.get() == nullptr, "PTX Compiler destruction clears only handle slot");
    }
};
void ptx_lifecycle_discovery() {
    ptx_lifecycle_guarded<unsigned> major, minor; require(nvPTXCompilerGetVersion(major.data(), minor.data()) == NVPTXCOMPILE_SUCCESS && major.get() == 13 && minor.get() == 4, "PTX Compiler pinned version");
    ptx_lifecycle_guarded<int> count; require(nvPTXCompilerGetNumSupportedArchs(count.data()) == NVPTXCOMPILE_SUCCESS && count.get() > 0, "PTX Compiler architecture count");
    const auto size = static_cast<size_t>(count.get()); std::vector<int> architectures(size + 16, -197), seen;
    require(nvPTXCompilerGetSupportedArchs(architectures.data() + 8) == NVPTXCOMPILE_SUCCESS, "PTX Compiler architecture list");
    for (size_t i = 0; i < architectures.size(); ++i) {
        if (i < 8 || i >= size + 8) require(architectures[i] == -197, "PTX Compiler architecture array guards");
        else {
            const auto arch = architectures[i]; require(arch > 0 && std::find(seen.begin(), seen.end(), arch) == seen.end(), "unique positive PTX Compiler architecture"); seen.push_back(arch);
            ptx_lifecycle_program program(static_cast<unsigned>(arch)); const auto option = "--gpu-name=compute_" + std::to_string(arch); const char* options[]{option.c_str()};
            require(nvPTXCompilerCompile(program.value, 1, options) == NVPTXCOMPILE_PARSE_ONLY_SUCCESS && program.binary().empty(), "advertised architecture accepts parse-only lifecycle input");
        }
    }
    require(nvPTXCompilerGetNumSupportedVersions(count.data()) == NVPTXCOMPILE_SUCCESS && count.get() > 0, "PTX Compiler language version count");
    const auto versions_count = static_cast<size_t>(count.get()); auto versions = std::make_unique<int[][2]>(versions_count + 16); std::vector<std::array<int, 2>> seen_versions;
    for (size_t i = 0; i < versions_count + 16; ++i) versions[i][0] = versions[i][1] = -197;
    require(nvPTXCompilerGetSupportedVersions(versions.get() + 8) == NVPTXCOMPILE_SUCCESS, "PTX Compiler language version list");
    for (size_t i = 0; i < versions_count + 16; ++i) {
        if (i < 8 || i >= versions_count + 8) require(versions[i][0] == -197 && versions[i][1] == -197, "PTX Compiler version-pair array guards");
        else { const std::array<int, 2> pair{versions[i][0], versions[i][1]}; require(pair[0] > 0 && pair[1] >= 0 && std::find(seen_versions.begin(), seen_versions.end(), pair) == seen_versions.end(), "unique valid PTX version pair"); seen_versions.push_back(pair); }
    }
}
void ptx_lifecycle_ownership() {
    for (bool nul : {false, true}) { ptx_lifecycle_program program(nul ? 79 : 17, nul); require(program.compile() == NVPTXCOMPILE_SUCCESS, "PTX Compiler retains explicitly sized source after caller overwrite"); require(!program.binary().empty(), "owned PTX produces opaque program"); program.destroy(); }
}
void ptx_lifecycle_outputs() {
    for (bool parse : {false, true}) {
        ptx_lifecycle_program program(parse ? 29 : 17); require(program.compile(parse) == (parse ? NVPTXCOMPILE_PARSE_ONLY_SUCCESS : NVPTXCOMPILE_SUCCESS), "PTX Compiler output-mode status");
        const auto retained = program.binary(), before = retained; require(retained.empty() == parse && program.binary() == retained, "PTX Compiler output availability and stable retrieval"); program.log(false); program.log(true);
        program.destroy(); require(retained == before, "PTX Compiler caller-owned result survives producer destruction");
    }
}
void ptx_lifecycle_retry() {
    ptx_lifecycle_program program(137); size_t size{}; unsigned char output{};
    require(nvPTXCompilerGetCompiledProgramSize(program.value, &size) == NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE && nvPTXCompilerGetCompiledProgram(program.value, &output) == NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE, "binary queries require completed compilation");
    const char* invalid[]{"--lifecycle-invalid-option"}; require(nvPTXCompilerCompile(program.value, 1, invalid) == NVPTXCOMPILE_ERROR_COMPILATION_FAILURE && program.log(true).size() > 1, "invalid option diagnostic");
    require(program.compile() == NVPTXCOMPILE_SUCCESS, "same PTX Compiler handle recovers after rejected options"); const auto retained = program.binary(), before = retained;
    require(program.compile(true) == NVPTXCOMPILE_PARSE_ONLY_SUCCESS && program.binary().empty() && retained == before, "later parse-only compilation replaces result while preserving caller image"); program.destroy(); require(retained == before, "retried producer destruction preserves caller image");
}
void ptx_lifecycle_validation() {
    nvPTXCompilerHandle rejected{}; const char input[] = "input";
    require(nvPTXCompilerCreate(nullptr, sizeof(input), input) == NVPTXCOMPILE_ERROR_INVALID_INPUT && nvPTXCompilerCreate(&rejected, 0, input) == NVPTXCOMPILE_ERROR_INVALID_INPUT && nvPTXCompilerCreate(&rejected, 1, nullptr) == NVPTXCOMPILE_ERROR_INVALID_INPUT, "invalid PTX Compiler create arguments");
    require(nvPTXCompilerDestroy(nullptr) == NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE && nvPTXCompilerCompile(nullptr, 0, nullptr) == NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE, "null PTX Compiler handles");
    ptx_lifecycle_program program(193); require(nvPTXCompilerCompile(program.value, -1, nullptr) == NVPTXCOMPILE_ERROR_INVALID_INPUT && nvPTXCompilerCompile(program.value, 1, nullptr) == NVPTXCOMPILE_ERROR_INVALID_INPUT, "invalid option counts/arrays");
    for (auto query : std::array<ptx_lifecycle_size_query, 3>{nvPTXCompilerGetCompiledProgramSize, nvPTXCompilerGetErrorLogSize, nvPTXCompilerGetInfoLogSize}) require(query(program.value, nullptr) == NVPTXCOMPILE_ERROR_INVALID_INPUT, "null PTX Compiler size output");
    require(program.compile() == NVPTXCOMPILE_SUCCESS && !program.binary().empty(), "valid compilation after rejected arguments"); require(nvPTXCompilerGetCompiledProgram(program.value, nullptr) == NVPTXCOMPILE_ERROR_INVALID_INPUT, "null binary storage"); program.destroy();
}
void ptx_lifecycle_threading() {
    std::array<std::thread, 4> workers; std::array<std::exception_ptr, 4> failures{}; const auto join = [&] { for (auto& worker : workers) if (worker.joinable()) worker.join(); };
    try { for (unsigned worker = 0; worker < workers.size(); ++worker) workers[worker] = std::thread([&, worker] {
        try { ptx_lifecycle_program program(211 + worker * 19); require(program.compile() == NVPTXCOMPILE_SUCCESS && !program.binary().empty(), "concurrent independent PTX Compiler results"); program.destroy(); }
        catch (...) { failures[worker] = std::current_exception(); }
    }); } catch (...) { join(); throw; }
    join(); for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
}
std::string ptx_lifecycle_case = "all";
void execute_ptx_lifecycle() {
    bool selected = false; const auto run = [&](const char* name, auto execute) { if (ptx_lifecycle_case == "all" || ptx_lifecycle_case == name) { execute(); selected = true; } };
    run("discovery", ptx_lifecycle_discovery); run("ownership", ptx_lifecycle_ownership); run("outputs", ptx_lifecycle_outputs); run("retry", ptx_lifecycle_retry); run("validation", ptx_lifecycle_validation); run("threading", ptx_lifecycle_threading);
    require(selected, "unknown nvptxcompiler_lifecycle case: " + ptx_lifecycle_case);
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::ptx_lifecycle_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_ptx_lifecycle);
}
