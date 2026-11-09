#include "test_framework.hpp"
#include <nvPTXCompiler.h>
#include <algorithm>
#include <atomic>
#include <utility>

namespace cudavm {
void options_ptx(nvPTXCompileResult result) {
    require(result == NVPTXCOMPILE_SUCCESS, "PTX Compiler option API status " + std::to_string(result));
}
template<class T> struct ptx_options_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    ptx_options_output() { before.fill(0x17); after.fill(0xcb); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0x17; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0xcb; }), "PTX options scalar output guards");
        return value;
    }
};
struct ptx_options_program {
    nvPTXCompilerHandle value{};
    explicit ptx_options_program(const std::string& source = ".version 6.5\n.target sm_75\n.address_size 64\n.visible .entry ptx_options_kernel() { ret; }\n") {
        ptx_options_output<nvPTXCompilerHandle> output;
        options_ptx(nvPTXCompilerCreate(&output.value, source.size(), source.data())); value = output.get();
        require(value != nullptr, "PTX options create publishes handle");
    }
    ~ptx_options_program() { if (value) nvPTXCompilerDestroy(&value); }
    ptx_options_program(const ptx_options_program&) = delete;
    ptx_options_program& operator=(const ptx_options_program&) = delete;
    nvPTXCompileResult compile(const std::vector<std::string>& options = {"--gpu-name=sm_75", "--compile-only"}) const {
        std::vector<const char*> pointers; for (const auto& option : options) pointers.push_back(option.c_str());
        return nvPTXCompilerCompile(value, static_cast<int>(pointers.size()), pointers.data());
    }
    std::vector<unsigned char> bytes(int kind) const {
        const auto size_query = kind == 0 ? nvPTXCompilerGetCompiledProgramSize : kind == 1 ? nvPTXCompilerGetErrorLogSize : nvPTXCompilerGetInfoLogSize;
        ptx_options_output<size_t> size, repeated;
        options_ptx(size_query(value, &size.value)); options_ptx(size_query(value, &repeated.value));
        const auto count = size.get(); require(count == repeated.get(), "PTX options output size is stable");
        const auto written = kind != 0 && count == 0 ? size_t{1} : count;
        std::vector<unsigned char> first(written + 32, 0x29), second(written + 32, 0x74);
        const auto get = [&](unsigned char* output) {
            return kind == 0 ? nvPTXCompilerGetCompiledProgram(value, output) : kind == 1
                ? nvPTXCompilerGetErrorLog(value, reinterpret_cast<char*>(output)) : nvPTXCompilerGetInfoLog(value, reinterpret_cast<char*>(output));
        };
        options_ptx(get(first.data() + 16)); options_ptx(get(second.data() + 16));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 16 || i >= written + 16) require(first[i] == 0x29 && second[i] == 0x74, "PTX options exact output guards");
            else require(first[i] == second[i], "PTX options Get writes complete output");
        }
        std::vector<unsigned char> result(first.begin() + 16, first.end() - 16);
        if (kind) require(result.back() == 0 && std::find(result.begin(), result.end() - 1, 0) == result.end() - 1, "PTX options logs have exact NUL termination");
        return result;
    }
    void incomplete() const {
        ptx_options_output<size_t> size;
        require(nvPTXCompilerGetCompiledProgramSize(value, &size.value) == NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE, "failed/cancelled compilation has no published binary result");
        size.get();
    }
};
void ptx_options_spellings() {
    ptx_options_program program;
    for (const auto& options : std::vector<std::vector<std::string>>{
        {"--gpu-name", "compute_75", "--opt-level", "0"}, {"-arch=compute_80", "-O1"},
        {"-arch", "compute_75", "-O", "2"}, {"--gpu-name=compute_75", "--opt-level=3"}}) {
        require(program.compile(options) == NVPTXCOMPILE_PARSE_ONLY_SUCCESS && program.bytes(0).empty(), "public target/optimization spellings accept parse-only input and publish no binary bytes");
        require(program.bytes(1).size() == 1, "successful parse-only input has empty error log");
    }
}
void ptx_options_physical() {
    ptx_options_program program;
    for (const auto* optimization : {"-O0", "-O1", "-O2", "-O3"}) {
        options_ptx(program.compile({"--gpu-name=sm_75", "-c", optimization, "-v"}));
        require(!program.bytes(0).empty() && program.bytes(1).size() == 1 && program.bytes(2).size() > 1, "compile-only optimization produces opaque image and verbose info without errors");
    }
    const auto retained = program.bytes(0);
    require(program.compile({"-arch=compute_75"}) == NVPTXCOMPILE_PARSE_ONLY_SUCCESS && program.bytes(0).empty()
        && program.bytes(2).size() == 1, "later nonverbose parse-only invocation replaces physical result and info log");
    require(!retained.empty(), "caller-owned opaque result remains after replacement");
}
void ptx_options_rejection() {
    ptx_options_program program;
    for (const auto* option : {"--conformance-invalid-option", "-O4", "--gpu-name=compute_0", "--opt-level=", "--gpu-name"}) {
        require(program.compile({"-arch=sm_75", option}) == NVPTXCOMPILE_ERROR_COMPILATION_FAILURE, "invalid compiler option has compilation-failure status");
        program.incomplete(); require(program.bytes(1).size() > 1, "invalid option has retrievable guarded error log");
        options_ptx(program.compile());
        require(!program.bytes(0).empty() && program.bytes(1).size() == 1, "valid retry restores output and clears rejected option diagnostic");
    }
    ptx_options_program version(".version 9.9\n.target sm_75\n.address_size 64\n.visible .entry unsupported_version() { ret; }\n");
    require(version.compile({"-arch=compute_75"}) == NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION && version.bytes(1).size() > 1, "unsupported PTX version has specific status and diagnostic");
    version.incomplete();
    ptx_options_program malformed("not valid PTX input");
    require(malformed.compile({"-arch=compute_75"}) == NVPTXCOMPILE_ERROR_COMPILATION_FAILURE && malformed.bytes(1).size() > 1, "malformed PTX has compilation-failure status and diagnostic");
    malformed.incomplete();
}
struct ptx_options_flow {
    std::atomic<unsigned> calls{0};
    std::atomic<bool> invalid_reserved{false}, cancel{false};
    static int callback(void* payload, void* reserved) noexcept {
        if (!payload) return 1;
        auto& state = *static_cast<ptx_options_flow*>(payload);
        state.calls.fetch_add(1);
        if (reserved) state.invalid_reserved.store(true);
        return state.cancel.load() ? 1 : 0;
    }
};
void ptx_options_callbacks() {
    // Payloads outlive the compiler and callbacks use only thread-safe state;
    // they neither throw nor call any compiler API.
    ptx_options_flow first, replacement;
    ptx_options_program program;
    options_ptx(nvPTXCompilerSetFlowCallback(program.value, ptx_options_flow::callback, &first));
    options_ptx(program.compile());
    require(first.calls.load() > 0 && !first.invalid_reserved.load() && !program.bytes(0).empty(), "continue callback receives caller payload, null reserved argument and permits output");
    first.cancel.store(true);
    const auto before_cancel = first.calls.load();
    require(program.compile() == NVPTXCOMPILE_ERROR_CANCELLED && first.calls.load() > before_cancel
        && !first.invalid_reserved.load(), "callback cancellation returns cancelled status");
    program.incomplete(); require(program.bytes(1).size() > 1, "cancelled compile retains guarded diagnostic");
    const auto retired_calls = first.calls.load();
    options_ptx(nvPTXCompilerSetFlowCallback(program.value, ptx_options_flow::callback, &replacement));
    options_ptx(program.compile());
    require(first.calls.load() == retired_calls && replacement.calls.load() > 0 && !replacement.invalid_reserved.load()
        && !program.bytes(0).empty() && program.bytes(1).size() == 1, "replacement callback and successful retry do not reuse old payload or cancelled state");
}
void ptx_options_validation() {
    ptx_options_flow state;
    ptx_options_program program;
    require(nvPTXCompilerSetFlowCallback(nullptr, ptx_options_flow::callback, &state) == NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE, "callback registration rejects null compiler handle");
    require(nvPTXCompilerSetFlowCallback(program.value, nullptr, &state) == NVPTXCOMPILE_ERROR_INVALID_INPUT, "callback registration rejects null function");
    const char* null_option = nullptr;
    require(nvPTXCompilerCompile(program.value, -1, nullptr) == NVPTXCOMPILE_ERROR_INVALID_INPUT
        && nvPTXCompilerCompile(program.value, 1, nullptr) == NVPTXCOMPILE_ERROR_INVALID_INPUT
        && nvPTXCompilerCompile(program.value, 1, &null_option) == NVPTXCOMPILE_ERROR_INVALID_INPUT, "malformed option arrays have invalid-input status");
    options_ptx(nvPTXCompilerSetFlowCallback(program.value, ptx_options_flow::callback, &state));
    options_ptx(program.compile());
    require(state.calls.load() > 0 && !program.bytes(0).empty(), "rejected option/callback arguments leave handle usable");
}
std::string_view nvptxcompiler_options_case = "all";
void execute_nvptxcompiler_options() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"spellings", ptx_options_spellings}, {"physical", ptx_options_physical}, {"rejection", ptx_options_rejection},
        {"callbacks", ptx_options_callbacks}, {"validation", ptx_options_validation}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (nvptxcompiler_options_case == "all" || nvptxcompiler_options_case == name) { selected = true; execute(); }
    require(selected, "unknown nvptxcompiler_options case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvptxcompiler_options_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvptxcompiler_options);
}
