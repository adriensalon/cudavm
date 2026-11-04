#include "conformance.hpp"
#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>
#include <algorithm>
#include <utility>

namespace cudavm {
#define JIT_DIAGNOSTIC_CALL(name, ...) (versioned ? __##name##_13_4(__VA_ARGS__) : name(__VA_ARGS__))
void diagnostic_jit(nvJitLinkResult result) {
    require(result == NVJITLINK_SUCCESS, "nvJitLink diagnostic API status " + std::to_string(result));
}
template<class T> struct jit_diagnostic_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    jit_diagnostic_output() { before.fill(0xd7); after.fill(0x45); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0xd7; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x45; }), "nvJitLink diagnostics scalar output guards");
        return value;
    }
};
const std::string diagnostic_jit_prefix = ".version 6.5\n.target sm_75\n.address_size 64\n";
struct jit_diagnostic_program {
    nvJitLinkHandle value{};
    bool versioned;
    explicit jit_diagnostic_program(bool pinned, const std::vector<std::string>& options = {"-arch=sm_75", "-r"}, nvJitLinkResult expected = NVJITLINK_SUCCESS) : versioned(pinned) {
        std::vector<const char*> pointers; for (const auto& option : options) pointers.push_back(option.c_str());
        jit_diagnostic_output<nvJitLinkHandle> output;
        const auto result = JIT_DIAGNOSTIC_CALL(nvJitLinkCreate, &output.value, static_cast<uint32_t>(pointers.size()), pointers.data());
        value = output.get();
        if (result != expected) { if (value) JIT_DIAGNOSTIC_CALL(nvJitLinkDestroy, &value); require(false, "nvJitLink Create diagnostic status: " + std::to_string(result)); }
        require(value != nullptr, "nvJitLink Create retains owned handle for success or configuration diagnostics");
    }
    ~jit_diagnostic_program() { if (value) JIT_DIAGNOSTIC_CALL(nvJitLinkDestroy, &value); }
    jit_diagnostic_program(const jit_diagnostic_program&) = delete;
    jit_diagnostic_program& operator=(const jit_diagnostic_program&) = delete;
    nvJitLinkResult add(const std::string& input, const char* name = "diagnostic_input.ptx") const {
        return JIT_DIAGNOSTIC_CALL(nvJitLinkAddData, value, NVJITLINK_INPUT_PTX, input.data(), input.size(), name);
    }
    nvJitLinkResult complete() const { return JIT_DIAGNOSTIC_CALL(nvJitLinkComplete, value); }
    std::vector<unsigned char> bytes(int kind) const {
        jit_diagnostic_output<size_t> size, repeated;
        const auto query = [&](size_t* output) {
            if (kind == 0) return JIT_DIAGNOSTIC_CALL(nvJitLinkGetLinkedCubinSize, value, output);
            if (kind == 1) return JIT_DIAGNOSTIC_CALL(nvJitLinkGetErrorLogSize, value, output);
            return JIT_DIAGNOSTIC_CALL(nvJitLinkGetInfoLogSize, value, output);
        };
        diagnostic_jit(query(&size.value)); diagnostic_jit(query(&repeated.value));
        const auto count = size.get(); require(count == repeated.get(), "nvJitLink diagnostics Size is stable");
        const auto written = kind && count == 0 ? size_t{1} : count;
        std::vector<unsigned char> first(written + 32, 0x2d), second(written + 32, 0x79);
        const auto get = [&](unsigned char* output) {
            if (kind == 0) return JIT_DIAGNOSTIC_CALL(nvJitLinkGetLinkedCubin, value, output);
            if (kind == 1) return JIT_DIAGNOSTIC_CALL(nvJitLinkGetErrorLog, value, reinterpret_cast<char*>(output));
            return JIT_DIAGNOSTIC_CALL(nvJitLinkGetInfoLog, value, reinterpret_cast<char*>(output));
        };
        diagnostic_jit(get(first.data() + 16)); diagnostic_jit(get(second.data() + 16));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 16 || i >= written + 16) require(first[i] == 0x2d && second[i] == 0x79, "nvJitLink diagnostics exact output guards");
            else require(first[i] == second[i], "nvJitLink diagnostics Get writes every reported byte");
        }
        std::vector<unsigned char> result(first.begin() + 16, first.end() - 16);
        if (kind) require(result.back() == 0 && std::find(result.begin(), result.end() - 1, 0) == result.end() - 1, "nvJitLink diagnostics text exact NUL termination");
        return result;
    }
    std::string log(bool error = true) const {
        const auto result = bytes(error ? 1 : 2); return {result.begin(), result.end() - 1};
    }
    void incomplete() const {
        jit_diagnostic_output<size_t> size;
        require(JIT_DIAGNOSTIC_CALL(nvJitLinkGetLinkedCubinSize, value, &size.value) == NVJITLINK_ERROR_INTERNAL, "unfinished/failed link publishes no binary");
        size.get();
    }
    void destroy() {
        jit_diagnostic_output<nvJitLinkHandle> output; output.value = value;
        diagnostic_jit(JIT_DIAGNOSTIC_CALL(nvJitLinkDestroy, &output.value)); value = nullptr;
        require(output.get() == nullptr, "diagnostic handle destruction clears exact caller slot");
    }
};
void jit_diagnostic_configuration() {
    const std::vector<std::pair<std::vector<std::string>, nvJitLinkResult>> cases{
        {{}, NVJITLINK_ERROR_MISSING_ARCH}, {{"-arch=sm_75", "-conformance-invalid-option"}, NVJITLINK_ERROR_UNRECOGNIZED_OPTION},
        {{"-arch=banana"}, NVJITLINK_ERROR_UNRECOGNIZED_ARCH}, {{"-arch=sm_0"}, NVJITLINK_ERROR_UNSUPPORTED_ARCH},
        {{"-arch=sm_75", "-ptx"}, NVJITLINK_ERROR_LTO_NOT_ENABLED},
        {{"-arch=compute_75"}, NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS},
        {{"-arch=sm_75", "-lto", "-ptx", "-r"}, NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS}};
    for (bool versioned : {false, true}) for (const auto& [options, expected] : cases) {
        jit_diagnostic_program program(versioned, options, expected);
        const auto log = program.log(); require(!log.empty(), "failed Create has retrievable configuration diagnostic");
        require(program.log() == log, "failed Create log is stable until destruction"); program.destroy();
    }
}
void jit_diagnostic_inputs() {
    for (bool versioned : {false, true}) {
        jit_diagnostic_program program(versioned);
        diagnostic_jit(program.add(diagnostic_jit_prefix + ".visible .entry first_diagnostic_kernel() { ret; }\n"));
        require(JIT_DIAGNOSTIC_CALL(nvJitLinkAddData, program.value, NVJITLINK_INPUT_PTX, nullptr, size_t{1}, "null.ptx") == NVJITLINK_ERROR_NULL_INPUT,
            "null input has null-input status");
        require(program.add("not valid PTX", "owned_malformed_diagnostic.ptx") == NVJITLINK_ERROR_PTX_COMPILE, "malformed explicit PTX has PTX-compile status");
        require(program.log().find("owned_malformed_diagnostic.ptx") != std::string::npos, "input diagnostic retains caller-provided label");
        diagnostic_jit(program.add(diagnostic_jit_prefix + ".visible .entry second_diagnostic_kernel() { ret; }\n"));
        diagnostic_jit(program.complete()); require(!program.bytes(0).empty(), "failed additions preserve accepted inputs for later successful completion");
    }
}
void jit_diagnostic_finalization() {
    for (bool versioned : {false, true}) {
        jit_diagnostic_program program(versioned, {"-arch=sm_75", "-O0"});
        diagnostic_jit(program.add(diagnostic_jit_prefix + R"ptx(
.extern .global .align 4 .u32 diagnostic_missing;
.visible .entry finalization_diagnostic(.param .u64 out) {
 .reg .b64 %out; .reg .b32 %value;
 ld.param.u64 %out,[out]; ld.global.u32 %value,[diagnostic_missing];
 st.global.u32 [%out],%value; ret;
}
)ptx"));
        require(program.complete() == NVJITLINK_ERROR_FINALIZE, "unresolved storage has finalize status");
        program.incomplete(); require(program.log().find("diagnostic_missing") != std::string::npos, "unresolved diagnostic identifies caller-owned symbol");
        diagnostic_jit(program.add(diagnostic_jit_prefix + ".visible .global .align 4 .u32 diagnostic_missing = 37;\n"));
        diagnostic_jit(program.complete()); require(!program.bytes(0).empty(), "missing definition can be added after failed completion for successful retry");
    }
}
void jit_diagnostic_logs() {
    for (bool versioned : {false, true}) for (bool verbose : {false, true}) {
        std::vector<std::string> options{"-arch=sm_75", "-r"};
        if (verbose) { options.push_back("-verbose"); options.push_back("-time"); }
        jit_diagnostic_program program(versioned, options);
        require(program.log().empty() && program.log(false).empty(), "new program's exact empty logs");
        diagnostic_jit(program.add(diagnostic_jit_prefix + ".visible .entry diagnostic_logs() { ret; }\n"));
        diagnostic_jit(program.complete());
        const auto error = program.bytes(1), info = program.bytes(2), binary = program.bytes(0);
        const auto saved_error = error, saved_info = info, saved_binary = binary;
        require(error.size() == 1 && (info.size() > 1) == verbose && !binary.empty(), "verbosity/timing control info availability without introducing errors");
        program.destroy();
        require(error == saved_error && info == saved_info && binary == saved_binary, "caller-owned diagnostic/code copies survive link-handle destruction");
    }
}
void jit_diagnostic_validation() {
    for (bool versioned : {false, true}) {
        require(JIT_DIAGNOSTIC_CALL(nvJitLinkGetErrorLogSize, nullptr, static_cast<size_t*>(nullptr)) == NVJITLINK_ERROR_INVALID_INPUT, "null log handle has invalid-input status");
        jit_diagnostic_program program(versioned);
        require(JIT_DIAGNOSTIC_CALL(nvJitLinkGetErrorLogSize, program.value, nullptr) == NVJITLINK_ERROR_INVALID_INPUT
            && JIT_DIAGNOSTIC_CALL(nvJitLinkGetErrorLog, program.value, nullptr) == NVJITLINK_ERROR_INVALID_INPUT, "null error-log outputs");
        require(JIT_DIAGNOSTIC_CALL(nvJitLinkGetInfoLogSize, program.value, nullptr) == NVJITLINK_ERROR_INVALID_INPUT
            && JIT_DIAGNOSTIC_CALL(nvJitLinkGetInfoLog, program.value, nullptr) == NVJITLINK_ERROR_INVALID_INPUT, "null info-log outputs");
        program.incomplete();
        jit_diagnostic_output<size_t> size;
        require(JIT_DIAGNOSTIC_CALL(nvJitLinkGetLinkedPtxSize, program.value, &size.value) == NVJITLINK_ERROR_INVALID_INPUT,
            "unselected output mode has invalid-input status"); size.get();
        diagnostic_jit(program.add(diagnostic_jit_prefix + ".visible .entry diagnostic_validation() { ret; }\n"));
        diagnostic_jit(program.complete()); require(!program.bytes(0).empty(), "diagnostic validation failures leave accepted link state usable");
    }
}
std::string_view nvjitlink_diagnostics_case = "all";
void execute_nvjitlink_diagnostics() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"configuration", jit_diagnostic_configuration}, {"inputs", jit_diagnostic_inputs}, {"finalization", jit_diagnostic_finalization},
        {"logs", jit_diagnostic_logs}, {"validation", jit_diagnostic_validation}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (nvjitlink_diagnostics_case == "all" || nvjitlink_diagnostics_case == name) { selected = true; execute(); }
    require(selected, "unknown nvjitlink_diagnostics case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvjitlink_diagnostics_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvjitlink_diagnostics);
}
