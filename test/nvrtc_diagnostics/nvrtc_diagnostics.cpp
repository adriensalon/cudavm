#include "test_framework.hpp"
#include <nvrtc.h>
#include <algorithm>
#include <utility>

namespace cudavm {
void diagnostics_rtc(nvrtcResult result) {
    require(result == NVRTC_SUCCESS, std::string("NVRTC diagnostic API: ") + nvrtcGetErrorString(result));
}
template<class T> struct diagnostics_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    diagnostics_output() { before.fill(0xbd); after.fill(0x61); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0xbd; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0x61; }), "NVRTC diagnostics scalar output guards");
        return value;
    }
};
struct diagnostics_program {
    nvrtcProgram value{};
    explicit diagnostics_program(const char* source, const char* header = nullptr) {
        diagnostics_output<nvrtcProgram> output;
        const char* header_name = "diagnostic_header.hpp";
        diagnostics_rtc(nvrtcCreateProgram(&output.value, source, "diagnostic_source.cu", header ? 1 : 0,
            header ? &header : nullptr, header ? &header_name : nullptr));
        value = output.get(); require(value != nullptr, "diagnostic program handle published");
    }
    ~diagnostics_program() { if (value) nvrtcDestroyProgram(&value); }
    diagnostics_program(const diagnostics_program&) = delete;
    diagnostics_program& operator=(const diagnostics_program&) = delete;
    nvrtcResult compile(const std::vector<std::string>& options = {"-arch=compute_75"}) const {
        std::vector<const char*> pointers; for (const auto& option : options) pointers.push_back(option.c_str());
        return nvrtcCompileProgram(value, static_cast<int>(pointers.size()), pointers.data());
    }
    std::string text(bool ptx = false) const {
        diagnostics_output<size_t> size, repeated;
        const auto query_size = ptx ? nvrtcGetPTXSize : nvrtcGetProgramLogSize;
        const auto query_text = ptx ? nvrtcGetPTX : nvrtcGetProgramLog;
        diagnostics_rtc(query_size(value, &size.value)); diagnostics_rtc(query_size(value, &repeated.value));
        const auto count = size.get(); require(count > 0 && repeated.get() == count, "diagnostic text size is stable and includes terminator");
        std::vector<char> first(count + 32, char(0x39)), second(count + 32, char(0x72));
        diagnostics_rtc(query_text(value, first.data() + 16)); diagnostics_rtc(query_text(value, second.data() + 16));
        for (size_t i = 0; i < first.size(); ++i) {
            if (i < 16 || i >= count + 16) require(first[i] == char(0x39) && second[i] == char(0x72), "diagnostic Get writes only exact reported extent");
            else require(first[i] == second[i], "diagnostic Get writes all bytes and repeat retrieval agrees");
        }
        require(first[count + 15] == 0 && std::find(first.begin() + 16, first.begin() + count + 15, '\0') == first.begin() + count + 15, "diagnostic text has exact terminal NUL");
        return {first.begin() + 16, first.begin() + count + 15};
    }
};
void diagnostics_source() {
    diagnostics_program program("#line 127 \"owned_diagnostic.cu\"\n#error conformance_owned_source_diagnostic\n");
    require(program.compile() == NVRTC_ERROR_COMPILATION, "source error returns compilation status");
    const auto log = program.text();
    require(log.find("owned_diagnostic.cu") != std::string::npos && log.find("127") != std::string::npos
        && log.find("conformance_owned_source_diagnostic") != std::string::npos, "source diagnostic retains caller filename, line and error directive payload");
    require(program.text(true).empty(), "failed source compilation publishes no PTX");
}
void diagnostics_headers() {
    diagnostics_program program("#include \"diagnostic_header.hpp\"\n", "#line 211 \"owned_diagnostic_header.hpp\"\n#error conformance_owned_header_diagnostic\n");
    require(program.compile() == NVRTC_ERROR_COMPILATION, "virtual header error returns compilation status");
    const auto log = program.text();
    require(log.find("owned_diagnostic_header.hpp") != std::string::npos && log.find("211") != std::string::npos
        && log.find("conformance_owned_header_diagnostic") != std::string::npos, "virtual header diagnostic retains caller location and payload");
    require(program.text(true).empty(), "failed header compilation publishes no PTX");
}
void diagnostics_options() {
    diagnostics_program program("extern \"C\" __global__ void diagnostics_options() {}\n");
    for (const auto* option : {"--conformance-invalid-option", "--std=c++98", "--gpu-architecture=compute_0", "--include-path", "--define-macro="}) {
        diagnostics_rtc(program.compile());
        require(program.text(true).find("diagnostics_options") != std::string::npos, "valid compilation before invalid option has PTX");
        require(program.compile({option}) == NVRTC_ERROR_INVALID_OPTION, "invalid public option has option-specific status");
        require(!program.text().empty() && program.text(true).empty(), "invalid option replaces successful output with a diagnostic and empty PTX");
    }
    diagnostics_rtc(program.compile());
    require(program.text().empty() && program.text(true).find("diagnostics_options") != std::string::npos, "valid retry clears previous option error and restores PTX");
}
void diagnostics_replacement() {
    diagnostics_program program(R"cuda(
#ifdef EMIT_ERROR
#line 313 "replacement_error.cu"
#error conformance_replacement_error
#endif
#ifdef EMIT_WARNING
#warning conformance_replacement_warning
#endif
extern "C" __global__ void diagnostics_replacement(unsigned* output) { *output = 37; }
)cuda");
    require(program.text().empty(), "new program has empty diagnostic text");
    diagnostics_rtc(program.compile());
    const auto clean_ptx = program.text(true);
    require(program.text().empty() && !clean_ptx.empty(), "clean compilation has empty diagnostic text");
    diagnostics_rtc(program.compile({"-DEMIT_WARNING"}));
    const auto warning = program.text();
    require(warning.find("conformance_replacement_warning") != std::string::npos && !program.text(true).empty(), "successful compilation retains warning and PTX");
    require(program.compile({"-DEMIT_ERROR"}) == NVRTC_ERROR_COMPILATION, "subsequent source error has compilation status");
    const auto error = program.text();
    require(error.find("conformance_replacement_error") != std::string::npos && error.find("conformance_replacement_warning") == std::string::npos
        && program.text(true).empty(), "error replaces previous warning and code without concatenating logs");
    diagnostics_rtc(program.compile());
    require(program.text().empty() && program.text(true) == clean_ptx, "clean retry removes stale diagnostics and reproduces original textual output");
    require(warning.find("conformance_replacement_warning") != std::string::npos && error.find("conformance_replacement_error") != std::string::npos, "caller-owned diagnostic snapshots survive later compilation");
}
void diagnostics_validation() {
    diagnostics_program program("extern \"C\" __global__ void diagnostics_valid() {}\n");
    diagnostics_output<size_t> size;
    char byte{};
    require(nvrtcGetProgramLogSize(nullptr, &size.value) == NVRTC_ERROR_INVALID_PROGRAM
        && nvrtcGetProgramLog(nullptr, &byte) == NVRTC_ERROR_INVALID_PROGRAM, "null diagnostic handles have invalid-program status");
    size.get();
    require(nvrtcGetProgramLogSize(program.value, nullptr) == NVRTC_ERROR_INVALID_INPUT
        && nvrtcGetProgramLog(program.value, nullptr) == NVRTC_ERROR_INVALID_INPUT, "null diagnostic outputs have invalid-input status");
    require(nvrtcCompileProgram(program.value, -1, nullptr) == NVRTC_ERROR_INVALID_INPUT
        && nvrtcCompileProgram(program.value, 1, nullptr) == NVRTC_ERROR_INVALID_INPUT, "invalid option-array shapes have invalid-input status");
    const char* null_option = nullptr;
    require(nvrtcCompileProgram(program.value, 1, &null_option) == NVRTC_ERROR_INVALID_INPUT, "null option entry has invalid-input status");
    diagnostics_rtc(program.compile());
    require(program.text().empty() && program.text(true).find("diagnostics_valid") != std::string::npos, "invalid argument calls leave program usable");
}
std::string_view nvrtc_diagnostics_case = "all";
void execute_nvrtc_diagnostics() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"source", diagnostics_source}, {"headers", diagnostics_headers}, {"options", diagnostics_options},
        {"replacement", diagnostics_replacement}, {"validation", diagnostics_validation}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (nvrtc_diagnostics_case == "all" || nvrtc_diagnostics_case == name) { selected = true; execute(); }
    require(selected, "unknown nvrtc_diagnostics case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvrtc_diagnostics_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvrtc_diagnostics);
}
