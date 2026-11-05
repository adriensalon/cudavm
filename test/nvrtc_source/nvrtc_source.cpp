#include "conformance.hpp"
#include <nvrtc.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <utility>

namespace cudavm {
void source_rtc(nvrtcResult result) {
    require(result == NVRTC_SUCCESS, std::string("NVRTC source API: ") + nvrtcGetErrorString(result));
}
template<class T> struct source_output {
    static_assert(alignof(T) <= 16);
    std::array<unsigned char, 16> before;
    T value{};
    std::array<unsigned char, 16> after;
    source_output() { before.fill(0x93); after.fill(0xe7); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto byte) { return byte == 0x93; })
            && std::all_of(after.begin(), after.end(), [](auto byte) { return byte == 0xe7; }), "NVRTC source scalar output guards");
        return value;
    }
};
struct source_program {
    nvrtcProgram value{};
    explicit source_program(const std::string& source, const std::vector<std::pair<std::string, std::string>>& headers = {}, const char* name = "nvrtc_source.cu") {
        std::vector<const char*> names, contents;
        for (const auto& [header, bytes] : headers) { names.push_back(header.c_str()); contents.push_back(bytes.c_str()); }
        source_output<nvrtcProgram> output;
        source_rtc(nvrtcCreateProgram(&output.value, source.c_str(), name, static_cast<int>(headers.size()), contents.data(), names.data()));
        value = output.get(); require(value != nullptr, "NVRTC source create publishes program");
    }
    ~source_program() { if (value) nvrtcDestroyProgram(&value); }
    source_program(const source_program&) = delete;
    source_program& operator=(const source_program&) = delete;
    std::string text(bool log = false) const {
        source_output<size_t> size;
        source_rtc(log ? nvrtcGetProgramLogSize(value, &size.value) : nvrtcGetPTXSize(value, &size.value));
        const auto count = size.get(); require(count > 0, "NVRTC source text includes terminator");
        std::vector<char> bytes(count + 32, char(0x5b));
        source_rtc(log ? nvrtcGetProgramLog(value, bytes.data() + 16) : nvrtcGetPTX(value, bytes.data() + 16));
        for (size_t i = 0; i < bytes.size(); ++i) if (i < 16 || i >= count + 16)
            require(bytes[i] == char(0x5b), "NVRTC source text exact output extent");
        require(bytes[count + 15] == 0 && std::find(bytes.begin() + 16, bytes.begin() + count + 15, '\0') == bytes.begin() + count + 15, "NVRTC source text exact NUL termination");
        return {bytes.begin() + 16, bytes.begin() + count + 15};
    }
    nvrtcResult compile(const std::vector<std::string>& options = {"-arch=compute_75", "--std=c++17"}) const {
        std::vector<const char*> pointers; for (const auto& option : options) pointers.push_back(option.c_str());
        return nvrtcCompileProgram(value, static_cast<int>(pointers.size()), pointers.data());
    }
    std::string success(const std::vector<std::string>& options = {"-arch=compute_75", "--std=c++17"}) const {
        const auto result = compile(options);
        require(result == NVRTC_SUCCESS, std::string(nvrtcGetErrorString(result)) + ": " + text(true));
        return text();
    }
    const char* lowered(const char* expression) const {
        source_output<const char*> output; source_rtc(nvrtcGetLoweredName(value, expression, &output.value));
        const auto name = output.get(); require(name && *name, "registered name expression has nonempty lowered name"); return name;
    }
};
void source_headers() {
    const std::vector<std::pair<std::string, std::string>> headers{
        {"source/entry.hpp", "#pragma once\n#include \"detail/value.hpp\"\n#include \"detail/value.hpp\"\n"},
        {"source/detail/value.hpp", "#ifndef SOURCE_VALUE_HPP\n#define SOURCE_VALUE_HPP\nstruct source_record { unsigned value; };\nstatic constexpr unsigned source_bias = 37;\n#endif\n"}};
    source_program program("#include \"source/entry.hpp\"\n#include \"source/entry.hpp\"\nstatic_assert(source_bias == 37, \"nested header value\");\nextern \"C\" __global__ void source_headers(unsigned* output) { source_record record{source_bias}; *output = record.value; }\n", headers);
    require(program.success().find("source_headers") != std::string::npos, "nested relative virtual headers and repeated inclusion retain kernel");
    source_program missing("#include \"conformance_missing_header.hpp\"\nextern \"C\" __global__ void source_missing() {}\n");
    require(missing.compile() == NVRTC_ERROR_COMPILATION && !missing.text(true).empty(), "missing source header is a compilation error with diagnostic");
}
struct source_include_tree {
    std::filesystem::path root;
    source_include_tree() {
        root = std::filesystem::current_path() / ("nvrtc_source_includes_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "create unique source include fixture directory");
        try {
            for (const auto* directory : {"first", "second"}) require(std::filesystem::create_directory(root / directory), "create source include search directory");
            write("first", 37); write("second", 91);
        } catch (...) { cleanup(); throw; }
    }
    void write(const char* directory, unsigned value) {
        std::ofstream file(root / directory / "selection.hpp", std::ios::binary);
        file << "#define SOURCE_SELECTION " << value << "\n";
        require(bool(file), "write source include search fixture");
    }
    void cleanup() const noexcept {
        std::error_code ignored;
        for (const auto* directory : {"first", "second"}) {
            std::filesystem::remove(root / directory / "selection.hpp", ignored);
            std::filesystem::remove(root / directory, ignored);
        }
        std::filesystem::remove(root, ignored);
    }
    ~source_include_tree() { cleanup(); }
};
void source_include_paths() {
    source_include_tree files;
    const std::string source = "#include <selection.hpp>\nstatic_assert(SOURCE_SELECTION == EXPECTED_SELECTION, \"include order\");\nextern \"C\" __global__ void source_paths(unsigned* output) { *output = SOURCE_SELECTION; }\n";
    for (bool reverse : {false, true}) {
        source_program program(source);
        const auto first = (files.root / (reverse ? "second" : "first")).string();
        const auto second = (files.root / (reverse ? "first" : "second")).string();
        require(program.success({"-arch=compute_75", "-I" + first, "--include-path=" + second,
            reverse ? "-DEXPECTED_SELECTION=91" : "--define-macro=EXPECTED_SELECTION=37"}).find("source_paths") != std::string::npos, "ordered disk include paths select independently asserted header");
    }
    source_program virtual_first(source, {{"selection.hpp", "#define SOURCE_SELECTION 113\n"}});
    require(virtual_first.success({"-arch=compute_75", "-I", (files.root / "first").string(), "-DEXPECTED_SELECTION=113"}).find("source_paths") != std::string::npos, "supplied virtual header takes precedence over include directory");
}
void source_preprocessor() {
    const std::string source = R"cuda(
#ifndef SOURCE_VALUE
#error source_value_required
#endif
#if SOURCE_VALUE != 73 || SOURCE_FLAG != 1
#error wrong_macro_value
#endif
#ifdef SOURCE_REMOVED
#error macro_not_undefined
#endif
#if !defined(__CUDACC_RTC__) || !defined(__CUDA_ARCH__)
#error missing_compilation_mode_macro
#endif
#if __CUDA_ARCH__ != EXPECTED_ARCH
#error wrong_source_architecture
#endif
extern "C" __global__ void source_macros(unsigned* output) { *output = SOURCE_VALUE + SOURCE_FLAG; }
)cuda";
    for (unsigned architecture : {75u, 80u, 90u}) {
        source_program program(source);
        const auto number = std::to_string(architecture);
        const auto text = program.success({"--gpu-architecture", "compute_" + number, "--define-macro", "SOURCE_VALUE=73", "-DSOURCE_FLAG",
            "-DSOURCE_REMOVED=1", "--undefine-macro=SOURCE_REMOVED", "-DEXPECTED_ARCH=" + number + "0"});
        require(text.find(".target sm_" + number) != std::string::npos, "preprocessor architecture agrees with emitted PTX target");
    }
    source_program physical(source);
    require(physical.success({"-arch=sm_90", "-DSOURCE_VALUE=73", "-DSOURCE_FLAG", "-DEXPECTED_ARCH=900"})
        .find(".target sm_90") != std::string::npos, "physical SM90 source macros and PTX agree");
    source_program removed("#ifdef SOURCE_FLAG\n#error short_undefine_failed\n#endif\nextern \"C\" __global__ void source_undefine() {}\n");
    removed.success({"-DSOURCE_FLAG", "-USOURCE_FLAG"});
}
void source_language() {
    const std::string source = R"cuda(
static_assert(__cplusplus == EXPECTED_CPP, "selected C++ standard");
template<class T> __device__ constexpr T source_identity(T value) { return value; }
extern "C" __global__ void source_language(unsigned* output) { *output = source_identity(19u); }
)cuda";
    const std::array<std::pair<const char*, const char*>, 4> modes{{
        {"c++11", "201103L"}, {"c++14", "201402L"}, {"c++17", "201703L"}, {"c++20", "202002L"}}};
    for (const auto& [standard, value] : modes) {
        source_program program(source);
        require(program.success({std::string("--std=") + standard, std::string("-DEXPECTED_CPP=") + value}).find("source_language") != std::string::npos, "standard option changes independently asserted language macro");
    }
    source_program modern("template<class T> concept source_unsigned = requires(T value) { value + 1u; };\ntemplate<source_unsigned T> __device__ T source_add(T value) { return value + 1u; }\nextern \"C\" __global__ void source_concept(unsigned* output) { *output = source_add(*output); }\n");
    require(modern.compile({"--std=c++17"}) == NVRTC_ERROR_COMPILATION && !modern.text(true).empty(), "C++20 concept is rejected in C++17");
    require(modern.success({"--std", "c++20"}).find("source_concept") != std::string::npos, "same concept program compiles in selected C++20 mode");
}
void source_names() {
    source_program program(R"cuda(
namespace source_names {
template<class T, unsigned Bias> __global__ void write(T* output) { *output = Bias; }
__device__ unsigned counter = 13;
__constant__ unsigned bias = 29;
}
extern "C" __global__ void source_c_name(unsigned* output) { *output = 41; }
)cuda");
    const std::array<const char*, 4> expressions{"&source_names::write<unsigned, 7>", "&source_names::counter", "&source_names::bias", "&source_c_name"};
    source_output<const char*> premature;
    require(nvrtcGetLoweredName(program.value, expressions[0], &premature.value) == NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION, "name query requires successful compilation");
    premature.get();
    for (const auto* expression : expressions) {
        std::string owned(expression); source_rtc(nvrtcAddNameExpression(program.value, owned.c_str()));
        std::fill(owned.begin(), owned.end(), '!');
    }
    const auto text = program.success();
    std::array<const char*, 4> pointers{};
    std::array<std::string, 4> names;
    for (size_t i = 0; i < expressions.size(); ++i) {
        pointers[i] = program.lowered(expressions[i]); names[i] = pointers[i];
        require(text.find(names[i]) != std::string::npos, "registered kernel/global/constant is retained in PTX under returned name");
        for (size_t earlier = 0; earlier < i; ++earlier) require(names[earlier] != names[i], "distinct named entities have distinct lowered names");
    }
    require(names.back() == "source_c_name", "C-linkage name is exact");
    require(nvrtcAddNameExpression(program.value, "&source_names::write<unsigned, 19>") == NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION, "name expressions cannot be added after compilation");
    require(nvrtcGetLoweredName(program.value, "source_c_name", &premature.value) == NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID, "query must use exact registered expression string");
    premature.get();
    program.success({"--std=c++20"});
    for (size_t i = 0; i < names.size(); ++i) require(std::string(pointers[i]) == names[i], "borrowed lowered-name storage survives successful recompilation");
    require(program.compile({"--source-invalid-option"}) == NVRTC_ERROR_INVALID_OPTION, "failed recompilation has defined option status");
    for (size_t i = 0; i < names.size(); ++i) require(std::string(pointers[i]) == names[i], "borrowed lowered-name storage survives failed recompilation until destroy");
}
std::string_view nvrtc_source_case = "all";
void execute_nvrtc_source() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"headers", source_headers}, {"include_paths", source_include_paths}, {"preprocessor", source_preprocessor},
        {"language", source_language}, {"names", source_names}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (nvrtc_source_case == "all" || nvrtc_source_case == name) { selected = true; execute(); }
    require(selected, "unknown nvrtc_source case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::nvrtc_source_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_nvrtc_source);
}
