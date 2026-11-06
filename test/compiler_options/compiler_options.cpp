#include "conformance.hpp"
#include <utility>

namespace cudavm {
void options_defaults() {
    compiler_fixture_program program(R"cuda(
#if !defined(__CUDACC_RTC__) || !defined(__CUDA_ARCH__)
#error missing_compiler_mode
#endif
static_assert(__cplusplus == 201703L, "default C++17");
static_assert(__CUDA_ARCH__ == 750, "default compute_75");
extern "C" __global__ void options_control(unsigned* out) { *out = __CUDA_ARCH__; }
)cuda", "compiler_defaults.cu");
    const auto ptx = program.success();
    require(ptx.find(".target sm_75") != std::string::npos && ptx.find(".entry options_control") != std::string::npos,
        "default source assertions agree with PTX target/kernel");
    struct { size_t before = 0x37591abd; size_t value = 19; size_t after = 0x9417c35b; } size;
    compiler_fixture_program::check(nvrtcGetCUBINSize(program.value, &size.value));
    require(size.before == 0x37591abd && size.after == 0x9417c35b && size.value == 0,
        "default virtual target has no CUBIN and preserves size output guards");
}
void options_dialects() {
    const std::string source = R"cuda(
static_assert(__cplusplus == EXPECTED_CPP, "effective language dialect");
static_assert(__CUDA_ARCH__ == EXPECTED_ARCH, "effective architecture");
extern "C" __global__ void options_control(unsigned* out) { *out = __CUDA_ARCH__; }
)cuda";
    const std::array<std::pair<const char*, const char*>, 5> dialects{{
        {"c++11", "201103L"}, {"c++14", "201402L"}, {"c++17", "201703L"}, {"c++20", "202002L"}, {"c++23", "202302L"}}};
    for (const auto& [dialect, value] : dialects) {
        compiler_fixture_program program(source, "compiler_dialects.cu");
        const auto ptx = program.success({"--std=c++17", "--std", dialect, "-arch=compute_75", "--gpu-architecture=compute_80",
            std::string("-DEXPECTED_CPP=") + value, "-DEXPECTED_ARCH=800"});
        require(ptx.find(".target sm_80") != std::string::npos, "last architecture option agrees with asserted source mode");
    }
    compiler_fixture_program reversed(source, "compiler_dialects_reversed.cu");
    require(reversed.success({"--std=c++20", "-std=c++14", "--gpu-architecture=compute_80", "-arch", "compute_75",
        "-DEXPECTED_CPP=201402L", "-DEXPECTED_ARCH=750"}).find(".target sm_75") != std::string::npos,
        "reversed dialect and architecture option order is observable");
}
void options_macros() {
    const std::string source = R"cuda(
#ifndef OPTIONS_VALUE
#define OPTIONS_VALUE 0
#endif
#ifndef OPTIONS_FUNCTION
#define OPTIONS_FUNCTION(x) 0
#endif
#ifndef OPTIONS_FLAG
#define OPTIONS_FLAG 0
#endif
static_assert(OPTIONS_VALUE == EXPECTED_VALUE, "ordered scalar definitions/removals");
static_assert(OPTIONS_FUNCTION(7) == EXPECTED_FUNCTION, "ordered function-like definitions/removals");
static_assert(OPTIONS_FLAG == EXPECTED_FLAG, "ordered valueless definitions/removals");
extern "C" __global__ void options_control(unsigned* out) { *out = OPTIONS_VALUE + OPTIONS_FUNCTION(7) + OPTIONS_FLAG; }
)cuda";
    const std::array<std::vector<std::string>, 4> options{{
        {"-UOPTIONS_VALUE", "-DOPTIONS_VALUE=23", "-UOPTIONS_FUNCTION", "-DOPTIONS_FUNCTION(x)=((x)*3)", "-UOPTIONS_FLAG", "-DOPTIONS_FLAG",
            "-DEXPECTED_VALUE=23", "-DEXPECTED_FUNCTION=21", "-DEXPECTED_FLAG=1"},
        {"--define-macro=OPTIONS_VALUE=11", "--undefine-macro", "OPTIONS_VALUE", "--define-macro", "OPTIONS_VALUE=47",
            "-DOPTIONS_FUNCTION(x)=((x)+5)", "-UOPTIONS_FUNCTION", "-DOPTIONS_FUNCTION(x)=((x)+9)", "-DOPTIONS_FLAG", "-UOPTIONS_FLAG", "-DOPTIONS_FLAG",
            "-DEXPECTED_VALUE=47", "-DEXPECTED_FUNCTION=16", "-DEXPECTED_FLAG=1"},
        {"-DOPTIONS_VALUE=59", "-UOPTIONS_VALUE", "-DOPTIONS_FUNCTION(x)=((x)+3)", "-UOPTIONS_FUNCTION", "-DOPTIONS_FLAG", "-UOPTIONS_FLAG",
            "-DEXPECTED_VALUE=0", "-DEXPECTED_FUNCTION=0", "-DEXPECTED_FLAG=0"},
        {"-D", "OPTIONS_VALUE=71", "-U", "OPTIONS_VALUE", "-D", "OPTIONS_VALUE=83", "--undefine-macro=OPTIONS_FUNCTION",
            "--define-macro=OPTIONS_FUNCTION(x)=((x)^3)", "--undefine-macro=OPTIONS_FLAG", "--define-macro=OPTIONS_FLAG",
            "-DEXPECTED_VALUE=83", "-DEXPECTED_FUNCTION=4", "-DEXPECTED_FLAG=1"}
    }};
    compiler_fixture_program program(source, "compiler_macro_order.cu");
    for (const auto& selection : options)
        require(program.success(selection).find(".entry options_control") != std::string::npos, "ordered macro source assertions retain kernel");
}
void options_isolation() {
    const std::string source = R"cuda(
#ifndef OPTIONS_VALUE
#error options_value_required
#endif
#ifdef OPTIONS_STALE
#error options_stale_leaked
#endif
static_assert(OPTIONS_VALUE == EXPECTED_VALUE, "per-call macro state");
extern "C" __global__ void options_control(unsigned* out) { *out = OPTIONS_VALUE; }
)cuda";
    compiler_fixture_program first(source, "compiler_isolation.cu"), second(source, "compiler_isolation.cu");
    const auto a = first.success({"-DOPTIONS_VALUE=17", "-DEXPECTED_VALUE=17", "-DOPTIONS_STALE", "-UOPTIONS_STALE"});
    const auto b = second.success({"-DOPTIONS_VALUE=29", "-DEXPECTED_VALUE=29"});
    require(a != b, "independent macro options produce distinct constant results");
    require(first.compile() == NVRTC_ERROR_COMPILATION, "omitted macros do not persist between compilations");
    require(first.text().empty() && first.text(true).find("options_value_required") != std::string::npos,
        "missing-option source rejection clears code and reports owned assertion");
    require(first.success({"-DOPTIONS_VALUE=29", "-DEXPECTED_VALUE=29"}) == b, "retry uses only newly supplied options");
    require(second.text() == b, "another handle's compilation and errors preserve this program's result");
}
void options_validation() {
    compiler_fixture_program program("extern \"C\" __global__ void options_control() {}\n", "compiler_option_validation.cu");
    const auto control = program.success();
    const std::array<std::vector<std::string>, 10> invalid{{
        {"--std"}, {"-arch"}, {"-D"}, {"-U"}, {"--relocatable-device-code"},
        {"--std=c++19"}, {"--gpu-architecture=compute_0"}, {"-rdc=maybe"}, {"--define-macro="}, {"--compiler-options-unknown"}
    }};
    for (const auto& options : invalid) {
        require(program.compile(options) == NVRTC_ERROR_INVALID_OPTION, "malformed compiler option has option-specific status");
        require(program.text().empty() && !program.text(true).empty(), "invalid option clears PTX and supplies diagnostic");
        require(program.success() == control, "valid retry repairs invalid option and reproduces control PTX");
    }
}
std::string_view compiler_options_case = "all";
void execute_compiler_options() {
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"defaults", options_defaults}, {"dialects", options_dialects}, {"macros", options_macros},
        {"isolation", options_isolation}, {"validation", options_validation}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (compiler_options_case == "all" || compiler_options_case == name) { selected = true; execute(); }
    require(selected, "unknown compiler_options case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::compiler_options_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_compiler_options);
}
