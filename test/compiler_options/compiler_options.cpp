#include "test_framework.hpp"
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
void options_architectures() {
    compiler_fixture_program program(R"cuda(
static_assert(__CUDA_ARCH__ == EXPECTED_ARCH, "effective source architecture");
#ifdef __CUDA_ARCH_FEAT_SM90_ALL
static_assert(EXPECTED_ARCH == 900 && EXPECT_SPECIFIC == 1, "Hopper-specific features enabled");
#elif defined(__CUDA_ARCH_FEAT_SM100_ALL)
static_assert(EXPECTED_ARCH == 1000 && EXPECT_SPECIFIC == 1, "SM100-specific features enabled");
#else
static_assert(EXPECT_SPECIFIC == 0, "generic architecture has no specific features");
#endif
extern "C" __global__ void options_control(unsigned* out) { *out = __CUDA_ARCH__; }
)cuda", "compiler_architectures.cu");
    for (unsigned architecture : {90u, 100u}) {
        const auto number = std::to_string(architecture), expected = "-DEXPECTED_ARCH=" + std::to_string(architecture * 10);
        const std::vector<std::string> generic_options{"-arch=compute_" + number, expected, "-DEXPECT_SPECIFIC=0"};
        const auto generic = program.success(generic_options);
        require(generic.find(".target sm_" + number + "\n") != std::string::npos, "generic source target remains generic");
        const auto specific = program.success({"-arch=compute_" + number + "a", expected, "-DEXPECT_SPECIFIC=1"});
        require(specific.find(".target sm_" + number + "a\n") != std::string::npos, "architecture-specific target survives native PTX emission");
        program.check(program.compile({"-arch=sm_" + number + "a", expected, "-DEXPECT_SPECIFIC=1", "-dlto"}));
        require(!program.lto().empty(), "physical architecture-specific option retains a relocatable public result");
        for (const auto& invalid : {"compute_" + number + "f", "sm_" + std::to_string(architecture + 1) + "a"}) {
            require(program.compile({"-arch=" + invalid, expected, "-DEXPECT_SPECIFIC=1"}) == NVRTC_ERROR_INVALID_OPTION,
                "undeclared architecture variants remain unsupported options");
            require(program.success(generic_options) == generic, "generic architecture repair clears architecture-specific features");
        }
    }
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
void options_assertions() {
    compiler_fixture_program program(R"cuda(
#ifndef assert
#error nvrtc_assert_required
#endif
#ifdef NDEBUG
__device__ constexpr unsigned disabled_assertion() {
    unsigned evaluated = 0;
    assert(++evaluated);
    return evaluated;
}
static_assert(disabled_assertion() == 0, "NDEBUG does not evaluate the expression");
extern "C" __global__ void options_control(unsigned* output) { *output = disabled_assertion(); }
#else
extern "C" __global__ void options_control(unsigned* output) {
    assert(output != nullptr);
    *output = 1;
}
#endif
)cuda", "compiler_assertions.cu");
    require(program.success().find("__assertfail") != std::string::npos, "enabled assert retains the device assertion ABI");
    require(program.success({"-DNDEBUG"}).find("__assertfail") == std::string::npos, "NDEBUG eliminates the assertion call");
    require(program.success({"-DNDEBUG", "-UNDEBUG"}).find("__assertfail") != std::string::npos, "ordered NDEBUG removal restores assertions");
}
std::string_view compiler_options_case = "all";
void execute_compiler_options() {
    const std::array<std::pair<std::string_view, void (*)()>, 7> cases{{
        {"defaults", options_defaults}, {"dialects", options_dialects}, {"macros", options_macros},
        {"isolation", options_isolation}, {"validation", options_validation}, {"assertions", options_assertions},
        {"architectures", options_architectures}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (compiler_options_case == "all" || compiler_options_case == name) { selected = true; execute(); }
    require(selected, "unknown compiler_options case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::compiler_options_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_compiler_options);
}
