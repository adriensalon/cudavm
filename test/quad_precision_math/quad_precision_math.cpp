#include "conformance.hpp"
#include <utility>

namespace cudavm {
constexpr const char* quad_windows_header = R"cuda(
#if !defined(_WIN64) || !defined(__CUDACC_RTC__)
#error quad_windows_rtc_profile_required
#endif
#include <crt/device_fp128_functions.h>
#include <crt/device_fp128_functions.h>
#ifndef __DEVICE_FP128_FUNCTIONS_H__
#error quad_header_was_not_processed
#endif
#if defined(__FLOAT128_CPP_SPELLING_ENABLED__) || defined(__FLOAT128_C_SPELLING_ENABLED__)
#error quad_type_leaked_through_windows_guard
#endif
)cuda";
void quad_disabled_headers() {
    compiler_fixture_program program(R"cuda(
#define __NV_DISABLE_DEVICE_FP128_FUNCTIONS__
#include <crt/device_fp128_functions.h>
#include <crt/device_fp128_functions.h>
#if defined(__DEVICE_FP128_FUNCTIONS_H__) || defined(__FLOAT128_CPP_SPELLING_ENABLED__) || defined(__FLOAT128_C_SPELLING_ENABLED__)
#error disabled_quad_header_leaked_declarations
#endif
extern "C" __global__ void quad_control(unsigned* out) { *out = 17; }
)cuda", "quad_disabled_headers.cu");
    require(program.success().find(".entry quad_control") != std::string::npos,
        "disabled repeated quad header leaves ordinary device code compilable");
}
void quad_windows_headers() {
    compiler_fixture_program program(std::string(quad_windows_header)
        + "extern \"C\" __global__ void quad_control(unsigned* out) { *out = 17; }\n", "quad_windows_headers.cu");
    require(program.success().find(".entry quad_control") != std::string::npos,
        "Windows RTC processes the quad header without exposing either spelling");
}
void quad_windows_rejection(const char* name, const char* statement, const char* entity) {
    const auto file = "quad_" + std::string(name) + ".cu";
    const auto source = std::string(quad_windows_header) + R"cuda(
extern "C" __global__ void quad_control(unsigned* out) {
#ifdef REJECT_QUAD
)cuda" + "#line 101 \"" + file + "\"\n" + statement + R"cuda(
#endif
    *out = 17;
}
)cuda";
    compiler_fixture_program program(source, file.c_str());
    const auto control = program.success();
    require(control.find(".entry quad_control") != std::string::npos, "legal Windows quad-header control retains kernel");
    require(program.compile({"-DREJECT_QUAD"}) == NVRTC_ERROR_COMPILATION, "unavailable Windows quad entity is a source rejection");
    const auto log = program.text(true);
    require(program.text().empty(), "quad source rejection publishes no PTX");
    require(log.find(file) != std::string::npos && log.find("101") != std::string::npos && log.find(entity) != std::string::npos,
        "quad rejection identifies the owned entity and source location: " + log);
    require(program.success() == control, "retry without the unavailable entity reproduces the legal result");
    require(program.text(true).empty(), "successful quad-source repair clears the diagnostic");
}
std::string_view quad_precision_case = "all";
void execute_quad_precision_math() {
    bool selected = false;
    auto run = [&](std::string_view name, auto execute) {
        if (quad_precision_case == "all" || quad_precision_case == name) { selected = true; execute(); }
    };
    run("disabled_headers", quad_disabled_headers);
#ifdef _WIN64
    run("windows_headers", quad_windows_headers);
    run("cpp_type", [] { quad_windows_rejection("cpp_type", "    __float128 value; (void)value;\n", "__float128"); });
    run("c_type", [] { quad_windows_rejection("c_type", "    _Float128 value; (void)value;\n", "_Float128"); });
    run("math_symbol", [] { quad_windows_rejection("math_symbol", "    (void)__nv_fp128_sqrt(1.0);\n", "__nv_fp128_sqrt"); });
#endif
    require(selected, "unknown or inapplicable quad_precision_math case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::quad_precision_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_quad_precision_math);
}
