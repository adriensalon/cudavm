#include "test_framework.hpp"

namespace cudavm {
constexpr const char* parallel_source = R"cuda(
#include <cuda_device_runtime_api.h>
#ifdef EXPECT_RDC
#ifndef __CUDACC_RDC__
#error dynamic_parallelism_rdc_macro_required
#endif
#else
#ifdef __CUDACC_RDC__
#error dynamic_parallelism_rdc_macro_unexpected
#endif
#endif
struct child_payload { unsigned first; unsigned long long second; unsigned third; };
static_assert(sizeof(child_payload) == 24 && alignof(child_payload) == 8, "child aggregate layout");
static_assert(__builtin_offsetof(child_payload, second) == 8 && __builtin_offsetof(child_payload, third) == 16, "child member offsets");
extern "C" __global__ void parallel_child(unsigned* output, unsigned scalar, child_payload value) {
    output[0] = scalar + value.first + unsigned(value.second) + value.third;
}
extern "C" __global__ void parallel_parent(unsigned* output) {
    child_payload value{3, 5, 11};
#line 101 "parallel_parent.cu"
    parallel_child<<<dim3(1), dim3(1), 0, 0>>>(output, 7, value);
}
)cuda";
const std::vector<std::string> parallel_options{"-arch=compute_75", "--std=c++17", "-rdc=true", "-DEXPECT_RDC"};
void parallel_check_ptx(const std::string& ptx) {
    require(ptx.find(".entry parallel_parent") != std::string::npos && ptx.find(".entry parallel_child") != std::string::npos,
        "relocatable child source preserves both kernel entries");
    require(ptx.find("__cudaCDP2GetParameterBuffer") != std::string::npos && ptx.find("__cudaCDP2LaunchDevice") != std::string::npos,
        "child source retains its declared CDP2 allocation and launch dependencies");
}
void parallel_source_rdc() {
    compiler_fixture_program program(parallel_source, "parallel_source.cu");
    parallel_check_ptx(program.success(parallel_options));
}
void parallel_rdc_rejection() {
    compiler_fixture_program program(parallel_source, "parallel_source.cu");
    const auto control = program.success(parallel_options);
    parallel_check_ptx(control);
    for (const auto& options : std::vector<std::vector<std::string>>{
        {"-arch=compute_75"}, {"-arch=compute_75", "-rdc=false"}, {"-arch=compute_75", "-rdc=true", "-rdc=false"}}) {
        require(program.compile(options) == NVRTC_ERROR_COMPILATION, "child launch without effective RDC is a source rejection");
        const auto log = program.text(true);
        require(program.text().empty(), "non-RDC child rejection publishes no PTX");
        require(log.find("parallel_parent.cu") != std::string::npos && log.find("101") != std::string::npos
            && (log.find("rdc") != std::string::npos || log.find("relocatable") != std::string::npos),
            "non-RDC rejection identifies the owned launch and required compilation mode: " + log);
        require(program.success(parallel_options) == control, "RDC retry repairs source rejection and reproduces the control result");
    }
    auto ordered = parallel_options; ordered.insert(ordered.begin(), "-rdc=false");
    require(program.success(ordered) == control, "last effective RDC option enables child source");
}
void parallel_source_lto() {
    compiler_fixture_program program(parallel_source, "parallel_source.cu");
    compiler_fixture_program::check(nvrtcAddNameExpression(program.value, "&parallel_child"));
    const auto status = program.compile({"-arch=compute_75", "--std=c++17", "-dlto", "-DEXPECT_RDC"});
    require(status == NVRTC_SUCCESS,
        "device LTO implies RDC for child source: " + program.text(true));
    const auto image = program.lto();
    require(!image.empty(), "child source has a nonempty retained LTO result");
    const char* name = nullptr;
    compiler_fixture_program::check(nvrtcGetLoweredName(program.value, "&parallel_child", &name));
    require(name && std::string_view(name) == "parallel_child", "retained child has its public extern-C kernel identity");
}
std::string_view parallel_case = "all";
void execute_dynamic_parallelism() {
    bool selected = false;
    auto run = [&](std::string_view name, auto execute) {
        if (parallel_case == "all" || parallel_case == name) { selected = true; execute(); }
    };
    run("source_rdc", parallel_source_rdc);
    run("rdc_rejection", parallel_rdc_rejection);
    run("source_lto", parallel_source_lto);
    require(selected, "unknown dynamic_parallelism case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::parallel_case = argv[1];
    return cudavm::run_host_test(argc, argv, cudavm::execute_dynamic_parallelism);
}
