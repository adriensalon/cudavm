#include "perf_fixtures.hpp"
#include <algorithm>
#include <array>

namespace cudavm {

void check_perf_compilation(const std::vector<char>& output, unsigned kernels) {
    perf_require(output.size() > 1 && output.back() == '\0', "PTX exact terminal NUL");
    perf_require(std::find(output.begin(), output.end() - 1, '\0') == output.end() - 1, "PTX has no embedded NUL");
    const std::string_view text(output.data(), output.size() - 1);
    perf_require(text.find(".version ") != text.npos && text.find(".target sm_75") != text.npos,
                 "PTX format and requested architecture");
    size_t count = 0, position = 0;
    while ((position = text.find(".entry ", position)) != text.npos) { ++count; position += 7; }
    perf_require(count == kernels, "PTX contains exactly the independently expected entry points");
    for (unsigned index = 0; index < kernels; ++index)
        perf_require(text.find(".entry perf_compile_" + std::to_string(index) + "(") != text.npos,
                     "PTX exports expected kernel " + std::to_string(index));
}

perf_result execute_source_compilation(const perf_options& options) {
    perf_require(options.backend == "host", "source compilation is a host-only workload");
    perf_require(options.case_name == "small" || options.case_name == "large", "unknown source compilation case");
    const bool large = options.case_name == "large";
    const std::array<const char*, 4> arguments{
        "--std=c++17", "--gpu-architecture=compute_75", large ? "-DPERF_STAGES=32" : "-DPERF_STAGES=8",
        large ? "-DPERF_LARGE=1" : "-DPERF_SMALL=1"};
    std::vector<char> output;
    const auto operation = [&] {
        perf_source_program program(perf_fixtures.source.data(), "source_compilation.cu");
        program.compile(arguments);
        output = program.ptx();
        program.close();
    };
    operation();
    check_perf_compilation(output, large ? 8 : 1);
    auto result = measure_perf(options, operation);
    check_perf_compilation(output, large ? 8 : 1);
    int major{}, minor{};
    perf_check_nvrtc(nvrtcVersion(&major, &minor), "nvrtcVersion");
    result.parameters = {{"api", "nvrtc"}, {"boundary", "create_compile_retrieve_destroy"},
        {"input", "source_compilation.cu"}, {"source_bytes", std::to_string(perf_fixtures.source.size())},
        {"output", "ptx"}, {"output_bytes", std::to_string(output.size())}, {"kernels", large ? "8" : "1"},
        {"template_stages", large ? "32..39" : "8"}, {"architecture", "compute_75"},
        {"standard", "c++17"}, {"optimization", "default"}, {"cache_policy", "fresh_program_warm_process"},
        {"nvrtc_version", std::to_string(major) + "." + std::to_string(minor)}};
    return result;
}

}

int main(int argc, char** argv) {
    return cudavm::run_perf(argc, argv, "source_compilation", cudavm::execute_source_compilation);
}
