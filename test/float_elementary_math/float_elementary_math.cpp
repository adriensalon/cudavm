#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
unsigned float_elementary_profile;
void execute_float_elementary_math(const loaded_program& program) {
    const auto values = elementary_inputs<uint32_t, 23, 8>();
    auto equivalent = [](uint32_t actual, uint32_t expected) {
        return numerical_binary32::is_nan(expected) ? numerical_binary32::is_nan(actual) : actual == expected;
    };
    if (!float_elementary_profile)
        execute_numerical_rows<uint32_t, 6>(program, "float_elementary_rounding", values, [](const auto& input) {
            const auto wide = elementary_rounding<numerical_binary32>(input[0]);
            std::array<uint32_t, 6> row; std::copy(wide.begin(), wide.end(), row.begin()); return row;
        }, equivalent);
    else execute_numerical_rows<uint32_t, 2>(program, "float_elementary_roots", values, [](const auto& input) {
        return std::array<uint32_t, 2>{uint32_t(elementary_root<numerical_binary32>(input[0], false)),
            uint32_t(elementary_root<numerical_binary32>(input[0], true))};
    }, equivalent);
}
}
int main(int argc, char** argv) {
    if (argc != 4 || (std::string_view(argv[3]) != "rounding" && std::string_view(argv[3]) != "roots")) {
        std::cerr << "usage: <executable> <backend> cu <rounding|roots>\n"; return 1;
    }
    cudavm::float_elementary_profile = std::string_view(argv[3]) == "roots";
    const auto source = "#define CUDAVM_ELEMENTARY_PROFILE " + std::to_string(cudavm::float_elementary_profile) + "\n"
        + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_float_elementary_math);
}
