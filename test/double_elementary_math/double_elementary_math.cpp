#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 3> double_elementary_profiles{"rounding", "sqrt", "reciprocal"};
unsigned double_elementary_profile;
void execute_double_elementary_math(const loaded_program& program) {
    const auto values = elementary_inputs<uint64_t, 52, 11>();
    auto equivalent = [](uint64_t actual, uint64_t expected) {
        return numerical_binary64::is_nan(expected) ? numerical_binary64::is_nan(actual) : actual == expected;
    };
    if (!double_elementary_profile)
        execute_numerical_rows<uint64_t, 6>(program, "double_elementary_rounding", values, [](const auto& input) {
            return elementary_rounding<numerical_binary64>(input[0]);
        }, equivalent);
    else {
        const auto name = std::string("double_elementary_") + double_elementary_profiles[double_elementary_profile];
        execute_numerical_rows<uint64_t, 1>(program, name.c_str(), values, [](const auto& input) {
            return std::array<uint64_t, 1>{elementary_root<numerical_binary64>(input[0], double_elementary_profile == 2)};
        }, equivalent);
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <rounding|sqrt|reciprocal>\n"; return 1; }
    const auto& profiles = cudavm::double_elementary_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown binary64 elementary profile\n"; return 1; }
    cudavm::double_elementary_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_ELEMENTARY_PROFILE " + std::to_string(cudavm::double_elementary_profile) + "\n"
        + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_double_elementary_math);
}
