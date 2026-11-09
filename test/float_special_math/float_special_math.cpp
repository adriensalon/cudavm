#include "test_fixtures.hpp"
#include "oracle.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 3> float_special_profiles{"hypot", "norm3", "norm4"};
unsigned float_special_profile;
void execute_float_special_math(const loaded_program& program) {
    const unsigned dimensions = float_special_profile + 2;
    const auto kernel = std::string("float_special_") + float_special_profiles[float_special_profile];
    execute_numerical_rows<uint64_t, 4>(program, kernel.c_str(), float_norm_inputs(dimensions), [dimensions](const auto& input) {
        const uint64_t expected = float_norm_oracle(float_norm_components(input), dimensions);
        return std::array<uint64_t, 4>{expected, expected, expected, expected};
    }, [](uint64_t actual, uint64_t expected) {
        if (actual >> 32) return false;
        if (numerical_binary32::is_nan(expected)) return numerical_binary32::is_nan(actual);
        if (numerical_binary32::is_zero(expected) || numerical_binary32::is_infinite(expected)) return actual == expected;
        if (actual & 0x80000000u || numerical_binary32::is_nan(actual) || numerical_binary32::is_infinite(actual)) return false;
        return (actual > expected ? actual - expected : expected - actual) <= 3;
    });
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <profile>\n"; return 1; }
    const auto& profiles = cudavm::float_special_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) return 1;
    cudavm::float_special_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_FLOAT_SPECIAL_PROFILE " + std::to_string(cudavm::float_special_profile) + "\n" + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_float_special_math);
}
