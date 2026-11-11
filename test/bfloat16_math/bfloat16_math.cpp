#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
using bfloat16_math_format = numerical_format<7, 8>;
constexpr std::array<const char*, 4> bfloat16_math_profiles{"rounding", "sqrt", "reciprocal", "rsqrt"};
unsigned bfloat16_math_profile;
std::vector<std::array<uint32_t, 3>> bfloat16_math_inputs() {
    std::vector<std::array<uint32_t, 3>> result;
    for (uint32_t bits = 0; bits < 65536; ++bits) result.push_back({bits, 0, 0});
    return result;
}
void execute_bfloat16_math(const loaded_program& program) {
    const auto values = bfloat16_math_inputs();
    auto equivalent = [](uint32_t actual, uint32_t expected) {
        return bfloat16_math_format::is_nan(expected) ? actual <= 0xffff && bfloat16_math_format::is_nan(actual) : actual == expected;
    };
    if (!bfloat16_math_profile)
        execute_numerical_rows<uint32_t, 4>(program, "bfloat16_math_rounding", values, [](const auto& input) {
            return std::array<uint32_t, 4>{uint32_t(elementary_integral<bfloat16_math_format>(input[0], 2)),
                uint32_t(elementary_integral<bfloat16_math_format>(input[0], 3)), uint32_t(elementary_integral<bfloat16_math_format>(input[0], 1)),
                uint32_t(elementary_integral<bfloat16_math_format>(input[0], 0))};
        }, equivalent);
    else {
        const auto kernel = std::string("bfloat16_math_") + bfloat16_math_profiles[bfloat16_math_profile];
        execute_numerical_rows<uint32_t, 1>(program, kernel.c_str(), values, [](const auto& input) {
            const auto answer = bfloat16_math_profile == 3 ? elementary_rsqrt<bfloat16_math_format>(input[0])
                : elementary_root<bfloat16_math_format>(input[0], bfloat16_math_profile == 2);
            return std::array<uint32_t, 1>{uint32_t(answer)};
        }, equivalent);
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <bfloat16-math-profile>\n"; return 1; }
    const auto& profiles = cudavm::bfloat16_math_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown scalar bfloat16 math profile\n"; return 1; }
    cudavm::bfloat16_math_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_BFLOAT16_MATH_PROFILE " + std::to_string(cudavm::bfloat16_math_profile) + "\n"
        + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_test(3, argv, fixtures, cudavm::execute_bfloat16_math);
}

