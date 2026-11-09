#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
using half_math_format = numerical_format<10, 5>;
constexpr std::array<const char*, 4> half_math_profiles{"rounding", "sqrt", "reciprocal", "rsqrt"};
unsigned half_math_profile;
std::vector<std::array<uint32_t, 3>> half_math_inputs() {
    std::vector<std::array<uint32_t, 3>> result;
    for (uint32_t bits = 0; bits < 65536; ++bits) result.push_back({bits, 0, 0});
    return result;
}
void execute_half_math(const loaded_program& program) {
    const auto values = half_math_inputs();
    auto equivalent = [](uint32_t actual, uint32_t expected) {
        return half_math_format::is_nan(expected) ? actual <= 0xffff && half_math_format::is_nan(actual) : actual == expected;
    };
    if (!half_math_profile)
        execute_numerical_rows<uint32_t, 4>(program, "half_math_rounding", values, [](const auto& input) {
            return std::array<uint32_t, 4>{uint32_t(elementary_integral<half_math_format>(input[0], 2)),
                uint32_t(elementary_integral<half_math_format>(input[0], 3)), uint32_t(elementary_integral<half_math_format>(input[0], 1)),
                uint32_t(elementary_integral<half_math_format>(input[0], 0))};
        }, equivalent);
    else {
        const auto kernel = std::string("half_math_") + half_math_profiles[half_math_profile];
        execute_numerical_rows<uint32_t, 1>(program, kernel.c_str(), values, [](const auto& input) {
            const auto answer = half_math_profile == 3 ? elementary_rsqrt<half_math_format>(input[0])
                : elementary_root<half_math_format>(input[0], half_math_profile == 2);
            return std::array<uint32_t, 1>{uint32_t(answer)};
        }, equivalent);
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <half-math-profile>\n"; return 1; }
    const auto& profiles = cudavm::half_math_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown scalar half math profile\n"; return 1; }
    cudavm::half_math_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_HALF_MATH_PROFILE " + std::to_string(cudavm::half_math_profile) + "\n"
        + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_half_math);
}
