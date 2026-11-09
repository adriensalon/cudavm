#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
constexpr std::array<const char*, 9> double_arithmetic_profiles{"add", "subtract", "multiply", "fma_rn", "fma_rz", "fma_rd", "fma_ru", "separate", "default"};
unsigned double_arithmetic_profile = 0;
std::array<uint64_t, 4> double_arithmetic_oracle(const std::array<uint64_t, 3>& input) {
    const auto [a, b, c] = input;
    std::array<uint64_t, 4> result{};
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
        if (double_arithmetic_profile == 0) result[rounding] = numerical_binary64::add(a, b, rounding);
        else if (double_arithmetic_profile == 1) result[rounding] = numerical_binary64::add(a, b ^ 0x8000000000000000ull, rounding);
        else if (double_arithmetic_profile == 2) result[rounding] = numerical_binary64::multiply(a, b, rounding);
        else if (double_arithmetic_profile == 7) result[rounding] = numerical_binary64::add(numerical_binary64::multiply(a, b, rounding), c, rounding);
        else {
            require(double_arithmetic_profile == 8, "known non-FMA binary64 profile");
            return {numerical_binary64::add(a, b, 0), numerical_binary64::add(a, b ^ 0x8000000000000000ull, 0),
                numerical_binary64::multiply(a, b, 0), numerical_binary64::fused(a, b, c, 0)};
        }
    }
    return result;
}
std::vector<std::array<uint64_t, 3>> double_arithmetic_inputs() {
    constexpr uint64_t edges[]{0, 0x8000000000000000ull, 1, 0x8000000000000001ull,
        0x000fffffffffffffull, 0x800fffffffffffffull, 0x0010000000000000ull, 0x8010000000000000ull,
        0x3c90000000000000ull, 0xbc90000000000000ull, 0x3ca0000000000000ull, 0xbca0000000000000ull,
        0x3fe0000000000000ull, 0xbfe0000000000000ull, 0x3fefffffffffffffull, 0x3ff0000000000000ull,
        0x3ff0000000000001ull, 0xbfefffffffffffffull, 0xbff0000000000000ull, 0xbff0000000000001ull,
        0x7fefffffffffffffull, 0xffefffffffffffffull, 0x7ff0000000000000ull, 0xfff0000000000000ull,
        0x7ff8123456789abcull, 0xfff0123456789abcull};
    std::vector<std::array<uint64_t, 3>> result;
    for (auto a : edges) for (auto b : edges) for (auto c : edges) result.push_back({a, b, c});
    for (uint64_t offset : {1ull, 2ull, 3ull, 127ull, 1023ull, 1048575ull})
        result.push_back({0x3ff0000000000000ull + offset, 0x3ff0000000000000ull - 2 * offset, 0xbff0000000000000ull});
    uint64_t state = 0x7b9135c3e497a651ull;
    auto random = [&] { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; };
    for (unsigned i = 0; i < 2049; ++i) {
        const auto a = random(), b = random(), c = random(); result.push_back({a, b, c});
        if (i % 7 == 0) result.push_back({a, a ^ 0x8000000000000000ull, c});
    }
    return result;
}
void execute_double_arithmetic(const loaded_program& program) {
    const auto values = double_arithmetic_inputs();
    auto equivalent = [](uint64_t actual, uint64_t expected) {
        return numerical_binary64::is_nan(expected) ? numerical_binary64::is_nan(actual) : actual == expected;
    };
    const auto kernel = std::string("double_arithmetic_") + double_arithmetic_profiles[double_arithmetic_profile];
    if (double_arithmetic_profile >= 3 && double_arithmetic_profile <= 6)
        execute_numerical_rows<uint64_t, 1>(program, kernel.c_str(), values, [](const auto& input) {
            return std::array<uint64_t, 1>{numerical_binary64::fused(input[0], input[1], input[2], double_arithmetic_profile - 3)};
        }, equivalent);
    else execute_numerical_rows<uint64_t, 4>(program, kernel.c_str(), values, double_arithmetic_oracle, equivalent);
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <arithmetic-profile>\n"; return 1; }
    const auto& profiles = cudavm::double_arithmetic_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown binary64 arithmetic profile\n"; return 1; }
    cudavm::double_arithmetic_profile = unsigned(selected - profiles.begin());
    // Compile only this profile's kernel; the host source check compiles all
    // profiles from the unprefixed fixture independently of GPU availability.
    const auto source = "#define CUDAVM_ARITHMETIC_PROFILE " + std::to_string(cudavm::double_arithmetic_profile) + "\n"
        + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_double_arithmetic);
}
