#include "test_fixtures.hpp"

namespace cudavm {
std::array<uint32_t, 24> float_arithmetic_oracle(const std::array<uint32_t, 3>& input) {
    const auto [a, b, c] = input;
    std::array<uint32_t, 24> result{};
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
        result[rounding] = uint32_t(numerical_binary32::add(a, b, rounding));
        result[4 + rounding] = uint32_t(numerical_binary32::add(a, b ^ 0x80000000u, rounding));
        result[8 + rounding] = uint32_t(numerical_binary32::multiply(a, b, rounding));
        result[12 + rounding] = uint32_t(numerical_binary32::fused(a, b, c, rounding));
        result[16 + rounding] = uint32_t(numerical_binary32::add(result[8 + rounding], c, rounding));
    }
    for (unsigned i = 0; i < 4; ++i) result[20 + i] = result[i * 4];
    return result;
}
std::vector<std::array<uint32_t, 3>> float_arithmetic_inputs() {
    constexpr uint32_t edges[]{0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
        0x33000000, 0xb3000000, 0x33800000, 0xb3800000, 0x3f000000, 0xbf000000,
        0x3f7fffff, 0x3f800000, 0x3f800001, 0xbf7fffff, 0xbf800000, 0xbf800001,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xff812345};
    std::vector<std::array<uint32_t, 3>> result;
    for (auto a : edges) for (auto b : edges) for (auto c : edges) result.push_back({a, b, c});
    // (1+epsilon)*(1-epsilon)-1 separates fused and rounded products.
    for (uint32_t offset : {1u, 2u, 3u, 127u, 1023u, 4095u})
        for (uint32_t sign : {0u, 0x80000000u}) result.push_back({0x3f800000u + offset, 0x3f800000u - 2 * offset, 0xbf800000u ^ sign});
    uint32_t state = 0x7b9135c3;
    auto random = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (unsigned i = 0; i < 2049; ++i) {
        const auto a = random(), b = random(), c = random(); result.push_back({a, b, c});
        if (i % 7 == 0) result.push_back({a, a ^ 0x80000000u, c});
    }
    return result;
}
void execute_float_arithmetic(const loaded_program& program) {
    const auto values = float_arithmetic_inputs();
    auto equivalent = [](uint32_t actual, uint32_t expected) {
        return numerical_binary32::is_nan(expected) ? numerical_binary32::is_nan(actual) : actual == expected;
    };
    execute_numerical_rows<uint32_t, 20>(program, "float_arithmetic_values", values, [](const auto& input) {
        const auto complete = float_arithmetic_oracle(input);
        std::array<uint32_t, 20> row; std::copy_n(complete.begin(), row.size(), row.begin()); return row;
    }, equivalent);
    execute_numerical_rows<uint32_t, 4>(program, "float_arithmetic_default", values, [](const auto& input) {
        const auto complete = float_arithmetic_oracle(input);
        std::array<uint32_t, 4> row; std::copy_n(complete.begin() + 20, row.size(), row.begin()); return row;
    }, equivalent);
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_float_arithmetic);
}
