#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
using bfloat16_arithmetic_format = numerical_format<7, 8>;
constexpr std::array<const char*, 8> bfloat16_arithmetic_profiles{"add_subtract", "multiply", "fused", "saturation", "operators", "divide", "divide_operator", "divide_assign"};
unsigned bfloat16_arithmetic_profile;
std::vector<std::array<uint32_t, 3>> bfloat16_arithmetic_inputs() {
    constexpr uint32_t edges[]{0, 0x8000, 1, 0x8001, 2, 0x8002, 0x007f, 0x807f, 0x0080, 0x8080,
        0x2000, 0x3eab, 0x3f00, 0xbf00, 0x3f7f, 0x3f80, 0x3f81, 0xbf80,
        0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7f81, 0xffc1, 0x7fff};
    std::vector<std::array<uint32_t, 3>> result;
    for (auto a : edges) for (auto b : edges) for (auto c : edges) result.push_back({a, b, c});
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const auto a = edges[bits % std::size(edges)], b = edges[(bits / std::size(edges)) % std::size(edges)];
        result.push_back({bits, a, b}); result.push_back({a, bits, b}); result.push_back({a, b, bits});
    }
    // (1+epsilon)*(1-epsilon)-1 separates single-rounding FMA from a
    // rounded intermediate product, including tiny signed results.
    for (uint32_t offset : {1u, 2u, 3u, 7u, 15u, 31u, 63u})
        for (uint32_t sign : {0u, 0x8000u}) result.push_back({0x3f80u + offset, 0x3f80u - 2 * offset, 0xbf80u ^ sign});
    // Tiny signed addends move an exact product midpoint to opposite sides.
    // An intermediate FP32/F64 rounding would erase that distinction.
    for (uint32_t a = 0x3f80; a < 0x4000; ++a)
        for (uint32_t b = 0x3f80; b < 0x4000; ++b)
            for (uint32_t c : {1u, 0x8001u}) result.push_back({a, b, c});
    for (uint32_t bits : {1u, 3u, 5u, 0x007fu, 0x0080u, 0x0081u, 0x7f7fu})
        for (uint32_t sign : {0u, 0x8000u}) for (uint32_t divisor : {0x4000u, 0x3f7fu, 0x3f81u})
            result.push_back({bits | sign, divisor, 0});
    uint32_t state = 0x71c95a3b;
    auto random = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state & 0xffffu; };
    for (unsigned i = 0; i < 4097; ++i) { const auto a = random(), b = random(), c = random(); result.push_back({a, b, c}); }
    return result;
}
uint32_t bfloat16_arithmetic_saturate(uint64_t bits) {
    if (bfloat16_arithmetic_format::is_nan(bits) || (bits & 0x8000)) return 0;
    return uint32_t(std::min(bits, uint64_t(0x3f80)));
}
void execute_bfloat16_arithmetic(const loaded_program& program) {
    const auto values = bfloat16_arithmetic_inputs();
    auto equivalent = [](uint32_t actual, uint32_t expected) {
        return bfloat16_arithmetic_format::is_nan(expected)
            ? actual <= 0xffff && bfloat16_arithmetic_format::is_nan(actual) : actual == expected;
    };
    auto add = [](uint32_t a, uint32_t b) { return uint32_t(bfloat16_arithmetic_format::add(a, b, 0)); };
    auto multiply = [](uint32_t a, uint32_t b) { return uint32_t(bfloat16_arithmetic_format::multiply(a, b, 0)); };
    if (!bfloat16_arithmetic_profile)
        execute_numerical_rows<uint32_t, 4>(program, "bfloat16_add_subtract", values, [&](const auto& input) {
            const auto sum = add(input[0], input[1]), difference = add(input[0], input[1] ^ 0x8000);
            return std::array<uint32_t, 4>{sum, difference, sum, difference};
        }, equivalent);
    else if (bfloat16_arithmetic_profile == 1)
        execute_numerical_rows<uint32_t, 2>(program, "bfloat16_multiply", values, [&](const auto& input) {
            const auto product = multiply(input[0], input[1]); return std::array<uint32_t, 2>{product, product};
        }, equivalent);
    else if (bfloat16_arithmetic_profile == 2)
        execute_numerical_rows<uint32_t, 2>(program, "bfloat16_fused", values, [&](const auto& input) {
            return std::array<uint32_t, 2>{uint32_t(bfloat16_arithmetic_format::fused(input[0], input[1], input[2], 0)),
                add(multiply(input[0], input[1]), input[2])};
        }, equivalent);
    else if (bfloat16_arithmetic_profile == 3)
        execute_numerical_rows<uint32_t, 5>(program, "bfloat16_saturation", values, [&](const auto& input) {
            const auto fused = bfloat16_arithmetic_format::fused(input[0], input[1], input[2], 0);
            const uint32_t relu = bfloat16_arithmetic_format::is_nan(fused) ? 0x7fff : fused & 0x8000 ? 0 : uint32_t(fused);
            return std::array<uint32_t, 5>{bfloat16_arithmetic_saturate(add(input[0], input[1])),
                bfloat16_arithmetic_saturate(add(input[0], input[1] ^ 0x8000)), bfloat16_arithmetic_saturate(multiply(input[0], input[1])),
                bfloat16_arithmetic_saturate(fused), relu};
        }, [](uint32_t actual, uint32_t expected) { return actual == expected; });
    else if (bfloat16_arithmetic_profile == 4) execute_numerical_rows<uint32_t, 10>(program, "bfloat16_operators", values, [&](const auto& input) {
        const auto a = input[0], b = input[1], sum = add(a, b), difference = add(a, b ^ 0x8000), product = multiply(a, b);
        return std::array<uint32_t, 10>{sum, difference, product, sum, difference, product, a, add(a, 0x3f80), a, add(a, 0xbf80)};
    }, equivalent);
    else {
        const auto kernel = std::string("bfloat16_") + bfloat16_arithmetic_profiles[bfloat16_arithmetic_profile];
        execute_numerical_rows<uint32_t, 1>(program, kernel.c_str(), values, [](const auto& input) {
            const auto quotient = uint32_t(numerical_divide<bfloat16_arithmetic_format>(input[0], input[1]));
            return std::array<uint32_t, 1>{quotient};
        }, equivalent);
    }
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <bfloat16-arithmetic-profile>\n"; return 1; }
    const auto& profiles = cudavm::bfloat16_arithmetic_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown scalar bfloat16 arithmetic profile\n"; return 1; }
    cudavm::bfloat16_arithmetic_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_BFLOAT16_PROFILE " + std::to_string(cudavm::bfloat16_arithmetic_profile) + "\n"
        + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_test(3, argv, fixtures, cudavm::execute_bfloat16_arithmetic);
}
