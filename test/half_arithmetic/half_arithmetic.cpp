#include "test_fixtures.hpp"
#include <iostream>

namespace cudavm {
using half_arithmetic_format = numerical_format<10, 5>;
constexpr std::array<const char*, 6> half_arithmetic_profiles{"add_subtract", "multiply", "fused", "saturation", "operators", "divide"};
unsigned half_arithmetic_profile;
std::vector<std::array<uint32_t, 3>> half_arithmetic_inputs() {
    constexpr uint32_t edges[]{0, 0x8000, 1, 0x8001, 2, 0x8002, 0x03ff, 0x83ff, 0x0400, 0x8400,
        0x1000, 0x3555, 0x3800, 0xb800, 0x3bff, 0x3c00, 0x3c01, 0xbc00,
        0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0xfe01, 0x7fff};
    std::vector<std::array<uint32_t, 3>> result;
    for (auto a : edges) for (auto b : edges) for (auto c : edges) result.push_back({a, b, c});
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const auto a = edges[bits % std::size(edges)], b = edges[(bits / std::size(edges)) % std::size(edges)];
        result.push_back({bits, a, b}); result.push_back({a, bits, b}); result.push_back({a, b, bits});
    }
    // (1+epsilon)*(1-epsilon)-1 separates single-rounding FMA from a
    // rounded intermediate product, including tiny signed results.
    for (uint32_t offset : {1u, 2u, 3u, 7u, 31u, 127u, 255u})
        for (uint32_t sign : {0u, 0x8000u}) result.push_back({0x3c00u + offset, 0x3c00u - 2 * offset, 0xbc00u ^ sign});
    for (uint32_t bits : {1u, 3u, 5u, 0x03ffu, 0x0400u, 0x0401u, 0x7bffu})
        for (uint32_t sign : {0u, 0x8000u}) for (uint32_t divisor : {0x4000u, 0x3bffu, 0x3c01u})
            result.push_back({bits | sign, divisor, 0});
    uint32_t state = 0x71c95a3b;
    auto random = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state & 0xffffu; };
    for (unsigned i = 0; i < 4097; ++i) { const auto a = random(), b = random(), c = random(); result.push_back({a, b, c}); }
    return result;
}
uint32_t half_arithmetic_saturate(uint64_t bits) {
    if (half_arithmetic_format::is_nan(bits) || (bits & 0x8000)) return 0;
    return uint32_t(std::min(bits, uint64_t(0x3c00)));
}
void execute_half_arithmetic(const loaded_program& program) {
    const auto values = half_arithmetic_inputs();
    auto equivalent = [](uint32_t actual, uint32_t expected) {
        return half_arithmetic_format::is_nan(expected)
            ? actual <= 0xffff && half_arithmetic_format::is_nan(actual) : actual == expected;
    };
    auto add = [](uint32_t a, uint32_t b) { return uint32_t(half_arithmetic_format::add(a, b, 0)); };
    auto multiply = [](uint32_t a, uint32_t b) { return uint32_t(half_arithmetic_format::multiply(a, b, 0)); };
    if (!half_arithmetic_profile)
        execute_numerical_rows<uint32_t, 4>(program, "half_add_subtract", values, [&](const auto& input) {
            const auto sum = add(input[0], input[1]), difference = add(input[0], input[1] ^ 0x8000);
            return std::array<uint32_t, 4>{sum, difference, sum, difference};
        }, equivalent);
    else if (half_arithmetic_profile == 1)
        execute_numerical_rows<uint32_t, 2>(program, "half_multiply", values, [&](const auto& input) {
            const auto product = multiply(input[0], input[1]); return std::array<uint32_t, 2>{product, product};
        }, equivalent);
    else if (half_arithmetic_profile == 2)
        execute_numerical_rows<uint32_t, 2>(program, "half_fused", values, [&](const auto& input) {
            return std::array<uint32_t, 2>{uint32_t(half_arithmetic_format::fused(input[0], input[1], input[2], 0)),
                add(multiply(input[0], input[1]), input[2])};
        }, equivalent);
    else if (half_arithmetic_profile == 3)
        execute_numerical_rows<uint32_t, 5>(program, "half_saturation", values, [&](const auto& input) {
            const auto fused = half_arithmetic_format::fused(input[0], input[1], input[2], 0);
            const uint32_t relu = half_arithmetic_format::is_nan(fused) ? 0x7fff : fused & 0x8000 ? 0 : uint32_t(fused);
            return std::array<uint32_t, 5>{half_arithmetic_saturate(add(input[0], input[1])),
                half_arithmetic_saturate(add(input[0], input[1] ^ 0x8000)), half_arithmetic_saturate(multiply(input[0], input[1])),
                half_arithmetic_saturate(fused), relu};
        }, [](uint32_t actual, uint32_t expected) { return actual == expected; });
    else if (half_arithmetic_profile == 4) execute_numerical_rows<uint32_t, 10>(program, "half_operators", values, [&](const auto& input) {
        const auto a = input[0], b = input[1], sum = add(a, b), difference = add(a, b ^ 0x8000), product = multiply(a, b);
        return std::array<uint32_t, 10>{sum, difference, product, sum, difference, product, a, add(a, 0x3c00), a, add(a, 0xbc00)};
    }, equivalent);
    else execute_numerical_rows<uint32_t, 3>(program, "half_divide", values, [](const auto& input) {
        const auto quotient = uint32_t(numerical_divide<half_arithmetic_format>(input[0], input[1]));
        return std::array<uint32_t, 3>{quotient, quotient, quotient};
    }, equivalent);
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: <executable> <backend> cu <half-arithmetic-profile>\n"; return 1; }
    const auto& profiles = cudavm::half_arithmetic_profiles;
    const auto selected = std::find(profiles.begin(), profiles.end(), std::string_view(argv[3]));
    if (selected == profiles.end()) { std::cerr << "unknown scalar half arithmetic profile\n"; return 1; }
    cudavm::half_arithmetic_profile = unsigned(selected - profiles.begin());
    const auto source = "#define CUDAVM_HALF_PROFILE " + std::to_string(cudavm::half_arithmetic_profile) + "\n"
        + std::string(cudavm::fixtures.source);
    auto fixtures = cudavm::fixtures; fixtures.source = source;
    return cudavm::run_conformance(3, argv, fixtures, cudavm::execute_half_arithmetic);
}
