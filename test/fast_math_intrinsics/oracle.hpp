#pragma once
#include "../test_framework.hpp"

namespace cudavm {
inline std::array<uint32_t, 2> fast_divide_oracle(const std::array<uint32_t, 3>& row) {
    const auto precise = uint32_t(numerical_divide<numerical_binary32>(row[0], row[1]));
    const auto x = row[0] & 0x7fffffffu, y = row[1] & 0x7fffffffu;
    const auto fast = y > 0x7e800000u && y < 0x7f800000u
        ? x >= 0x7f800000u ? uint32_t(numerical_binary32::nan) : (row[0] ^ row[1]) & 0x80000000u : precise;
    return {precise, fast};
}
inline bool fast_divide_equivalent(uint32_t actual, uint32_t expected, size_t field, unsigned profile) {
    if (numerical_binary32::is_nan(expected)) return numerical_binary32::is_nan(actual);
    if (!field) return actual == expected;
    if (numerical_binary32::is_zero(expected))
        return profile == 1 ? numerical_binary32::is_zero(actual) : actual == expected;
    if (numerical_binary32::is_infinite(expected)) return actual == expected;
    if (numerical_binary32::is_nan(actual) || numerical_binary32::is_infinite(actual)) return false;
    auto order = [](uint32_t bits) { return bits & 0x80000000u ? ~bits : bits | 0x80000000u; };
    const auto a = order(actual), b = order(expected);
    return (a > b ? a - b : b - a) <= 2;
}
inline std::vector<std::array<uint32_t, 3>> fast_divide_inputs(unsigned profile) {
    std::vector<std::array<uint32_t, 3>> rows;
    auto append = [&](uint32_t a, uint32_t b) { rows.push_back({a, b, std::rotl(a, 9) ^ b ^ 0xa55ac3d2u}); };
    auto signed_pair = [&](uint32_t a, uint32_t b) {
        for (uint32_t x : {0u, 0x80000000u}) for (uint32_t y : {0u, 0x80000000u}) append(a | x, b | y);
    };
    if (!profile) {
        // Normal inputs and normal results throughout the documented bound.
        for (int y_exponent = 1; y_exponent <= 253; ++y_exponent)
            for (uint32_t y_fraction : {0u, 1u, 0x7fffffu}) {
                if (y_exponent == 253 && y_fraction) continue;
                for (int difference : {-125, -64, -1, 0, 1, 64, 125}) {
                    const int x_exponent = y_exponent + difference;
                    if (x_exponent < 1 || x_exponent > 254) continue;
                    for (uint32_t x_fraction : {0u, 1u, 0x7fffffu})
                        signed_pair((uint32_t(x_exponent) << 23) | x_fraction, (uint32_t(y_exponent) << 23) | y_fraction);
                }
            }
        uint32_t state = 0x3c6ef372;
        for (unsigned i = 0; i != 8192; ++i) {
            state = state * 1664525u + 1013904223u; const auto y_bits = state;
            const int y_exponent = 1 + int((state >> 24) % 253);
            state = state * 1664525u + 1013904223u;
            const int x_exponent = y_exponent + int((state >> 24) % 251) - 125;
            if (x_exponent < 1 || x_exponent > 254) continue;
            append((state & 0x807fffffu) | (uint32_t(x_exponent) << 23),
                (y_bits & 0x80000000u) | (uint32_t(y_exponent) << 23) | (y_exponent == 253 ? 0 : y_bits & 0x7fffffu));
        }
    } else if (profile == 1) {
        constexpr uint32_t denominators[]{0x7e800001, 0x7eb504f3, 0x7effffff, 0x7f000000, 0x7f000001, 0x7f400000, 0x7f7ffffe, 0x7f7fffff};
        for (uint32_t exponent = 0; exponent != 256; ++exponent)
            for (uint32_t fraction : {0u, 1u, 0x7fffffu}) for (auto y : denominators)
                signed_pair((exponent << 23) | fraction, y);
    } else {
        constexpr uint32_t numerators[]{0, 0x80000000, 0x800000, 0x80800000, 0x3f800000, 0xbf800000, 0x40000000,
            0xc0000000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321};
        constexpr uint32_t denominators[]{0, 0x80000000, 0x800000, 0x80800000, 0x3f800000, 0xbf800000,
            0x7e7fffff, 0xfe7fffff, 0x7e800000, 0xfe800000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321};
        for (auto x : numerators) for (auto y : denominators) append(x, y);
    }
    return rows;
}
}
