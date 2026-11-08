#pragma once
#include "../conformance.hpp"

namespace cudavm {
inline std::array<uint32_t, 4> float_norm_components(const std::array<uint64_t, 3>& row) {
    return {uint32_t(row[0]), uint32_t(row[0] >> 32), uint32_t(row[1]), uint32_t(row[1] >> 32)};
}
inline uint32_t float_norm_oracle(const std::array<uint32_t, 4>& words, unsigned dimensions) {
    bool nan = false;
    for (unsigned i = 0; i != dimensions; ++i) {
        if (numerical_binary32::is_infinite(words[i])) return uint32_t(numerical_binary32::infinity);
        nan |= numerical_binary32::is_nan(words[i]);
    }
    if (nan) return uint32_t(numerical_binary32::nan);
    numerical_dyadic sum;
    for (unsigned i = 0; i != dimensions; ++i) {
        const auto value = numerical_binary32::decode(words[i]);
        sum = numerical_sum(sum, {value.magnitude.times(value.magnitude), value.scale * 2, false}, 0);
    }
    return uint32_t(elementary_root_value<numerical_binary32>(sum));
}
inline std::vector<std::array<uint64_t, 3>> float_norm_inputs(unsigned dimensions) {
    std::vector<std::array<uint64_t, 3>> rows;
    auto append = [&](std::array<uint32_t, 4> a) {
        rows.push_back({uint64_t(a[0]) | (uint64_t(a[1]) << 32), uint64_t(a[2]) | (uint64_t(a[3]) << 32),
            0xa55a0123456789efull ^ (uint64_t(a[0]) * 65537 + a[3])});
    };
    // Every finite exponent in each coordinate, with exact single-axis norms.
    for (uint32_t exponent = 0; exponent != 255; ++exponent)
        for (uint32_t fraction : {0u, 1u, 0x3fffffu, 0x400000u, 0x7ffffeu, 0x7fffffu})
            for (uint32_t sign : {0u, 0x80000000u}) for (unsigned axis = 0; axis != 4; ++axis) {
                std::array<uint32_t, 4> a{}; a[axis] = sign | (exponent << 23) | fraction; append(a);
            }
    constexpr uint32_t finite[]{0, 0x80000000, 1, 0x7fffff, 0x800000, 0x3f000000, 0x3f800000, 0xbf800000, 0x40000000, 0x7dffffff};
    for (unsigned i = 0; i != std::size(finite); ++i) for (unsigned j = 0; j != std::size(finite); ++j)
        for (unsigned k = 0; k != std::size(finite); ++k)
            append({finite[i], finite[j], finite[k], finite[(i * 3 + j * 7 + k * 9) % std::size(finite)]});
    constexpr uint32_t special[]{0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x800000, 0x3f800000, 0xbf800000,
        0x7dffffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321};
    for (unsigned x = 0; x != 4; ++x) for (unsigned y = 0; y != 4; ++y) if (x != y)
        for (auto a : special) for (auto b : special) {
            std::array<uint32_t, 4> words{0x3f800000, 0x40000000, 0x3f000000, 0}; words[x] = a; words[y] = b; append(words);
        }
    uint32_t state = 0x6a09e667;
    for (unsigned i = 0; i != 4097; ++i) {
        std::array<uint32_t, 4> a;
        for (auto& bits : a) { state = state * 1664525u + 1013904223u; bits = (state & 0x807fffffu) | (((state >> 24) % 252) << 23); }
        append(a);
    }
    for (int exponent = -149; exponent <= 122; ++exponent) {
        auto scaled = [&](unsigned n) { return uint32_t(numerical_binary32::encode({numerical_integer(n), exponent, false}, 0)); };
        append({scaled(3), scaled(4), 0, 0});
        append({scaled(1), scaled(2), scaled(2), 0});
        append({scaled(1), scaled(2), scaled(2), scaled(4)});
    }
    // Derive secondary coordinates independently at squared result midpoints.
    // The sum and square root used here retain every exact dyadic bit.
    for (uint32_t exponent = 1; exponent <= 251; exponent += 17)
        for (uint32_t fraction : {0u, 0x123456u, 0x7ffffeu}) {
            const uint32_t x = (exponent << 23) | fraction;
            const auto value = numerical_binary32::decode(x);
            auto midpoint = numerical_sum(value, numerical_binary32::decode(x + 1), 0); --midpoint.scale;
            numerical_dyadic square{midpoint.magnitude.times(midpoint.magnitude), midpoint.scale * 2, false};
            const auto remainder = numerical_sum(square, {value.magnitude.times(value.magnitude), value.scale * 2, true}, 0);
            const auto y = uint32_t(elementary_root_value<numerical_binary32>(remainder));
            for (int step = -3; step <= 3; ++step) if (int64_t(y) + step >= 0)
                for (unsigned axis = 0; axis != dimensions; ++axis) {
                    std::array<uint32_t, 4> a{}; a[axis] = x; a[(axis + 1) % dimensions] = uint32_t(int64_t(y) + step); append(a);
                }
        }
    // Unambiguous large finite and overflowing controls. Closely rounded
    // overflow boundaries remain a separate, undeclared input profile.
    for (auto a : {std::array<uint32_t, 4>{0x7f7fffff, 0x7f7fffff, 0, 0},
            {0x7f7fffff, 0, 0x7f7fffff, 0}, {0, 0, 0x7f7fffff, 0x7f7fffff},
            {0x7f000000, 0x7f000000, 0x7f000000, 0x7f000000},
            {0x7f7fffff, 1, 0, 0}, {0x7f7fffff, 0, 0, 1}}) append(a);
    return rows;
}
}
