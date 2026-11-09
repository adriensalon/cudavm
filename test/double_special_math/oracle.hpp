#pragma once
#include "../test_framework.hpp"

namespace cudavm {
inline uint64_t double_norm_oracle(const std::array<uint64_t, 4>& words, unsigned dimensions) {
    bool nan = false; unsigned nonzero = 0; uint64_t single = 0;
    for (unsigned i = 0; i != dimensions; ++i) {
        if (numerical_binary64::is_infinite(words[i])) return numerical_binary64::infinity;
        nan |= numerical_binary64::is_nan(words[i]);
        if (!numerical_binary64::is_zero(words[i])) { ++nonzero; single = words[i] & (numerical_binary64::sign - 1); }
    }
    if (nan) return numerical_binary64::nan;
    if (nonzero <= 1) return single;
    numerical_dyadic sum;
    for (unsigned i = 0; i != dimensions; ++i) {
        const auto value = numerical_binary64::decode(words[i]);
        sum = numerical_sum(sum, {value.magnitude.times(value.magnitude), value.scale * 2, false}, 0);
    }
    return elementary_root_value<numerical_binary64>(sum);
}
inline std::vector<std::array<uint64_t, 4>> double_norm_inputs(unsigned dimensions) {
    std::vector<std::array<uint64_t, 4>> rows;
    auto append = [&](std::array<uint64_t, 4> a) { rows.push_back(a); };
    for (uint64_t exponent = 0; exponent != 2047; ++exponent)
        for (uint64_t fraction : {0ull, 1ull, 0xfffffffffffffull})
            for (uint64_t sign : {0ull, 0x8000000000000000ull}) for (unsigned axis = 0; axis != 4; ++axis) {
                std::array<uint64_t, 4> a{}; a[axis] = sign | (exponent << 52) | fraction; append(a);
            }
    constexpr uint64_t finite[]{0, 0x8000000000000000, 1, 0xfffffffffffff, 0x10000000000000,
        0x3fe0000000000000, 0x3ff0000000000000, 0xbff0000000000000, 0x4000000000000000, 0x7fbfffffffffffff};
    for (unsigned i = 0; i != std::size(finite); ++i) for (unsigned j = 0; j != std::size(finite); ++j)
        for (unsigned k = 0; k != std::size(finite); ++k)
            append({finite[i], finite[j], finite[k], finite[(i * 3 + j * 7 + k * 9) % std::size(finite)]});
    constexpr uint64_t special[]{0, 0x8000000000000000, 1, 0x8000000000000001, 0xfffffffffffff, 0x10000000000000,
        0x3ff0000000000000, 0xbff0000000000000, 0x7fbfffffffffffff, 0x7ff0000000000000, 0xfff0000000000000,
        0x7ff8123456789abc, 0xfff8abcdef012345, 0x7ff0123456789abc, 0xfff0543210abcdef};
    for (unsigned x = 0; x != 4; ++x) for (unsigned y = 0; y != 4; ++y) if (x != y)
        for (auto a : special) for (auto b : special) {
            std::array<uint64_t, 4> words{0x3ff0000000000000, 0x4000000000000000, 0x3fe0000000000000, 0};
            words[x] = a; words[y] = b; append(words);
        }
    uint64_t state = 0xbb67ae8584caa73bull;
    for (unsigned i = 0; i != 4097; ++i) {
        std::array<uint64_t, 4> a;
        for (auto& bits : a) { state = state * 6364136223846793005ull + 1442695040888963407ull; bits = (state & 0x800fffffffffffffull) | (((state >> 53) % 2044) << 52); }
        append(a);
    }
    for (int exponent = -1074; exponent <= 1018; exponent += 17) {
        auto scaled = [&](unsigned n) { return numerical_binary64::encode({numerical_integer(n), exponent, false}, 0); };
        append({scaled(3), scaled(4), 0, 0}); append({scaled(1), scaled(2), scaled(2), 0});
        append({scaled(1), scaled(2), scaled(2), scaled(4)});
    }
    for (uint64_t exponent = 1; exponent <= 2043; exponent += 127)
        for (uint64_t fraction : {0ull, 0x123456789abull, 0xffffffffffffeull}) {
            const auto x = (exponent << 52) | fraction;
            const auto value = numerical_binary64::decode(x);
            auto midpoint = numerical_sum(value, numerical_binary64::decode(x + 1), 0); --midpoint.scale;
            numerical_dyadic square{midpoint.magnitude.times(midpoint.magnitude), midpoint.scale * 2, false};
            const auto remainder = numerical_sum(square, {value.magnitude.times(value.magnitude), value.scale * 2, true}, 0);
            const auto y = elementary_root_value<numerical_binary64>(remainder);
            for (int step = -3; step <= 3; ++step) if (step >= 0 || y >= uint64_t(-step))
                for (unsigned axis = 0; axis != dimensions; ++axis) {
                    std::array<uint64_t, 4> a{}; a[axis] = x; a[(axis + 1) % dimensions] = step < 0 ? y - unsigned(-step) : y + unsigned(step); append(a);
                }
        }
    for (auto a : {std::array<uint64_t, 4>{0x7fefffffffffffff, 0x7fefffffffffffff, 0, 0},
            {0x7fefffffffffffff, 0, 0x7fefffffffffffff, 0}, {0, 0, 0x7fefffffffffffff, 0x7fefffffffffffff},
            {0x7fe0000000000000, 0x7fe0000000000000, 0x7fe0000000000000, 0x7fe0000000000000},
            {0x7fefffffffffffff, 1, 0, 0}, {0x7fefffffffffffff, 0, 0, 1}}) append(a);
    return rows;
}
}
