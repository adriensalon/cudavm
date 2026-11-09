#pragma once
#include "../test_framework.hpp"

namespace cudavm {
template<class Format> uint64_t special_neighbor(uint64_t x, uint64_t y) {
    if (Format::is_nan(x) || Format::is_nan(y)) return Format::nan;
    // A signed rank collapses the two zeros into one point in the numerical
    // lattice. Adjacency is one rank; encoding restores the source zero sign.
    auto rank = [](uint64_t bits) {
        const auto magnitude = bits & (Format::sign - 1);
        return bits & Format::sign ? Format::sign - magnitude : Format::sign + magnitude;
    };
    const auto a = rank(x), b = rank(y);
    if (a == b) return y;
    const auto next = a < b ? a + 1 : a - 1;
    if (next == Format::sign) return x & Format::sign;
    return next < Format::sign ? Format::sign | (Format::sign - next) : next - Format::sign;
}
template<class Format> std::array<uint64_t, 8> special_neighbor_oracle(const std::array<uint64_t, 3>& row) {
    const auto x = row[0], y = row[1], first = special_neighbor<Format>(x, y);
    const auto magnitude = first & (Format::sign - 1);
    constexpr auto minimum_normal = Format::infinity & (0 - Format::infinity);
    const uint64_t classification = Format::is_nan(first) ? 1 : unsigned(Format::is_infinite(first)) * 2
        | unsigned(!magnitude) * 4 | unsigned(magnitude && magnitude < minimum_normal) * 8
        | unsigned(magnitude >= minimum_normal && magnitude < Format::infinity) * 16 | unsigned(bool(first & Format::sign)) * 32;
    return {first, special_neighbor<Format>(y, x), special_neighbor<Format>(x, 0), special_neighbor<Format>(x, Format::sign),
        special_neighbor<Format>(x, Format::infinity), special_neighbor<Format>(x, Format::sign | Format::infinity),
        special_neighbor<Format>(first, x), classification};
}
template<unsigned Fraction, unsigned Exponent> std::vector<std::array<uint64_t, 3>> special_neighbor_inputs() {
    using format = numerical_format<Fraction, Exponent>;
    constexpr auto sign = format::sign, infinity = format::infinity, fraction = (uint64_t(1) << Fraction) - 1;
    constexpr uint64_t edges[]{0, sign, 1, sign | 1, 2, sign | 2, fraction - 1, fraction, sign | fraction,
        fraction + 1, fraction + 2, sign | (fraction + 1), sign | (fraction + 2),
        uint64_t(format::bias) << Fraction, (uint64_t(format::bias) << Fraction) - 1, (uint64_t(format::bias) << Fraction) + 1,
        infinity - 1, sign | (infinity - 1), infinity, sign | infinity, format::nan, sign | format::nan, infinity | 1, sign | infinity | 1};
    std::vector<std::array<uint64_t, 3>> rows;
    auto append = [&](uint64_t x, uint64_t y) { rows.push_back({x, y, std::rotl(x, 19) ^ y ^ 0xa55a5917c3d2e4f6ull}); };
    for (auto x : edges) for (auto y : edges) append(x, y);
    for (uint64_t exponent = 0; exponent != (uint64_t(1) << Exponent); ++exponent)
        for (uint64_t mantissa : {uint64_t(0), uint64_t(1), fraction >> 1, (fraction >> 1) + 1, fraction - 1, fraction})
            for (auto negative : {uint64_t(0), sign}) {
                const auto x = negative | (exponent << Fraction) | mantissa;
                for (auto y : {uint64_t(0), sign, infinity, sign | infinity, x, x ^ sign}) append(x, y);
            }
    uint64_t state = 0xbb67ae8584caa73bull;
    auto next = [&] { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state & (sign | (sign - 1)); };
    for (unsigned i = 0; i < 4096; ++i) { const auto x = next(), y = next(); append(x, y); }
    return rows;
}
}
