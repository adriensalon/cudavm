#pragma once
#include "../test_framework.hpp"

namespace cudavm {
inline uint32_t mode_flush(uint32_t bits) {
    return (bits & 0x7fffffffu) < 0x00800000u ? bits & 0x80000000u : bits;
}
inline uint32_t mode_arithmetic(uint32_t a, uint32_t b, uint32_t c, unsigned operation, unsigned rounding, bool flush) {
    auto result = uint32_t(!operation ? numerical_binary32::add(a, b, rounding)
        : operation == 1 ? numerical_binary32::multiply(a, b, rounding) : numerical_binary32::fused(a, b, c, rounding));
    // Multiplication detects tininess before rounding. FMA's narrow tininess
    // boundary has its own open profile, excluded below using exact magnitudes.
    if (flush && operation == 1 && (result & 0x7fffffffu) == 0x00800000u) {
        const auto left = numerical_binary32::decode(a), right = numerical_binary32::decode(b);
        const numerical_dyadic exact{left.magnitude.times(right.magnitude), left.scale + right.scale, left.negative != right.negative};
        if (exact.magnitude.width() && int(exact.magnitude.width()) - 1 + exact.scale < -126)
            return exact.negative ? 0x80000000u : 0u;
    }
    return flush ? mode_flush(result) : result;
}
inline std::array<uint32_t, 8> floating_mode_oracle(const std::array<uint32_t, 3>& input, unsigned operation) {
    std::array<uint32_t, 8> output{};
    for (unsigned flush = 0; flush != 2; ++flush) for (unsigned rounding = 0; rounding != 4; ++rounding) {
        const auto a = flush ? mode_flush(input[0]) : input[0];
        const auto b = flush ? mode_flush(input[1]) : input[1];
        const auto c = flush ? mode_flush(input[2]) : input[2];
        uint32_t result;
        if (operation < 3) result = mode_arithmetic(a, b, c, operation, rounding, flush);
        else {
            const auto product = mode_arithmetic(a, b, 0, 1, rounding, flush);
            result = mode_arithmetic(product, c, 0, 0, rounding, flush);
        }
        output[flush * 4 + rounding] = flush ? mode_flush(result) : result;
    }
    return output;
}
inline std::vector<std::array<uint32_t, 3>> floating_mode_inputs(unsigned operation = 0) {
    constexpr uint32_t edges[]{0, 0x80000000, 1, 0x80000001, 2, 0x80000002,
        0x003fffff, 0x00400000, 0x00400001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x00800001, 0x01000000, 0x33000000,
        0x3f000000, 0xbf000000, 0x3f7fffff, 0x3f800000, 0x3f800001,
        0xbf800000, 0x40000000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
        0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345};
    std::vector<std::array<uint32_t, 3>> rows;
    for (auto a : edges) for (auto b : edges) for (auto c : edges) rows.push_back({a, b, c});
    auto signed_triple = [&](uint32_t a, uint32_t b, uint32_t c) {
        for (auto x : {0u, 0x80000000u}) for (auto y : {0u, 0x80000000u}) for (auto z : {0u, 0x80000000u})
            rows.push_back({a ^ x, b ^ y, c ^ z});
    };
    for (unsigned bit = 0; bit < 23; ++bit) for (int delta : {-1, 0, 1}) {
        const auto tiny = uint32_t(int(1u << bit) + delta);
        signed_triple(tiny, 0x3f800000, 0);
        signed_triple(0x00800000, 0x3f000000, tiny);
        signed_triple(0x00800000, 0x3f7fffff, tiny);
        signed_triple(0x00800000, tiny | 0x80000000, 0);
        signed_triple(0x00800000 + tiny, 0x80800000, 0);
    }
    // Cancellation exposes the rounded intermediate in separate multiply/add.
    for (uint32_t offset : {1u, 2u, 3u, 127u, 1023u, 4095u})
        signed_triple(0x3f800000 + offset, 0x3f800000 - 2 * offset, 0xbf800000);
    uint32_t state = 0x6a09e667;
    auto next = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (unsigned i = 0; i != 4096; ++i) {
        const auto a = next(), b = next(), c = next(); rows.push_back({a, b, c});
        if (i % 8 == 0) rows.push_back({a, a ^ 0x80000000u, c});
    }
    if (operation == 2) {
        // The native FMA tininess boundary does not share multiplication's
        // simple pre-rounding rule. Keep this narrow interval explicitly open.
        std::erase_if(rows, [](const auto& row) {
            const auto a = mode_flush(row[0]), b = mode_flush(row[1]), c = mode_flush(row[2]);
            if ((a & 0x7fffffffu) >= 0x7f800000u || (b & 0x7fffffffu) >= 0x7f800000u || (c & 0x7fffffffu) >= 0x7f800000u) return false;
            const auto left = numerical_binary32::decode(a), right = numerical_binary32::decode(b);
            const auto exact = numerical_sum({left.magnitude.times(right.magnitude), left.scale + right.scale,
                left.negative != right.negative}, numerical_binary32::decode(c), 0);
            return elementary_compare(exact, numerical_binary32::decode(0x007fffff)) > 0
                && elementary_compare(exact, numerical_binary32::decode(0x00800000)) < 0;
        });
    }
    return rows;
}
}
