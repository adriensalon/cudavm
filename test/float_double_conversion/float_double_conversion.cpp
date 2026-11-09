#include "test_fixtures.hpp"

namespace cudavm {
uint64_t float_double_widen_bits(uint32_t bits) {
    if (numerical_binary32::is_nan(bits)) return numerical_binary64::nan;
    if (numerical_binary32::is_infinite(bits))
        return (uint64_t(bits >> 31) << 63) | numerical_binary64::infinity;
    return numerical_binary64::encode(numerical_binary32::decode(bits), 0);
}
std::array<uint64_t, 5> float_double_narrow_bits(const std::array<uint64_t, 3>& input) {
    const auto bits = input[0];
    std::array<uint64_t, 5> row{};
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
        if (numerical_binary64::is_nan(bits)) row[rounding] = numerical_binary32::nan;
        else if (numerical_binary64::is_infinite(bits))
            row[rounding] = ((bits >> 32) & numerical_binary32::sign) | numerical_binary32::infinity;
        else row[rounding] = numerical_binary32::encode(numerical_binary64::decode(bits), rounding);
    }
    row[4] = row[0];
    return row;
}
std::vector<std::array<uint64_t, 3>> float_double_widen_inputs() {
    std::vector<std::array<uint64_t, 3>> result;
    for (uint32_t exponent = 0; exponent < 256; ++exponent)
        for (uint32_t fraction : {0u, 1u, 2u, 3u, 0x3fffffu, 0x400000u, 0x400001u, 0x7ffffeu, 0x7fffffu})
            for (uint32_t sign : {0u, 0x80000000u}) result.push_back({sign | (exponent << 23) | fraction, 0, 0});
    uint32_t state = 0x719c3a5b;
    for (unsigned i = 0; i < 8193; ++i) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        result.push_back({state, 0, 0});
    }
    return result;
}
std::vector<std::array<uint64_t, 3>> float_double_narrow_inputs() {
    std::vector<std::array<uint64_t, 3>> result;
    auto append = [&](uint64_t bits) {
        for (uint64_t sign : {uint64_t(0), numerical_binary64::sign}) result.push_back({bits | sign, 0, 0});
    };
    for (uint64_t exponent = 0; exponent < 2048; ++exponent)
        for (uint64_t fraction : {0ull, 1ull, 2ull, 0x7ffffffffffffull, 0x8000000000000ull, 0xfffffffffffffull})
            append((exponent << 52) | fraction);
    // Exact adjacent binary32 midpoints, including zero, the normal/subnormal
    // boundary, significand carry and overflow to the conceptual value 2^128.
    for (uint32_t exponent = 0; exponent < 255; ++exponent)
        for (uint32_t fraction : {0u, 1u, 2u, 0x3fffffu, 0x400000u, 0x7ffffeu, 0x7fffffu}) {
            const auto lower = (exponent << 23) | fraction;
            auto upper_value = lower == 0x7f7fffffu ? numerical_dyadic{numerical_integer(1), 128, false}
                : numerical_binary32::decode(lower + 1);
            auto midpoint = numerical_sum(numerical_binary32::decode(lower), upper_value, 0);
            --midpoint.scale;
            const auto bits = numerical_binary64::encode(midpoint, 0);
            for (auto value : {bits - 1, bits, bits + 1}) append(value);
            append(float_double_widen_bits(lower));
        }
    uint64_t state = 0xc7e153a9b2714d65ull;
    for (unsigned i = 0; i < 8193; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        result.push_back({state, 0, 0});
    }
    return result;
}
void execute_float_double_conversion(const loaded_program& program) {
    execute_numerical_rows<uint64_t, 1>(program, "float_double_widen", float_double_widen_inputs(), [](const auto& input) {
        return std::array<uint64_t, 1>{float_double_widen_bits(uint32_t(input[0]))};
    }, [](uint64_t actual, uint64_t expected) {
        return numerical_binary64::is_nan(expected) ? numerical_binary64::is_nan(actual) : actual == expected;
    });
    execute_numerical_rows<uint64_t, 5>(program, "float_double_narrow", float_double_narrow_inputs(), float_double_narrow_bits,
        [](uint64_t actual, uint64_t expected) {
            return numerical_binary32::is_nan(expected) ? actual <= UINT32_MAX && numerical_binary32::is_nan(actual) : actual == expected;
        });
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_float_double_conversion);
}
