#include "test_fixtures.hpp"
#include <algorithm>

namespace cudavm {
uint32_t half_widen_bits(uint32_t bits) {
    const auto sign = (bits & 0x8000u) << 16;
    unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    if (exponent == 31) return sign | 0x7f800000u | (fraction << 13);
    if (exponent) return sign | ((exponent + 112) << 23) | (fraction << 13);
    if (!fraction) return sign;
    int scale = -14;
    while (!(fraction & 1024)) { fraction <<= 1; --scale; }
    return sign | (unsigned(scale + 127) << 23) | ((fraction & 1023) << 13);
}
uint32_t half_narrow_bits(uint32_t bits, unsigned rounding) {
    const unsigned sign = (bits >> 16) & 0x8000, exponent = (bits >> 23) & 255, fraction = bits & 0x7fffff;
    if (exponent == 255) return sign | 0x7c00 | (fraction ? 0x200 : 0);
    if (!(bits & 0x7fffffff)) return sign;
    const int scale = int(exponent) - 127;
    auto round_up = [&](unsigned truncated, unsigned remainder, unsigned halfway) {
        if (!remainder) return false;
        if (rounding == 0) return remainder > halfway || (remainder == halfway && (truncated & 1));
        if (rounding == 1) return false;
        return rounding == 2 ? sign != 0 : sign == 0;
    };
    if (scale > 15) {
        const bool infinity = rounding == 0 || (rounding == 2 && sign) || (rounding == 3 && !sign);
        return sign | (infinity ? 0x7c00 : 0x7bff);
    }
    if (scale >= -14) {
        auto magnitude = (unsigned(scale + 15) << 10) | (fraction >> 13);
        if (round_up(magnitude, fraction & 8191, 4096)) ++magnitude;
        return sign | magnitude;
    }
    if (scale < -25) return sign | unsigned((rounding == 2 && sign) || (rounding == 3 && !sign));
    const unsigned significand = 0x800000 | fraction, shift = unsigned(-scale - 1);
    auto magnitude = significand >> shift;
    if (round_up(magnitude, significand & ((1u << shift) - 1), 1u << (shift - 1))) ++magnitude;
    return sign | magnitude;
}
bool half_conversion_nan(uint32_t bits, bool half) {
    return half ? (bits & 0x7c00) == 0x7c00 && (bits & 1023) && !(bits & 0xffff0000) :
                  (bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff);
}
std::vector<uint32_t> half_conversion_inputs() {
    std::vector<uint32_t> result;
    for (uint32_t half = 0; half < 65536; ++half) result.push_back(half);
    for (unsigned exponent = 0; exponent < 256; ++exponent)
        for (uint32_t fraction : {0u, 1u, 1023u, 4095u, 4096u, 4097u, 8191u, 8192u, 0x3fffffu, 0x7fefffu, 0x7ff000u, 0x7ff001u, 0x7ffffeu, 0x7fffffu})
            for (uint32_t sign : {0u, 0x80000000u}) result.push_back(sign | (exponent << 23) | fraction);
    for (unsigned half = 1; half < 0x7bff; ++half) {
        const auto lower = half_widen_bits(half), upper = half_widen_bits(half + 1);
        // Adjacent nonzero half values lie in one binary32 binade, possibly
        // ending at its upper boundary, so bit midpoint arithmetic is exact.
        const auto midpoint = lower + (upper - lower) / 2;
        for (auto value : {midpoint - 1, midpoint, midpoint + 1})
            for (uint32_t sign : {0u, 0x80000000u}) result.push_back(sign | value);
    }
    for (uint32_t value : {0x32ffffffu, 0x33000000u, 0x33000001u, 0x387fffffu, 0x38800000u,
                           0x477fdfffu, 0x477fe000u, 0x477fe001u, 0x477fefffu, 0x477ff000u, 0x477ff001u})
        for (uint32_t sign : {0u, 0x80000000u}) result.push_back(sign | value);
    return result;
}
void execute_half_conversion(const loaded_program& program) {
    constexpr uint32_t guard = 0xd937a15b;
    constexpr unsigned guards = 13, fields = 5, threads = 128;
    const auto values = half_conversion_inputs();
    const unsigned capacity = unsigned(values.size()), blocks = (capacity + threads - 1) / threads;
    std::vector<uint32_t> input(capacity + guards * 2, guard), preserved(input.size());
    std::copy(values.begin(), values.end(), input.begin() + guards);
    std::vector<uint32_t> initial((capacity + 3) * fields + guards * 2, guard), actual(initial.size());
    device_buffer source(input.size() * 4), output(initial.size() * 4);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    for (unsigned count : {0u, 1u, 37u, capacity}) {
        TEST_CUDA(cuMemcpyHtoD(output.value, initial.data(), initial.size() * 4));
        void* arguments[]{&to, &from, &count};
        TEST_CUDA(cuLaunchKernel(program.function("convert_half"), blocks, 1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (i < guards || i >= guards + size_t(count) * fields) {
                if (actual[i] != guard) throw std::runtime_error("half conversion inactive output and allocation guards");
                continue;
            }
            const auto record = (i - guards) / fields, field = (i - guards) % fields;
            const auto bits = values[record];
            const auto expected = field == 0 ? half_widen_bits(bits & 0xffff) : half_narrow_bits(bits, unsigned(field - 1));
            if (half_conversion_nan(expected, field != 0)) {
                require(half_conversion_nan(actual[i], field != 0), "half conversion retains NaN class"); continue;
            }
            if (actual[i] != expected) throw std::runtime_error("half conversion input=" + std::to_string(bits) + " field=" + std::to_string(field)
                + " got=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected));
        }
        TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "half conversion preserves complete source and guards");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_half_conversion);
}
