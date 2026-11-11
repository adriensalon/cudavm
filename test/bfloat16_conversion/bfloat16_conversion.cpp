#include "test_fixtures.hpp"
#include <algorithm>

namespace cudavm {
bool bfloat_conversion_nan(uint32_t bits, bool narrow) {
    if (narrow) return !(bits & 0xffff0000) && (bits & 0x7f80) == 0x7f80 && (bits & 127);
    return (bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff);
}
uint32_t bfloat_narrow_bits(uint32_t bits, unsigned rounding) {
    if (bfloat_conversion_nan(bits, false)) return 0x7fc0;
    auto retained = bits >> 16;
    const auto discarded = bits & 0xffff;
    const bool negative = (bits & 0x80000000) != 0;
    if (rounding == 0) retained += discarded > 0x8000 || (discarded == 0x8000 && (retained & 1));
    else if (rounding == 2) retained += discarded && negative;
    else if (rounding == 3) retained += discarded && !negative;
    return retained;
}
std::vector<uint32_t> bfloat_conversion_inputs() {
    std::vector<uint32_t> result;
    for (uint32_t word = 0; word < 65536; ++word) result.push_back(word);
    for (uint32_t word = 0; word < 0x7f80; ++word) {
        const auto midpoint = (word << 16) | 0x8000;
        for (auto value : {midpoint - 1, midpoint, midpoint + 1})
            for (uint32_t sign : {0u, 0x80000000u}) result.push_back(sign | value);
    }
    for (unsigned exponent = 0; exponent < 256; ++exponent)
        for (uint32_t fraction : {0u, 1u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu, 0x10000u, 0x7f7fffu, 0x7f8000u, 0x7f8001u, 0x7fffffu})
            for (uint32_t sign : {0u, 0x80000000u}) result.push_back(sign | (exponent << 23) | fraction);
    return result;
}
void execute_bfloat16_conversion(const loaded_program& program) {
    constexpr uint32_t guard = 0xa739c15b;
    constexpr unsigned guards = 11, fields = 5, threads = 128;
    const auto values = bfloat_conversion_inputs();
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
        TEST_CUDA(cuLaunchKernel(program.function("convert_bfloat16"), blocks, 1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (i < guards || i >= guards + size_t(count) * fields) {
                if (actual[i] != guard) throw std::runtime_error("bfloat16 conversion inactive outputs and guards");
                continue;
            }
            const auto record = (i - guards) / fields, field = (i - guards) % fields;
            const auto bits = values[record];
            const auto expected = field ? bfloat_narrow_bits(bits, unsigned(field - 1)) : (bits & 0xffff) << 16;
            if (bfloat_conversion_nan(expected, field != 0)) {
                require(bfloat_conversion_nan(actual[i], field != 0), "bfloat16 conversion retains NaN class"); continue;
            }
            if (actual[i] != expected) throw std::runtime_error("bfloat16 conversion input=" + std::to_string(bits) + " field=" + std::to_string(field)
                + " got=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected));
        }
        TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "bfloat16 conversion preserves complete source and guards");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_bfloat16_conversion);
}
