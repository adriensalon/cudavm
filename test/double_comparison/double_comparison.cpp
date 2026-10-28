#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
constexpr uint64_t double_sign = 0x8000000000000000ull, double_magnitude = 0x7fffffffffffffffull;
constexpr uint64_t double_infinity = 0x7ff0000000000000ull, double_quiet_nan = 0x7ff8000000000000ull;
bool double_nan(uint64_t bits) { return (bits & double_magnitude) > double_infinity; }
bool double_less(uint64_t a, uint64_t b) {
    if (double_nan(a) || double_nan(b) || ((a | b) & double_magnitude) == 0) return false;
    if ((a ^ b) & double_sign) return (a & double_sign) != 0;
    return a & double_sign ? (a & double_magnitude) > (b & double_magnitude)
        : (a & double_magnitude) < (b & double_magnitude);
}
uint64_t double_extreme(uint64_t a, uint64_t b, bool maximum) {
    if (double_nan(a)) return double_nan(b) ? double_quiet_nan : b;
    if (double_nan(b)) return a;
    if (((a | b) & double_magnitude) == 0) return maximum ? a & b : a | b;
    return (maximum ? double_less(b, a) : double_less(a, b)) ? a : b;
}
uint64_t promoted_float_bits(uint32_t bits) {
    const uint64_t sign = uint64_t(bits >> 31) << 63;
    unsigned exponent = (bits >> 23) & 255, fraction = bits & 0x7fffff;
    if (exponent == 255) return sign | double_infinity | (uint64_t(fraction) << 29);
    if (exponent) return sign | (uint64_t(exponent + 896) << 52) | (uint64_t(fraction) << 29);
    if (!fraction) return sign;
    unsigned shift = 0;
    while ((fraction & 0x800000) == 0) { fraction <<= 1; ++shift; }
    return sign | (uint64_t(897 - shift) << 52) | (uint64_t(fraction & 0x7fffff) << 29);
}
uint64_t comparison_word_pair(std::span<const uint32_t> words, size_t at) {
    return words[at] | (uint64_t(words[at + 1]) << 32);
}
std::array<uint32_t, 44> double_comparison_expected(uint64_t a, uint64_t b, uint32_t c) {
    std::array<uint32_t, 44> row{};
    const uint64_t ax = a & double_magnitude, cx = promoted_float_bits(c);
    const bool ordered = !double_nan(a) && !double_nan(b);
    const bool equal = ordered && (a == b || ((a | b) & double_magnitude) == 0);
    const bool less = double_less(a, b), greater = double_less(b, a);
    const bool le = less || equal, ge = greater || equal;
    const std::array<uint32_t, 14> predicates{equal, !equal, less, le, greater, ge,
        !ge, !greater, !le, !less, double_nan(a), ax == double_infinity, ax < double_infinity, (a >> 63) != 0};
    std::copy(predicates.begin(), predicates.end(), row.begin());
    auto put = [&](unsigned at, uint64_t bits) { row[at] = uint32_t(bits); row[at + 1] = uint32_t(bits >> 32); };
    const uint64_t minimum = double_extreme(a, b, false), maximum = double_extreme(a, b, true);
    put(14, minimum); put(16, maximum);
    put(18, double_nan(a) ? double_quiet_nan : ax);
    put(20, (double_nan(a) ? double_quiet_nan : ax) | (b & double_sign));
    put(22, minimum); put(24, maximum); put(26, minimum); put(28, maximum);
    row[30] = ax == 0 || (ax >= 0x0010000000000000ull && ax < double_infinity);
    row[31] = ax != 0 && ax < 0x0010000000000000ull;
    put(32, double_extreme(a, cx, false)); put(34, double_extreme(cx, a, false));
    put(36, double_extreme(a, cx, true)); put(38, double_extreme(cx, a, true));
    put(40, double_nan(a) ? a : ax); put(42, ax | (b & double_sign));
    return row;
}
void execute_double_comparison(const loaded_program& program) {
    const auto kernel = program.function("compare_double");
    constexpr uint64_t edges[]{0, 0x8000000000000000ull, 1, 0x8000000000000001ull,
        0x000fffffffffffffull, 0x800fffffffffffffull, 0x0010000000000000ull, 0x8010000000000000ull,
        0x3fefffffffffffffull, 0x3ff0000000000000ull, 0x3ff0000000000001ull,
        0xbfefffffffffffffull, 0xbff0000000000000ull, 0xbff0000000000001ull,
        0x47efffffe0000000ull, 0x47f0000000000000ull, 0x3690000000000000ull, 0x36a0000000000000ull,
        0x7fefffffffffffffull, 0xffefffffffffffffull, double_infinity, double_infinity | double_sign,
        0x7ff8123456789abcull, 0xfff8abcdef012345ull, 0x7ff0000000000001ull, 0xfff0000000000001ull};
    constexpr uint32_t floats[]{0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x807fffff,
        0x800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x7f800001, 0xff800001};
    constexpr unsigned edge_count = sizeof(edges) / sizeof(edges[0]), float_count = sizeof(floats) / sizeof(floats[0]);
    constexpr unsigned cases = edge_count * edge_count + 256, fields = 44;
    constexpr uint32_t guard = 0x23c579ab;
    for (unsigned threads : {1u, 13u, 32u, 33u, 96u}) {
        const unsigned blocks = (cases + threads - 1) / threads + 1, total = blocks * threads;
        std::vector<uint32_t> input(total * 6 + 4, guard), preserved(input.size());
        uint32_t state = 0xabcdef13;
        for (unsigned i = 0; i < total; ++i) {
            uint64_t a, b;
            if (i < edge_count * edge_count) { a = edges[i / edge_count]; b = edges[i % edge_count]; }
            else {
                state = state * 1664525u + 1013904223u; a = uint64_t(state) << 32;
                state = state * 1664525u + 1013904223u; a |= state;
                state = state * 1664525u + 1013904223u; b = uint64_t(state) << 32;
                state = state * 1664525u + 1013904223u; b |= state;
            }
            input[2 + i * 6] = uint32_t(a); input[3 + i * 6] = uint32_t(a >> 32);
            input[4 + i * 6] = uint32_t(b); input[5 + i * 6] = uint32_t(b >> 32);
            input[6 + i * 6] = floats[(i * 5 + 7) % float_count];
        }
        std::vector<uint32_t> actual(total * fields + 4), expected(actual.size(), guard);
        device_buffer source(input.size() * 4), destination(actual.size() * 4);
        auto in = source.value + 8, out = destination.value + 8;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            for (unsigned i = 0; i < std::min(total, count); ++i) {
                const uint64_t a = comparison_word_pair(input, 2 + i * 6), b = comparison_word_pair(input, 4 + i * 6);
                const uint32_t c = input[6 + i * 6];
                const uint64_t cx = promoted_float_bits(c);
                const auto row = double_comparison_expected(a, b, c);
                const size_t base = 2 + i * fields;
                std::copy(row.begin(), row.end(), expected.begin() + base);
                auto normalize = [&](unsigned field) { actual[base + field] = row[field]; actual[base + field + 1] = row[field + 1]; };
                for (unsigned field : {14u, 16u, 18u, 20u, 22u, 24u, 26u, 28u, 32u, 34u, 36u, 38u}) {
                    const uint64_t result = comparison_word_pair(actual, base + field);
                    if (double_nan(comparison_word_pair(row, field))) {
                        require(double_nan(result), "double comparison expected NaN field=" + std::to_string(field));
                        if (field == 20) require((result & double_sign) == (b & double_sign), "double copysign NaN sign");
                        normalize(field);
                    }
                    const uint64_t second = field >= 32 ? cx : b;
                    if ((field == 14 || field == 16 || field == 26 || field == 28 || field >= 32)
                        && ((a | second) & double_magnitude) == 0 && a != second) {
                        require((result & double_magnitude) == 0, "double min/max zero class"); normalize(field);
                    }
                }
                if (double_nan(a)) {
                    const uint64_t result = comparison_word_pair(actual, base + 40);
                    // SM75 abs.f64 quiets signaling NaNs while retaining their
                    // payload. Accept that quiet-bit transition and either
                    // documented sign behavior without losing payload checks.
                    const uint64_t quiet = 0x0008000000000000ull;
                    require(result == a || result == (a & double_magnitude)
                        || result == (a | quiet) || result == ((a & double_magnitude) | quiet), "raw double abs NaN payload row="
                        + std::to_string(i) + " input=" + std::to_string(a) + " result=" + std::to_string(result)); normalize(40);
                }
            }
            expect_words(actual, expected, "double comparison threads=" + std::to_string(threads) + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "double comparison source preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_double_comparison);
}
