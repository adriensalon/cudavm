#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
bool float_nan(uint32_t bits) { return (bits & 0x7fffffffu) > 0x7f800000u; }
std::array<uint32_t, 24> float_comparison_expected(uint32_t a, uint32_t b) {
    const uint32_t ax = a & 0x7fffffffu, bx = b & 0x7fffffffu;
    const bool unordered = float_nan(a) || float_nan(b);
    const bool zero_pair = (ax | bx) == 0;
    const bool equal = !unordered && (a == b || zero_pair);
    // Compare signs and magnitudes directly, without host floating arithmetic.
    const bool negative = (a >> 31) != 0;
    const bool less = !unordered && !zero_pair &&
        ((a ^ b) >> 31 ? negative : negative ? ax > bx : ax < bx);
    const bool greater = !unordered && !less && !equal;
    const bool le = less || equal, ge = greater || equal;
    const uint32_t minimum = float_nan(a) ? (float_nan(b) ? 0x7fc00000u : b)
        : float_nan(b) ? a : zero_pair ? (a | b) : less ? a : b;
    const uint32_t maximum = float_nan(a) ? (float_nan(b) ? 0x7fc00000u : b)
        : float_nan(b) ? a : zero_pair ? (a & b) : greater ? a : b;
    return {equal, !equal, less, le, greater, ge, !ge, !greater, !le, !less,
        float_nan(a), ax == 0x7f800000u, ax < 0x7f800000u, negative,
        minimum, maximum, float_nan(a) ? 0x7fc00000u : ax,
        float_nan(a) ? (0x7fc00000u | (b & 0x80000000u)) : ax | (b & 0x80000000u),
        minimum, maximum, ax == 0 || (ax >= 0x00800000u && ax < 0x7f800000u),
        ax != 0 && ax < 0x00800000u, minimum, maximum};
}
void execute_float_comparison(const loaded_program& program) {
    const auto kernel = program.function("compare_float");
    constexpr std::array<uint32_t, 22> edges{0, 0x80000000, 1, 0x80000001,
        0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x3f7fffff, 0x3f800000,
        0x3f800001, 0xbf7fffff, 0xbf800000, 0xbf800001, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321};
    constexpr unsigned cases = edges.size() * edges.size() + 256, fields = 24;
    constexpr uint32_t guard = 0x15d3c79b;
    for (unsigned threads : {1u, 13u, 32u, 33u, 96u}) {
        const unsigned blocks = (cases + threads - 1) / threads + 1, total = blocks * threads;
        std::vector<uint32_t> input(total * 2 + 2, guard), preserved(input.size());
        uint32_t state = 0x9e3779b9;
        for (unsigned i = 0; i < total; ++i) {
            if (i < edges.size() * edges.size()) {
                input[1 + i * 2] = edges[i / edges.size()];
                input[2 + i * 2] = edges[i % edges.size()];
            } else {
                state = state * 1664525u + 1013904223u; input[1 + i * 2] = state;
                state = state * 1664525u + 1013904223u; input[2 + i * 2] = state;
            }
        }
        std::vector<uint32_t> actual(total * fields + 2), expected(actual.size(), guard);
        device_buffer source(input.size() * 4), destination(actual.size() * 4);
        auto in = source.value + 4, out = destination.value + 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            for (unsigned i = 0; i < std::min(total, count); ++i) {
                const uint32_t a = input[1 + i * 2], b = input[2 + i * 2];
                const auto row = float_comparison_expected(a, b);
                const size_t base = 1 + i * fields;
                std::copy(row.begin(), row.end(), expected.begin() + base);
                for (unsigned field : {14u, 15u, 16u, 17u, 18u, 19u, 22u, 23u}) {
                    // Only the NaN class is specified for these math results.
                    if (float_nan(row[field])) {
                        require(float_nan(actual[base + field]), "float comparison expected NaN field=" + std::to_string(field));
                        if (field == 17) require((actual[base + field] >> 31) == (b >> 31), "copysign NaN sign");
                        actual[base + field] = row[field];
                    }
                    // Public fminf/fmaxf do not pin opposite-zero tie signs.
                    // The raw PTX fields below enforce their stronger rule.
                    if ((field == 14 || field == 15 || field == 22 || field == 23)
                        && ((a | b) & 0x7fffffffu) == 0 && a != b) {
                        require((actual[base + field] & 0x7fffffffu) == 0, "min/max zero class");
                        actual[base + field] = row[field];
                    }
                }
            }
            expect_words(actual, expected, "float comparison threads=" + std::to_string(threads)
                + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "float comparison source preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_float_comparison);
}
