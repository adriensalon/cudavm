#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_numeric_representations(const loaded_program& program) {
    const auto kernel = program.function("numeric_bits");
    constexpr std::array<uint32_t, 18> float_bits{0, 0x80000000, 1, 0x80000001,
        0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321,
        0x7f812345, 0xff854321};
    constexpr std::array<uint64_t, 18> double_bits{0, 0x8000000000000000ull,
        1, 0x8000000000000001ull, 0x000fffffffffffffull, 0x800fffffffffffffull,
        0x0010000000000000ull, 0x8010000000000000ull, 0x3ff0000000000000ull,
        0xbff0000000000000ull, 0x7fefffffffffffffull, 0xffefffffffffffffull,
        0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8123456789abcull,
        0xfff8abcdef012345ull, 0x7ff0123456789abcull, 0xfff0abcdef012345ull};
    constexpr uint32_t guard = 0xd1e2f3a4;
    constexpr unsigned fields = 12, blocks = 3;
    for (unsigned threads : {1u, 13u, 33u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> input(total * 4 + 4, guard), preserved(input.size());
        // Two guard words keep every binary64 load/store naturally aligned.
        for (unsigned i = 0; i < total; ++i) {
            const auto real = double_bits[i % double_bits.size()];
            input[2 + i * 4] = float_bits[i % float_bits.size()];
            input[3 + i * 4] = float_bits[(i * 5 + 7) % float_bits.size()];
            input[4 + i * 4] = uint32_t(real); input[5 + i * 4] = uint32_t(real >> 32);
        }
        std::vector<uint32_t> expected(total * fields + 4, guard), actual(expected.size());
        device_buffer source(input.size() * 4), destination(expected.size() * 4);
        auto in = source.value + 8, out = destination.value + 8;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            for (unsigned i = 0; i < std::min(total, count); ++i) {
                const auto a = input[2 + i * 4], b = input[3 + i * 4];
                const auto lo = input[4 + i * 4], hi = input[5 + i * 4];
                const std::array<uint32_t, fields> row{a, b, a, b, lo, hi, lo, hi, lo, hi, b, a};
                std::copy(row.begin(), row.end(), expected.begin() + 2 + i * fields);
            }
            expect_words(actual, expected, "numeric representations threads=" + std::to_string(threads)
                + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "numeric representation source preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_numeric_representations);
}
