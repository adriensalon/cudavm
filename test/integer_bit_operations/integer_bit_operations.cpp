#include "fixtures.hpp"
#include <algorithm>
#include <limits>

namespace cudavm {
struct bit_observation {
    uint32_t population = 0, leading, first = 0;
    uint64_t reversed = 0;
    bit_observation(uint64_t value, unsigned width) : leading(width) {
        for (unsigned bit = 0; bit < width; ++bit) {
            if (((value >> bit) & 1) == 0) continue;
            ++population;
            leading = width - bit - 1;
            if (!first) first = bit + 1;
            reversed |= uint64_t(1) << (width - bit - 1);
        }
    }
};

void execute_integer_bit_operations(const loaded_program& program) {
    const auto kernel = program.function("count_reverse_bits");
    constexpr uint32_t guard = 0xdeadbeef;
    constexpr unsigned blocks = 3, fields = 9;
    std::vector<uint64_t> corpus{0, 1, std::numeric_limits<uint64_t>::max(),
        0x8000000000000000ull, 0xffffffffull, 0xffffffff00000000ull,
        0x0123456789abcdefull, 0xfedcba9876543210ull,
        0xaaaaaaaaaaaaaaaaull, 0x5555555555555555ull};
    for (unsigned bit = 0; bit < 64; ++bit) {
        corpus.push_back(uint64_t(1) << bit);
        corpus.push_back(~(uint64_t(1) << bit));
    }
    for (unsigned threads : {1u, 13u, 33u, 64u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> input(total * 2 + 4, guard), preserved(input.size());
        std::vector<uint32_t> expected(total * fields + 4, guard), actual(expected.size());
        for (unsigned i = 0; i < total; ++i) {
            const auto value = corpus[i % corpus.size()];
            input[i * 2 + 2] = uint32_t(value);
            input[i * 2 + 3] = uint32_t(value >> 32);
        }
        device_buffer source(input.size() * sizeof(uint32_t)), destination(expected.size() * sizeof(uint32_t));
        auto in = source.value + 2 * sizeof(uint32_t), out = destination.value + sizeof(uint32_t);
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * sizeof(uint32_t)));
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * sizeof(uint32_t)));
            void* args[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * sizeof(uint32_t)));
            for (unsigned i = 0; i < std::min(total, count); ++i) {
                const auto value = corpus[i % corpus.size()];
                const bit_observation narrow(uint32_t(value), 32), wide(value, 64);
                const std::array<uint32_t, fields> values{narrow.population, narrow.leading,
                    uint32_t(narrow.reversed), narrow.first, wide.population, wide.leading,
                    uint32_t(wide.reversed), uint32_t(wide.reversed >> 32), wide.first};
                std::copy(values.begin(), values.end(), expected.begin() + 1 + i * fields);
            }
            expect_words(actual, expected, "bit operations threads=" + std::to_string(threads)
                + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * sizeof(uint32_t)));
        expect_words(preserved, input, "bit-operation input preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_integer_bit_operations);
}
