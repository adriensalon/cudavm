#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_block_synchronization(const loaded_program& program) {
    const auto kernel = program.function("block_rounds");
    constexpr uint32_t guard = 0xc5d4e3f2;
    constexpr unsigned blocks = 3, fields = 5;
    for (unsigned threads : {1u, 13u, 33u, 64u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> input(total + 2, guard), preserved(input.size());
        std::vector<uint32_t> expected(total * fields + 2, guard), actual(expected.size());
        std::vector<uint32_t> scratch_expected(total + 2, guard), scratch_actual(scratch_expected.size());
        for (unsigned i = 0; i < total; ++i) input[i + 1] = i * 0x1020305u + 0x80000001u;
        device_buffer source(input.size() * 4), destination(expected.size() * 4), scratch(scratch_expected.size() * 4);
        auto in = source.value + 4, out = destination.value + 4, temporary = scratch.value + 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned rounds : {0u, 1u, 4u})
        for (unsigned pattern = 0; pattern < 4; ++pattern)
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            std::fill(scratch_expected.begin(), scratch_expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            CONFORMANCE_CUDA(cuMemcpyHtoD(scratch.value, scratch_expected.data(), scratch_expected.size() * 4));
            void* args[]{&out, &in, &temporary, &count, &rounds, &pattern};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            CONFORMANCE_CUDA(cuMemcpyDtoH(scratch_actual.data(), scratch.value, scratch_actual.size() * 4));
            for (unsigned block = 0; block < blocks; ++block) {
                std::vector<std::array<uint32_t, fields>> sums(threads);
                for (unsigned round = 0; round < rounds; ++round) {
                    unsigned truth = 0;
                    for (unsigned lane = 0; lane < threads; ++lane) {
                        const unsigned i = block * threads + lane;
                        const bool predicate = pattern == 0 ? false : pattern == 1 ? true
                            : pattern == 2 ? lane == round % threads : (input[i + 1] + round) % 4 != 0;
                        truth += unsigned(predicate);
                        scratch_expected[i + 1] = input[i + 1] ^ (round * 17u + i);
                    }
                    for (unsigned lane = 0; lane < threads; ++lane) {
                        const unsigned peer = block * threads + (lane + round + 1) % threads;
                        sums[lane][0] += truth;
                        sums[lane][1] += unsigned(truth == threads);
                        sums[lane][2] += unsigned(truth != 0);
                        sums[lane][3] += scratch_expected[peer + 1];
                        sums[lane][4] += scratch_expected[peer + 1] + 0x12345u;
                    }
                }
                for (unsigned lane = 0; lane < threads; ++lane) {
                    const unsigned i = block * threads + lane;
                    if (i < count) std::copy(sums[lane].begin(), sums[lane].end(), expected.begin() + 1 + i * fields);
                    if (rounds) scratch_expected[i + 1] += 0x12345u;
                }
            }
            const auto context = " threads=" + std::to_string(threads) + " rounds=" + std::to_string(rounds)
                + " pattern=" + std::to_string(pattern) + " count=" + std::to_string(count);
            expect_words(actual, expected, "block synchronization" + context);
            expect_words(scratch_actual, scratch_expected, "block publication" + context);
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "block-synchronization source preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_block_synchronization);
}
