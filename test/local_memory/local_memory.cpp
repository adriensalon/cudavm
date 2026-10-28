#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_local_memory(const loaded_program& program) {
    const auto kernel = program.function("local_arrays");
    constexpr unsigned blocks = 3, fields = 5;
    constexpr uint32_t guard = 0xe3f2a190;
    for (unsigned threads : {1u, 13u, 33u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> input(total + 2, guard), preserved(input.size());
        for (unsigned i = 0; i < total; ++i) input[i + 1] = i * 0x7654321u + 0xfffffffdu;
        std::vector<uint32_t> expected(total * fields + 2, guard), actual(expected.size());
        device_buffer source(input.size() * 4), destination(expected.size() * 4);
        auto in = source.value + 4, out = destination.value + 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned rounds : {0u, 1u, 5u, 19u})
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in, &count, &rounds};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            for (unsigned i = 0; i < std::min(total, count); ++i) {
                const uint32_t seed = input[i + 1];
                std::array<uint32_t, 17> values{};
                for (unsigned slot = 0; slot < values.size(); ++slot)
                    values[slot] = (seed + slot * 0x10203u) ^ (i * 17u + slot);
                for (unsigned round = 0; round < rounds + i % 4; ++round) {
                    const unsigned slot = (seed + round * 5u) % 17;
                    values[slot] ^= values[(slot + 7) % 17] + round;
                }
                uint32_t sum = 0, hash = 0;
                for (auto value : values) { sum += value; hash = (hash * 33u) ^ value; }
                const std::array<uint32_t, fields> row{sum, hash, values[seed % 17],
                    0x13579bdfu ^ i, 0xeca86420u ^ seed};
                std::copy(row.begin(), row.end(), expected.begin() + 1 + i * fields);
            }
            expect_words(actual, expected, "local memory threads=" + std::to_string(threads)
                + " rounds=" + std::to_string(rounds) + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "local-memory source preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_local_memory);
}
