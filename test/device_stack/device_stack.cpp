#include "test_fixtures.hpp"
#include <algorithm>

namespace cudavm {
uint32_t stack_frame_oracle(uint32_t seed, unsigned rounds) {
    std::array<uint32_t, 5> state;
    for (unsigned j = 0; j < state.size(); ++j) state[j] = seed + j * 0x1020304u;
    for (unsigned round = 0; round < rounds; ++round) {
        const uint32_t value = seed + round * 7u;
        const uint32_t leaf = (value * 3u + (value ^ 0x89abcdefu)) ^ (value + 0x7654321u);
        state[(seed + round) % 5] ^= leaf + round;
    }
    uint32_t hash = 0;
    for (auto value : state) hash = hash * 33u + value;
    return hash;
}
void execute_device_stack(const loaded_program& program) {
    const auto function = program.function("frame_results");
    constexpr unsigned blocks = 3, fields = 5, guards = 4;
    constexpr uint32_t guard = 0xa1b2c3d4;
    for (unsigned lanes : {1u, 13u, 33u, 64u}) {
        const unsigned total = blocks * lanes;
        std::vector<uint32_t> input(total + guards * 2, guard), preserved(input.size());
        std::vector<uint32_t> expected(total * fields + guards * 2 + fields, guard), actual(expected.size());
        device_buffer source(input.size() * 4), output(expected.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4;
        for (unsigned phase = 0; phase < 2; ++phase) {
            for (unsigned i = 0; i < total; ++i) input[guards + i] = 0xfffffff0u + i * 0x7654321u + phase * 0xfedcba98u;
            CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
            for (unsigned rounds : {0u, 1u, 7u}) for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 3}) {
                std::fill(expected.begin(), expected.end(), guard);
                CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
                void* args[]{&to, &from, &count, &rounds};
                CONFORMANCE_CUDA(cuLaunchKernel(function, blocks, 1, 1, lanes, 1, 1, 0, nullptr, args, nullptr));
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
                for (unsigned i = 0; i < std::min(total, count); ++i) {
                    const uint32_t seed = input[guards + i], first = stack_frame_oracle(seed, rounds + i % 3);
                    const std::array<uint32_t, fields> row{first, stack_frame_oracle(seed ^ first, rounds + 2),
                        seed + 17, seed ^ 0x13579bdfu, i + 0xeca86420u};
                    std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
                }
                expect_words(actual, expected, "nested local frames lanes=" + std::to_string(lanes)
                    + " rounds=" + std::to_string(rounds) + " count=" + std::to_string(count));
            }
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "nested frame input and guard preservation");
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_stack);
}
