#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
std::array<uint32_t, 2> control_flow_expected(uint32_t value, uint32_t seed) {
    uint32_t state = value ^ seed;
    const unsigned iterations = std::min(value & 7u, (value >> 5) & 7u);
    for (unsigned step = 0; step < iterations; ++step) {
        uint32_t next = step % 2 ? (state / 2) ^ (value + step) : state * 3u + step;
        state = next % 4 ? next ^ 0x9e3779b9u : next + 17u;
    }
    const unsigned selector = (value / 8) % 4;
    if (selector == 0) state += seed;
    if (selector == 1) state ^= seed;
    if (selector == 2) state = (state << 3) | (state >> 29);
    if (selector == 3) state = ~state;
    return {state, iterations};
}

void execute_control_flow(const loaded_program& program) {
    const auto kernel = program.function("branch_loop");
    constexpr uint32_t guard = 0xdeadbeef;
    constexpr unsigned blocks = 3;
    for (unsigned threads : {1u, 13u, 33u, 64u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> input(total + 2, guard), preserved(input.size());
        std::vector<uint32_t> expected(total * 2 + 4, guard), actual(expected.size());
        // Low bytes vary loop/break/selector combinations and include early returns.
        for (unsigned i = 0; i < total; ++i)
            input[i + 1] = i % 17 == 0 ? 0u : (i & 255u) | (i % 2 ? 0xffffff00u : 0u);
        device_buffer source(input.size() * sizeof(uint32_t)), destination(expected.size() * sizeof(uint32_t));
        auto in = source.value + sizeof(uint32_t), out = destination.value + sizeof(uint32_t);
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * sizeof(uint32_t)));
        for (uint32_t seed : {0u, 0x80000000u, 0xffffffffu})
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * sizeof(uint32_t)));
            void* args[]{&out, &in, &count, &seed};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * sizeof(uint32_t)));
            for (unsigned i = 0; i < std::min(total, count); ++i) {
                if (!input[i + 1]) continue;
                const auto values = control_flow_expected(input[i + 1], seed);
                std::copy(values.begin(), values.end(), expected.begin() + 1 + i * 2);
            }
            expect_words(actual, expected, "branch/loop threads=" + std::to_string(threads)
                + " count=" + std::to_string(count) + " seed=" + std::to_string(seed));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * sizeof(uint32_t)));
        expect_words(preserved, input, "control-flow input preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_control_flow);
}
