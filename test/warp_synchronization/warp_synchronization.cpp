#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_warp_synchronization(const loaded_program& program) {
    constexpr uint32_t guard = 0x936ad5c7;
    constexpr unsigned blocks = 3;
    const auto kernel = program.function("warp_publish");
    for (auto shape : {std::array<unsigned, 3>{1, 1, 1}, {13, 1, 1}, {32, 1, 1},
                       {45, 1, 1}, {8, 3, 3}, {8, 4, 3}}) {
        const unsigned threads = shape[0] * shape[1] * shape[2], total = blocks * threads;
        std::vector<uint32_t> input(total + 4, guard), preserved(input.size());
        std::vector<uint32_t> actual(total * 16 + 4), expected(actual.size());
        std::vector<uint32_t> scratch(total + 4), expected_scratch(scratch.size());
        for (unsigned i = 0; i < total; ++i) input[i + 2] = 0xfedcba98u ^ (i * 0x07654321u);
        device_buffer source(input.size() * 4), destination(actual.size() * 4), workspace(scratch.size() * 4);
        auto in = source.value + 8, out = destination.value + 8, temporary = workspace.value + 8;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned mode : {0u, 1u, 2u}) for (unsigned rounds : {0u, 1u, 4u})
        for (uint32_t mask : {UINT32_MAX, 0x55555555u, 0xaaaaaaaau, 0x80010001u, 0x80000000u, 0u}) {
            std::fill(expected.begin(), expected.end(), guard);
            std::fill(expected_scratch.begin(), expected_scratch.end(), guard);
            CONFORMANCE_CUDA(cuMemsetD32(destination.value, guard, actual.size()));
            CONFORMANCE_CUDA(cuMemsetD32(workspace.value, guard, scratch.size()));
            void* args[]{&out, &in, &temporary, &mask, &rounds, &mode};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, shape[0], shape[1], shape[2],
                0, nullptr, args, nullptr));
            for (unsigned i = 0; i < total; ++i) {
                const unsigned local = i % threads, lane = local % 32, warp_base = local - lane;
                const unsigned live = std::min(32u, threads - warp_base);
                uint32_t members = mask & (UINT32_MAX >> (32 - live));
                if (mode == 1) members &= lane & 1 ? 0xaaaaaaaau : 0x55555555u;
                if (!(members & (uint32_t(1) << lane))) continue;
                unsigned peer = (lane + 1) % 32;
                while (!(members & (uint32_t(1) << peer))) peer = (peer + 1) % 32;
                const unsigned peer_i = i - lane + peer;
                const unsigned count = rounds + (rounds && mode == 1 ? (lane & 1) + (local / 32) % 3 : 0);
                const unsigned n = mode == 2 ? peer % 3 : 0;
                for (unsigned round = 0; round < count; ++round) {
                    const uint32_t value = (input[peer_i + 2] ^ (round * 0x10203u + peer_i)) + n * (n + 1) / 2;
                    expected[2 + i * 16 + round * 2] = value;
                    expected[3 + i * 16 + round * 2] = value + 0x9e3779b9u;
                }
                if (count) {
                    const unsigned own_n = mode == 2 ? lane % 3 : 0;
                    expected_scratch[i + 2] = (input[i + 2] ^ ((count - 1) * 0x10203u + i))
                        + 0x9e3779b9u + own_n * (own_n + 1) / 2;
                }
            }
            const auto context = "warp publication threads=" + std::to_string(threads)
                + " mode=" + std::to_string(mode) + " rounds=" + std::to_string(rounds)
                + " mask=" + std::to_string(mask);
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            expect_words(actual, expected, context);
            CONFORMANCE_CUDA(cuMemcpyDtoH(scratch.data(), workspace.value, scratch.size() * 4));
            expect_words(scratch, expected_scratch, context + " final scratch and guards");
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "warp publication source preservation");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_warp_synchronization);
}
