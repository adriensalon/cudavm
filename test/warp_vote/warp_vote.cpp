#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_warp_vote(const loaded_program& program) {
    constexpr uint32_t guard = 0xdeadbeef;
    constexpr unsigned blocks = 3;
    const auto masked = program.function("masked_vote"), retiring = program.function("retiring_vote");
    auto low_bits = [](unsigned count) { return count == 32 ? UINT32_MAX : (uint32_t(1) << count) - 1; };
    for (unsigned threads : {1u, 13u, 31u, 32u, 33u, 45u, 64u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> input(total + 2, guard), preserved(input.size());
        std::vector<uint32_t> actual(total * 12 + 4), expected(actual.size());
        device_buffer source(input.size() * sizeof(uint32_t));
        device_buffer destination(actual.size() * sizeof(uint32_t));
        auto in = source.value + sizeof(uint32_t), out = destination.value + sizeof(uint32_t);
        const auto shape = " threads=" + std::to_string(threads);
        auto reset = [&] {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemsetD32(destination.value, guard, actual.size()));
        };
        auto fetch = [&](const std::string& context) {
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * sizeof(uint32_t)));
            expect_words(actual, expected, context);
        };
        for (unsigned pattern : {0u, 1u, 2u, 3u}) {
            for (unsigned i = 0; i < total; ++i)
                input[i + 1] = pattern == 0 ? 0 : pattern == 1 ? UINT32_MAX
                    : pattern == 2 ? ((i & 1) ? 0x80000000u : 5u) : i * 0x07654321u + 0xfffffff1u;
            CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * sizeof(uint32_t)));
            for (uint32_t mask : {UINT32_MAX, 0x55555555u, 0xaaaaaaaau, 0x80010001u, 0u}) {
                reset();
                void* args[]{&out, &in, &mask};
                CONFORMANCE_CUDA(cuLaunchKernel(masked, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
                for (unsigned i = 0; i < total; ++i) {
                    const unsigned local = i % threads, lane = local % 32;
                    if (!(mask & (uint32_t(1) << lane))) continue;
                    const unsigned live = std::min(32u, threads - (local / 32) * 32);
                    const uint32_t members = mask & low_bits(live);
                    for (unsigned phase = 0; phase < 4; ++phase) {
                        const unsigned shift = phase == 3 ? 31 : phase;
                        uint32_t ballot = 0;
                        for (unsigned peer = 0; peer < live; ++peer)
                            if ((members & (uint32_t(1) << peer))
                                && (input[i - lane + peer + 1] & (uint32_t(1) << shift)))
                                ballot |= uint32_t(1) << peer;
                        const unsigned offset = i * 12 + phase * 3 + 1;
                        expected[offset] = ballot;
                        expected[offset + 1] = ballot == members;
                        expected[offset + 2] = ballot != 0;
                    }
                }
                fetch("masked ballot/all/any" + shape + " pattern=" + std::to_string(pattern)
                    + " mask=" + std::to_string(mask));
            }
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * sizeof(uint32_t)));
            expect_words(preserved, input, "vote input preservation" + shape);
        }
        for (unsigned keep : {0u, 1u, 7u, 13u, 31u, 32u}) {
            reset();
            void* args[]{&out, &keep};
            CONFORMANCE_CUDA(cuLaunchKernel(retiring, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            for (unsigned i = 0; i < total; ++i) {
                const unsigned local = i % threads, lane = local % 32;
                const unsigned live = std::min({32u, threads - (local / 32) * 32, keep});
                for (unsigned phase = 0; phase < 4; ++phase) {
                    if (lane >= live || lane < phase) continue;
                    uint32_t members = 0, ballot = 0;
                    for (unsigned peer = phase; peer < live; ++peer) {
                        members |= uint32_t(1) << peer;
                        if (peer % 3 == phase) ballot |= uint32_t(1) << peer;
                    }
                    const unsigned offset = i * 12 + phase * 3 + 1;
                    expected[offset] = ballot;
                    expected[offset + 1] = ballot == members;
                    expected[offset + 2] = ballot != 0;
                }
            }
            fetch("retiring ballot/all/any" + shape + " keep=" + std::to_string(keep));
        }
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_warp_vote);
}
