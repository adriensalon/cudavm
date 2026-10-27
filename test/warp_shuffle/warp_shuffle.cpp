#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_warp_shuffle(const loaded_program& program) {
    constexpr uint32_t guard = 0xdeadbeef;
    constexpr unsigned blocks = 3;
    const auto modes = program.function("shuffle_modes");
    const auto scan = program.function("shuffle_scan");
    for (unsigned threads : {1u, 13u, 31u, 32u, 33u, 45u, 64u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> input(total + 2, guard), preserved(input.size());
        std::vector<uint32_t> actual(total * 4 + 4);
        device_buffer source(input.size() * sizeof(uint32_t));
        device_buffer destination(actual.size() * sizeof(uint32_t));
        auto in = source.value + sizeof(uint32_t), out = destination.value + sizeof(uint32_t);
        const auto shape = " threads=" + std::to_string(threads);
        auto fetch = [&](unsigned words, const std::string& context) {
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * sizeof(uint32_t)));
            require(actual.front() == guard, context + ": prefix guard");
            for (size_t i = words + 1; i < actual.size(); ++i)
                require(actual[i] == guard, context + ": tail guard " + std::to_string(i));
        };
        for (uint32_t seed : {0u, 0xfffffff1u}) {
            for (unsigned i = 0; i < total; ++i)
                input[i + 1] = seed + i * 0x07654321u;
            CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * sizeof(uint32_t)));
            for (unsigned width : {1u, 2u, 8u, 16u, 32u}) {
                std::vector<unsigned> distances{0u, 1u, width - 1, width, 31u};
                std::sort(distances.begin(), distances.end());
                distances.erase(std::unique(distances.begin(), distances.end()), distances.end());
                for (unsigned distance : distances) {
                    const auto context = "shuffle" + shape + " width=" + std::to_string(width)
                        + " distance=" + std::to_string(distance) + " seed=" + std::to_string(seed);
                    CONFORMANCE_CUDA(cuMemsetD32(destination.value, guard, actual.size()));
                    void* args[]{&out, &in, &width, &distance};
                    CONFORMANCE_CUDA(cuLaunchKernel(modes, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
                    fetch(total * 4, context);
                    for (unsigned i = 0; i < total; ++i) {
                        const unsigned local = i % threads, lane = local % 32;
                        const unsigned live = std::min(32u, threads - (local / 32) * 32);
                        const unsigned first = lane / width * width, last = first + width - 1;
                        const unsigned delta = distance & 31;
                        const std::array<unsigned, 4> peers{
                            first + delta % width,
                            delta <= lane - first ? lane - delta : lane,
                            lane + delta <= last ? lane + delta : lane,
                            (lane ^ delta) <= last ? lane ^ delta : lane};
                        for (unsigned mode = 0; mode < peers.size(); ++mode) {
                            // Data from an inactive source lane is unspecified by CUDA.
                            if (peers[mode] >= live) continue;
                            require(actual[i * 4 + mode + 1] == input[i - lane + peers[mode] + 1],
                                context + ": thread=" + std::to_string(i) + " mode=" + std::to_string(mode));
                        }
                    }
                }
            }
            CONFORMANCE_CUDA(cuMemsetD32(destination.value, guard, actual.size()));
            void* args[]{&out, &in};
            CONFORMANCE_CUDA(cuLaunchKernel(scan, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            fetch(total * 2, "scan" + shape);
            for (unsigned i = 0; i < total; ++i) {
                const unsigned local = i % threads, lane = local % 32;
                const unsigned live = std::min(32u, threads - (local / 32) * 32);
                uint32_t prefix = 0, sum = 0;
                for (unsigned peer = 0; peer < live; ++peer) {
                    const auto value = input[i - lane + peer + 1];
                    if (peer <= lane) prefix += value;
                    sum += value;
                }
                const std::array<uint32_t, 2> expected{prefix, sum};
                expect_words(std::span(actual).subspan(i * 2 + 1, 2), expected,
                    "inclusive scan and total broadcast" + shape + " thread=" + std::to_string(i));
            }
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * sizeof(uint32_t)));
            expect_words(preserved, input, "shuffle input preservation" + shape);
        }
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_warp_shuffle);
}
