#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
uint32_t match_low_bits(unsigned count) {
    return count == 32 ? UINT32_MAX : (uint32_t(1) << count) - 1;
}
void execute_warp_match(const loaded_program& program) {
    constexpr uint32_t guard = 0x8b73e5a9;
    constexpr unsigned blocks = 24;
    const auto masked = program.function("masked_match"), retiring = program.function("retiring_match");
    constexpr uint64_t words[]{0, 0x8000000080000000ull, 0x7ff800017fc12345ull,
        0x7ff800027fc12345ull, 0x7ff000007f800000ull, 0xfff00000ff800000ull,
        0x7ff000017f812345ull, 1, 0xffffffffffffffffull};
    for (unsigned threads : {1u, 13u, 31u, 32u, 45u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint64_t> input(total + 2, 0x8b73e5a98b73e5a9ull), preserved(input.size());
        std::vector<uint32_t> actual(total * 24 + 4), expected(actual.size());
        device_buffer source(input.size() * 8), destination(actual.size() * 4);
        auto in = source.value + 8, out = destination.value + 8;
        auto check_row = [&](unsigned i, unsigned offset, uint32_t members, bool wide) {
            uint32_t peers = 0;
            const unsigned lane = (i % threads) % 32;
            for (unsigned peer = 0; peer < 32; ++peer) {
                if (!(members & (uint32_t(1) << peer))) continue;
                const auto a = input[i + 1], b = input[i - lane + peer + 1];
                if (wide ? a == b : uint32_t(a) == uint32_t(b)) peers |= uint32_t(1) << peer;
            }
            expected[offset + 2] = peers;
            expected[offset + 3] = peers == members ? members : 0;
            expected[offset + 4] = peers == members;
        };
        auto reset = [&] {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemsetD32(destination.value, guard, actual.size()));
        };
        auto fetch = [&](const std::string& context) {
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            expect_words(actual, expected, context + " threads=" + std::to_string(threads));
        };
        for (unsigned i = 0; i < total; ++i) {
            const unsigned pattern = (i / threads) / 3;
            const unsigned lane = (i % threads) % 32;
            input[i + 1] = pattern == 0 ? words[2] : pattern == 1 ? uint64_t(lane)
                : pattern == 2 ? (uint64_t(lane % 5) << 32) | 0x7fc12345u
                : pattern == 3 ? 0x7ff8000100000000ull | uint64_t(lane % 5)
                : pattern == 4 ? words[lane % std::size(words)]
                : pattern == 5 ? (lane & 1 ? 0x8000000080000000ull : 0)
                : pattern == 6 ? words[6] : words[(i / threads) % std::size(words)];
        }
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 8));
        for (uint32_t mask : {UINT32_MAX, 0x55555555u, 0xaaaaaaaau, 0x80010001u, 0x80000000u, 0u}) {
            reset();
            void* args[]{&out, &in, &mask};
            CONFORMANCE_CUDA(cuLaunchKernel(masked, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            for (unsigned i = 0; i < total; ++i) {
                const unsigned local = i % threads, lane = local % 32;
                if (!(mask & (uint32_t(1) << lane))) continue;
                const auto members = mask & match_low_bits(std::min(32u, threads - (local / 32) * 32));
                for (unsigned type = 0; type < 6; ++type)
                    check_row(i, i * 18 + type * 3, members, type == 2 || type == 3 || type == 5);
            }
            fetch("match overloads mask=" + std::to_string(mask));
        }
        // Every pattern also checks shrinking membership, with named exited
        // lanes retained in the mask and the predicate overwritten per phase.
        for (unsigned keep : {0u, 1u, 7u, 31u, 32u}) for (uint32_t mask : {UINT32_MAX, 0xaaaaaaaau}) {
            reset();
            void* args[]{&out, &in, &mask, &keep};
            CONFORMANCE_CUDA(cuLaunchKernel(retiring, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            for (unsigned i = 0; i < total; ++i) {
                const unsigned local = i % threads, lane = local % 32;
                const unsigned live = std::min({32u, threads - (local / 32) * 32, keep});
                if (lane >= live || !(mask & (uint32_t(1) << lane))) continue;
                for (unsigned phase = 0; phase < 4 && phase <= lane; ++phase) {
                    const auto members = mask & match_low_bits(live) & ~match_low_bits(phase);
                    check_row(i, i * 24 + phase * 6, members, false);
                    check_row(i, i * 24 + phase * 6 + 3, members, true);
                }
            }
            fetch("retiring match mask=" + std::to_string(mask) + " keep=" + std::to_string(keep));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 8));
        require(preserved == input, "match source words and guards preserved");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_warp_match);
}
