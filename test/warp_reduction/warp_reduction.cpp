#include "fixtures.hpp"
#include <algorithm>
#include <bit>
#include <limits>

namespace cudavm {
std::array<uint32_t, 9> reduction_oracle(std::span<const uint32_t> values, unsigned first,
    unsigned live, uint32_t mask, unsigned phase) {
    uint64_t sum = 0;
    uint32_t minimum = UINT32_MAX, maximum = 0, all = UINT32_MAX, any = 0, parity = 0;
    int32_t signed_minimum = INT32_MAX, signed_maximum = INT32_MIN;
    for (unsigned peer = phase * 3; peer < live; ++peer) {
        if (!(mask & (uint32_t(1) << peer))) continue;
        const uint32_t value = values[first + peer] ^ (phase * 0x10010001u);
        const auto signed_value = std::bit_cast<int32_t>(value);
        sum += value; minimum = std::min(minimum, value); maximum = std::max(maximum, value);
        signed_minimum = std::min(signed_minimum, signed_value); signed_maximum = std::max(signed_maximum, signed_value);
        all &= value; any |= value; parity ^= value;
    }
    return {uint32_t(sum), minimum, maximum, uint32_t(sum), std::bit_cast<uint32_t>(signed_minimum),
        std::bit_cast<uint32_t>(signed_maximum), all, any, parity};
}
void execute_warp_reduction(const loaded_program& program) {
    constexpr unsigned blocks = 3, guards = 4;
    constexpr uint32_t guard = 0xa1b2c3d4;
    constexpr std::array<uint32_t, 8> edges{0, 1, 0x80000000, 0x7fffffff, 0xffffffff, 0xfffffffe, 0xaaaa5555, 0x5555aaaa};
    for (unsigned threads : {1u, 13u, 32u, 33u, 64u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> input(total + guards * 2, guard), preserved(input.size());
        std::vector<uint32_t> expected((total + 2) * 36 + guards * 2, guard), actual(expected.size());
        device_buffer source(input.size() * 4), output(expected.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4;
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
            for (unsigned i = 0; i < total; ++i)
                input[guards + i] = pattern == 0 ? 0 : pattern == 1 ? UINT32_MAX : pattern == 2 ? edges[i % edges.size()]
                    : i * 0x9e3779b9u + 0x12345678u;
            CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
            const auto values = std::span(input).subspan(guards, total);
            for (uint32_t mask : {UINT32_MAX, 0x55555555u, 0xaaaaaaaau, 0x80010001u, 0u})
            for (bool retirement : {false, true}) {
                std::fill(expected.begin(), expected.end(), guard);
                CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
                void* args[]{&to, &from, &mask};
                const auto function = program.function(retirement ? "retiring_reductions" : "masked_reductions");
                CONFORMANCE_CUDA(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
                for (unsigned i = 0; i < total; ++i) {
                    const auto local = i % threads, lane = local & 31u;
                    if (!(mask & (uint32_t(1) << lane))) continue;
                    const auto first = i - lane, live = std::min(32u, threads - (local & ~31u));
                    for (unsigned phase = 0; phase < (retirement ? 4u : 1u); ++phase) {
                        if (lane < phase * 3) break;
                        const auto row = reduction_oracle(values, first, live, mask, phase);
                        std::copy(row.begin(), row.end(), expected.begin() + guards + i * (retirement ? 36 : 9) + phase * 9);
                    }
                }
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
                expect_words(actual, expected, "all pinned 32-bit warp reduction overloads, masks, retirement and guards");
            }
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "warp reduction input and allocation preservation");
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_warp_reduction);
}
