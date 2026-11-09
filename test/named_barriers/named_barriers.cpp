#include "test_fixtures.hpp"
#include <algorithm>
#include <iostream>

namespace cudavm {
bool named_cubin_profile = false;
void execute_named_barriers(const loaded_program& program) {
    constexpr unsigned blocks = 3, fields = 5, guards = 4;
    constexpr uint32_t guard = 0xc5d4e3f2;
    if (named_cubin_profile) std::cout << "profile=cubin implicit whole-block barriers; counted/reduction/handoff SASS remains open\n";
    for (const std::string name : {"named_whole", "named_groups", "named_handoff"}) {
        if (named_cubin_profile && name != "named_whole") continue;
        const auto kernel = program.function(name.c_str());
        const bool whole = name == "named_whole", groups = name == "named_groups";
        for (unsigned threads : {1u, 13u, 32u, 33u, 64u, 96u, 128u}) {
            if (!whole && threads != 128) continue;
            const unsigned total = blocks * threads;
            std::vector<uint32_t> input(total + 2 * guards, guard), preserved(input.size());
            std::vector<uint32_t> expected(total * fields + 2 * guards, guard), actual(expected.size());
            std::vector<uint32_t> scratch_expected(input.size(), guard), scratch_actual(input.size());
            device_buffer source(input.size() * 4), output(expected.size() * 4), scratch(input.size() * 4);
            auto in = source.value + guards * 4, out = output.value + guards * 4, temporary = scratch.value + guards * 4;
            for (unsigned seed : {0u, 0x98765432u}) {
                for (unsigned i = 0; i < total; ++i) input[guards + i] = i * 0x1020305u + seed;
                CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
                for (unsigned rounds : {0u, 1u, 7u})
                for (unsigned id : {0u, 1u, 7u, 15u})
                for (unsigned pattern = 0; pattern < (groups ? 4u : 1u); ++pattern) {
                    if (!whole && id) continue;
                    std::fill(expected.begin(), expected.end(), guard);
                    std::fill(scratch_expected.begin(), scratch_expected.end(), guard);
                    CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
                    CONFORMANCE_CUDA(cuMemcpyHtoD(scratch.value, scratch_expected.data(), scratch_expected.size() * 4));
                    void* args[]{&out, &in, &temporary, &rounds, &id, &pattern};
                    CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
                    for (unsigned i = 0; i < total; ++i) {
                        const unsigned lane = i % threads, block_base = i - lane;
                        std::array<uint32_t, fields> sums{};
                        const bool producer = !whole && !groups && lane < 32;
                        const bool consumer = !whole && !groups && lane >= 32 && lane < 64;
                        const unsigned group_base = whole ? 0 : consumer ? 0 : lane & ~63u;
                        const unsigned members = whole ? threads : consumer ? 32 : 64;
                        for (unsigned round = 0; round < rounds; ++round) {
                            if (!consumer) scratch_expected[guards + i] = (input[guards + i] ^ (round * 17u + i)) + 0x12345u;
                            if (!producer) {
                                const unsigned peer = block_base + group_base + (lane + round + 1) % members;
                                const auto value = input[guards + peer] ^ (round * 17u + peer);
                                sums[0] += value; sums[1] += value + 0x12345u;
                            }
                            if (groups) {
                                unsigned votes = 0;
                                for (unsigned rank = 0; rank < members; ++rank) {
                                    const auto peer = block_base + group_base + rank;
                                    votes += unsigned(pattern == 0 ? false : pattern == 1 ? true
                                        : pattern == 2 ? rank == round % members : (input[guards + peer] + round) % 4 != 0);
                                }
                                sums[2] += votes; sums[3] += unsigned(votes == members); sums[4] += unsigned(votes != 0);
                            }
                        }
                        std::copy(sums.begin(), sums.end(), expected.begin() + guards + i * fields);
                    }
                    const auto context = name + " threads=" + std::to_string(threads) + " rounds=" + std::to_string(rounds)
                        + " id=" + std::to_string(id) + " pattern=" + std::to_string(pattern) + " seed=" + std::to_string(seed);
                    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
                    expect_words(actual, expected, context + " results and guards");
                    CONFORMANCE_CUDA(cuMemcpyDtoH(scratch_actual.data(), scratch.value, scratch_actual.size() * 4));
                    expect_words(scratch_actual, scratch_expected, context + " published storage and guards");
                }
                CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
                expect_words(preserved, input, name + " source preservation");
            }
        }
    }
}
}
int main(int argc, char** argv) {
    if (argc == 3) cudavm::named_cubin_profile = std::string_view(argv[2]) == "cubin";
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_named_barriers);
}
