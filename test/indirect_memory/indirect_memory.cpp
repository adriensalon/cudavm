#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_indirect_memory(const loaded_program& program) {
    const auto kernel = program.function("follow_pointers");
    constexpr uint32_t guard = 0xdeadbeef;
    constexpr unsigned words = 32, prefix = 2, blocks = 3;
    std::array<std::vector<uint32_t>, 3> payload;
    for (unsigned allocation = 0; allocation < payload.size(); ++allocation) {
        payload[allocation].assign(words + prefix * 2, guard);
        for (unsigned word = 0; word < words; ++word)
            payload[allocation][word + prefix] = 0xfffffffdu + allocation * 0x7654321u + word * 0x1020305u;
    }
    device_buffer first(payload[0].size() * 4), second(payload[1].size() * 4), third(payload[2].size() * 4);
    const std::array<CUdeviceptr, 3> allocations{first.value, second.value, third.value};
    for (unsigned allocation = 0; allocation < payload.size(); ++allocation)
        CONFORMANCE_CUDA(cuMemcpyHtoD(allocations[allocation], payload[allocation].data(), payload[allocation].size() * 4));
    device_buffer leaf0(4 * sizeof(CUdeviceptr)), leaf1(4 * sizeof(CUdeviceptr)), root(2 * sizeof(CUdeviceptr));
    const std::array<CUdeviceptr, 2> leaf_addresses{leaf0.value, leaf1.value};
    // Every table entry has an independent host description. Repeated owners
    // are interior aliases; null leaves must not be dereferenced.
    const std::array<std::array<int, 4>, 2> owners{{{0, 1, 0, -1}, {2, 0, -1, 1}}};
    const std::array<std::array<unsigned, 4>, 2> offsets{{{0, 4, 8, 0}, {3, 12, 0, 16}}};
    for (unsigned threads : {1u, 13u, 33u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint32_t> selection(total + 2, guard), preserved(selection.size());
        for (unsigned i = 0; i < total; ++i) selection[i + 1] = (i & 3u) | (((i >> 3) & 7u) << 2);
        std::vector<uint32_t> expected(total * 3 + 4, guard), actual(expected.size());
        device_buffer selectors(selection.size() * 4), destination(expected.size() * 4);
        auto indices = selectors.value + 4, out = destination.value + 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(selectors.value, selection.data(), selection.size() * 4));
        for (unsigned phase : {0u, 1u, 2u}) {
            std::array<std::array<CUdeviceptr, 4>, 2> leaves{};
            for (unsigned table = 0; table < leaves.size(); ++table)
            for (unsigned entry = 0; entry < leaves[table].size(); ++entry) {
                const int owner = owners[table][entry];
                if (owner >= 0) leaves[table][entry] = allocations[owner]
                    + (prefix + offsets[table][entry] + unsigned(phase == 2)) * 4;
            }
            const std::array<CUdeviceptr, 2> roots{
                leaf_addresses[phase == 1 ? 1 : 0], leaf_addresses[phase == 1 ? 0 : 1]};
            CONFORMANCE_CUDA(cuMemcpyHtoD(leaf0.value, leaves[0].data(), sizeof(leaves[0])));
            CONFORMANCE_CUDA(cuMemcpyHtoD(leaf1.value, leaves[1].data(), sizeof(leaves[1])));
            CONFORMANCE_CUDA(cuMemcpyHtoD(root.value, roots.data(), sizeof(roots)));
            for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
                std::fill(expected.begin(), expected.end(), guard);
                CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
                void* args[]{&out, &root.value, &indices, &count};
                CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
                for (unsigned i = 0; i < std::min(total, count); ++i) {
                    unsigned table = (i / 4) % 2;
                    if (phase == 1) table = 1 - table;
                    const unsigned entry = selection[i + 1] % 4, element = (selection[i + 1] / 4) % 8;
                    const int owner = owners[table][entry];
                    if (owner < 0) continue;
                    const unsigned offset = prefix + offsets[table][entry] + unsigned(phase == 2) + element;
                    const auto pointer = leaves[table][entry];
                    expected[i * 3 + 1] = payload[owner][offset] + i * 0x101u;
                    expected[i * 3 + 2] = uint32_t(pointer);
                    expected[i * 3 + 3] = uint32_t(pointer >> 32);
                }
                expect_words(actual, expected, "pointer chains threads=" + std::to_string(threads)
                    + " phase=" + std::to_string(phase) + " count=" + std::to_string(count));
            }
            std::array<CUdeviceptr, 4> leaf_copy{};
            for (unsigned table = 0; table < leaves.size(); ++table) {
                CONFORMANCE_CUDA(cuMemcpyDtoH(leaf_copy.data(), leaf_addresses[table], sizeof(leaf_copy)));
                require(leaf_copy == leaves[table], "leaf pointer-table preservation");
            }
            std::array<CUdeviceptr, 2> root_copy{};
            CONFORMANCE_CUDA(cuMemcpyDtoH(root_copy.data(), root.value, sizeof(root_copy)));
            require(root_copy == roots, "root pointer-table preservation");
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), selectors.value, preserved.size() * 4));
        expect_words(preserved, selection, "pointer selection preservation");
    }
    for (unsigned allocation = 0; allocation < payload.size(); ++allocation) {
        std::vector<uint32_t> actual(payload[allocation].size());
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocations[allocation], actual.size() * 4));
        expect_words(actual, payload[allocation], "pointer payload/guard preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_indirect_memory);
}
