#include "fixtures.hpp"
#include <algorithm>
#include <numeric>

namespace cudavm {
void execute_split_barriers(const loaded_program& program) {
    constexpr unsigned guards = 4, blocks = 3;
    constexpr uint32_t guard = 0xa1b2c3d4;
    for (unsigned count : {1u, 2u, 7u, 31u})
    for (unsigned rounds : {0u, 1u, 5u}) {
        std::vector<uint32_t> expected(guards * 2 + count * rounds * 2, guard), actual(expected.size());
        device_buffer output(expected.size() * 4);
        auto to = output.value + guards * 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        void* args[]{&to, &count, &rounds};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("split_serial"), 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
        for (unsigned r = 0; r < rounds; ++r) for (unsigned a = 0; a < count; ++a) {
            expected[guards + (r * count + a) * 2] = count - a;
            expected[guards + (r * count + a) * 2 + 1] = a + 1 == count;
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "serial expected/pending counts, incomplete waits, completion and reinitialization");
    }
    for (unsigned threads : {1u, 13u, 32u, 33u, 64u, 96u, 128u})
    for (unsigned rounds : {0u, 1u, 5u})
    for (unsigned drop : {0u, 1u}) {
        const unsigned total = blocks * threads;
        unsigned group_size = std::min(threads, 64u);
        std::vector<uint32_t> input(guards * 2 + total, guard), preserved(input.size());
        std::vector<uint32_t> expected(guards * 2 + total * rounds * 4, guard), actual(expected.size());
        std::vector<uint32_t> scratch(guards * 2 + total, guard), actual_scratch(scratch.size());
        for (unsigned i = 0; i < total; ++i) input[guards + i] = i * 0x9e3779b9u + rounds * 17u;
        device_buffer source(input.size() * 4), output(expected.size() * 4), temporary(scratch.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4, temp = temporary.value + guards * 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(temporary.value, scratch.data(), scratch.size() * 4));
        void* args[]{&to, &from, &temp, &rounds, &group_size, &drop};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("split_publication"), blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        for (unsigned r = 0; r < rounds; ++r) for (unsigned block = 0; block < blocks; ++block)
        for (unsigned base = 0; base < threads; base += group_size) {
            unsigned size = std::min(group_size, threads - base);
            unsigned live = size - (drop ? std::min(r, size - 1) : 0);
            std::vector<uint32_t> counts;
            for (unsigned rank = 0; rank < live; ++rank) {
                unsigned id = block * threads + base + rank;
                unsigned peer = block * threads + base + (rank + r + 1) % live;
                unsigned row = guards + (r * total + id) * 4;
                uint32_t value = input[guards + peer] ^ (r * 0x10001u + peer);
                expected[row] = value;
                expected[row + 1] = value + 0x12345u;
                expected[row + 2] = (input[guards + id] + r) * 33u + rank;
                counts.push_back(actual[row + 3]);
                expected[row + 3] = actual[row + 3];
                scratch[guards + id] = (input[guards + id] ^ (r * 0x10001u + id)) + 0x12345u;
            }
            std::sort(counts.begin(), counts.end());
            std::vector<uint32_t> pending(live);
            std::iota(pending.begin(), pending.end(), 1u);
            expect_words(counts, pending, "each phase has exactly one arrival for every pending count");
        }
        expect_words(actual, expected, "split publication, independent groups, dropped arrivals, phases and complete guards");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual_scratch.data(), temporary.value, actual_scratch.size() * 4));
        expect_words(actual_scratch, scratch, "last live publication and scratch guards");
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "split-barrier source preservation");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_split_barriers);
}
