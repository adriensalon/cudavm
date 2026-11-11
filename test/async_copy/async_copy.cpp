#include "test_fixtures.hpp"
#include <algorithm>
#include <cstring>

namespace cudavm {
void execute_async_copy(const loaded_program& program) {
    constexpr unsigned blocks = 3, guards = 4;
    constexpr uint32_t guard = 0xa1b2c3d4;
    for (unsigned bytes : {4u, 8u, 16u})
    for (unsigned zfill = 0; zfill <= bytes; ++zfill)
    for (unsigned threads : {1u, 13u, 64u})
    for (unsigned rounds : {0u, 1u, 3u})
    for (unsigned mode : {0u, 1u, 2u}) {
        const bool complete = threads == 64 && rounds == 3 && mode == 2;
        const bool boundary = (zfill == 0 || zfill == bytes) && threads != 64 && rounds != 3 && mode != 2;
        if (!complete && !boundary) continue;
        unsigned total = blocks * threads, count = mode == 0 ? 0 : mode == 1 ? total - 1 : total;
        unsigned seed = mode * 0x1234567u + rounds;
        std::vector<uint32_t> input(guards * 2 + total * 16, guard), preserved(input.size());
        std::vector<uint32_t> expected(guards * 2 + (total * rounds + 2) * 20, guard), actual(expected.size());
        for (unsigned word = 0; word < total * 16; ++word) input[guards + word] = word * 0x9e3779b9u + seed;
        for (unsigned round = 0; round < rounds; ++round) for (unsigned id = 0; id < total; ++id) {
            unsigned peer = id - id % threads + (id % threads + round + 1) % threads;
            auto parameters = [&](unsigned index) {
                return std::array<unsigned, 3>{bytes, zfill, ((index + round) & 1) * bytes};
            };
            const auto p = parameters(peer), own = parameters(id);
            auto* row = expected.data() + guards + (round * total + id) * 20;
            if (peer < count) {
                auto* destination = reinterpret_cast<unsigned char*>(row) + 16 + p[2];
                const auto* source = reinterpret_cast<const unsigned char*>(input.data() + guards + peer * 16) + p[2];
                std::copy_n(source, p[0] - p[1], destination);
                std::fill_n(destination + p[0] - p[1], p[1], 0);
            }
            row[16] = own[0]; row[17] = own[1]; row[18] = own[2];
            row[19] = (input[guards + id * 16] ^ seed) * 33u + round;
        }
        device_buffer source(input.size() * 4), output(expected.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4;
        std::vector<uint32_t> initialized(expected.size(), guard);
        TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        TEST_CUDA(cuMemcpyHtoD(output.value, initialized.data(), initialized.size() * 4));
        void* args[]{&to, &from, &rounds, &count, &seed};
        const auto name = "async_copy_" + std::to_string(bytes) + "_" + std::to_string(zfill);
        TEST_CUDA(cuLaunchKernel(program.function(name.c_str()), blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "copy sizes, every zero-fill extent, aligned offsets, peer visibility and guards");
        TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "async copy source and allocation preservation");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_async_copy);
}
