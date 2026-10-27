#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_shared_memory(const loaded_program& program) {
    const auto exchange = program.function("shared_exchange");
    const auto reduce = program.function("shared_reduce");
    const auto transpose = program.function("shared_transpose");
    constexpr uint32_t guard = 0xdeadbeef;
    for (unsigned lanes : {1u, 13u, 32u, 45u, 64u, 128u}) {
        const unsigned blocks = 3, total = blocks * lanes;
        std::vector<uint32_t> input(total), actual(total + 2), expected(total + 2, guard);
        for (unsigned i = 0; i < total; ++i) input[i] = i * 0x7654321u + 0xfffffffdu;
        device_buffer in(total * 4), allocation(actual.size() * 4);
        auto out = allocation.value + 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(in.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 1, total - 1, total}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in.value, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(exchange, blocks, 1, 1, lanes, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocation.value, actual.size() * 4));
            // Two reversals restore the original value, with the other lane's
            // index added between the barriers. All arithmetic is modulo 2^32.
            for (unsigned i = 0; i < count; ++i) {
                uint32_t value = input[i];
                for (unsigned step = 0; step < ((i % lanes) >> 4); ++step) value = value * 3u + step;
                expected[i + 1] = value * 3u + lanes - 1 - i % lanes;
            }
            auto context = " lanes=" + std::to_string(lanes) + " count=" + std::to_string(count);
            expect_words(actual, expected, "shared exchange" + context);
            if ((lanes & (lanes - 1)) != 0) continue;
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.value, expected.data(), expected.size() * 4));
            CONFORMANCE_CUDA(cuLaunchKernel(reduce, blocks, 1, 1, lanes, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocation.value, actual.size() * 4));
            for (unsigned block = 0; block < blocks; ++block) {
                uint32_t sum = 0;
                for (unsigned lane = 0; lane < lanes; ++lane) {
                    unsigned i = block * lanes + lane;
                    if (i < count) sum += input[i];
                }
                expected[block + 1] = sum;
            }
            expect_words(actual, expected, "block reduction" + context);
        }
        std::vector<uint32_t> preserved(input.size());
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), in.value, preserved.size() * 4));
        expect_words(preserved, input, "shared input preservation");
    }
    const unsigned blocks = 3, total = blocks * 64;
    std::vector<uint32_t> input(total), actual(total + 2), expected(total + 2, guard);
    for (unsigned i = 0; i < total; ++i) input[i] = 0xfffffffeu + i * 0x1020305u;
    device_buffer in(total * 4), allocation(actual.size() * 4);
    auto out = allocation.value + 4;
    CONFORMANCE_CUDA(cuMemcpyHtoD(in.value, input.data(), input.size() * 4));
    for (unsigned tiles : {0u, 1u, blocks}) {
        std::fill(expected.begin(), expected.end(), guard);
        CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.value, expected.data(), expected.size() * 4));
        void* args[]{&out, &in.value, &tiles};
        CONFORMANCE_CUDA(cuLaunchKernel(transpose, blocks, 1, 1, 8, 8, 1, 0, nullptr, args, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocation.value, actual.size() * 4));
        for (unsigned i = 0; i < total; ++i)
            if (i / 64 < tiles) expected[i + 1] = input[i / 64 * 64 + (i % 8) * 8 + i % 64 / 8];
        expect_words(actual, expected, "shared transpose tiles=" + std::to_string(tiles));
    }
    std::vector<uint32_t> preserved(input.size());
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), in.value, preserved.size() * 4));
    expect_words(preserved, input, "transpose input preservation");
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_shared_memory);
}
