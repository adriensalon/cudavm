#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
uint32_t permutation_expected(uint32_t low, uint32_t high, uint32_t selector, bool sign) {
    std::array<unsigned, 8> bytes{};
    for (unsigned byte = 0; byte < 4; ++byte) {
        bytes[byte] = (low >> (byte * 8)) & 255u;
        bytes[byte + 4] = (high >> (byte * 8)) & 255u;
    }
    uint32_t result = 0;
    for (unsigned lane = 0; lane < 4; ++lane) {
        const unsigned choice = (selector >> (lane * 4)) & 15u;
        const unsigned source = bytes[choice % 8];
        const unsigned value = sign && choice >= 8 ? (source >= 128 ? 255 : 0) : source;
        result |= uint32_t(value) << (lane * 8);
    }
    return result;
}

void execute_integer_permutation(const loaded_program& program) {
    const auto kernel = program.function("permute_bytes");
    constexpr uint32_t guard = 0xdeadbeef;
    // The large cohort covers every low-16-bit selector. The small cohort
    // independently exercises partial logical warps and per-block indexing.
    const std::array<std::array<unsigned, 2>, 2> shapes{{{3, 13}, {683, 96}}};
    for (const auto& shape : shapes) {
        const unsigned total = shape[0] * shape[1];
        std::vector<uint32_t> input(total * 3 + 4, guard), preserved(input.size());
        std::vector<uint32_t> expected(total * 2 + 4, guard), actual(expected.size());
        for (unsigned i = 0; i < total; ++i) {
            input[i * 3 + 1] = 0x80ff017fu ^ (i * 0x01010101u);
            input[i * 3 + 2] = 0xfe02aa55u + i * 0x1020305u;
            input[i * 3 + 3] = 0xa5810000u | (i & 65535u);
        }
        device_buffer source(input.size() * 4), destination(expected.size() * 4);
        auto in = source.value + 4, out = destination.value + 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, shape[0], 1, 1, shape[1], 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            for (unsigned i = 0; i < std::min(total, count); ++i) {
                const auto low = input[i * 3 + 1], high = input[i * 3 + 2], selector = input[i * 3 + 3];
                expected[i * 2 + 1] = permutation_expected(low, high, selector, false);
                expected[i * 2 + 2] = permutation_expected(low, high, selector, true);
            }
            expect_words(actual, expected, "byte permutation blocks=" + std::to_string(shape[0])
                + " threads=" + std::to_string(shape[1]) + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "permutation source preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_integer_permutation);
}
