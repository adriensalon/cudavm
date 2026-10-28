#include "fixtures.hpp"
#include <algorithm>
#include <bit>

namespace cudavm {
// A bit-at-a-time product, independent of the device's widened multiplication.
uint64_t arithmetic_high_product(uint64_t a, uint64_t b) {
    uint64_t low = 0, high = 0;
    for (unsigned bit = 0; bit < 64; ++bit) {
        if (!(b & (uint64_t(1) << bit))) continue;
        const uint64_t previous = low;
        low += a << bit;
        high += (bit ? a >> (64 - bit) : 0) + uint64_t(low < previous);
    }
    return high;
}

void execute_integer_arithmetic(const loaded_program& program) {
    const auto kernel = program.function("integer_values");
    constexpr unsigned fields = 12;
    constexpr uint32_t guard = 0xa7b6c5d4;
    const std::vector<uint64_t> edges{0, 1, 2, 3, 7, 31, 0x7fffffff, 0x80000000,
        0xfffffffe, 0xffffffff, 0x100000000, 0x100000001, 0x7fffffffffffffff,
        0x8000000000000000ull, 0xffffffff00000001ull, 0xfffffffffffffffeull,
        0xffffffffffffffffull, 0x55555555aaaaaaaaull, 0xaaaaaaaa55555555ull};
    std::vector<std::array<uint64_t, 2>> corpus;
    for (auto a : edges) for (auto b : edges) corpus.push_back({a, b});
    uint64_t random = 0x3141592653589793ull;
    for (unsigned i = 0; i < 257; ++i) {
        random ^= random << 13; random ^= random >> 7; random ^= random << 17;
        const auto a = random;
        random ^= random << 13; random ^= random >> 7; random ^= random << 17;
        corpus.push_back({a, random});
    }
    for (unsigned threads : {13u, 33u, 96u}) {
        const unsigned blocks = (unsigned(corpus.size()) + threads - 1) / threads;
        const unsigned total = blocks * threads;
        std::vector<uint64_t> input(total * 2 + 4, 0xd4c5b6a7d4c5b6a7ull), preserved(input.size());
        std::vector<uint32_t> expected(total * fields * 2 + 4, guard), actual(expected.size());
        for (unsigned i = 0; i < total; ++i) {
            input[2 + i * 2] = corpus[i % corpus.size()][0];
            input[3 + i * 2] = corpus[i % corpus.size()][1];
        }
        device_buffer source(input.size() * 8), destination(expected.size() * 4);
        auto in = source.value + 16, out = destination.value + 8;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 8));
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            for (unsigned i = 0; i < std::min(count, total); ++i) {
                const uint64_t a = input[2 + i * 2], b = input[3 + i * 2];
                const uint32_t x = uint32_t(a), y = uint32_t(b);
                const int32_t sx = std::bit_cast<int32_t>(x), sy = std::bit_cast<int32_t>(y);
                const uint32_t divisor = y == 0 || (x == 0x80000000u && y == 0xffffffffu) ? 1 : y;
                const int32_t signed_divisor = std::bit_cast<int32_t>(divisor);
                const uint64_t signed_product = uint64_t(int64_t(sx) * int64_t(sy));
                const uint64_t high = arithmetic_high_product(a, b);
                const std::array<uint64_t, fields> row{x / divisor, x % divisor, uint32_t(sx / signed_divisor),
                    uint32_t(sx % signed_divisor), uint32_t(signed_product >> 32), uint64_t(x) * y >> 32,
                    uint64_t(x) * y, signed_product, a + b, a - b, high,
                    high - ((a >> 63) ? b : 0) - ((b >> 63) ? a : 0)};
                for (unsigned field = 0; field < fields; ++field) {
                    expected[2 + (i * fields + field) * 2] = uint32_t(row[field]);
                    expected[3 + (i * fields + field) * 2] = uint32_t(row[field] >> 32);
                }
            }
            expect_words(actual, expected, "integer arithmetic threads=" + std::to_string(threads)
                + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 8));
        require(preserved == input, "integer-arithmetic source preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_integer_arithmetic);
}
