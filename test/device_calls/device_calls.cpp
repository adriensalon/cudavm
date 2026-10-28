#include "fixtures.hpp"
#include <algorithm>
#include <bit>

namespace cudavm {
uint32_t call_fold(uint32_t a, uint32_t b) {
    const uint32_t value = a ^ (b * 33u);
    return value & 0x80000000u ? value - 0x13579bdfu : value + 0x89abcdefu;
}
uint64_t call_mix(uint64_t a, uint64_t b, unsigned shift) {
    return std::rotl(a, int(shift & 63)) + b;
}
std::array<uint32_t, 6> call_oracle(uint64_t a, uint64_t b, unsigned i, unsigned rounds) {
    const uint64_t mixed = call_mix(a, b, i);
    uint64_t nested = a;
    for (unsigned round = 0; round < rounds + (i & 3); ++round) {
        const uint64_t value = call_mix(nested, b + round, round * 7);
        const uint64_t folded = call_fold(uint32_t(value), uint32_t(value >> 32));
        nested = value ^ (folded * 0x100000001ull);
    }
    return {uint32_t(mixed), uint32_t(mixed >> 32), call_fold(uint32_t(mixed), uint32_t(mixed >> 32)),
        uint32_t(nested), uint32_t(nested >> 32), i & 1 ? call_fold(uint32_t(a), uint32_t(b))
            : call_fold(uint32_t(b >> 32), uint32_t(a >> 32))};
}
void execute_device_calls(const loaded_program& program) {
    constexpr uint32_t guard = 0x97e35ac1;
    constexpr unsigned blocks = 3;
    const auto kernel = program.function("call_results");
    constexpr uint64_t words[]{0, 1, UINT64_MAX, 0x8000000000000000ull, 0x7fffffffffffffffull,
        0x00000000ffffffffull, 0xffffffff00000000ull, 0x8000000080000000ull,
        0x0123456789abcdefull, 0xfedcba9876543210ull, 0x55555555aaaaaaaaull};
    for (unsigned threads : {1u, 13u, 32u, 45u, 96u}) {
        const unsigned total = blocks * threads;
        std::vector<uint64_t> input(total * 2 + 4, 0x97e35ac197e35ac1ull), preserved(input.size());
        uint64_t state = 0x9265ac37d8914befull;
        for (unsigned i = 0; i < total; ++i) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            input[2 + i * 2] = i % 3 ? words[i % std::size(words)] : state;
            input[3 + i * 2] = i % 5 ? words[(i / std::size(words) + i * 3) % std::size(words)] : ~state;
        }
        std::vector<uint32_t> actual(total * 6 + 4), expected(actual.size());
        device_buffer source(input.size() * 8), destination(actual.size() * 4);
        auto in = source.value + 16, out = destination.value + 8;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 8));
        for (unsigned rounds : {0u, 1u, 4u}) {
            std::vector<std::array<uint32_t, 6>> oracle;
            for (unsigned i = 0; i < total; ++i)
                oracle.push_back(call_oracle(input[2 + i * 2], input[3 + i * 2], i, rounds));
            for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 17}) {
                std::fill(expected.begin(), expected.end(), guard);
                for (unsigned i = 0; i < std::min(count, total); ++i)
                    std::copy(oracle[i].begin(), oracle[i].end(), expected.begin() + 2 + i * 6);
                CONFORMANCE_CUDA(cuMemsetD32(destination.value, guard, actual.size()));
                void* args[]{&out, &in, &count, &rounds};
                CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
                expect_words(actual, expected, "device calls threads=" + std::to_string(threads)
                    + " rounds=" + std::to_string(rounds) + " count=" + std::to_string(count));
            }
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 8));
        require(preserved == input, "device call source words and guards preserved");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_calls);
}
