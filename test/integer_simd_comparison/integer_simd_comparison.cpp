#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
constexpr unsigned simd_comparison_fields = 48;
int64_t simd_comparison_lane(uint32_t word, unsigned lane, unsigned bits, bool sign) {
    const uint32_t value = (word >> (lane * bits)) & ((1u << bits) - 1);
    return sign && value >= (1u << (bits - 1)) ? int64_t(value) - (int64_t(1) << bits) : value;
}
bool simd_comparison_match(int64_t x, int64_t y, int64_t sx, int64_t sy, unsigned predicate) {
    switch (predicate) {
    case 0: return x == y;
    case 1: return sx >= sy;
    case 2: return x >= y;
    case 3: return sx > sy;
    case 4: return x > y;
    case 5: return sx <= sy;
    case 6: return x <= y;
    case 7: return sx < sy;
    case 8: return x < y;
    default: return x != y;
    }
}
std::array<uint32_t, simd_comparison_fields> simd_comparison_row(const std::array<uint32_t, 3>& input) {
    const auto [a, b, c] = input;
    std::array<uint32_t, simd_comparison_fields> row{};
    constexpr unsigned reductions[]{0, 9, 7, 2};
    for (unsigned width = 0; width < 2; ++width) {
        const unsigned bits = width == 0 ? 16 : 8;
        const uint32_t mask = (1u << bits) - 1;
        for (unsigned op = 0; op < 4; ++op) row[40 + op * 2 + width] = c;
        for (unsigned lane = 0; lane < 32 / bits; ++lane) {
            const auto x = simd_comparison_lane(a, lane, bits, false), y = simd_comparison_lane(b, lane, bits, false);
            const auto sx = simd_comparison_lane(a, lane, bits, true), sy = simd_comparison_lane(b, lane, bits, true);
            for (unsigned predicate = 0; predicate < 10; ++predicate) {
                const bool match = simd_comparison_match(x, y, sx, sy, predicate);
                row[predicate * 2 + width] |= uint32_t(match) << (lane * bits);
                row[20 + predicate * 2 + width] |= (match ? mask : 0) << (lane * bits);
            }
            for (unsigned op = 0; op < 4; ++op)
                row[40 + op * 2 + width] += simd_comparison_match(x, y, sx, sy, reductions[op]);
        }
    }
    return row;
}
void execute_integer_simd_comparison(const loaded_program& program) {
    const auto kernel = program.function("simd_comparison");
    constexpr uint32_t guard = 0x8bc7a359;
    std::vector<std::array<uint32_t, 3>> corpus;
    constexpr uint32_t accumulators[]{0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe};
    // Every ordered byte pair in every position, with different neighbors.
    for (unsigned a = 0; a < 256; ++a) for (unsigned b = 0; b < 256; ++b) {
        const uint32_t x = a | ((a ^ 0x80) << 8) | ((a ^ 0x55) << 16) | ((255 - a) << 24);
        const uint32_t y = b | ((b ^ 0x80) << 8) | ((255 - b) << 16) | ((b ^ 0xaa) << 24);
        corpus.push_back({x, y, accumulators[(a + 3 * b) % 6]});
    }
    constexpr uint32_t halves[]{0, 1, 2, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff};
    // Exhaust halfword inputs against representative signed/unsigned boundaries.
    for (uint32_t a = 0; a < 65536; ++a) {
        const uint32_t b = halves[a % std::size(halves)], high = halves[(a / std::size(halves)) % std::size(halves)];
        corpus.push_back({a | ((65535 - a) << 16), b | (high << 16), accumulators[a % 6]});
    }
    for (auto a : halves) for (auto b : halves) for (auto c : accumulators)
        corpus.push_back({a | ((a ^ 0x8000) << 16), b | ((65535 - b) << 16), c});
    uint32_t state = 0x23a59c71;
    auto random = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (unsigned i = 0; i < 513; ++i) { const auto a = random(), b = random(), c = random(); corpus.push_back({a, b, c}); }
    std::vector<std::array<uint32_t, simd_comparison_fields>> oracle;
    for (const auto& input : corpus) oracle.push_back(simd_comparison_row(input));
    for (unsigned threads : {1u, 13u, 32u, 33u, 96u}) {
        const unsigned blocks = (unsigned(corpus.size()) + threads - 1) / threads + 1;
        const unsigned grid_x = std::min(blocks, 251u), grid_y = (blocks + grid_x - 1) / grid_x;
        const unsigned total = grid_x * grid_y * threads;
        std::vector<uint32_t> input(total * 3 + 4, guard), preserved(input.size());
        std::vector<uint32_t> complete(total * simd_comparison_fields + 4, guard), expected(complete.size()), actual(complete.size());
        for (unsigned i = 0; i < total; ++i) {
            std::copy(corpus[i % corpus.size()].begin(), corpus[i % corpus.size()].end(), input.begin() + 2 + i * 3);
            std::copy(oracle[i % oracle.size()].begin(), oracle[i % oracle.size()].end(), complete.begin() + 2 + i * simd_comparison_fields);
        }
        device_buffer source(input.size() * 4), destination(complete.size() * 4);
        auto in = source.value + 8, out = destination.value + 8;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, grid_x, grid_y, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            std::copy_n(complete.begin() + 2, std::min(total, count) * simd_comparison_fields, expected.begin() + 2);
            expect_words(actual, expected, "SIMD comparison threads=" + std::to_string(threads) + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "SIMD comparison source preservation");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_integer_simd_comparison);
}
