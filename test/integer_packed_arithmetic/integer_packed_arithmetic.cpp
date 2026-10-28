#include "fixtures.hpp"
#include <algorithm>
#include <bit>
#include <limits>

namespace cudavm {
constexpr unsigned packed_fields = 48;
int64_t packed_lane(uint32_t word, unsigned lane, unsigned bits, bool sign) {
    const uint32_t value = (word >> (lane * bits)) & ((1u << bits) - 1);
    return sign && (value & (1u << (bits - 1))) ? int64_t(value) - (int64_t(1) << bits) : value;
}
int64_t packed_dot(uint32_t a, uint32_t b, unsigned kind, bool sign_a, bool sign_b) {
    int64_t sum = 0;
    const unsigned lanes = kind == 0 ? 4 : 2, width = kind == 0 ? 8 : 16;
    for (unsigned lane = 0; lane < lanes; ++lane)
        sum += packed_lane(a, lane, width, sign_a) * packed_lane(b, lane + (kind == 2 ? 2 : 0), 8, sign_b);
    return sum;
}
int64_t packed_floor_half(int64_t value) { return value / 2 - (value < 0 && value % 2 != 0); }
int64_t packed_ceil_half(int64_t value) { return value / 2 + (value > 0 && value % 2 != 0); }
uint32_t packed_lanes_expected(uint32_t a, uint32_t b, unsigned bits, unsigned operation) {
    uint32_t result = 0;
    const int64_t lower = -(int64_t(1) << (bits - 1)), upper = -lower - 1;
    const uint32_t mask = (1u << bits) - 1;
    for (unsigned lane = 0; lane < 32 / bits; ++lane) {
        const int64_t sx = packed_lane(a, lane, bits, true), sy = packed_lane(b, lane, bits, true);
        const int64_t ux = packed_lane(a, lane, bits, false), uy = packed_lane(b, lane, bits, false);
        int64_t value = 0;
        switch (operation) {
        case 0: value = std::clamp(sx + sy, lower, upper); break;
        case 1: value = std::min(ux + uy, int64_t(mask)); break;
        case 2: value = std::clamp(sx - sy, lower, upper); break;
        case 3: value = std::max(ux - uy, int64_t(0)); break;
        case 4: value = sx + sy; value = value < 0 ? packed_floor_half(value) : packed_ceil_half(value); break;
        case 5: value = packed_ceil_half(ux + uy); break;
        case 6: value = (ux + uy) / 2; break;
        case 7: value = std::min(sx, sy); break;
        case 8: value = std::min(ux, uy); break;
        case 9: value = std::max(sx, sy); break;
        case 10: value = std::max(ux, uy); break;
        case 11: value = std::min(sx < 0 ? -sx : sx, upper); break;
        case 12: value = std::clamp(-sx, lower, upper); break;
        }
        result |= (uint32_t(value) & mask) << (lane * bits);
    }
    return result;
}
std::array<uint32_t, 4> packed_input(uint32_t a, uint32_t b, uint32_t c) {
    int64_t minimum = 0, maximum = 0;
    for (unsigned kind = 0; kind < 3; ++kind) for (auto signs : {1u, 2u, 3u}) {
        const auto delta = packed_dot(a, b, kind, (signs & 1) != 0, (signs & 2) != 0);
        minimum = std::min(minimum, delta); maximum = std::max(maximum, delta);
    }
    const auto safe = std::clamp(int64_t(std::bit_cast<int32_t>(c)),
        int64_t(std::numeric_limits<int32_t>::min()) - minimum,
        int64_t(std::numeric_limits<int32_t>::max()) - maximum);
    return {a, b, c, uint32_t(safe)};
}
std::array<uint32_t, packed_fields> packed_expected(const std::array<uint32_t, 4>& input) {
    const auto [a, b, c, signed_c] = input;
    const int64_t accumulator = std::bit_cast<int32_t>(signed_c);
    std::array<uint32_t, packed_fields> row{};
    for (unsigned kind = 0; kind < 3; ++kind) {
        row[kind * 2] = uint32_t(accumulator + packed_dot(a, b, kind, true, true));
        row[kind * 2 + 1] = c + uint32_t(packed_dot(a, b, kind, false, false));
        row[6 + kind * 2] = row[kind * 2]; row[7 + kind * 2] = row[kind * 2 + 1];
        row[42 + kind * 2] = uint32_t(accumulator + packed_dot(a, b, kind, true, false));
        row[43 + kind * 2] = uint32_t(accumulator + packed_dot(a, b, kind, false, true));
    }
    const int64_t sum = int64_t(std::bit_cast<int32_t>(a)) + std::bit_cast<int32_t>(b);
    row[12] = uint32_t(packed_floor_half(sum)); row[13] = uint32_t(packed_ceil_half(sum));
    row[14] = uint32_t((uint64_t(a) + b) / 2); row[15] = uint32_t((uint64_t(a) + b + 1) / 2);
    for (unsigned operation = 0; operation < 13; ++operation)
        for (unsigned size = 0; size < 2; ++size)
            row[16 + operation * 2 + size] = packed_lanes_expected(a, b, size == 0 ? 16 : 8, operation);
    return row;
}
void execute_integer_packed_arithmetic(const loaded_program& program) {
    const auto kernel = program.function("packed_values");
    constexpr uint32_t guard = 0x8bc7a359;
    std::vector<std::array<uint32_t, 4>> corpus;
    // Exhaust every ordered byte pair in all byte positions, with distinct
    // neighbors. Halfword edges and random words exercise wider correlations.
    for (unsigned a = 0; a < 256; ++a) for (unsigned b = 0; b < 256; ++b) {
        const uint32_t x = a | ((a ^ 0x80) << 8) | ((a ^ 0x55) << 16) | ((255 - a) << 24);
        const uint32_t y = b | ((b ^ 0x80) << 8) | ((255 - b) << 16) | ((b ^ 0xaa) << 24);
        const uint32_t accumulators[]{0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe};
        corpus.push_back(packed_input(x, y, accumulators[(a + b * 3) % 6]));
    }
    constexpr uint32_t edges[]{0, 1, 0x7f, 0x80, 0xff, 0x7fff, 0x8000, 0xffff,
        0x7fff7fff, 0x80008000, 0x7fffffff, 0x80000000, 0xffffffff, 0xffff0001,
        0x0001ffff, 0x7f807f80, 0x807f807f, 0x01234567, 0x89abcdef};
    for (auto a : edges) for (auto b : edges) for (auto c : {0u, 0x7fffffffu, 0x80000000u, 0xffffffffu})
        corpus.push_back(packed_input(a, b, c));
    uint32_t state = 0x23a59c71;
    auto random = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (unsigned i = 0; i < 513; ++i) { const auto a = random(), b = random(), c = random(); corpus.push_back(packed_input(a, b, c)); }
    std::vector<std::array<uint32_t, packed_fields>> oracle;
    oracle.reserve(corpus.size());
    for (const auto& input : corpus) oracle.push_back(packed_expected(input));
    for (unsigned threads : {1u, 13u, 32u, 33u, 96u}) {
        const unsigned blocks = (unsigned(corpus.size()) + threads - 1) / threads + 1;
        const unsigned grid_x = std::min(blocks, 251u), grid_y = (blocks + grid_x - 1) / grid_x;
        const unsigned total = grid_x * grid_y * threads;
        std::vector<uint32_t> input(total * 4 + 4, guard), preserved(input.size());
        std::vector<uint32_t> complete(total * packed_fields + 4, guard), expected(complete.size()), actual(complete.size());
        for (unsigned i = 0; i < total; ++i) {
            std::copy(corpus[i % corpus.size()].begin(), corpus[i % corpus.size()].end(), input.begin() + 2 + i * 4);
            std::copy(oracle[i % oracle.size()].begin(), oracle[i % oracle.size()].end(), complete.begin() + 2 + i * packed_fields);
        }
        device_buffer source(input.size() * 4), destination(complete.size() * 4);
        auto in = source.value + 8, out = destination.value + 8;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
            void* args[]{&out, &in, &count};
            try {
                CONFORMANCE_CUDA(cuLaunchKernel(kernel, grid_x, grid_y, 1, threads, 1, 1, 0, nullptr, args, nullptr));
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            } catch (const std::exception& error) {
                throw std::runtime_error("packed launch threads=" + std::to_string(threads) + " count="
                    + std::to_string(count) + ": " + error.what());
            }
            std::copy_n(complete.begin() + 2, std::min(total, count) * packed_fields, expected.begin() + 2);
            expect_words(actual, expected, "packed arithmetic threads=" + std::to_string(threads) + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "packed arithmetic source preservation");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_integer_packed_arithmetic);
}
