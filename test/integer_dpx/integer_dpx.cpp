#include "test_fixtures.hpp"

namespace cudavm {
constexpr unsigned integer_dpx_fields = 48;
int64_t integer_dpx_lane(uint32_t word, unsigned lane, unsigned bits, bool sign) {
    const uint64_t modulus = uint64_t(1) << bits;
    const auto value = (uint64_t(word) >> (lane * bits)) & (modulus - 1);
    return sign && value >= modulus / 2 ? int64_t(value) - int64_t(modulus) : int64_t(value);
}
std::array<uint32_t, integer_dpx_fields> integer_dpx_oracle(const std::array<uint32_t, 3>& input) {
    std::array<uint32_t, integer_dpx_fields> result{};
    // Each pair contains a scalar result followed by two independent halfwords.
    // Wide mathematical sums are reduced to a lane before signed comparison.
    for (unsigned pair = 0; pair < 18; ++pair) for (unsigned packed = 0; packed < 2; ++packed) {
        const unsigned bits = packed ? 16 : 32, lanes = packed ? 2 : 1;
        const bool sign = pair < 3 || pair == 4 || pair == 6 || pair == 7
            || pair == 8 || pair == 10 || pair == 12 || pair == 13 || pair == 14 || pair == 16;
        const bool maximum = pair == 0 || pair == 2 || pair == 3 || pair == 6
            || pair == 8 || pair == 9 || pair == 12 || pair == 14 || pair == 15;
        const bool add = pair >= 8 && pair < 14;
        const bool relu = pair < 2 || pair == 6 || pair == 7 || pair == 12 || pair == 13;
        const uint64_t modulus = uint64_t(1) << bits, mask = modulus - 1;
        for (unsigned lane = 0; lane < lanes; ++lane) {
            auto a = integer_dpx_lane(input[0], lane, bits, sign);
            auto b = integer_dpx_lane(input[1], lane, bits, sign);
            const auto c = integer_dpx_lane(input[2], lane, bits, sign);
            const bool predicate = maximum ? a >= b : a <= b;
            if (add) {
                const auto sum = uint64_t(a + b) & mask;
                a = sign && sum >= modulus / 2 ? int64_t(sum) - int64_t(modulus) : int64_t(sum);
                b = c;
            }
            auto value = maximum ? std::max(a, b) : std::min(a, b);
            if (pair >= 2 && pair < 8) value = maximum ? std::max(value, c) : std::min(value, c);
            if (relu) value = std::max(value, int64_t(0));
            result[pair * 2 + packed] |= uint32_t(uint64_t(value) & mask) << (lane * bits);
            if (pair >= 14) {
                // Public packed predicate pointers are ordered high, low.
                const unsigned base = 36 + (pair - 14) * 3;
                result[base + (packed ? 2 - lane : 0)] = uint32_t(predicate);
            }
        }
    }
    return result;
}
std::vector<std::array<uint32_t, 3>> integer_dpx_inputs() {
    constexpr uint32_t edges[]{0, 1, 2, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff,
        0x7ffffffe, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe, 0xffffffff};
    std::vector<std::array<uint32_t, 3>> inputs;
    for (auto a : edges) for (auto b : edges) for (auto c : edges) inputs.push_back({a, b, c});
    constexpr uint32_t halves[]{0, 1, 2, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff};
    for (auto a : halves) for (auto b : halves) for (auto c : halves) {
        inputs.push_back({a | ((a ^ 0x8000) << 16), b | ((0xffff - b) << 16), c | ((c ^ 0x7fff) << 16)});
        inputs.push_back({a | (b << 16), a | (c << 16), c | (a << 16)});
    }
    uint32_t state = 0x37a5c921;
    auto random = [&] { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (unsigned i = 0; i < 4097; ++i) {
        const auto a = random(), b = random(), c = random();
        inputs.push_back({a, b, c});
        if (i % 7 == 0) inputs.push_back({a, a, c});
    }
    return inputs;
}
void execute_integer_dpx(const loaded_program& program) {
    constexpr uint32_t guard = 0xc59b173d;
    constexpr unsigned guards = 11;
    const auto values = integer_dpx_inputs();
    const auto capacity = unsigned(values.size());
    std::vector<uint32_t> input(capacity * 3 + guards * 2, guard), preserved(input.size());
    std::vector<uint32_t> initial((capacity + 3) * integer_dpx_fields + guards * 2, guard), actual(initial.size()), expected(initial);
    for (unsigned i = 0; i < capacity; ++i) std::copy(values[i].begin(), values[i].end(), input.begin() + guards + i * 3);
    device_buffer source(input.size() * 4), output(initial.size() * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    for (unsigned threads : {33u, 128u}) for (unsigned count : {0u, 1u, 37u, capacity}) {
        expected = initial;
        for (unsigned i = 0; i < count; ++i) {
            const auto row = integer_dpx_oracle(values[i]);
            std::copy(row.begin(), row.end(), expected.begin() + guards + i * integer_dpx_fields);
        }
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, initial.data(), initial.size() * 4));
        void* arguments[]{&to, &from, &count};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("integer_dpx_values"), (capacity + threads - 1) / threads,
            1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "DPX threads=" + std::to_string(threads) + " count=" + std::to_string(count));
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "DPX complete source preservation");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_integer_dpx);
}
