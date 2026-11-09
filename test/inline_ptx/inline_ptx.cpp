#include "test_fixtures.hpp"

namespace cudavm {
std::array<uint32_t, 13> inline_ptx_oracle(uint32_t value, unsigned index) {
    const uint64_t sum = uint64_t(value) + 0xfffffff0ull;
    const uint64_t wide = (uint64_t(value) << 32 | (value ^ 0xabcdef01u)) + 0x1234567887654321ull;
    return {value + 17u, value ^ 0x193759bdu, uint32_t(sum), uint32_t(sum >> 32), value + (value & 7u) * 11u,
        uint32_t(wide), uint32_t(wide >> 32), (value & 0xffffu) ^ 0x5a39u,
        ((value & 0x807fffffu) | 0x3f000000u) ^ 0x80000000u, value + index,
        value ^ 0x76543210u, (value ^ 0x76543210u) + 1u, ~value};
}
void execute_inline_ptx(const loaded_program& program) {
    constexpr unsigned capacity = 269, fields = 13, guards = 13;
    constexpr uint32_t guard = 0xc93a175b;
    constexpr std::array<uint32_t, 12> edges{0, 1, 7, 8, 15, 16, 17, 0x7fffffff, 0x80000000, 0xffffffef, 0xfffffff0, 0xffffffff};
    std::vector<uint32_t> input(capacity + guards * 2, guard), preserved(input.size());
    std::vector<uint32_t> expected((capacity + 3) * fields + guards * 2, guard), actual(expected.size());
    device_buffer source(input.size() * 4), output(expected.size() * 4);
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    for (unsigned phase = 0; phase < 2; ++phase) {
        for (unsigned i = 0; i < capacity; ++i)
            input[guards + i] = i < edges.size() ? edges[(i + phase) % edges.size()] : (i * 0x7654321u) ^ (phase * 0x193759bdu);
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned threads : {32u, 71u}) for (unsigned count : {0u, 1u, 33u, capacity}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            void* arguments[]{&to, &from, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(program.function("inline_results"), (capacity + threads - 1) / threads, 1, 1,
                threads, 1, 1, 0, nullptr, arguments, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) {
                const auto row = inline_ptx_oracle(input[guards + i], i);
                std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
            }
            expect_words(actual, expected, "inline PTX operands, scope, carry, side effects and guards");
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "inline PTX preserves complete input and guards");
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_inline_ptx);
}
