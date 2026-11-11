#include "test_fixtures.hpp"

namespace cudavm {
std::array<uint32_t, 13> language_oracle(uint32_t value, unsigned index) {
    return {value + 31u, value * 7u + 11u, value ^ 0x193759bdu,
        (value & 0x7fffffffu) + 43u, value * 19u, 0xabcdef01u, 49u,
        value + 52u, 52u, value ^ 0x76543210u, value * 23u + index + 28u,
        index % 3 == 0 ? value + 59u : index % 3 == 1 ? ~value : value * 61u, value + 53u};
}
void execute_cuda_language(const loaded_program& program) {
    constexpr unsigned capacity = 259, fields = 13, guards = 11;
    constexpr uint32_t guard = 0x95c3a71b;
    constexpr std::array<uint32_t, 8> edges{0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe, 0x13579bdf, 0xabcdef01};
    std::vector<uint32_t> input(capacity + guards * 2, guard), preserved(input.size());
    std::vector<uint32_t> expected((capacity + 3) * fields + guards * 2, guard), actual(expected.size());
    device_buffer source(input.size() * 4), output(expected.size() * 4);
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    for (unsigned phase = 0; phase < 2; ++phase) {
        for (unsigned i = 0; i < capacity; ++i)
            input[guards + i] = i < edges.size() ? edges[(i + phase) % edges.size()] : (i * 0x193759bdu) ^ (phase * 0xabcdef01u);
        TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        for (unsigned threads : {32u, 67u}) for (unsigned count : {0u, 1u, 33u, capacity}) {
            std::fill(expected.begin(), expected.end(), guard);
            TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            void* arguments[]{&to, &from, &count};
            TEST_CUDA(cuLaunchKernel(program.function("language_results"), (capacity + threads - 1) / threads, 1, 1,
                threads, 1, 1, 0, nullptr, arguments, nullptr));
            TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            for (unsigned i = 0; i < count; ++i) {
                const auto row = language_oracle(input[guards + i], i);
                std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
            }
            expect_words(actual, expected, "CUDA C++ qualifier, overload, template and initialization results/guards");
            TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "CUDA C++ preserves complete source and guards");
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_cuda_language);
}
