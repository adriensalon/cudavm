#include "test_fixtures.hpp"

namespace cudavm {
void cache_case(const loaded_program& program, unsigned x, unsigned y, unsigned count) {
    constexpr unsigned blocks = 3, guards = 16;
    constexpr uint32_t guard = 0x8c5917a3;
    const unsigned capacity = blocks * x * y;
    std::vector<uint32_t> input(guards * 2 + capacity, guard), preserved(input.size());
    for (unsigned i = 0; i < capacity; ++i) input[guards + i] = 0x93b17d5au ^ (i * 0x10203u);
    std::vector<uint32_t> initial(guards * 2 + (capacity + 1) * 16, guard), expected(initial), actual(initial.size());
    for (unsigned i = 0; i < count; ++i) {
        const auto base = guards + i * 16;
        for (unsigned mode = 0; mode < 5; ++mode) expected[base + mode] = input[guards + i];
        for (unsigned mode = 0; mode < 4; ++mode)
            expected[base + 5 + mode] = expected[base + 9 + mode] = input[guards + i] ^ (0x1020304u * (mode + 1));
        expected[base + 13] = input[guards + i] + 0x17u;
    }
    device_buffer source(input.size() * 4), destination(initial.size() * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
    auto in = source.value + guards * 4, out = destination.value + guards * 4;
    void* arguments[]{&out, &in, &count};
    CONFORMANCE_CUDA(cuLaunchKernel(program.function("cache_access"), blocks, 1, 1, x, y, 1, 0, nullptr, arguments, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
    expect_words(actual, expected, "cache load/store values, same-thread readback, prefetch, padding and inactive records");
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, input.size() * 4));
    expect_words(preserved, input, "cache controls preserve source values and guards");
}
void execute_cache_memory_controls(const loaded_program& program) {
    for (auto geometry : {std::array<unsigned, 2>{1, 1}, {11, 3}, {16, 4}}) {
        const auto capacity = geometry[0] * geometry[1] * 3;
        for (unsigned count : {0u, 1u, capacity - 1, capacity}) cache_case(program, geometry[0], geometry[1], count);
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_cache_memory_controls);
}
