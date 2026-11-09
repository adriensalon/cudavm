#include "test_fixtures.hpp"

namespace cudavm {
void tensor_memory_case(const loaded_program& program, std::array<unsigned, 3> shape,
    unsigned count, unsigned columns, unsigned rounds) {
    constexpr unsigned blocks = 3, capacity = blocks * 128, guards = 16;
    constexpr uint32_t guard = 0x8c5917a3;
    std::vector<uint32_t> input(guards * 2 + capacity, guard), preserved(input.size());
    for (unsigned i = 0; i < capacity; ++i) input[guards + i] = 0x93b17d5au ^ (i * 0x10203u);
    std::vector<uint32_t> initial(guards * 2 + (capacity + 1) * 16, guard), expected(initial), actual(initial.size());
    for (unsigned i = 0; i < count; ++i) for (unsigned round = 0; round < rounds; ++round)
        for (unsigned slot = 0; slot < 3; ++slot) {
            const unsigned column = slot == 0 ? 0 : slot == 1 ? columns / 2 : columns - 1;
            expected[guards + i * 16 + round * 4 + slot] = input[guards + i]
                ^ ((round + 1) * 0x5917ab3du) ^ (column * 0x1020304u);
        }
    device_buffer source(input.size() * 4), destination(initial.size() * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
    auto in = source.value + guards * 4, out = destination.value + guards * 4;
    void* arguments[]{&out, &in, &count, &columns, &rounds};
    CONFORMANCE_CUDA(cuLaunchKernel(program.function("tensor_memory_exchange"), blocks, 1, 1,
        shape[0], shape[1], shape[2], 0, nullptr, arguments, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
    expect_words(actual, expected, "TMEM allocation/store/load completion, independent rows/columns/CTAs, guards and inactive records");
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, input.size() * 4));
    expect_words(preserved, input, "TMEM source values and allocation guards are preserved");
}
void execute_tensor_memory(const loaded_program& program) {
    for (auto shape : {std::array<unsigned, 3>{128, 1, 1}, {16, 8, 1}, {4, 4, 8}})
        for (unsigned count : {0u, 1u, 383u, 384u}) for (unsigned columns : {32u, 64u, 128u}) for (unsigned rounds : {0u, 1u, 3u})
            tensor_memory_case(program, shape, count, columns, rounds);
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_tensor_memory);
}
