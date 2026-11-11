#include "test_fixtures.hpp"

namespace cudavm {
void tile_view_case(const loaded_program& program, std::array<unsigned, 3> shape,
    unsigned rows, unsigned columns, unsigned granularity, unsigned mode, unsigned count) {
    constexpr unsigned blocks = 3, slot = 256, guards = 16;
    constexpr uint32_t guard = 0x8c5917a3, dynamic = 0xffffffffu;
    auto round_up = [](unsigned value, unsigned multiple) { return value + (multiple - value % multiple) % multiple; };
    const unsigned row_stride = mode == 0 ? columns : mode == 1 || mode == 3 ? 1 : mode == 2 ? round_up(columns, granularity)
        : mode == 4 ? columns * 2 + 3 : 8;
    const unsigned column_stride = mode == 1 ? rows : mode == 3 ? round_up(rows, granularity) : mode == 4 ? 2 : 1;
    std::vector<uint32_t> input(guards * 2 + blocks * slot, guard), preserved(input.size());
    std::vector<uint32_t> initial(guards * 2 + (blocks + 1) * slot, guard), expected(initial), actual(initial.size());
    for (unsigned block = 0; block < blocks; ++block) {
        const auto base = guards + block * slot;
        for (unsigned row = 0; row < rows; ++row) for (unsigned column = 0; column < columns; ++column) {
            const auto index = base + row * row_stride + column * column_stride;
            input[index] = 0x93b17d5au ^ (block * 0x1000001u + row * 0x831079u + column * 0x7b15u);
            if (row * columns + column < count)
                expected[base + 16 + row * (columns + 3) + column] = input[index] ^ (row * 0x10203u + column * 0x105u);
        }
        const uint32_t metadata[]{2, mode == 5 ? 0u : 2u, rows, columns, row_stride, column_stride,
            mode == 5 ? 3u : dynamic, mode == 5 ? 5u : dynamic,
            mode == 5 ? 8u : mode == 1 || mode == 3 ? 1u : dynamic,
            mode == 1 || mode == 3 || mode == 4 ? dynamic : 1u,
            1, 1, 8, 1, 1, 0x5917ab3d};
        std::copy(std::begin(metadata), std::end(metadata), expected.begin() + base);
    }
    device_buffer source(input.size() * 4), destination(initial.size() * 4);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    TEST_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
    auto in = source.value + guards * 4, out = destination.value + guards * 4;
    void* arguments[]{&out, &in, &rows, &columns, &granularity, &mode, &count};
    TEST_CUDA(cuLaunchKernel(program.function("tile_header_views"), blocks, 1, 1,
        shape[0], shape[1], shape[2], 0, nullptr, arguments, nullptr));
    TEST_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
    expect_words(actual, expected, "CUDA tile header extents, layout, span pointer, mapped data and guards mode=" + std::to_string(mode));
    TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, input.size() * 4));
    expect_words(preserved, input, "CUDA tile header view source, gaps, padding and guards");
}
void execute_tile_operations(const loaded_program& program) {
    for (unsigned mode = 0; mode < 6; ++mode) for (auto dimensions : {std::array<unsigned, 2>{3, 5}, {4, 8}, {7, 11}}) {
        if (mode == 5 && dimensions != std::array<unsigned, 2>{3, 5}) continue;
        const unsigned cells = dimensions[0] * dimensions[1];
        for (auto shape : {std::array<unsigned, 3>{11, 3, 1}, {4, 4, 4}})
            for (unsigned count : {0u, 1u, cells - 1, cells}) for (unsigned granularity : {4u, 8u})
                tile_view_case(program, shape, dimensions[0], dimensions[1], granularity, mode, count);
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_tile_operations);
}
