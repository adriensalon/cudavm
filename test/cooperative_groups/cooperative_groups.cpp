#include "test_fixtures.hpp"
#include <algorithm>

namespace cudavm {
void group_case(const loaded_program& program, bool tiles, unsigned width,
    std::array<unsigned, 3> shape, unsigned rounds, uint32_t seed) {
    constexpr unsigned guards = 4, fields = 20;
    constexpr uint32_t guard = 0xa1b2c3d4;
    const std::array<unsigned, 3> grid = tiles ? std::array<unsigned, 3>{3, 1, 1} : std::array<unsigned, 3>{2, 2, 2};
    unsigned threads = shape[0] * shape[1] * shape[2], blocks = grid[0] * grid[1] * grid[2], total = blocks * threads;
    std::vector<uint32_t> input(guards * 2 + total, guard), preserved(input.size());
    std::vector<uint32_t> expected(guards * 2 + (total * rounds + 2) * fields, guard), actual(expected.size());
    std::vector<uint32_t> scratch(guards * 2 + total, guard), actual_scratch(scratch.size());
    for (unsigned id = 0; id < total; ++id) input[guards + id] = seed ^ (id * 0x9e3779b9u);
    auto value = [&](unsigned id, unsigned round) { return input[guards + id] ^ (round * 0x10001u + id); };
    for (unsigned round = 0; round < rounds; ++round) for (unsigned id = 0; id < total; ++id) {
        unsigned block = id / threads, lane = id % threads;
        auto* row = expected.data() + guards + (round * total + id) * fields;
        if (!tiles) {
            row[0] = threads; row[1] = lane;
            row[2] = block % grid[0]; row[3] = block / grid[0] % grid[1]; row[4] = block / (grid[0] * grid[1]);
            row[5] = lane % shape[0]; row[6] = lane / shape[0] % shape[1]; row[7] = lane / (shape[0] * shape[1]);
            std::copy(shape.begin(), shape.end(), row + 8);
            unsigned peer = block * threads + (lane + round + 1) % threads;
            row[11] = value(peer, round); row[12] = row[11] + 0x12345u;
        } else {
            unsigned rank = lane % width, base = id - rank, mask = 0;
            uint32_t prefix = 0, sum = 0;
            for (unsigned member = 0; member < width; ++member) {
                sum += value(base + member, round);
                if (member <= rank) prefix += value(base + member, round);
                if ((input[guards + base + member] + round) % 3 != 0) mask |= uint32_t(1) << member;
            }
            row[0] = width; row[1] = rank; row[2] = lane / width; row[3] = threads / width;
            row[4] = value(base + (rank + round + 1) % width, round); row[5] = row[4] + 0x12345u;
            row[6] = value(base + width - 1, round); row[7] = value(base + (rank ^ (width - 1)), round);
            row[8] = prefix; row[9] = sum; row[10] = mask != 0;
            row[11] = mask == (UINT32_MAX >> (32 - width)); row[12] = mask;
            row[13] = width; row[14] = rank; row[15] = lane % 32 / 4; row[16] = 8;
            row[17] = lane % 4; row[18] = 4;
        }
        scratch[guards + id] = value(id, round) + 0x12345u;
    }
    device_buffer source(input.size() * 4), output(expected.size() * 4), temporary(scratch.size() * 4);
    auto from = source.value + guards * 4, to = output.value + guards * 4, temp = temporary.value + guards * 4;
    std::vector<uint32_t> initialized(expected.size(), guard), empty(scratch.size(), guard);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    TEST_CUDA(cuMemcpyHtoD(output.value, initialized.data(), initialized.size() * 4));
    TEST_CUDA(cuMemcpyHtoD(temporary.value, empty.data(), empty.size() * 4));
    void* args[]{&to, &from, &temp, &rounds, &width};
    const auto name = tiles ? "cooperative_tile_" + std::to_string(width) : std::string("cooperative_block");
    TEST_CUDA(cuLaunchKernel(program.function(name.c_str()),
        grid[0], grid[1], grid[2], shape[0], shape[1], shape[2], 0, nullptr, args, nullptr));
    TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
    expect_words(actual, expected, tiles ? "tile metadata, publication, shuffle, scan, reduction and guards" : "block geometry, ranks, publication and guards");
    TEST_CUDA(cuMemcpyDtoH(actual_scratch.data(), temporary.value, actual_scratch.size() * 4));
    expect_words(actual_scratch, scratch, "cooperative publication scratch and guards");
    TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
    expect_words(preserved, input, "cooperative source preservation");
}
void execute_cooperative_groups(const loaded_program& program) {
    for (unsigned rounds : {0u, 1u, 3u}) for (uint32_t seed : {0u, 0xffffffffu}) {
        for (auto shape : {std::array<unsigned, 3>{1, 1, 1}, {13, 1, 1}, {4, 3, 2}, {8, 4, 2}})
            group_case(program, false, 0, shape, rounds, seed);
        for (unsigned threads : {32u, 64u, 96u}) for (unsigned width : {1u, 2u, 4u, 8u, 16u, 32u})
            group_case(program, true, width, {threads, 1, 1}, rounds, seed);
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_cooperative_groups);
}
