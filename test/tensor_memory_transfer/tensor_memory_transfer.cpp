#include "test_fixtures.hpp"

namespace cudavm {
static_assert(sizeof(CUtensorMap) == 128 && alignof(CUtensorMap) == 128);
void tensor_load_case(const loaded_program& program, unsigned pitch, unsigned threads, unsigned rounds, int x, int y) {
    constexpr unsigned width = 16, height = 8, blocks = 3, guards = 32, max_rounds = 3;
    constexpr uint32_t guard = 0x8c5917a3;
    std::vector<uint32_t> input(pitch * height + guards * 2, guard), preserved(input.size());
    for (unsigned row = 0; row < height; ++row) for (unsigned column = 0; column < width; ++column)
        input[guards + row * pitch + column] = 0x5917c3d2u ^ (row << 20) ^ (column * 0x10001u);
    std::vector<uint32_t> initial(blocks * max_rounds * 96 + guards * 2, guard), expected(initial), actual(initial.size());
    for (unsigned round = 0; round < rounds; ++round) for (unsigned block = 0; block < blocks; ++block)
        for (unsigned i = 0; i < 32; ++i) {
            const auto record = guards + (round * blocks + block) * 96;
            expected[record + i] = 0x9b3d715a; expected[record + 64 + i] = 0xc6e8294f;
            const int source_x = x + int(block) * 4 + int(i % 8), source_y = y + int(round) + int(i / 8);
            expected[record + 32 + i] = source_x >= 0 && source_x < int(width) && source_y >= 0 && source_y < int(height)
                ? input[guards + unsigned(source_y) * pitch + unsigned(source_x)] : 0;
        }
    device_buffer source(input.size() * 4), destination(initial.size() * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
    auto from = source.value + guards * 4, to = destination.value + guards * 4;
    struct { std::array<unsigned char, 128> before; CUtensorMap value; std::array<unsigned char, 128> after; } descriptor;
    descriptor.before.fill(0x93); descriptor.after.fill(0x57); std::memset(&descriptor.value, 0xa5, sizeof(descriptor.value));
    const cuuint64_t dimensions[]{width, height}, strides[]{pitch * 4};
    const cuuint32_t box[]{8, 4}, elements[]{1, 1};
    CONFORMANCE_CUDA(cuTensorMapEncodeTiled(&descriptor.value, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2,
        reinterpret_cast<void*>(static_cast<uintptr_t>(from)), dimensions, strides, box, elements,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    const auto original = descriptor;
    void* arguments[]{&descriptor.value, &to, &rounds, &x, &y};
    CONFORMANCE_CUDA(cuLaunchKernel(program.function("tensor_load"), blocks, 1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
    expect_words(actual, expected, "TMA tile, shared/output guards and inactive phases pitch=" + std::to_string(pitch)
        + " threads=" + std::to_string(threads) + " rounds=" + std::to_string(rounds) + " x=" + std::to_string(x) + " y=" + std::to_string(y));
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
    expect_words(preserved, input, "TMA source, row padding and allocation guards");
    require(std::all_of(descriptor.before.begin(), descriptor.before.end(), [](auto b) { return b == 0x93; })
        && std::all_of(descriptor.after.begin(), descriptor.after.end(), [](auto b) { return b == 0x57; })
        && !std::memcmp(&descriptor, &original, sizeof(descriptor)), "TMA host descriptor guards and parameter preservation");
}
void execute_tensor_memory_transfer(const loaded_program& program) {
    require_capability(native_cuda_version() >= 12000, "TMA requires a native CUDA 12.0 Driver service");
    for (unsigned pitch : {20u, 24u}) for (unsigned threads : {33u, 64u}) for (unsigned rounds : {0u, 1u, 3u})
        for (auto origin : {std::array<int, 2>{0, 0}, {4, 1}, {12, 6}, {-4, -1}})
            tensor_load_case(program, pitch, threads, rounds, origin[0], origin[1]);
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_tensor_memory_transfer);
}
