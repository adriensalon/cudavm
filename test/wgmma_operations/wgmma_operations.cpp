#include "test_fixtures.hpp"

namespace cudavm {
std::string_view wgmma_profile = "f16";
int wgmma_a(unsigned tile, unsigned row, unsigned column) { return int((tile * 7 + row * 3 + column * 5) % 9) - 4; }
int wgmma_b(unsigned tile, unsigned row, unsigned column) { return int((tile * 3 + row * 7 + column * 2) % 9) - 4; }
int wgmma_c(unsigned tile, unsigned row, unsigned column) { return int((tile + row * 5 + column * 3) % 11) - 5; }
uint16_t wgmma_half(int value) {
    constexpr uint16_t magnitude[]{0, 0x3c00, 0x4000, 0x4200, 0x4400};
    return uint16_t(magnitude[value < 0 ? -value : value] | (value < 0 ? 0x8000 : 0));
}
void wgmma_case(const loaded_program& program, unsigned groups, unsigned rounds, unsigned accumulate, bool fractional) {
    constexpr unsigned blocks = 3, half_guards = 16, float_guards = 8;
    constexpr uint16_t half_guard = 0x59a3;
    constexpr uint32_t float_guard = 0x4b5917a3;
    const unsigned tiles = blocks * groups;
    unsigned half_stride = groups == 1 ? 24 : 32, float_stride = groups == 1 ? 12 : 16;
    std::vector<uint16_t> a(half_guards * 2 + (tiles + 1) * half_stride * 64, half_guard);
    std::vector<uint16_t> b(half_guards * 2 + (tiles + 1) * half_stride * 8, half_guard);
    std::vector<uint32_t> initial(float_guards * 2 + (tiles + 1) * float_stride * 64, float_guard);
    auto expected = initial;
    for (unsigned tile = 0; tile < tiles; ++tile) {
        for (unsigned row = 0; row < 64; ++row) for (unsigned k = 0; k < 16; ++k)
            a[half_guards + tile * half_stride * 64 + row * half_stride + k] = wgmma_half(wgmma_a(tile, row, k));
        for (unsigned k = 0; k < 16; ++k) for (unsigned column = 0; column < 8; ++column)
            b[half_guards + tile * half_stride * 8 + column * half_stride + k] = wgmma_half(wgmma_b(tile, k, column));
        for (unsigned row = 0; row < 64; ++row) for (unsigned column = 0; column < 8; ++column) {
            const auto index = float_guards + tile * float_stride * 64 + row * float_stride + column;
            const float offset = fractional ? 0x1p-10f : 0.0f;
            initial[index] = std::bit_cast<uint32_t>(float(wgmma_c(tile, row, column)) + offset);
            int dot = 0;
            for (unsigned k = 0; k < 16; ++k) dot += wgmma_a(tile, row, k) * wgmma_b(tile, k, column);
            expected[index] = std::bit_cast<uint32_t>(float((!rounds || accumulate ? wgmma_c(tile, row, column) : 0) + int(rounds) * dot)
                + (!rounds || accumulate ? offset : 0.0f));
        }
    }
    device_buffer left(a.size() * 2), right(b.size() * 2), accumulator(initial.size() * 4), destination(initial.size() * 4);
    std::vector<uint32_t> empty(initial.size(), float_guard), actual(initial.size()), preserved(initial.size());
    CONFORMANCE_CUDA(cuMemcpyHtoD(left.value, a.data(), a.size() * 2));
    CONFORMANCE_CUDA(cuMemcpyHtoD(right.value, b.data(), b.size() * 2));
    CONFORMANCE_CUDA(cuMemcpyHtoD(accumulator.value, initial.data(), initial.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, empty.data(), empty.size() * 4));
    auto ap = left.value + half_guards * 2, bp = right.value + half_guards * 2;
    auto cp = accumulator.value + float_guards * 4, out = destination.value + float_guards * 4;
    void* arguments[]{&out, &ap, &bp, &cp, &half_stride, &float_stride, &rounds, &accumulate};
    CONFORMANCE_CUDA(cuLaunchKernel(program.function("wgmma_f16"), blocks, 1, 1, 16, groups * 8, 1, 0, nullptr, arguments, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
    if (rounds) for (size_t i = 0; i < actual.size(); ++i)
        if (!expected[i] && !(actual[i] & 0x7fffffffu)) actual[i] = 0;
    expect_words(actual, expected, "WGMMA exact matrix, layout, completion, padding and guards rounds=" + std::to_string(rounds)
        + " groups=" + std::to_string(groups) + " accumulate=" + std::to_string(accumulate));
    for (auto pair : {std::pair{left.value, &a}, std::pair{right.value, &b}}) {
        std::vector<uint16_t> copy(pair.second->size());
        CONFORMANCE_CUDA(cuMemcpyDtoH(copy.data(), pair.first, copy.size() * 2));
        require(copy == *pair.second, "WGMMA half sources, row padding and guards are preserved");
    }
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), accumulator.value, preserved.size() * 4));
    expect_words(preserved, initial, "WGMMA initial accumulator, padding and guards are preserved");
}
void execute_wgmma_operations(const loaded_program& program) {
    require(wgmma_profile == "f16" || wgmma_profile == "f16_precision", "unknown WGMMA profile");
    for (unsigned groups : {1u, 2u}) for (unsigned rounds : {0u, 1u, 3u}) for (unsigned accumulate : {0u, 1u})
        wgmma_case(program, groups, rounds, accumulate, wgmma_profile == "f16_precision");
}
}
int main(int argc, char** argv) {
    if (argc == 4) { cudavm::wgmma_profile = argv[3]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_wgmma_operations);
}
