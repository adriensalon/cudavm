#include "test_fixtures.hpp"

namespace cudavm {
std::string_view wmma_profile = "all";
int wmma_a(unsigned tile, unsigned row, unsigned column) { return int((tile * 3 + row * 5 + column * 7) % 9) - 4; }
int wmma_b(unsigned tile, unsigned row, unsigned column) { return int((tile * 7 + row * 3 + column * 5) % 9) - 4; }
int wmma_c(unsigned tile, unsigned row, unsigned column) { return int((tile + row * 3 + column) % 11) - 5; }
uint16_t wmma_half(int value) {
    constexpr uint16_t magnitude[]{0, 0x3c00, 0x4000, 0x4200, 0x4400};
    return uint16_t(magnitude[value < 0 ? -value : value] | (value < 0 ? 0x8000 : 0));
}
size_t wmma_index(unsigned row, unsigned column, unsigned stride, bool column_major) {
    return column_major ? column * stride + row : row * stride + column;
}
void wmma_case(const loaded_program& program, unsigned mode, unsigned control, unsigned rounds, unsigned warps, bool fractional = false) {
    constexpr unsigned blocks = 2, half_guards = 16, float_guards = 8;
    const unsigned tiles = blocks * warps;
    unsigned half_stride = warps == 1 ? 24 : 32, float_stride = warps == 1 ? 20 : 24;
    constexpr uint16_t half_guard = 0x59a3;
    constexpr uint32_t float_guard = 0x4b5917a3;
    std::vector<uint16_t> a(half_guards * 2 + (tiles + 1) * half_stride * 16, half_guard), b(a.size()), half_copy(a.size());
    std::fill(b.begin(), b.end(), half_guard);
    std::vector<uint32_t> c(float_guards * 2 + (tiles + 1) * float_stride * 16, float_guard), expected(c), actual(c.size());
    for (unsigned tile = 0; tile < tiles; ++tile) for (unsigned row = 0; row < 16; ++row) for (unsigned col = 0; col < 16; ++col) {
        a[half_guards + tile * half_stride * 16 + wmma_index(row, col, half_stride, mode & 2)] = wmma_half(wmma_a(tile, row, col));
        b[half_guards + tile * half_stride * 16 + wmma_index(row, col, half_stride, mode & 1)] = wmma_half(wmma_b(tile, row, col));
        const auto index = float_guards + tile * float_stride * 16 + wmma_index(row, col, float_stride, control & 1);
        float initial_value = float(wmma_c(tile, row, col));
        if (fractional) initial_value += 0x1p-10f;
        c[index] = std::bit_cast<uint32_t>(initial_value);
        int product = 0;
        for (unsigned k = 0; k < 16; ++k) product += wmma_a(tile, row, k) * wmma_b(tile, k, col);
        const int value = -(control & 2 ? 2 : wmma_c(tile, row, col)) + int(rounds) * product;
        float result = float(value);
        if (fractional && !(control & 2)) result -= 0x1p-10f;
        expected[index] = !fractional && !rounds && !(control & 2) && wmma_c(tile, row, col) == 0
            ? 0x80000000u : std::bit_cast<uint32_t>(result);
    }
    device_buffer left(a.size() * 2), right(b.size() * 2), initial(c.size() * 4), destination(expected.size() * 4);
    std::vector<uint32_t> empty(expected.size(), float_guard), preserved(c.size());
    CONFORMANCE_CUDA(cuMemcpyHtoD(left.value, a.data(), a.size() * 2));
    CONFORMANCE_CUDA(cuMemcpyHtoD(right.value, b.data(), b.size() * 2));
    CONFORMANCE_CUDA(cuMemcpyHtoD(initial.value, c.data(), c.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, empty.data(), empty.size() * 4));
    auto ap = left.value + half_guards * 2, bp = right.value + half_guards * 2;
    auto cp = initial.value + float_guards * 4, out = destination.value + float_guards * 4;
    void* arguments[]{&out, &ap, &bp, &cp, &half_stride, &float_stride, &control, &rounds};
    constexpr const char* kernels[]{"wmma_rr", "wmma_rc", "wmma_cr", "wmma_cc"};
    CONFORMANCE_CUDA(cuLaunchKernel(program.function(kernels[mode]), blocks, 1, 1, 16, warps * 2, 1, 0, nullptr, arguments, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
    // WMMA accumulation rounding is unspecified; either sign of an exact zero
    // is valid. With no multiply, fragment negation prescribes the sign exactly.
    if (rounds) for (size_t i = 0; i < actual.size(); ++i)
        if (expected[i] == 0 && !(actual[i] & 0x7fffffffu)) actual[i] = 0;
    expect_words(actual, expected, "WMMA exact integer matrix results and padding mode=" + std::to_string(mode)
        + " control=" + std::to_string(control) + " rounds=" + std::to_string(rounds) + " warps=" + std::to_string(warps)
        + (fractional ? " fractional accumulator" : ""));
    for (auto pair : {std::pair{left.value, &a}, std::pair{right.value, &b}}) {
        CONFORMANCE_CUDA(cuMemcpyDtoH(half_copy.data(), pair.first, half_copy.size() * 2));
        require(half_copy == *pair.second, "WMMA half inputs, row padding and guards are preserved");
    }
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), initial.value, preserved.size() * 4));
    expect_words(preserved, c, "WMMA accumulator source, padding and guards are preserved");
}
void wmma_fill_case(const loaded_program& program) {
    constexpr unsigned guards = 8, stride = 20;
    constexpr uint32_t guard = 0x4b5917a3;
    std::vector<uint32_t> initial(guards * 2 + stride * 16, guard), actual(initial.size());
    device_buffer destination(initial.size() * 4);
    auto output = destination.value + guards * 4;
    for (unsigned bits : {0x3fa00000u, 0x7f800000u, 0xff800000u, 0x7fc12345u}) for (unsigned saturate : {0u, 1u}) {
        CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
        void* arguments[]{&output, &bits, &saturate};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("wmma_fill"), 1, 1, 1, 32, 1, 1, 0, nullptr, arguments, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (i < guards || i >= guards + stride * 16 || (i - guards) % stride >= 16)
                require(actual[i] == guard, "WMMA filled fragments preserve output padding and guards");
            else {
                const bool nan = bits == 0x7fc12345u;
                uint32_t expected = bits;
                if (saturate && nan) expected = 0;
                else if (saturate && (bits & 0x7fffffffu) == 0x7f800000u) expected = (bits & 0x80000000u) | 0x7f7fffffu;
                require(nan && !saturate ? (actual[i] & 0x7fffffffu) > 0x7f800000u : actual[i] == expected,
                    "WMMA filled half operands and float accumulator saturation control");
            }
        }
    }
}
void execute_wmma_operations(const loaded_program& program) {
    bool selected = false;
    constexpr const char* profiles[]{"rr", "rc", "cr", "cc"};
    for (unsigned mode = 0; mode < 4; ++mode) if (wmma_profile == "all" || wmma_profile == profiles[mode]) {
        selected = true;
        for (unsigned control : {0u, 1u, 2u, 3u, 4u, 7u}) for (unsigned rounds : {0u, 1u, 3u})
            for (unsigned warps : {1u, 2u}) wmma_case(program, mode, control, rounds, warps);
    }
    if (wmma_profile == "all" || wmma_profile == "fill") { selected = true; wmma_fill_case(program); }
    if (wmma_profile == "all" || wmma_profile == "precision") {
        selected = true;
        for (unsigned control : {0u, 1u}) for (unsigned rounds : {1u, 3u}) for (unsigned warps : {1u, 2u})
            wmma_case(program, 0, control, rounds, warps, true);
    }
    require(selected, "unknown WMMA profile");
}
}
int main(int argc, char** argv) {
    if (argc == 4) { cudavm::wmma_profile = argv[3]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_wmma_operations);
}
