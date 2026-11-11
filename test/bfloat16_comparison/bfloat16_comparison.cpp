#include "test_fixtures.hpp"
#include <algorithm>

namespace cudavm {
bool bfloat16_comparison_nan(uint32_t bits) { return (bits & 0x7f80) == 0x7f80 && (bits & 127); }
std::array<uint32_t, 20> bfloat16_comparison_oracle(uint32_t a, uint32_t b) {
    const bool an = bfloat16_comparison_nan(a), bn = bfloat16_comparison_nan(b), unordered = an || bn;
    const bool az = !(a & 0x7fff), bz = !(b & 0x7fff);
    const bool equal = !unordered && (a == b || (az && bz));
    const bool less = !unordered && !equal && ((a ^ b) & 0x8000 ? (a & 0x8000) != 0 : (a & 0x8000) ? a > b : a < b);
    const bool greater = !unordered && !equal && !less;
    uint32_t minimum, maximum;
    if (an && bn) minimum = maximum = 0x7fff;
    else if (an) minimum = maximum = b;
    else if (bn) minimum = maximum = a;
    else if (az && bz) { minimum = a | b; maximum = a & b; }
    else { minimum = less ? a : b; maximum = greater ? a : b; }
    const uint32_t infinite = (a & 0x7fff) == 0x7f80 ? (a & 0x8000 ? UINT32_MAX : 1u) : 0;
    return {uint32_t(equal), uint32_t(!unordered && !equal), uint32_t(less), uint32_t(equal || less), uint32_t(greater), uint32_t(equal || greater),
        uint32_t(unordered || equal), uint32_t(unordered || !equal), uint32_t(unordered || less), uint32_t(unordered || equal || less),
        uint32_t(unordered || greater), uint32_t(unordered || equal || greater), uint32_t(an), infinite,
        minimum, maximum, unordered ? 0x7fffu : minimum, unordered ? 0x7fffu : maximum, a & 0x7fffu, a ^ 0x8000u};
}
std::vector<uint32_t> bfloat16_comparison_inputs() {
    constexpr std::array<uint32_t, 25> edges{0, 0x8000, 1, 0x8001, 2, 0x8002, 0x007f, 0x807f, 0x0080, 0x8080,
        0x3f7f, 0x3f80, 0x3f81, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fc0, 0xffc0, 0x7fff, 0xffff, 0x3eab};
    std::vector<uint32_t> result;
    auto append = [&](uint32_t a, uint32_t b) { result.push_back(a | (b << 16)); };
    for (auto a : edges) for (auto b : edges) append(a, b);
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        append(bits, bits); append(bits, bits ^ 0x8000); append(bits, (bits + 1) & 0xffff);
        append(bits, edges[bits % edges.size()]); append(edges[(bits / 17) % edges.size()], bits);
    }
    return result;
}
void execute_bfloat16_comparison(const loaded_program& program) {
    constexpr unsigned fields = 20, guards = 17, threads = 128;
    constexpr uint32_t guard = 0xc739a15b;
    const auto values = bfloat16_comparison_inputs();
    const unsigned capacity = unsigned(values.size()), blocks = (capacity + threads - 1) / threads;
    std::vector<uint32_t> input(capacity + guards * 2, guard), preserved(input.size());
    std::copy(values.begin(), values.end(), input.begin() + guards);
    std::vector<uint32_t> initial((capacity + 3) * fields + guards * 2, guard), actual(initial.size());
    device_buffer source(input.size() * 4), output(initial.size() * 4);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    for (unsigned count : {0u, 33u, capacity}) {
        TEST_CUDA(cuMemcpyHtoD(output.value, initial.data(), initial.size() * 4));
        void* arguments[]{&to, &from, &count};
        TEST_CUDA(cuLaunchKernel(program.function("compare_bfloat16"), blocks, 1, 1, threads, 1, 1, 0, nullptr, arguments, nullptr));
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        for (size_t i = 0; i < actual.size(); ++i) if (i < guards || i >= guards + size_t(count) * fields)
            if (actual[i] != guard) throw std::runtime_error("bfloat16 comparison output guards/inactive records");
        for (unsigned record = 0; record < count; ++record) {
            const auto a = values[record] & 0xffff, b = values[record] >> 16;
            const auto expected = bfloat16_comparison_oracle(a, b);
            for (unsigned field = 0; field < fields; ++field) {
                const auto got = actual[guards + size_t(record) * fields + field];
                if (field >= 18 && bfloat16_comparison_nan(a)) {
                    require(!(got & 0xffff0000) && bfloat16_comparison_nan(got), "bfloat16 absolute/negation retains NaN class"); continue;
                }
                if (got != expected[field]) throw std::runtime_error("bfloat16 comparison a=" + std::to_string(a) + " b=" + std::to_string(b)
                    + " field=" + std::to_string(field) + " got=" + std::to_string(got) + " expected=" + std::to_string(expected[field]));
            }
        }
        TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "bfloat16 comparison preserves complete input and guards");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_bfloat16_comparison);
}
