#include "test_fixtures.hpp"

namespace cudavm {
void architecture_case(const loaded_program& program, std::array<unsigned, 3> shape, unsigned count, unsigned dynamic_bytes) {
    constexpr unsigned blocks = 4, guards = 16;
    constexpr uint32_t guard = 0x8c5917a3;
    const unsigned threads = shape[0] * shape[1] * shape[2], capacity = blocks * threads;
    std::vector<uint32_t> input(guards * 2 + capacity, guard), preserved(input.size());
    for (unsigned i = 0; i < capacity; ++i) input[guards + i] = 0x93b17d5au ^ (i * 0x10203u);
    std::vector<uint32_t> initial(guards * 2 + (capacity + 1) * 16, guard), expected(initial), actual(initial.size());
    const auto kernel = program.function("architecture_state");
    int static_bytes = -1;
    CONFORMANCE_CUDA(cuFuncGetAttribute(&static_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel));
    require(static_bytes >= 64 * 4, "architecture fixture retains its volatile static shared allocation");
    device_buffer source(input.size() * 4), destination(initial.size() * 4);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
    auto in = source.value + guards * 4, out = destination.value + guards * 4;
    void* arguments[]{&out, &in, &count, &dynamic_bytes};
    CONFORMANCE_CUDA(cuLaunchKernel(kernel, 2, 2, 1, shape[0], shape[1], shape[2], dynamic_bytes, nullptr, arguments, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
    for (unsigned i = 0; i < count; ++i) {
        const auto base = guards + i * 16;
        require(actual[base + 1] && actual[base] < actual[base + 1], "native SM identifier lies below the identifier-space bound");
        require(actual[base + 3] && actual[base + 2] < actual[base + 3], "native warp identifier lies below the warp identifier-space bound");
        require(actual[base + 4] == actual[guards + 4] && actual[base + 5] == actual[guards + 5],
            "native grid identity is shared by all active threads/CTAs of the same launch");
        require(actual[base + 6] == dynamic_bytes, "dynamic shared-size register reports the requested launch size");
        require(actual[base + 7] >= unsigned(static_bytes) + dynamic_bytes, "total shared-size register includes static and dynamic allocation");
        // Placement/identity values are constrained above, not predicted from
        // the thread's logical CTA/warp index or an assumed physical SM count.
        for (unsigned field = 0; field < 8; ++field) expected[base + field] = actual[base + field];
        const unsigned block = i / threads, local = i % threads;
        expected[base + 8] = (input[guards + i] ^ (block * 0x10203u + local))
            ^ (dynamic_bytes ? (input[guards + i] & 255u) << 16 : 0u);
    }
    expect_words(actual, expected, "architecture register data, shared accesses, guards, padding and inactive records");
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, input.size() * 4));
    expect_words(preserved, input, "architecture intrinsic source values and guards are preserved");
}
void execute_architecture_intrinsics(const loaded_program& program) {
    for (auto shape : {std::array<unsigned, 3>{1, 1, 1}, {11, 3, 1}, {4, 4, 4}}) {
        const auto capacity = 4 * shape[0] * shape[1] * shape[2];
        for (unsigned count : {0u, 1u, capacity - 1, capacity}) for (unsigned bytes : {0u, 256u, 1024u})
            architecture_case(program, shape, count, bytes);
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_architecture_intrinsics);
}
