#include "fixtures.hpp"
#include <algorithm>
#include <cstring>

namespace cudavm {
void scalar_parameters(const loaded_program& program) {
    const auto scalar = program.function("cubin_scalar");
    const auto branch = program.function("cubin_branch");
    size_t offset = 0, size = 0;
    CONFORMANCE_CUDA(cuFuncGetParamInfo(scalar, 2, &offset, &size));
    require(offset == 16 && size == 4, "scalar parameter layout");
    device_buffer input(12), output(12);
    for (uint32_t value : {0u, 6u, 7u, 8u, 0x7fffffffu, 0xfffffffdu, 0xffffffffu}) {
        std::array<uint32_t, 3> words{value, value, value};
        const std::array<uint32_t, 3> guards{0x12345678, 0x23456789, 0x34567890};
        std::array<uint32_t, 3> actual{};
        uint32_t add = 7;
        void* args[]{&output.value, &input.value, &add};
        CONFORMANCE_CUDA(cuMemcpyHtoD(input.value, words.data(), sizeof(words)));
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, guards.data(), sizeof(guards)));
        CONFORMANCE_CUDA(cuLaunchKernel(scalar, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
        auto expected = guards;
        expected[1] = value + add;
        expect_words(actual, expected, "pointer-list scalar launch");

        std::array<char, 20> packed{};
        std::memcpy(packed.data(), &output.value, 8);
        std::memcpy(packed.data() + 8, &input.value, 8);
        std::memcpy(packed.data() + 16, &add, 4);
        size_t packed_size = packed.size();
        void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(),
            CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size, CU_LAUNCH_PARAM_END};
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, guards.data(), sizeof(guards)));
        CONFORMANCE_CUDA(cuLaunchKernel(branch, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, extra));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
        expected = guards;
        expected[0] = value < add ? value * 3u : value + 5u;
        expect_words(actual, expected, "packed scalar launch");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), input.value, sizeof(actual)));
        expect_words(actual, words, "scalar input preservation");
    }
}

void launch_geometry(const loaded_program& program) {
    const auto kernel = program.function("cubin_indexed");
    const std::array<std::array<unsigned, 6>, 4> shapes{{
        {3, 1, 1, 32, 1, 1}, {2, 3, 1, 8, 4, 4},
        {2, 3, 4, 8, 4, 4}, {3, 2, 2, 1, 1, 1}}};
    constexpr uint32_t guard = 0xcafebabe;
    for (const auto& shape : shapes) {
        const unsigned blocks = shape[0] * shape[1] * shape[2];
        const unsigned total = blocks * shape[3] * shape[4] * shape[5];
        std::vector<uint32_t> input(total + 2), actual(total + 2), expected(total + 2, guard);
        for (unsigned i = 0; i < input.size(); ++i) input[i] = 0xfffffff0u + 0x1020305u * i;
        device_buffer in(input.size() * 4), out(actual.size() * 4);
        auto input_pointer = in.value + 4, output_pointer = out.value + 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(in.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 3, total - 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(out.value, expected.data(), expected.size() * 4));
            void* args[]{&output_pointer, &input_pointer, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, shape[0], shape[1], shape[2],
                shape[3], shape[4], shape[5], 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), out.value, actual.size() * 4));
            for (unsigned i = 0; i < std::min(total, count); ++i)
                expected[i + 1] = input[i + 1] + i + blocks;
            expect_words(actual, expected, "3D launch count=" + std::to_string(count)
                + " invocations=" + std::to_string(total));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), in.value, actual.size() * 4));
        expect_words(actual, input, "input preservation");
    }
}

void wide_parameters(const loaded_program& program) {
    const auto kernel = program.function("cubin_wide");
    size_t offset = 0, size = 0;
    CONFORMANCE_CUDA(cuFuncGetParamInfo(kernel, 3, &offset, &size));
    require(offset == 16 && size == 8, "64-bit parameter alignment");
    device_buffer allocation(16);
    CUdeviceptr out = allocation.value + 4;
    constexpr std::array<uint32_t, 6> edges{0, 1, 0x7fffffff, 0x80000000, 0xfffffffd, 0xffffffff};
    constexpr std::array<uint64_t, 5> addends{0, 0xffffffff, 0x100000000,
        0xffffffffffffffff, 0x12345678fedcba98};
    const std::array<uint32_t, 4> guards{0x12345678, 0x23456789, 0x34567890, 0x45678901};
    for (unsigned i = 0; i < edges.size() * edges.size(); ++i) {
        uint32_t left = edges[i / edges.size()], right = edges[i % edges.size()];
        uint64_t add = addends[i % addends.size()];
        void* args[]{&out, &left, &right, &add};
        CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.value, guards.data(), sizeof(guards)));
        CONFORMANCE_CUDA(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
        std::array<uint32_t, 4> actual{};
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocation.value, sizeof(actual)));
        const uint64_t value = uint64_t(left) * right + add;
        auto expected = guards;
        expected[1] = uint32_t(value);
        expected[2] = uint32_t(value >> 32);
        expect_words(actual, expected, "wide multiply/add case=" + std::to_string(i));
    }
}
void execute_kernel_launch(const loaded_program& program) {
    scalar_parameters(program);
    launch_geometry(program);
    wide_parameters(program);
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_kernel_launch);
}
