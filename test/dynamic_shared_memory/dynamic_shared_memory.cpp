#include "fixtures.hpp"
#include <algorithm>
#include <limits>

namespace cudavm {
void execute_dynamic_shared_memory(const loaded_program& program) {
    const auto reduce = program.function("dynamic_reduce");
    const auto exchange = program.function("mixed_exchange");
    constexpr uint32_t guard = 0xdeadbeef;
    constexpr unsigned blocks = 3;
    for (unsigned lanes : {1u, 4u, 16u, 45u, 128u}) {
        const unsigned total = blocks * lanes;
        std::vector<uint32_t> input(total + 2, guard), actual(total + 2), expected(total + 2, guard);
        for (unsigned i = 0; i < total; ++i) input[i + 1] = 0xfffffffdu + i * 0x7654321u;
        device_buffer in(input.size() * 4), allocation(actual.size() * 4);
        auto source = in.value + 4, out = allocation.value + 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(in.value, input.data(), input.size() * 4));
        for (unsigned count : {0u, 1u, total / 2 + 1, total - 1, total, total + 7}) {
            void* args[]{&out, &source, &count};
            auto context = " lanes=" + std::to_string(lanes) + " count=" + std::to_string(count);
            for (unsigned padding : {0u, 16u}) {
                std::fill(expected.begin(), expected.end(), guard);
                CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.value, expected.data(), expected.size() * 4));
                CONFORMANCE_CUDA(cuLaunchKernel(exchange, blocks, 1, 1, lanes, 1, 1,
                    lanes * 4 + padding, nullptr, args, nullptr));
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocation.value, actual.size() * 4));
                for (unsigned i = 0; i < std::min(total, count); ++i) {
                    const unsigned block = i / lanes;
                    const unsigned peer = block * lanes + lanes - 1 - i % lanes;
                    expected[i + 1] = (peer < count ? input[peer + 1] : 0u) + block + 7;
                }
                expect_words(actual, expected, "mixed static/dynamic exchange" + context
                    + " padding=" + std::to_string(padding));
            }
            if ((lanes & (lanes - 1)) != 0) continue;
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.value, expected.data(), expected.size() * 4));
            CONFORMANCE_CUDA(cuLaunchKernel(reduce, blocks, 1, 1, lanes, 1, 1,
                lanes * 4, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocation.value, actual.size() * 4));
            for (unsigned block = 0; block < blocks; ++block) {
                uint32_t sum = 0;
                for (unsigned lane = 0; lane < lanes; ++lane) {
                    const unsigned i = block * lanes + lane;
                    if (i < count) sum += input[i + 1];
                }
                expected[block + 1] = sum;
            }
            expect_words(actual, expected, "dynamic block reduction" + context);
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), in.value, actual.size() * 4));
        expect_words(actual, input, "dynamic-shared input preservation");
    }

    // Invalid requests must fail before dispatch and preserve the whole output.
    device_buffer in(4), out(16);
    const std::array<uint32_t, 4> guards{guard, guard, guard, guard};
    const uint32_t input_value = 7;
    CONFORMANCE_CUDA(cuMemcpyHtoD(in.value, &input_value, sizeof(input_value)));
    for (auto kernel : {reduce, exchange}) {
        int limit = 0;
        CONFORMANCE_CUDA(cuFuncGetAttribute(&limit, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel));
        require(limit >= 512 && unsigned(limit) < std::numeric_limits<unsigned>::max(),
            "invalid reported dynamic shared-memory limit");
        unsigned count = 1;
        void* args[]{&out.value, &in.value, &count};
        CONFORMANCE_CUDA(cuMemcpyHtoD(out.value, guards.data(), sizeof(guards)));
        const auto status = cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1,
            unsigned(limit) + 1, nullptr, args, nullptr);
        require(status == CUDA_ERROR_INVALID_VALUE || status == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
            "launch above reported dynamic shared-memory limit was not rejected as a resource error");
        std::array<uint32_t, 4> actual{};
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), out.value, sizeof(actual)));
        expect_words(actual, guards, "over-limit launch output preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_dynamic_shared_memory);
}
