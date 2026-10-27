#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_device_builtins(const loaded_program& program) {
    const auto kernel = program.function("record_builtins");
    constexpr uint32_t guard = 0xdeadbeef;
    constexpr unsigned fields = 16;
    const std::array<std::array<unsigned, 6>, 5> shapes{{
        {1, 1, 1, 1, 1, 1}, {3, 2, 2, 7, 5, 3}, {2, 3, 2, 8, 4, 3},
        {2, 1, 2, 33, 1, 1}, {2, 2, 1, 8, 4, 4}}};
    for (const auto& shape : shapes) {
        const unsigned threads = shape[3] * shape[4] * shape[5];
        const unsigned total = shape[0] * shape[1] * shape[2] * threads;
        std::vector<uint32_t> expected(total * fields + 4, guard), actual(expected.size());
        device_buffer allocation(expected.size() * sizeof(uint32_t));
        auto out = allocation.value + sizeof(uint32_t);
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(allocation.value, expected.data(), expected.size() * sizeof(uint32_t)));
            void* args[]{&out, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(kernel, shape[0], shape[1], shape[2],
                shape[3], shape[4], shape[5], 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), allocation.value, actual.size() * sizeof(uint32_t)));
            unsigned block = 0, index = 0;
            // Count the x-fastest coordinates independently of device flattening.
            for (unsigned bz = 0; bz < shape[2]; ++bz)
            for (unsigned by = 0; by < shape[1]; ++by)
            for (unsigned bx = 0; bx < shape[0]; ++bx, ++block) {
                unsigned local = 0;
                for (unsigned tz = 0; tz < shape[5]; ++tz)
                for (unsigned ty = 0; ty < shape[4]; ++ty)
                for (unsigned tx = 0; tx < shape[3]; ++tx, ++local, ++index) {
                    if (index >= count) continue;
                    const std::array<uint32_t, fields> values{tx, ty, tz,
                        shape[3], shape[4], shape[5], bx, by, bz,
                        shape[0], shape[1], shape[2], local % 32, 32, local, block};
                    std::copy(values.begin(), values.end(), expected.begin() + 1 + index * fields);
                }
            }
            const auto context = "builtins grid=" + std::to_string(shape[0]) + ","
                + std::to_string(shape[1]) + "," + std::to_string(shape[2])
                + " block=" + std::to_string(shape[3]) + "," + std::to_string(shape[4])
                + "," + std::to_string(shape[5]) + " count=" + std::to_string(count);
            expect_words(actual, expected, context);
        }
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_builtins);
}
