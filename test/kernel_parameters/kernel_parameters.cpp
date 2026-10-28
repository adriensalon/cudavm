#include "fixtures.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace cudavm {
struct alignas(16) parameter_record {
    uint32_t tag;
    uint16_t small;
    uint8_t low, high;
    uint64_t wide;
    uint32_t values[3], tail;
};
struct alignas(16) parameter_vector { uint32_t x, y, z, w; };
static_assert(sizeof(parameter_record) == 32 && alignof(parameter_record) == 16);
static_assert(offsetof(parameter_record, small) == 4 && offsetof(parameter_record, wide) == 8);
static_assert(offsetof(parameter_record, values) == 16 && offsetof(parameter_record, tail) == 28);
static_assert(sizeof(parameter_vector) == 16 && alignof(parameter_vector) == 16);

void execute_kernel_parameters(const loaded_program& program) {
    const auto kernel = program.function("parameter_layout");
    constexpr std::array<size_t, 7> offsets{0, 8, 16, 48, 64, 80, 88};
    constexpr std::array<size_t, 7> sizes{8, 1, 32, 2, 16, 8, 4};
    for (size_t index = 0; index < offsets.size(); ++index) {
        size_t offset = 999, size = 999;
        CONFORMANCE_CUDA(cuFuncGetParamInfo(kernel, index, &offset, &size));
        require(offset == offsets[index] && size == sizes[index],
            "aggregate parameter layout index=" + std::to_string(index));
    }
    constexpr uint32_t guard = 0xead13c87;
    constexpr unsigned fields = 17, blocks = 3;
    const std::array<parameter_record, 3> records{{
        {0, 0, 0, 255, 0, {0, 1, 0xffffffffu}, 0x12345678},
        {0xffffffffu, 65535, 128, 1, 0xffffffffffffffffull,
            {0x80000000u, 0x7fffffffu, 0}, 0xfedcba98},
        {0x89abcdef, 0x8123, 0x5a, 0xa5, 0x12345678ffffffffull,
            {0x13579bdf, 0x2468ace0, 0xfffeffff}, 0x98765432}}};
    for (unsigned threads : {1u, 17u, 96u}) {
        const unsigned total = threads * blocks;
        std::vector<uint32_t> expected(total * fields + 2, guard), actual(expected.size());
        device_buffer destination(expected.size() * 4);
        auto out = destination.value + 4;
        for (unsigned variant = 0; variant < records.size(); ++variant) {
            auto record = records[variant];
            uint8_t marker = uint8_t(variant ? 0xffu - variant : 0);
            uint16_t bias = uint16_t(variant ? 65535u - variant : 0);
            parameter_vector vector{variant, 0x80000000u + variant, 0xffffffffu - variant, 0x76543210};
            uint64_t salt = variant == 1 ? 1 : 0xfedcba9876543210ull;
            for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
                std::fill(expected.begin(), expected.end(), guard);
                for (unsigned i = 0; i < std::min(total, count); ++i) {
                    const uint64_t sum = record.wide + salt + i;
                    const std::array<uint32_t, fields> row{marker, uint32_t(record.small) + bias,
                        record.low, record.high, record.tag ^ i, uint32_t(record.wide),
                        uint32_t(record.wide >> 32), record.values[0] + i, record.values[1] ^ i,
                        record.values[2] - i, record.tail, vector.x, vector.y, vector.z, vector.w,
                        uint32_t(sum), uint32_t(sum >> 32)};
                    std::copy(row.begin(), row.end(), expected.begin() + 1 + i * fields);
                }
                void* args[]{&out, &marker, &record, &bias, &vector, &salt, &count};
                // Give padding distinct nonzero bytes; only declared fields may affect results.
                std::array<unsigned char, 92> packed;
                packed.fill(0xb7);
                for (size_t index = 0; index < offsets.size(); ++index)
                    std::memcpy(packed.data() + offsets[index], args[index], sizes[index]);
                size_t packed_size = packed.size();
                void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(),
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size, CU_LAUNCH_PARAM_END};
                for (bool use_packed : {false, true}) {
                    std::vector<uint32_t> initial(expected.size(), guard);
                    CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
                    CONFORMANCE_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0,
                        nullptr, use_packed ? nullptr : args, use_packed ? extra : nullptr));
                    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
                    expect_words(actual, expected, "kernel parameters threads=" + std::to_string(threads)
                        + " variant=" + std::to_string(variant) + " count=" + std::to_string(count)
                        + (use_packed ? " packed" : " pointer list"));
                    for (size_t index = 0; index < offsets.size(); ++index)
                        require(std::memcmp(packed.data() + offsets[index], args[index], sizes[index]) == 0,
                            "host parameter preservation");
                }
            }
        }
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_kernel_parameters);
}
