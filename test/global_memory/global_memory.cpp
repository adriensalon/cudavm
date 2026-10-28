#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
uint64_t global_memory_read(const std::vector<unsigned char>& bytes, size_t offset, unsigned width) {
    uint64_t value = 0;
    for (unsigned byte = 0; byte < width; ++byte) value |= uint64_t(bytes[offset + byte]) << (byte * 8);
    return value;
}
void global_memory_write(std::vector<unsigned char>& bytes, size_t offset, unsigned width, uint64_t value) {
    for (unsigned byte = 0; byte < width; ++byte) bytes[offset + byte] = static_cast<unsigned char>(value >> (byte * 8));
}
void expect_memory_bytes(const std::vector<unsigned char>& actual,
                         const std::vector<unsigned char>& expected, const std::string& context) {
    require(actual.size() == expected.size(), context + ": output extent");
    for (size_t byte = 0; byte < actual.size(); ++byte)
        require(actual[byte] == expected[byte], context + ": byte " + std::to_string(byte)
            + " got " + std::to_string(actual[byte]) + ", expected " + std::to_string(expected[byte]));
}

void execute_global_memory(const loaded_program& program) {
    const auto mixed = program.function("mixed_width_access");
    const auto neighbors = program.function("neighbor_byte_access");
    constexpr unsigned blocks = 3, prefix = 16;
    constexpr unsigned char guard = 0xa7;
    for (unsigned threads : {1u, 13u, 33u, 64u}) {
        const unsigned total = blocks * threads;
        std::vector<unsigned char> input(total * 16 + prefix * 2, guard), preserved(input.size());
        for (unsigned byte = 0; byte < total * 16; ++byte)
            input[prefix + byte] = static_cast<unsigned char>(byte * 73u + byte / 7u);
        std::vector<unsigned char> expected(input.size(), guard), actual(expected.size());
        device_buffer source(input.size()), destination(expected.size());
        auto in = source.value + prefix;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size()));
        for (bool byte_access : {false, true})
        for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
            std::fill(expected.begin(), expected.end(), guard);
            CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size()));
            // Neighboring byte stores deliberately share transport words and
            // start at an interior, unaligned byte address.
            const unsigned output_prefix = prefix + unsigned(byte_access);
            auto out = destination.value + output_prefix;
            void* args[]{&out, &in, &count};
            CONFORMANCE_CUDA(cuLaunchKernel(byte_access ? neighbors : mixed,
                blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size()));
            for (unsigned i = 0; i < std::min(total, count); ++i) {
                if (byte_access) {
                    expected[output_prefix + i] = input[prefix + i] ^ 0x5a;
                    continue;
                }
                const auto source_offset = prefix + i * 16;
                const auto target_offset = output_prefix + i * 16;
                expected[target_offset + 1] = input[source_offset] ^ 0xa5;
                global_memory_write(expected, target_offset + 2, 2,
                    global_memory_read(input, source_offset + 2, 2) ^ 0x5aa5);
                global_memory_write(expected, target_offset + 4, 4,
                    global_memory_read(input, source_offset + 4, 4) + 0x1020305u);
                global_memory_write(expected, target_offset + 8, 8,
                    global_memory_read(input, source_offset + 8, 8) ^ 0x0123456789abcdefull);
            }
            expect_memory_bytes(actual, expected, std::string(byte_access ? "neighbor bytes" : "mixed widths")
                + " threads=" + std::to_string(threads) + " count=" + std::to_string(count));
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size()));
        expect_memory_bytes(preserved, input, "global-memory source preservation");
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_global_memory);
}
