#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_global_symbols(const loaded_program& program) {
    auto globals = program.global("mutable_words", 19 * 4);
    auto constants = program.global("constant_words", 7 * 4);
    require(globals % 4 == 0 && constants % 4 == 0, "global symbol alignment");
    require(program.global("mutable_words", 19 * 4) == globals
        && program.global("constant_words", 7 * 4) == constants, "stable symbol lookup");
    std::vector<uint32_t> words{0x13579bdf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0xeca86420};
    std::vector<uint32_t> constant_words{0, 1, 0x80000000, 0xffffffff, 0x12345678, 0xabcdef01, 17};
    auto check_storage = [&] {
        std::vector<uint32_t> actual(words.size()), constant_actual(constant_words.size());
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), globals, actual.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyDtoH(constant_actual.data(), constants, constant_actual.size() * 4));
        expect_words(actual, words, "mutable symbol contents and guards");
        expect_words(constant_actual, constant_words, "constant symbol contents");
    };
    check_storage();
    const auto read = program.function("read_symbols"), update = program.function("update_symbols");
    constexpr uint32_t guard = 0xa1b2c3d4;
    constexpr unsigned fields = 7, blocks = 3;
    for (unsigned phase = 0; phase < 3; ++phase) {
        if (phase == 1) {
            for (unsigned i = 1; i <= 17; ++i) words[i] = 0xfffffff0u + i * 0x1234567u;
            for (unsigned i = 0; i < constant_words.size(); ++i) constant_words[i] ^= 0xfedcba98u + i;
            CONFORMANCE_CUDA(cuMemcpyHtoD(globals + 4, words.data() + 1, 17 * 4));
            CONFORMANCE_CUDA(cuMemcpyHtoD(constants, constant_words.data(), constant_words.size() * 4));
        }
        if (phase == 2) {
            // Each slot has exactly one writer. Synchronize before host observation.
            for (uint32_t salt : {0u, 0xffffffffu, 0x98765432u}) {
                void* args[]{&salt};
                CONFORMANCE_CUDA(cuLaunchKernel(update, 3, 1, 1, 13, 1, 1, 0, nullptr, args, nullptr));
                for (unsigned i = 0; i < 17; ++i) words[i + 1] += salt ^ constant_words[i % 7];
                CONFORMANCE_CUDA(cuCtxSynchronize());
                check_storage();
            }
        }
        check_storage();
        for (unsigned threads : {1u, 13u, 33u}) {
            const unsigned total = blocks * threads;
            std::vector<uint32_t> actual(total * fields + 2), expected(actual.size(), guard);
            device_buffer destination(actual.size() * 4);
            auto out = destination.value + 4;
            for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
                std::fill(expected.begin(), expected.end(), guard);
                CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, expected.data(), expected.size() * 4));
                void* args[]{&out, &count};
                CONFORMANCE_CUDA(cuLaunchKernel(read, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
                for (unsigned i = 0; i < std::min(total, count); ++i) {
                    const unsigned slot = i % 17 + 1, constant_slot = i % 7;
                    const auto address = globals + slot * 4, constant_address = constants + constant_slot * 4;
                    const std::array<uint32_t, fields> row{words[slot], constant_words[constant_slot],
                        (words[slot] ^ constant_words[constant_slot]) + i, uint32_t(address), uint32_t(address >> 32),
                        uint32_t(constant_address), uint32_t(constant_address >> 32)};
                    std::copy(row.begin(), row.end(), expected.begin() + 1 + i * fields);
                }
                expect_words(actual, expected, "global symbols phase=" + std::to_string(phase)
                    + " threads=" + std::to_string(threads) + " count=" + std::to_string(count));
            }
        }
        check_storage();
    }
}
}

int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_global_symbols, 2);
}
