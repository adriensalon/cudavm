#include "test_fixtures.hpp"
#include <algorithm>

namespace cudavm {
std::array<uint32_t, 15> object_oracle(uint32_t seed) {
    const uint64_t initial = uint64_t(seed) << 32 | (seed ^ 0x89abcdefu);
    const uint32_t tag = seed + 17, salt = seed ^ 0x13579bdfu, da = seed + 5, db = seed ^ 0x10203040u;
    const uint64_t original = initial + (uint64_t(da) << 32 | da), copy = (initial ^ 0x100000001ull) + (uint64_t(db) << 32 | db);
    const uint32_t a_tag = tag ^ da, a_salt = salt + da * 7, b_tag = (tag + 3) ^ db, b_salt = (salt ^ 0xffffffffu) + db * 7;
    auto checksum = [](uint64_t value, uint32_t t, uint32_t s) { return uint32_t(value) ^ uint32_t(value >> 32) ^ t ^ s; };
    uint32_t journal = seed * 33 ^ 0xc001cafeu;
    journal = journal * 33 ^ checksum(copy, b_tag, b_salt);
    journal = journal * 33 ^ checksum(original, a_tag, a_salt);
    return {journal, uint32_t(original), uint32_t(original >> 32), a_tag, a_salt,
        uint32_t(copy), uint32_t(copy >> 32), b_tag, b_salt, 32, 16, 0, 8, 12, 16};
}
void execute_device_objects(const loaded_program& program) {
    const auto function = program.function("object_results");
    constexpr uint32_t guard = 0xa1b2c3d4;
    constexpr unsigned blocks = 3, guards = 4, fields = 15;
    const std::array<uint32_t, 8> edges{0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffb, 0x13579bdf, 0x89abcdef};
    for (unsigned lanes : {1u, 13u, 33u, 64u}) {
        const unsigned total = blocks * lanes;
        std::vector<uint32_t> input(total + guards * 2, guard), preserved(input.size());
        std::vector<uint32_t> expected((total + 2) * fields + guards * 2, guard), actual(expected.size());
        device_buffer source(input.size() * 4), output(expected.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4;
        for (unsigned phase = 0; phase < 2; ++phase) {
            for (unsigned i = 0; i < total; ++i) input[guards + i] = i < edges.size() ? edges[(i + phase) % edges.size()] : 0xfffffff0u + i * 0x7654321u + phase * 0x89abcdefu;
            CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
            for (unsigned count : {0u, 1u, total / 2 + 1, total - 1, total, total + 7}) {
                std::fill(expected.begin(), expected.end(), guard);
                CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
                void* args[]{&to, &from, &count};
                CONFORMANCE_CUDA(cuLaunchKernel(function, blocks, 1, 1, lanes, 1, 1, 0, nullptr, args, nullptr));
                CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
                for (unsigned i = 0; i < std::min(total, count); ++i) {
                    const auto row = object_oracle(input[guards + i]);
                    std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
                }
                std::string differences;
                unsigned mismatches = 0;
                for (size_t word = 0; word < actual.size(); ++word) if (actual[word] != expected[word] && mismatches++ < 12)
                    differences += " word " + std::to_string(word) + " got " + std::to_string(actual[word]) + " expected " + std::to_string(expected[word]);
                require(differences.empty(), "device object fields, lifecycle journal and layout:" + differences);
            }
            CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "device object source and guard preservation");
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_objects);
}
