#include "test_fixtures.hpp"
#include "address_nvcc.hpp"
#include <algorithm>
#include <iostream>

namespace cudavm {
void execute_address_spaces(const loaded_program& program) {
    const auto function = program.function("address_results");
    const auto constant_address = program.global("address_constants", 9 * 4);
    std::array<uint32_t, 9> constants{0x13579bdf, 0, 1, 0x80000000, 0xffffffff, 0x12345678, 0x89abcdef, 17, 0xeca86420};
    std::array<uint32_t, 9> constant_actual{};
    TEST_CUDA(cuMemcpyDtoH(constant_actual.data(), constant_address, sizeof(constants)));
    expect_words(constant_actual, constants, "constant-space initialization");
    constexpr unsigned blocks = 3, fields = 36, guards = 4;
    constexpr uint32_t guard = 0xa1b2c3d4;
    for (unsigned lanes : {1u, 13u, 33u, 64u}) {
        const unsigned total = blocks * lanes;
        std::vector<uint32_t> input(total * 2 + guards * 2, guard), preserved(input.size());
        std::vector<uint32_t> expected((total + 2) * fields + guards * 2, guard), actual(expected.size());
        device_buffer source(input.size() * 4), output(expected.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4;
        for (unsigned phase = 0; phase < 2; ++phase) {
            for (unsigned i = 0; i < total * 2; ++i) input[guards + i] = 0xfffffff0u + i * 0x7654321u + phase * 0x89abcdefu;
            for (unsigned i = 1; i + 1 < constants.size(); ++i) constants[i] = (i & 1 ? 0xffffffffu : 0x80000000u) ^ (phase * 0x1234567u + i * 0x13579bdu);
            TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
            TEST_CUDA(cuMemcpyHtoD(constant_address, constants.data(), sizeof(constants)));
            for (unsigned selector : {0u, 1u, 2u, 3u})
            for (unsigned count : {0u, 1u, total / 2 + 1, total, total + 7}) {
                std::fill(expected.begin(), expected.end(), guard);
                TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
                void* args[]{&to, &from, &count, &selector};
                TEST_CUDA(cuLaunchKernel(function, blocks, 1, 1, lanes, 1, 1, 0, nullptr, args, nullptr));
                TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
                for (unsigned i = 0; i < std::min(total, count); ++i) {
                    const uint32_t a = input[guards + i * 2], b = input[guards + i * 2 + 1];
                    const std::array<std::array<uint32_t, 2>, 4> values{{{a, b}, {a ^ 0xaaaaaaaa, b + i / lanes + 7},
                        {a + i * 17, b ^ (i * 0x10203)}, {constants[i % 6 + 1], constants[i % 6 + 2]}}};
                    std::array<uint32_t, fields> row{};
                    for (unsigned space = 0; space < 4; ++space) {
                        const std::array<uint32_t, 6> group{1u << space, 1, values[space][0], values[space][1], 4, 1};
                        std::copy(group.begin(), group.end(), row.begin() + space * 6);
                    }
                    const auto global_bits = from + i * 8, constant_bits = constant_address + (i % 6 + 1) * 4;
                    row[24] = uint32_t(global_bits); row[25] = uint32_t(global_bits >> 32);
                    row[26] = uint32_t(constant_bits); row[27] = uint32_t(constant_bits >> 32);
                    const unsigned space = (i + selector) & 3;
                    row[28] = 1u << space; row[29] = values[space][0]; row[30] = values[space][1]; row[31] = 1;
                    row[32] = 0x13579bdfu ^ i; row[33] = 0xeca86420u ^ a;
                    row[34] = 0x89abcdefu ^ b; row[35] = 0x76543210u ^ i;
                    std::copy(row.begin(), row.end(), expected.begin() + guards + i * fields);
                }
                expect_words(actual, expected, "address-space masks, conversions, aliases and guards lanes=" + std::to_string(lanes)
                    + " selector=" + std::to_string(selector) + " count=" + std::to_string(count));
            }
            TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "address-space source preservation");
            TEST_CUDA(cuMemcpyDtoH(constant_actual.data(), constant_address, sizeof(constants)));
            expect_words(constant_actual, constants, "constant-space content and guard preservation");
        }
    }
}
}
int main(int argc, char** argv) {
    int status = cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_address_spaces);
    if (status == 0 && argc == 3 && std::string_view(argv[2]) == "cubin") {
        std::cout << "producer=nvcc-cuda-cpp" << std::endl;
        auto fixtures = cudavm::fixtures;
        fixtures.cubin = cudavm::address_nvcc_cubin;
        status = cudavm::run_test(argc, argv, fixtures, cudavm::execute_address_spaces);
    }
    return status;
}
