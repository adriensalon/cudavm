#include "test_fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_atomic_memory_order(const loaded_program& program) {
    constexpr unsigned blocks = 3, guards = 4, pairs = blocks * 32;
    constexpr uint32_t guard = 0xa1b2c3d4;
    for (const auto* profile : {"cta_global", "cta_shared", "gpu_global"}) {
        const bool shared = std::string_view(profile) == "cta_shared";
        for (unsigned seed : {0u, UINT32_MAX, 0x80000000u}) {
            std::vector<uint32_t> expected(guards * 2 + blocks * 12, guard), actual(expected.size());
            std::vector<uint32_t> state(guards * 2 + blocks, guard), actual_state(state.size());
            device_buffer output(expected.size() * 4), storage(state.size() * 4);
            auto to = output.value + guards * 4, pointer = storage.value + guards * 4;
            TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            TEST_CUDA(cuMemcpyHtoD(storage.value, state.data(), state.size() * 4));
            void* args[]{&to, &pointer, &seed};
            const auto name = std::string("orders_") + profile;
            TEST_CUDA(cuLaunchKernel(program.function(name.c_str()), blocks, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
            uint32_t value = seed;
            for (unsigned mode = 0; mode < 4; ++mode) {
                const auto old = value;
                value = uint32_t(uint64_t(value) + (mode & 1 ? UINT32_MAX : 0x80000001u));
                for (unsigned block = 0; block < blocks; ++block) {
                    expected[guards + block * 12 + mode * 2] = old;
                    expected[guards + block * 12 + mode * 2 + 1] = value;
                }
            }
            if (!shared) std::fill_n(state.begin() + guards, blocks, value);
            TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            expect_words(actual, expected, "scoped relaxed/acquire/release/acq_rel add, acquire load and guards");
            TEST_CUDA(cuMemcpyDtoH(actual_state.data(), storage.value, actual_state.size() * 4));
            expect_words(actual_state, state, "scoped atomic final storage and neighbors");
        }
        for (unsigned rounds : {0u, 1u, 7u}) for (unsigned seed : {0u, 0x87654321u}) {
            std::vector<uint32_t> input(guards * 2 + pairs, guard), preserved(input.size());
            std::vector<uint32_t> expected(guards * 2 + (pairs * rounds + 2) * 2, guard), actual(expected.size());
            std::vector<uint32_t> flags(guards * 2 + blocks * 64, guard), actual_flags(flags.size());
            std::vector<uint32_t> values(guards * 2 + pairs, guard), actual_values(values.size());
            for (unsigned id = 0; id < pairs; ++id) input[guards + id] = id * 0x9e3779b9u ^ seed;
            device_buffer source(input.size() * 4), output(expected.size() * 4), flags_storage(flags.size() * 4), payload(values.size() * 4);
            auto from = source.value + guards * 4, to = output.value + guards * 4;
            auto flag_pointer = flags_storage.value + guards * 4, value_pointer = payload.value + guards * 4;
            TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
            TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
            TEST_CUDA(cuMemcpyHtoD(flags_storage.value, flags.data(), flags.size() * 4));
            TEST_CUDA(cuMemcpyHtoD(payload.value, values.data(), values.size() * 4));
            void* args[]{&to, &from, &flag_pointer, &value_pointer, &rounds};
            const auto name = std::string("publish_") + profile;
            TEST_CUDA(cuLaunchKernel(program.function(name.c_str()), blocks, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
            for (unsigned round = 0; round < rounds; ++round) for (unsigned id = 0; id < pairs; ++id) {
                auto value = input[guards + id] ^ (round * 0x10001u + id);
                expected[guards + (round * pairs + id) * 2] = value;
                expected[guards + (round * pairs + id) * 2 + 1] = round + 1;
                if (!shared) values[guards + id] = value;
            }
            if (!shared) std::fill(flags.begin() + guards, flags.end() - guards, rounds);
            TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
            expect_words(actual, expected, "release/acquire publication, acknowledgement, reuse and output guards");
            TEST_CUDA(cuMemcpyDtoH(actual_flags.data(), flags_storage.value, actual_flags.size() * 4));
            expect_words(actual_flags, flags, "atomic generations and flag allocation guards");
            TEST_CUDA(cuMemcpyDtoH(actual_values.data(), payload.value, actual_values.size() * 4));
            expect_words(actual_values, values, "published payload and surrounding allocation words");
            TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "atomic publication source preservation");
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_atomic_memory_order);
}
