#include "test_fixtures.hpp"
#include <algorithm>
#include <iostream>

namespace cudavm {
void execute_memory_model_litmus(const loaded_program& program) {
    constexpr unsigned blocks = 3, guards = 4, pairs = blocks * 32;
    constexpr uint32_t guard = 0xa1b2c3d4;
    for (const auto* profile : {"cta_global", "cta_shared", "gpu_global"})
    for (bool fenced : {false, true}) for (unsigned forced : {0u, 1u})
    for (unsigned trials : {0u, 1u, 29u}) {
        const bool shared = std::string_view(profile) == "cta_shared";
        std::vector<uint32_t> input(guards * 2 + pairs, guard), preserved(input.size());
        std::vector<uint32_t> expected(guards * 2 + (pairs * trials + 2) * 4, guard), actual(expected.size());
        std::vector<uint32_t> flags(guards * 2 + pairs, guard), actual_flags(flags.size());
        std::vector<uint32_t> values(guards * 2 + pairs, guard), actual_values(values.size());
        for (unsigned id = 0; id < pairs; ++id) input[guards + id] = id * 0x9e3779b9u ^ (trials * 0x1234567u);
        device_buffer source(input.size() * 4), output(expected.size() * 4), flags_storage(flags.size() * 4), payload(values.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4;
        auto flag_pointer = flags_storage.value + guards * 4, value_pointer = payload.value + guards * 4;
        TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        TEST_CUDA(cuMemcpyHtoD(flags_storage.value, flags.data(), flags.size() * 4));
        TEST_CUDA(cuMemcpyHtoD(payload.value, values.data(), values.size() * 4));
        void* args[]{&to, &from, &flag_pointer, &value_pointer, &trials, &forced};
        const auto name = std::string(fenced ? "mp_fenced_" : "mp_ordered_") + profile;
        TEST_CUDA(cuLaunchKernel(program.function(name.c_str()), blocks, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
        TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        std::array<unsigned, 3> outcomes{};
        for (unsigned trial = 0; trial < trials; ++trial) for (unsigned id = 0; id < pairs; ++id) {
            unsigned row = guards + (trial * pairs + id) * 4;
            uint32_t initial = input[guards + id] + trial * 17u, published = initial ^ 0xa5a55a5au;
            auto observed = actual[row], value = actual[row + 1];
            require((observed == 0 && (value == initial || value == published)) || (observed == 1 && value == published),
                name + " MP forbidden observation at trial=" + std::to_string(trial) + " pair=" + std::to_string(id)
                    + " flag=" + std::to_string(observed) + " payload=" + std::to_string(value));
            if (forced) require(observed == 1 && value == published, "barrier control must observe the exact published pair");
            ++outcomes[observed ? 2 : value == initial ? 0 : 1];
            expected[row] = observed; expected[row + 1] = value;
            if (!shared) { flags[guards + id] = 1; values[guards + id] = published; }
        }
        expect_words(actual, expected, "litmus observation extents, inactive fields and all output guards");
        TEST_CUDA(cuMemcpyDtoH(actual_flags.data(), flags_storage.value, actual_flags.size() * 4));
        expect_words(actual_flags, flags, "litmus final flags and neighboring words");
        TEST_CUDA(cuMemcpyDtoH(actual_values.data(), payload.value, actual_values.size() * 4));
        expect_words(actual_values, values, "litmus final payload and allocation guards");
        TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "litmus source preservation");
        std::cout << name << " forced=" << forced << " trials=" << trials
            << " outcomes=" << outcomes[0] << ',' << outcomes[1] << ',' << outcomes[2] << '\n';
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_memory_model_litmus);
}
