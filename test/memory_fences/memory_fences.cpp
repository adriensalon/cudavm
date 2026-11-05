#include "fixtures.hpp"
#include <algorithm>

namespace cudavm {
void execute_memory_fences(const loaded_program& program) {
    constexpr unsigned blocks = 3, guards = 4, pairs = blocks * 32;
    constexpr uint32_t guard = 0xa1b2c3d4;
    for (const auto* profile : {"cta_global", "cta_shared", "gpu_global"})
    for (unsigned rounds : {0u, 1u, 7u}) for (unsigned seed : {0u, 0x87654321u}) {
        const bool shared = std::string_view(profile) == "cta_shared";
        std::vector<uint32_t> input(guards * 2 + pairs, guard), preserved(input.size());
        std::vector<uint32_t> expected(guards * 2 + (pairs * rounds + 2) * 3, guard), actual(expected.size());
        std::vector<uint32_t> flags(guards * 2 + blocks * 64, guard), actual_flags(flags.size());
        std::vector<uint32_t> values(guards * 2 + pairs * 2, guard), actual_values(values.size());
        for (unsigned id = 0; id < pairs; ++id) input[guards + id] = id * 0x9e3779b9u ^ seed;
        device_buffer source(input.size() * 4), output(expected.size() * 4), flags_storage(flags.size() * 4), payload(values.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4;
        auto flag_pointer = flags_storage.value + guards * 4, value_pointer = payload.value + guards * 4;
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, expected.data(), expected.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(flags_storage.value, flags.data(), flags.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(payload.value, values.data(), values.size() * 4));
        void* args[]{&to, &from, &flag_pointer, &value_pointer, &rounds};
        const auto name = std::string("fence_") + profile;
        CONFORMANCE_CUDA(cuLaunchKernel(program.function(name.c_str()), blocks, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
        for (unsigned round = 0; round < rounds; ++round) for (unsigned id = 0; id < pairs; ++id) {
            auto value = input[guards + id] ^ (round * 0x10001u + id);
            expected[guards + (round * pairs + id) * 3] = value;
            expected[guards + (round * pairs + id) * 3 + 1] = value + 0x12345u;
            expected[guards + (round * pairs + id) * 3 + 2] = round + 1;
            if (!shared) { values[guards + id * 2] = value; values[guards + id * 2 + 1] = value + 0x12345u; }
        }
        if (!shared) std::fill(flags.begin() + guards, flags.end() - guards, rounds);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "fenced publication, relaxed flag handoff, acknowledgment and output guards");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual_flags.data(), flags_storage.value, actual_flags.size() * 4));
        expect_words(actual_flags, flags, "fence handoff generations and flag neighbors");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual_values.data(), payload.value, actual_values.size() * 4));
        expect_words(actual_values, values, "two-word fenced payload and allocation guards");
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "fence source preservation");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_memory_fences);
}
