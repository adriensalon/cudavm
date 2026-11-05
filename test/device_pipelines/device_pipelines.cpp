#include "fixtures.hpp"

namespace cudavm {
void execute_device_pipelines(const loaded_program& program) {
    constexpr unsigned blocks = 3, guards = 4;
    constexpr uint32_t guard = 0xa1b2c3d4;
    for (unsigned threads : {1u, 13u, 32u, 64u}) for (unsigned rounds : {0u, 1u, 4u})
    for (unsigned seed : {0u, 0x87654321u}) {
        unsigned total = threads * blocks;
        std::vector<uint32_t> input(guards * 2 + 4 * total, guard), preserved(input.size());
        std::vector<uint32_t> expected(guards * 2 + (rounds * total + 2) * 16, guard), actual(expected.size());
        for (unsigned i = 0; i < total * 4; ++i) input[guards + i] = i * 0x9e3779b9u ^ seed;
        for (unsigned round = 0; round < rounds; ++round) for (unsigned id = 0; id < total; ++id) {
            auto* row = expected.data() + guards + (round * total + id) * 16;
            unsigned peer = id - id % threads + (id % threads + round + 1) % threads;
            for (unsigned stage = 0; stage < 4; ++stage) {
                const unsigned source_stage = (stage + round) % 4;
                row[stage] = input[guards + source_stage * total + id];
                row[4 + stage] = round * 6 + stage;
                row[8 + stage] = input[guards + source_stage * total + peer];
            }
            row[12] = input[guards + ((round + 3) % 4) * total + id];
            row[13] = round * 6 + 5;
        }
        device_buffer source(input.size() * 4), output(expected.size() * 4);
        auto from = source.value + guards * 4, to = output.value + guards * 4;
        std::vector<uint32_t> initialized(expected.size(), guard);
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
        CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, initialized.data(), initialized.size() * 4));
        void* args[]{&to, &from, &rounds};
        CONFORMANCE_CUDA(cuLaunchKernel(program.function("pipeline_batches"), blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        expect_words(actual, expected, "pipeline batch sequence, staged waits, peer publication, reuse and guards");
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
        expect_words(preserved, input, "pipeline source and allocation preservation");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_pipelines);
}
