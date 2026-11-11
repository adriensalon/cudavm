#include "test_fixtures.hpp"

namespace cudavm {
void execute_distributed_shared_memory(const loaded_program& program) {
    CUdevice device{}; int clusters = 0;
    TEST_CUDA(cuCtxGetDevice(&device));
    TEST_CUDA(cuDeviceGetAttribute(&clusters, CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, device));
    require_capability(clusters != 0, "native cluster launch support is required");
    constexpr unsigned blocks = 4, capacity = 128, fields = 4, guards = 12;
    constexpr uint32_t guard = 0xc6e8294f;
    std::vector<uint32_t> input(blocks * capacity + guards * 2, guard), preserved(input.size());
    for (unsigned block = 0; block < blocks; ++block) for (unsigned lane = 0; lane < capacity; ++lane)
        input[guards + block * capacity + lane] = 0x5917c3d2u ^ (block << 24) ^ (lane * 0x10101u);
    std::vector<uint32_t> initial(blocks * capacity * fields + guards * 2, guard), expected(initial), actual(initial.size());
    device_buffer source(input.size() * 4), destination(initial.size() * 4);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    auto from = source.value + guards * 4, to = destination.value + guards * 4;
    CUlaunchAttribute cluster{}; cluster.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION; cluster.value.clusterDim = {2, 1, 1};
    CUlaunchConfig launch{}; launch.gridDimX = blocks; launch.gridDimY = launch.gridDimZ = 1;
    launch.blockDimY = launch.blockDimZ = 1; launch.attrs = &cluster; launch.numAttrs = 1;
    const auto kernel = program.function("cluster_exchange");
    for (unsigned threads : {33u, 128u}) {
        launch.blockDimX = threads;
        int maximum_cluster = 0;
        TEST_CUDA(cuOccupancyMaxPotentialClusterSize(&maximum_cluster, kernel, &launch));
        require_capability(maximum_cluster >= 2, "two-CTA cluster capacity is required for this block size");
        for (unsigned count : {0u, 1u, 37u, capacity}) {
            expected = initial;
            for (unsigned block = 0; block < blocks; ++block) for (unsigned lane = 0; lane < std::min(count, threads); ++lane) {
                const auto row = guards + (block * capacity + lane) * fields;
                expected[row] = input[guards + (block ^ 1) * capacity + lane];
                expected[row + 1] = input[guards + block * capacity + lane];
                expected[row + 2] = block % 2; expected[row + 3] = 1 - block % 2;
            }
            TEST_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size() * 4));
            void* arguments[]{&to, &from, &count};
            TEST_CUDA(cuLaunchKernelEx(&launch, kernel, arguments, nullptr));
            TEST_CUDA(cuMemcpyDtoH(actual.data(), destination.value, actual.size() * 4));
            expect_words(actual, expected, "cluster exchange threads=" + std::to_string(threads) + " count=" + std::to_string(count));
            TEST_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
            expect_words(preserved, input, "cluster exchange preserves source and guards");
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_distributed_shared_memory);
}
