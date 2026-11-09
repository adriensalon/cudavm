#include "test_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>

namespace cudavm {
void cluster_case(const loaded_program& program, std::array<unsigned, 3> cluster, unsigned rounds, bool runtime) {
    constexpr unsigned guards = 4, threads = 32;
    constexpr uint32_t guard = 0xa1b2c3d4;
    const std::array<unsigned, 3> grid{cluster[0] * 2, cluster[1] * 2, cluster[2]};
    unsigned blocks = grid[0] * grid[1] * grid[2], total = blocks * threads;
    std::vector<uint32_t> input(guards * 2 + total, guard), preserved(input.size());
    std::vector<uint32_t> expected(guards * 2 + (total * rounds + 2) * 10, guard), actual(expected.size());
    std::vector<uint32_t> scratch(guards * 2 + total, guard), actual_scratch(scratch.size());
    for (unsigned id = 0; id < total; ++id) input[guards + id] = id * 0x9e3779b9u + rounds * 17u;
    for (unsigned round = 0; round < rounds; ++round) for (unsigned id = 0; id < total; ++id) {
        unsigned block = id / threads, bx = block % grid[0], by = block / grid[0] % grid[1], bz = block / (grid[0] * grid[1]);
        unsigned x = bx % cluster[0], y = by % cluster[1], z = bz % cluster[2];
        unsigned size = cluster[0] * cluster[1] * cluster[2], rank = x + cluster[0] * (y + cluster[1] * z);
        unsigned next = (rank + 1) % size;
        unsigned peer_block = bx - x + next % cluster[0]
            + grid[0] * (by - y + next / cluster[0] % cluster[1] + grid[1] * (bz - z + next / (cluster[0] * cluster[1])));
        unsigned peer = peer_block * threads + (id % threads + 1) % threads;
        auto* row = expected.data() + guards + (round * total + id) * 10;
        row[0] = rank; row[1] = size; row[2] = x; row[3] = y; row[4] = z;
        std::copy(cluster.begin(), cluster.end(), row + 5);
        row[8] = input[guards + peer] ^ (round * 0x10001u + peer); row[9] = peer;
        scratch[guards + id] = input[guards + id] ^ (round * 0x10001u + id);
    }
    device_buffer source(input.size() * 4), output(expected.size() * 4), temporary(scratch.size() * 4);
    auto from = source.value + guards * 4, to = output.value + guards * 4, temp = temporary.value + guards * 4;
    std::vector<uint32_t> initialized(expected.size(), guard), empty(scratch.size(), guard);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, initialized.data(), initialized.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(temporary.value, empty.data(), empty.size() * 4));
    void* args[]{&to, &from, &temp, &rounds};
    if (runtime) {
        cudaLaunchAttribute attribute{}; attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim = {cluster[0], cluster[1], cluster[2]};
        cudaLaunchConfig_t config{}; config.gridDim = dim3(grid[0], grid[1], grid[2]); config.blockDim = dim3(threads);
        config.attrs = &attribute; config.numAttrs = 1;
        auto kernel = program.library_kernel("cluster_rounds"); require(kernel, "Runtime cluster launch uses a library kernel");
        auto status = cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), args);
        require(status == cudaSuccess, std::string("Runtime cluster launch: ") + cudaGetErrorString(status));
    } else {
        CUlaunchAttribute attribute{}; attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attribute.value.clusterDim = {cluster[0], cluster[1], cluster[2]};
        CUlaunchConfig config{}; config.gridDimX = grid[0]; config.gridDimY = grid[1]; config.gridDimZ = grid[2];
        config.blockDimX = threads; config.blockDimY = config.blockDimZ = 1; config.attrs = &attribute; config.numAttrs = 1;
        int active = 0; CONFORMANCE_CUDA(cuOccupancyMaxActiveClusters(&active, program.function("cluster_rounds"), &config));
        require(active > 0, "declared portable cluster has positive residency");
        CONFORMANCE_CUDA(cuLaunchKernelEx(&config, program.function("cluster_rounds"), args, nullptr));
    }
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
    expect_words(actual, expected, "cluster membership, repeated barriers, cross-block publication and guards");
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual_scratch.data(), temporary.value, actual_scratch.size() * 4));
    expect_words(actual_scratch, scratch, "cluster scratch and guards");
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
    expect_words(preserved, input, "cluster source preservation");
}
void execute_cluster_synchronization(const loaded_program& program) {
    CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    int supported = 0; CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, device));
    require_capability(supported != 0, "native device does not advertise cluster launch");
    CUlaunchConfig configuration{}; configuration.gridDimX = 4; configuration.gridDimY = 2; configuration.gridDimZ = 2;
    configuration.blockDimX = 32; configuration.blockDimY = configuration.blockDimZ = 1;
    int maximum = 0;
    CONFORMANCE_CUDA(cuOccupancyMaxPotentialClusterSize(&maximum, program.function("cluster_rounds"), &configuration));
    require_capability(maximum >= 2, "native partition cannot admit the declared two-block cluster profile");
    for (auto cluster : {std::array<unsigned, 3>{1, 1, 1}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}})
    for (unsigned rounds : {0u, 1u, 3u}) {
        cluster_case(program, cluster, rounds, false);
        if (program.library_kernel("cluster_rounds")) cluster_case(program, cluster, rounds, true);
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_cluster_synchronization);
}
