#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>

namespace cudavm {
void cooperative_runtime(cudaError_t status, const char* operation) {
    require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}
#define COOPERATIVE_RUNTIME(call) ::cudavm::cooperative_runtime((call), #call)
struct cooperative_stream {
    CUstream value{};
    cooperative_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~cooperative_stream() { if (value) cuStreamDestroy(value); }
};
void cooperative_launch_case(const loaded_program& program, unsigned threads, unsigned shared, unsigned rounds,
    std::array<unsigned, 3> grid, unsigned api, bool reject) {
    constexpr unsigned guards = 4;
    constexpr uint32_t guard = 0xa1b2c3d4;
    unsigned blocks = grid[0] * grid[1] * grid[2], total = blocks * threads;
    const auto function = program.function("cooperative_grid");
    const auto kernel = program.library_kernel("cooperative_grid");
    require(api < 2 || kernel, "Runtime cooperative execution requires a library kernel");
    std::vector<uint32_t> input(guards * 2 + total, guard), preserved(input.size());
    std::vector<uint32_t> expected(guards * 2 + (total * rounds + 2) * 8, guard), actual(expected.size());
    std::vector<uint32_t> scratch(guards * 2 + total, guard), actual_scratch(scratch.size());
    for (unsigned id = 0; id < total; ++id) input[guards + id] = id * 0x9e3779b9u + rounds * 17u;
    auto value = [&](unsigned id, unsigned round) { return input[guards + id] ^ (round * 0x10001u + id); };
    if (!reject) for (unsigned round = 0; round < rounds; ++round) for (unsigned id = 0; id < total; ++id) {
        unsigned peer = (id + threads + round + 1) % total;
        auto* row = expected.data() + guards + (round * total + id) * 8;
        row[0] = value(peer, round); row[1] = total; row[2] = id; row[3] = id / threads;
        row[4] = 1; row[5] = ((input[guards + id] + round) * 33u + id) ^ (round & 1 ? round : 0);
        row[6] = blocks; row[7] = peer;
        scratch[guards + id] = value(id, round);
    }
    device_buffer source(input.size() * 4), output(expected.size() * 4), temporary(scratch.size() * 4);
    auto from = source.value + guards * 4, to = output.value + guards * 4, temp = temporary.value + guards * 4;
    std::vector<uint32_t> initialized(expected.size(), guard), empty(scratch.size(), guard);
    CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(output.value, initialized.data(), initialized.size() * 4));
    CONFORMANCE_CUDA(cuMemcpyHtoD(temporary.value, empty.data(), empty.size() * 4));
    cooperative_stream stream;
    void* args[]{&to, &from, &temp, &rounds};
    if (api < 2) {
        CUresult status;
        if (api == 0) status = cuLaunchCooperativeKernel(function, grid[0], grid[1], grid[2], threads, 1, 1, shared, stream.value, args);
        else {
            CUlaunchAttribute attribute{}; attribute.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE; attribute.value.cooperative = 1;
            CUlaunchConfig config{}; config.gridDimX = grid[0]; config.gridDimY = grid[1]; config.gridDimZ = grid[2];
            config.blockDimX = threads; config.blockDimY = config.blockDimZ = 1;
            config.sharedMemBytes = shared; config.hStream = stream.value; config.attrs = &attribute; config.numAttrs = 1;
            status = cuLaunchKernelEx(&config, function, args, nullptr);
        }
        if (reject) require(status == CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, "Driver rejects a grid one block beyond residency");
        else check_driver(status, "Driver cooperative launch");
        COOPERATIVE_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.value)));
    } else {
        cudaError_t status;
        auto token = reinterpret_cast<const void*>(kernel);
        if (api == 2) status = cudaLaunchCooperativeKernel(token, dim3(grid[0], grid[1], grid[2]), dim3(threads), args, shared, reinterpret_cast<cudaStream_t>(stream.value));
        else {
            cudaLaunchAttribute attribute{}; attribute.id = cudaLaunchAttributeCooperative; attribute.val.cooperative = 1;
            cudaLaunchConfig_t config{}; config.gridDim = dim3(grid[0], grid[1], grid[2]); config.blockDim = dim3(threads);
            config.dynamicSmemBytes = shared; config.stream = reinterpret_cast<cudaStream_t>(stream.value);
            config.attrs = &attribute; config.numAttrs = 1;
            status = cudaLaunchKernelExC(&config, token, args);
        }
        if (reject) require(status == cudaErrorCooperativeLaunchTooLarge && cudaGetLastError() == cudaErrorCooperativeLaunchTooLarge,
            "Runtime translates and records oversized cooperative admission");
        else cooperative_runtime(status, "Runtime cooperative launch");
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
    }
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
    expect_words(actual, expected, "cooperative grid metadata, cross-block publication, repeated phases and output guards");
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual_scratch.data(), temporary.value, actual_scratch.size() * 4));
    expect_words(actual_scratch, scratch, "cooperative admission and scratch guards");
    CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size() * 4));
    expect_words(preserved, input, "cooperative source preservation");
}
void execute_cooperative_launch(const loaded_program& program) {
    CUdevice device{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device));
    int supported = 0, multiprocessors = 0;
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device));
    require_capability(supported != 0, "native device does not advertise cooperative launch");
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    require(multiprocessors > 0, "cooperative launch requires positive native residency");
    const auto function = program.function("cooperative_grid");
    for (unsigned shared : {0u, 16384u}) {
        unsigned threads = shared ? 64u : 32u;
        int active = 0; CONFORMANCE_CUDA(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active, function, threads, shared));
        require(active > 0 && uint64_t(active) * unsigned(multiprocessors) < UINT32_MAX / threads,
            "positive cooperative residency fits the fixture's rank representation");
        unsigned capacity = unsigned(active) * unsigned(multiprocessors);
        for (unsigned api = 0; api < (program.library_kernel("cooperative_grid") ? 4u : 2u); ++api) {
            cooperative_launch_case(program, threads, shared, 0, {1, 1, 1}, api, false);
            if (capacity >= 4) cooperative_launch_case(program, threads, shared, 3, {2, 2, 1}, api, false);
            cooperative_launch_case(program, threads, shared, 3, {capacity + 1, 1, 1}, api, true);
            // A repaired launch at the admitted boundary must still execute.
            cooperative_launch_case(program, threads, shared, 3, {capacity, 1, 1}, api, false);
        }
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_cooperative_launch);
}
