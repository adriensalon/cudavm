#include "test_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>

namespace cudavm {
void capability_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CAPABILITY_RUNTIME(call) ::cudavm::capability_runtime((call), #call)
int capability_attribute(CUdevice device, CUdevice_attribute attribute) {
    struct { uint32_t before = 0x12345678; int value = -1; uint32_t after = 0x87654321; } output;
    CONFORMANCE_CUDA(cuDeviceGetAttribute(&output.value, attribute, device));
    require(output.before == 0x12345678 && output.after == 0x87654321, "device attribute output width");
    require(output.value > 0, "geometry capability is positive"); return output.value;
}
using capability_shape = std::array<unsigned, 3>;
void capability_launch(CUfunction function, capability_shape grid, capability_shape block) {
    constexpr unsigned fields = 16; constexpr uint32_t guard = 0xa1b2c3d4;
    const unsigned total = grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2];
    std::vector<uint32_t> expected((total + 3) * fields + 2, guard), actual(expected.size());
    device_buffer output(expected.size() * 4);
    CUdeviceptr destination = output.value + 4;
    for (unsigned count : {total - 1, total}) {
        std::fill(expected.begin(), expected.end(), guard);
        CONFORMANCE_CUDA(cuMemsetD32(output.value, guard, expected.size()));
        void* arguments[]{&destination, &count};
        CONFORMANCE_CUDA(cuLaunchKernel(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, nullptr, arguments, nullptr));
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, actual.size() * 4));
        unsigned index = 0, block_rank = 0;
        for (unsigned bz = 0; bz < grid[2]; ++bz)
        for (unsigned by = 0; by < grid[1]; ++by)
        for (unsigned bx = 0; bx < grid[0]; ++bx, ++block_rank) {
            unsigned rank = 0;
            for (unsigned tz = 0; tz < block[2]; ++tz)
            for (unsigned ty = 0; ty < block[1]; ++ty)
            for (unsigned tx = 0; tx < block[0]; ++tx, ++rank, ++index) {
                if (index >= count) continue;
                const std::array<uint32_t, fields> row{tx, ty, tz, block[0], block[1], block[2], bx, by, bz,
                    grid[0], grid[1], grid[2], rank % 32, 32, rank, block_rank};
                std::copy(row.begin(), row.end(), expected.begin() + 1 + index * fields);
            }
        }
        expect_words(actual, expected, "reported geometry boundary threads=" + std::to_string(block[0] * block[1] * block[2]));
    }
}
void capability_rejection(CUfunction function, capability_shape grid, capability_shape block) {
    constexpr uint32_t guard = 0xa1b2c3d4;
    std::array<uint32_t, 20> actual{}, expected; expected.fill(guard);
    device_buffer output(sizeof(expected)); CONFORMANCE_CUDA(cuMemsetD32(output.value, guard, expected.size()));
    CUdeviceptr destination = output.value + 4; unsigned count = 1;
    void* arguments[]{&destination, &count};
    require(cuLaunchKernel(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, nullptr, arguments, nullptr) == CUDA_ERROR_INVALID_VALUE,
            "geometry outside the reported device limit is rejected before execution");
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
    expect_words(actual, expected, "rejected geometry preserves initialized output");
    CONFORMANCE_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr));
    CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), output.value, sizeof(actual)));
    const std::array<uint32_t, 16> row{0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 32, 0, 0};
    std::copy(row.begin(), row.end(), expected.begin() + 1);
    expect_words(actual, expected, "legal launch after admission failure");
}
void execute_device_capabilities(const loaded_program& program) {
    CUdevice device{}; CUcontext before{}, after{};
    CONFORMANCE_CUDA(cuCtxGetDevice(&device)); CONFORMANCE_CUDA(cuCtxGetCurrent(&before));
    const unsigned maximum = capability_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    require(capability_attribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE) == 32, "reported logical CUDA warp width");
    const std::array<CUdevice_attribute, 3> block_attributes{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z};
    const std::array<CUdevice_attribute, 3> grid_attributes{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z};
    capability_shape limits{}, grid_limits{};
    for (unsigned axis = 0; axis < 3; ++axis) {
        limits[axis] = capability_attribute(device, block_attributes[axis]); grid_limits[axis] = capability_attribute(device, grid_attributes[axis]);
    }
    CUdevprop legacy{}; CONFORMANCE_CUDA(cuDeviceGetProperties(&legacy, device));
    cudaDeviceProp runtime{}; CAPABILITY_RUNTIME(cudaGetDeviceProperties(&runtime, device));
    require(legacy.maxThreadsPerBlock == maximum && runtime.maxThreadsPerBlock == maximum && runtime.warpSize == 32 && legacy.SIMDWidth == 32,
            "Driver, legacy and Runtime thread/warp limits agree");
    for (unsigned axis = 0; axis < 3; ++axis)
        require(legacy.maxThreadsDim[axis] == limits[axis] && runtime.maxThreadsDim[axis] == limits[axis]
            && legacy.maxGridSize[axis] == grid_limits[axis] && runtime.maxGridSize[axis] == grid_limits[axis], "per-axis query surfaces agree");
    const auto function = program.function("record_builtins");
    int function_maximum = 0; CONFORMANCE_CUDA(cuFuncGetAttribute(&function_maximum, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
    require(function_maximum >= maximum, "geometry fixture admits the reported maximum block thread count");
    std::vector<capability_shape> shapes{{1, 1, 1}};
    for (unsigned axis = 0; axis < 3; ++axis) { capability_shape shape{1, 1, 1}; shape[axis] = std::min(limits[axis], maximum); shapes.push_back(shape); }
    bool saturated = false;
    for (unsigned z = 1; z <= std::min(limits[2], maximum); ++z) {
        if (maximum % z) continue;
        for (unsigned y = 1; y <= std::min(limits[1], maximum / z); ++y) {
            if ((maximum / z) % y) continue;
            const unsigned x = maximum / z / y;
            if (x > limits[0]) continue;
            saturated = true;
            // Distinct x/y/z layouts exercise dimension-specific admission.
            if (shapes.size() < 10) shapes.push_back({x, y, z});
        }
    }
    require(saturated, "reported block limits have an executable maximum-thread factorization");
    for (unsigned x = 3; x <= 22 && x <= limits[0]; ++x)
        if (limits[1] >= 3 && limits[2] >= 2 && x * 6 <= maximum) shapes.push_back({x, 3, 2});
    if (limits[0] >= 3 && limits[1] >= 3 && limits[2] >= 2 && maximum >= 18) shapes.push_back({3, 3, 2});
    const capability_shape rectangular{std::min(grid_limits[0], 2u), std::min(grid_limits[1], 2u), std::min(grid_limits[2], 2u)};
    for (auto block : shapes) { capability_launch(function, {1, 1, 1}, block); capability_launch(function, rectangular, block); }
    capability_rejection(function, {0, 1, 1}, {1, 1, 1});
    for (unsigned axis = 0; axis < 3; ++axis) {
        capability_shape zero{1, 1, 1}, oversized{1, 1, 1}; zero[axis] = 0; oversized[axis] = limits[axis] + 1;
        capability_rejection(function, {1, 1, 1}, zero); capability_rejection(function, {1, 1, 1}, oversized);
    }
    capability_rejection(function, {1, 1, 1}, {maximum + 1, 1, 1});
    bool checked_total = false;
    for (auto block : shapes) {
        if (checked_total || block[0] * block[1] * block[2] != maximum) continue;
        for (unsigned axis = 0; axis < 3; ++axis) if (block[axis] < limits[axis]) {
            ++block[axis]; capability_rejection(function, {1, 1, 1}, block); checked_total = true; break;
        }
    }
    require(checked_total || uint64_t(limits[0]) * limits[1] * limits[2] == maximum,
            "total-thread admission boundary checked independently of per-axis limits where applicable");
    CONFORMANCE_CUDA(cuCtxGetCurrent(&after)); require(before == after, "capability queries/execution preserve current context");
}
}
int main(int argc, char** argv) { return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_device_capabilities); }
