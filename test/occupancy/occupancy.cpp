#include "test_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <limits>

namespace cudavm {
int occupancy_device_attribute(CUdevice device, CUdevice_attribute attribute) {
    int value = -1;
    TEST_CUDA(cuDeviceGetAttribute(&value, attribute, device));
    require(value > 0, "native residency resource is positive");
    return value;
}
int occupancy_function_attribute(CUfunction function, CUfunction_attribute attribute) {
    int value = -1;
    TEST_CUDA(cuFuncGetAttribute(&value, attribute, function));
    require(value >= 0, "function resource is nonnegative");
    return value;
}
int occupancy_blocks(CUfunction function, int block, size_t shared, unsigned flags = 0) {
    struct { uint32_t before = 0x12345678; int value = -1; uint32_t after = 0x87654321; } result;
    TEST_CUDA(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&result.value, function, block, shared, flags));
    require(result.before == 0x12345678 && result.after == 0x87654321 && result.value >= 0, "occupancy output bounds and sign");
    return result.value;
}
void occupancy_launch(CUfunction function, int grid, int block, unsigned shared) {
    require(grid > 0 && block > 0 && shared >= unsigned(block) * 4, "recommendation has executable dynamic storage");
    const size_t total = size_t(grid) * unsigned(block), guards = 4, words = total + guards * 2 + 3;
    require(total < std::numeric_limits<unsigned>::max(), "recommendation fits fixture index width");
    constexpr uint32_t guard = 0xa1b2c3d4;
    std::vector<uint32_t> input(words, guard), expected(words, guard), actual(words);
    for (size_t i = 0; i < total; ++i) input[guards + i] = 0xfffffff0u + uint32_t(i) * 0x7654321u;
    device_buffer source(words * 4), output(words * 4);
    TEST_CUDA(cuMemcpyHtoD(source.value, input.data(), words * 4));
    TEST_CUDA(cuMemcpyHtoD(output.value, expected.data(), words * 4));
    unsigned count = unsigned(total - 1);
    auto from = source.value + guards * 4, to = output.value + guards * 4;
    void* args[]{&to, &from, &count};
    TEST_CUDA(cuLaunchKernel(function, unsigned(grid), 1, 1, unsigned(block), 1, 1, shared, nullptr, args, nullptr));
    TEST_CUDA(cuMemcpyDtoH(actual.data(), output.value, words * 4));
    for (size_t i = 0; i < count; ++i) {
        const size_t group = i / unsigned(block), peer = group * unsigned(block) + unsigned(block) - 1 - i % unsigned(block);
        expected[guards + i] = (peer < count ? input[guards + peer] : 0u) + uint32_t(group) + 7;
    }
    expect_words(actual, expected, "occupancy-recommended launch results and guards");
    TEST_CUDA(cuMemcpyDtoH(actual.data(), source.value, words * 4));
    expect_words(actual, input, "occupancy-recommended launch source preservation");
}
CUfunction occupancy_callback_function{};
CUdevice occupancy_callback_device{};
unsigned occupancy_callback_calls = 0;
bool occupancy_callback_valid = true;
size_t CUDA_CB occupancy_dynamic_bytes(int block) {
    ++occupancy_callback_calls;
    int maximum = 0, warp = 0;
    if (cuFuncGetAttribute(&maximum, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, occupancy_callback_function) != CUDA_SUCCESS
        || cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, occupancy_callback_device) != CUDA_SUCCESS
        || block <= 0 || block > maximum || warp <= 0) occupancy_callback_valid = false;
    return size_t(block) * 4;
}
void occupancy_potential(CUfunction function, CUdevice device, bool callback, unsigned flags, int limit) {
    const int maximum = occupancy_function_attribute(function, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    const int warp = occupancy_device_attribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    const int processors = occupancy_device_attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    const int cap = limit ? std::min(maximum, limit) : maximum;
    constexpr size_t fixed_shared = 4096;
    require(size_t(cap) * 4 <= fixed_shared, "fixture fixed storage covers each recommended lane");
    occupancy_callback_function = function; occupancy_callback_device = device;
    occupancy_callback_calls = 0; occupancy_callback_valid = true;
    int grid = -1, block = -1;
    TEST_CUDA(cuOccupancyMaxPotentialBlockSizeWithFlags(&grid, &block, function,
        callback ? occupancy_dynamic_bytes : nullptr, callback ? std::numeric_limits<size_t>::max() : fixed_shared, limit, flags));
    require(block > 0 && block <= cap && grid > 0, "potential configuration respects requested ceiling");
    if (callback) require(occupancy_callback_calls > 0 && occupancy_callback_valid, "sizing callback reenters queries with legal block sizes");
    const auto bytes = [&](int candidate) { return callback ? size_t(candidate) * 4 : fixed_shared; };
    int64_t best = 0;
    // Enumerate in ascending order independently of the configurator's search.
    for (int candidate = warp; candidate <= cap; candidate += warp)
        best = std::max(best, int64_t(occupancy_blocks(function, candidate, bytes(candidate), flags)) * candidate);
    best = std::max(best, int64_t(occupancy_blocks(function, cap, bytes(cap), flags)) * cap);
    const int active = occupancy_blocks(function, block, bytes(block), flags);
    require(int64_t(active) * block == best, "recommendation maximizes resident threads among eligible candidates");
    require(int64_t(grid) == int64_t(active) * processors, "minimum occupancy grid follows native per-SM residency");
    if (!flags && !callback) {
        int plain_grid = -1, plain_block = -1;
        TEST_CUDA(cuOccupancyMaxPotentialBlockSize(&plain_grid, &plain_block, function, nullptr, fixed_shared, limit));
        require(grid == plain_grid && block == plain_block, "default potential flags agree with unflagged API");
    }
    occupancy_launch(function, grid, block, unsigned(bytes(block)));
}
void execute_occupancy(const loaded_program& program) {
    const auto function = program.function("mixed_exchange");
    const auto kernel = program.library_kernel("mixed_exchange");
    CUdevice device{}; TEST_CUDA(cuCtxGetDevice(&device));
    const int maximum = occupancy_function_attribute(function, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    const int static_shared = occupancy_function_attribute(function, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    const int dynamic_maximum = occupancy_function_attribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
    const int sm_threads = occupancy_device_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
    const int sm_blocks = occupancy_device_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR);
    const int sm_shared = occupancy_device_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    require(maximum >= 128 && dynamic_maximum >= 4096, "native fixture permits selected block/storage profiles");
    for (int block : {1, 31, 32, 33, 64, 128, maximum, maximum + 1}) {
        int previous = std::numeric_limits<int>::max();
        for (size_t bytes : {size_t(0), size_t(64), size_t(1024), size_t(dynamic_maximum)}) {
            int plain = -1;
            TEST_CUDA(cuOccupancyMaxActiveBlocksPerMultiprocessor(&plain, function, block, bytes));
            const int active = occupancy_blocks(function, block, bytes);
            require(active == plain && active <= previous && active <= sm_blocks
                && int64_t(active) * block <= sm_threads
                && uint64_t(active) * (bytes + unsigned(static_shared)) <= unsigned(sm_shared), "residency respects resource limits and storage monotonicity");
            if (block > maximum) require(active == 0, "oversized block has no residency");
            else if (bytes <= 1024) require(active > 0, "small legal fixture has positive residency");
            previous = active;
            if (kernel) {
                int runtime = -1, runtime_flagged = -1;
                require(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&runtime, reinterpret_cast<const void*>(kernel), block, bytes) == cudaSuccess
                    && cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&runtime_flagged, reinterpret_cast<const void*>(kernel), block, bytes, 0) == cudaSuccess,
                    "Runtime occupancy queries succeed");
                require(runtime == active && runtime_flagged == active
                    && occupancy_blocks(reinterpret_cast<CUfunction>(kernel), block, bytes) == active, "Runtime and contextless occupancy agree");
            }
        }
    }
    const int target = occupancy_blocks(function, 128, 512);
    size_t budget = 999;
    TEST_CUDA(cuOccupancyAvailableDynamicSMemPerBlock(&budget, function, target, 128));
    require(budget >= 512 && uint64_t(target) * (budget + unsigned(static_shared)) <= unsigned(sm_shared), "inverse shared budget respects physical capacity");
    require(occupancy_blocks(function, 128, std::min(budget, size_t(dynamic_maximum))) >= target, "inverse shared budget preserves requested residency");
    if (kernel) {
        size_t runtime_budget = 999;
        require(cudaOccupancyAvailableDynamicSMemPerBlock(&runtime_budget, reinterpret_cast<const void*>(kernel), target, 128) == cudaSuccess
            && runtime_budget == budget, "Runtime inverse budget agrees");
    }
    for (unsigned flags : {unsigned(CU_OCCUPANCY_DEFAULT), unsigned(CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE)})
        for (int limit : {0, 97}) for (bool callback : {false, true}) occupancy_potential(function, device, callback, flags, limit);
    int active = 123, grid = 234, block = 345; size_t shared = 456;
    require(cuOccupancyMaxActiveBlocksPerMultiprocessor(nullptr, function, 32, 0) == CUDA_ERROR_INVALID_VALUE
        && cuOccupancyMaxActiveBlocksPerMultiprocessor(&active, function, 0, 0) == CUDA_ERROR_INVALID_VALUE
        && cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&active, function, 32, 0, 2) == CUDA_ERROR_INVALID_VALUE && active == 123,
        "invalid active-block queries preserve output");
    require(cuOccupancyMaxPotentialBlockSize(nullptr, &block, function, nullptr, 0, 0) == CUDA_ERROR_INVALID_VALUE
        && cuOccupancyMaxPotentialBlockSize(&grid, &block, function, nullptr, 0, -1) == CUDA_ERROR_INVALID_VALUE
        && cuOccupancyMaxPotentialBlockSizeWithFlags(&grid, &block, function, nullptr, 0, 0, 2) == CUDA_ERROR_INVALID_VALUE
        && grid == 234 && block == 345, "invalid potential queries preserve both outputs");
    require(cuOccupancyAvailableDynamicSMemPerBlock(nullptr, function, 1, 32) == CUDA_ERROR_INVALID_VALUE
        && cuOccupancyAvailableDynamicSMemPerBlock(&shared, function, 0, 32) == CUDA_ERROR_INVALID_VALUE && shared == 456,
        "invalid inverse budget preserves output");
    if (kernel) {
        require(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, reinterpret_cast<const void*>(kernel), 0, 0) == cudaErrorInvalidValue
            && active == 123 && cudaGetLastError() == cudaErrorInvalidValue, "Runtime validation records its error and preserves output");
    }
}
}
int main(int argc, char** argv) {
    return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_occupancy);
}
