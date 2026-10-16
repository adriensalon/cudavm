void potential_configuration(const std::shared_ptr<context>& owner, CUfunction handle, const std::shared_ptr<kernel>& function,
                             int& grid, int& block, CUoccupancyB2DSize callback, size_t shared, int limit, unsigned flags) {
    if (!callback) {
        context_configuration(owner, [&] { function->potential_block_size(grid, block, nullptr, shared, limit, flags); });
        return;
    }
    // User sizing callbacks run outside registry/execution locks. Obtain every
    // residency result through the public handle translation, so callbacks may
    // query CUDA without deadlocking and unloading still invalidates the handle.
    driver_wait([&] {
        auto check = [](CUresult code) { require(code == CUDA_SUCCESS, code, "occupancy configuration query failed"); };
        int maximum = 0, warp = 0, threads = 0, processors = 0;
        check(cuFuncGetAttribute(&maximum, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, handle));
        check(cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, owner->device));
        check(cuDeviceGetAttribute(&threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, owner->device));
        check(cuDeviceGetAttribute(&processors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, owner->device));
        require(maximum > 0 && warp > 0 && threads > 0 && processors > 0, CUDA_ERROR_NOT_SUPPORTED, "native occupancy resource limits are unavailable");
        if (limit) maximum = std::min(maximum, limit);
        int64_t best_threads = 0; int best_blocks = 0;
        for (int64_t aligned = ((int64_t(maximum) + warp - 1) / warp) * warp; aligned > 0; aligned -= warp) {
            int candidate = int(std::min<int64_t>(maximum, aligned)), blocks = 0;
            auto bytes = callback(candidate);
            check(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&blocks, handle, candidate, bytes, flags));
            auto active = int64_t(blocks) * candidate;
            if (active > best_threads) { best_threads = active; best_blocks = blocks; block = candidate; }
            if (best_threads == threads) break;
        }
        require(int64_t(best_blocks) * processors <= std::numeric_limits<int>::max(), CUDA_ERROR_INVALID_VALUE, "occupancy grid size overflow");
        grid = best_blocks * processors;
    });
}
