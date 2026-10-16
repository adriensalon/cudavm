DRIVER_API(cuFuncGetAttribute, (int* output, CUfunction_attribute attribute, CUfunction handle),
    driver_ready(); require(output && attribute >= 0 && attribute < CU_FUNC_ATTRIBUTE_MAX, CUDA_ERROR_INVALID_VALUE, "invalid function attribute query");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native;
    *output = context_configuration(owner, [&] { return function->attribute(attribute); }))
DRIVER_API(cuFuncSetAttribute, (CUfunction handle, CUfunction_attribute attribute, int value),
    driver_ready(); require(attribute >= CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES && attribute < CU_FUNC_ATTRIBUTE_MAX &&
        attribute != CU_FUNC_ATTRIBUTE_DEVICE_NODE_UPDATE_SUPPORTED, CUDA_ERROR_INVALID_VALUE, "function attribute is not writable");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native;
    native_call(owner, [&] { function->set_attribute(attribute, value);
        for (auto& [id, entry] : global().functions) if (entry.native == function) entry.overrides.insert(attribute); }, true))
DRIVER_API(cuFuncSetCacheConfig, (CUfunction handle, CUfunc_cache config),
    driver_ready(); require(config >= CU_FUNC_CACHE_PREFER_NONE && config <= CU_FUNC_CACHE_PREFER_EQUAL, CUDA_ERROR_INVALID_VALUE, "invalid function cache preference");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native;
    native_call(owner, [&] { function->set_cache_config(config);
        for (auto& [id, entry] : global().functions) if (entry.native == function) entry.cache_override = true; }, true))
DRIVER_API(cuFuncSetSharedMemConfig, (CUfunction handle, CUsharedconfig config),
    driver_ready(); require(config >= CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE && config <= CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE,
        CUDA_ERROR_INVALID_VALUE, "invalid function shared memory bank preference");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native;
    native_call(owner, [&] { function->set_shared_config(config); }, true))
DRIVER_API(cuOccupancyMaxActiveBlocksPerMultiprocessor, (int* output, CUfunction handle, int block, size_t shared),
    driver_ready(); require(output && block > 0, CUDA_ERROR_INVALID_VALUE, "invalid occupancy query");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native;
    *output = context_configuration(owner, [&] { return function->active_blocks(block, shared, CU_OCCUPANCY_DEFAULT); }))
DRIVER_API(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, (int* output, CUfunction handle, int block, size_t shared, unsigned flags),
    driver_ready(); require(output && block > 0 && !(flags & ~unsigned(CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE)), CUDA_ERROR_INVALID_VALUE, "invalid occupancy query or flags");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native;
    *output = context_configuration(owner, [&] { return function->active_blocks(block, shared, flags); }))
DRIVER_API(cuOccupancyMaxPotentialBlockSize, (int* grid, int* block, CUfunction handle, CUoccupancyB2DSize callback, size_t shared, int limit),
    driver_ready(); require(grid && block && limit >= 0, CUDA_ERROR_INVALID_VALUE, "invalid occupancy launch configuration query");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native; int grid_value = 0, block_value = 0;
    potential_configuration(owner, handle, function, grid_value, block_value, callback, shared, limit, CU_OCCUPANCY_DEFAULT);
    *grid = grid_value; *block = block_value)
DRIVER_API(cuOccupancyMaxPotentialBlockSizeWithFlags, (int* grid, int* block, CUfunction handle, CUoccupancyB2DSize callback, size_t shared, int limit, unsigned flags),
    driver_ready(); require(grid && block && limit >= 0 && !(flags & ~unsigned(CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE)), CUDA_ERROR_INVALID_VALUE, "invalid occupancy launch configuration query or flags");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native; int grid_value = 0, block_value = 0;
    potential_configuration(owner, handle, function, grid_value, block_value, callback, shared, limit, flags);
    *grid = grid_value; *block = block_value)
DRIVER_API(cuOccupancyAvailableDynamicSMemPerBlock, (size_t* output, CUfunction handle, int blocks, int block),
    driver_ready(); require(output && blocks > 0 && block > 0, CUDA_ERROR_INVALID_VALUE, "invalid shared memory occupancy query");
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native;
    *output = context_configuration(owner, [&] { return function->available_dynamic_shared(blocks, block); }))
