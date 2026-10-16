RUNTIME_API(cudaFuncGetAttributes, (cudaFuncAttributes* output, const void* token),
    require(output, CUDA_ERROR_INVALID_VALUE, "null function attributes output"); auto function = runtime_kernel(token);
    cudaFuncAttributes result{};
    auto attribute = [&](CUfunction_attribute which) { int value = 0; check(cuFuncGetAttribute(&value, which, function)); return value; };
    result.sharedSizeBytes = attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    result.constSizeBytes = attribute(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
    result.localSizeBytes = attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
    result.maxThreadsPerBlock = attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    result.numRegs = attribute(CU_FUNC_ATTRIBUTE_NUM_REGS);
    result.ptxVersion = attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION);
    result.binaryVersion = attribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION);
    result.cacheModeCA = attribute(CU_FUNC_ATTRIBUTE_CACHE_MODE_CA);
    result.maxDynamicSharedSizeBytes = attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
    result.preferredShmemCarveout = attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
    // Older native drivers cannot query attributes introduced after their ABI.
    auto optional_attribute = [&](CUfunction_attribute which) {
        int value = 0; auto status = cuFuncGetAttribute(&value, which, function);
        if (status != CUDA_ERROR_INVALID_VALUE && status != CUDA_ERROR_NOT_SUPPORTED) check(status);
        return status == CUDA_SUCCESS ? value : 0;
    };
    result.clusterDimMustBeSet = optional_attribute(CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET);
    result.requiredClusterWidth = optional_attribute(CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH);
    result.requiredClusterHeight = optional_attribute(CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT);
    result.requiredClusterDepth = optional_attribute(CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH);
    result.nonPortableClusterSizeAllowed = optional_attribute(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED);
    result.clusterSchedulingPolicyPreference = optional_attribute(CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE);
    result.deviceNodeUpdateStatus = optional_attribute(CU_FUNC_ATTRIBUTE_DEVICE_NODE_UPDATE_SUPPORTED);
    result.sharedMemoryMode = static_cast<cudaSharedMemoryMode>(optional_attribute(CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE));
    *output = result)
RUNTIME_API(cudaFuncSetAttribute, (const void* token, cudaFuncAttribute attribute, int value),
    auto which = attribute == cudaFuncAttributeSharedMemoryMode ? CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE : static_cast<CUfunction_attribute>(attribute);
    require(attribute >= cudaFuncAttributeMaxDynamicSharedMemorySize && attribute <= cudaFuncAttributeSharedMemoryMode,
        CUDA_ERROR_INVALID_VALUE, "invalid Runtime function attribute");
    auto function = runtime_kernel(token); check(runtime_driver_wait([&] { return cuFuncSetAttribute(function, which, value); })))
RUNTIME_API(cudaFuncSetCacheConfig, (const void* token, cudaFuncCache config),
    auto function = runtime_kernel(token); check(runtime_driver_wait([&] { return cuFuncSetCacheConfig(function, static_cast<CUfunc_cache>(config)); })))
RUNTIME_API(cudaFuncSetSharedMemConfig, (const void* token, cudaSharedMemConfig config),
    auto function = runtime_kernel(token); check(runtime_driver_wait([&] { return cuFuncSetSharedMemConfig(function, static_cast<CUsharedconfig>(config)); })))
RUNTIME_API(cudaFuncGetName, (const char** output, const void* token),
    require(output, CUDA_ERROR_INVALID_VALUE, "null function name output"); check(cuFuncGetName(output, runtime_kernel(token))))
RUNTIME_API(cudaFuncGetParamInfo, (const void* token, size_t index, size_t* offset, size_t* size),
    require(offset && size, CUDA_ERROR_INVALID_VALUE, "null function parameter output"); check(cuFuncGetParamInfo(runtime_kernel(token), index, offset, size)))
RUNTIME_API(cudaFuncGetParamCount, (const void* token, size_t* count),
    require(count, CUDA_ERROR_INVALID_VALUE, "null function parameter count output"); check(cuFuncGetParamCount(runtime_kernel(token), count)))
RUNTIME_API(cudaOccupancyMaxActiveBlocksPerMultiprocessor, (int* output, const void* token, int block, size_t shared),
    require(output, CUDA_ERROR_INVALID_VALUE, "null occupancy output"); check(cuOccupancyMaxActiveBlocksPerMultiprocessor(output, runtime_kernel(token), block, shared)))
RUNTIME_API(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, (int* output, const void* token, int block, size_t shared, unsigned flags),
    require(output, CUDA_ERROR_INVALID_VALUE, "null occupancy output"); check(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(output, runtime_kernel(token), block, shared, flags)))
RUNTIME_API(cudaOccupancyAvailableDynamicSMemPerBlock, (size_t* output, const void* token, int blocks, int block),
    require(output, CUDA_ERROR_INVALID_VALUE, "null shared memory occupancy output"); check(cuOccupancyAvailableDynamicSMemPerBlock(output, runtime_kernel(token), blocks, block)))
