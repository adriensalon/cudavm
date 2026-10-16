RUNTIME_API(cudaSetDeviceFlags, (unsigned flags),
    runtime_device_flags(flags);
    check(cuInit(0)); CUcontext owner = nullptr; check(cuCtxGetCurrent(&owner));
    if (owner) { runtime_current(); check(cuCtxSetFlags(flags)); }
    else { check(cuDevicePrimaryCtxSetFlags(runtime_device, flags)); runtime_current(); })
RUNTIME_API(cudaGetDeviceFlags, (unsigned* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null device flags output"); check(cuInit(0));
    CUcontext owner = nullptr; check(cuCtxGetCurrent(&owner));
    if (owner && cudavmDriverContextExists(owner)) check(cuCtxGetFlags(output));
    else {
        int device = owner ? cudavmDriverPrimaryDevice(owner) : runtime_device;
        require(device >= 0, CUDA_ERROR_INVALID_CONTEXT, "current Driver context was destroyed");
        int active = 0; check(cuDevicePrimaryCtxGetState(device, output, &active));
        if (active) *output |= cudaDeviceMapHost;
    })
RUNTIME_API(cudaDeviceGetLimit, (size_t* output, cudaLimit limit),
    require(output && ((limit >= cudaLimitStackSize && limit <= cudaLimitPersistingL2CacheSize) || limit == cudaLimitPerBlockMemorySize),
        CUDA_ERROR_INVALID_VALUE, "invalid device limit query"); runtime_current(); check(cuCtxGetLimit(output, static_cast<CUlimit>(limit))))
RUNTIME_API(cudaDeviceSetLimit, (cudaLimit limit, size_t value),
    require((limit >= cudaLimitStackSize && limit <= cudaLimitPersistingL2CacheSize) || limit == cudaLimitPerBlockMemorySize,
        CUDA_ERROR_INVALID_VALUE, "invalid device limit");
    runtime_current(); check(runtime_driver_wait([&] { return cuCtxSetLimit(static_cast<CUlimit>(limit), value); })))
RUNTIME_API(cudaDeviceGetCacheConfig, (cudaFuncCache* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null device cache configuration output"); runtime_current();
    CUfunc_cache config{}; check(cuCtxGetCacheConfig(&config)); *output = static_cast<cudaFuncCache>(config))
RUNTIME_API(cudaDeviceSetCacheConfig, (cudaFuncCache config), runtime_current(); check(cuCtxSetCacheConfig(static_cast<CUfunc_cache>(config))))
RUNTIME_API(cudaCtxResetPersistingL2Cache, (), runtime_current(); check(cuCtxResetPersistingL2Cache()))
RUNTIME_API(cudaDeviceGetSharedMemConfig, (cudaSharedMemConfig* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null device shared memory configuration output"); runtime_current();
    CUsharedconfig config{}; check(cuCtxGetSharedMemConfig(&config)); *output = static_cast<cudaSharedMemConfig>(config))
RUNTIME_API(cudaDeviceSetSharedMemConfig, (cudaSharedMemConfig config), runtime_current(); check(cuCtxSetSharedMemConfig(static_cast<CUsharedconfig>(config))))
