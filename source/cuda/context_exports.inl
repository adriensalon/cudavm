// Preserve the ABI used by clients built before CUDA 12.5.
DRIVER_API(cuCtxCreate_v2, (CUcontext* output, unsigned flags, CUdevice device),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null context output");
    auto owner = create(device, false, flags); context_stack().push_back(owner->handle); *output = owner->handle)
DRIVER_API(cuCtxCreate_v3, (CUcontext* output, CUexecAffinityParam* parameters, int count, unsigned flags, CUdevice device),
    driver_ready(); require(output && count >= 0 && (!count || parameters), CUDA_ERROR_INVALID_VALUE, "invalid context affinity parameters");
    require(count == 0, CUDA_ERROR_NOT_SUPPORTED, "context affinity is not implemented");
    auto owner = create(device, false, flags); context_stack().push_back(owner->handle); *output = owner->handle)
DRIVER_API(cuCtxGetDevice_v2, (CUdevice* device, CUcontext handle),
    driver_ready(); require(device, CUDA_ERROR_INVALID_VALUE, "null device output"); *device = (handle ? lookup(handle) : current())->device)
DRIVER_API(cuCtxSynchronize_v2, (CUcontext handle), driver_ready(); synchronize_context(handle ? lookup(handle) : current()))
DRIVER_API(cuCtxSetFlags, (unsigned flags), driver_ready(); set_context_flags(current(), flags))
DRIVER_API(cuCtxResetPersistingL2Cache, (), driver_ready(); auto owner = current(); context_configuration(owner, [&] { owner->native->reset_persisting_cache(); }))
DRIVER_API(cuDevicePrimaryCtxSetFlags, (CUdevice device, unsigned flags), driver_ready(); set_primary_flags(device, flags))
// The old ABI rejects an active primary context; the current ABI updates it.
#undef cuDevicePrimaryCtxSetFlags
DRIVER_API(cuDevicePrimaryCtxSetFlags, (CUdevice device, unsigned flags), driver_ready(); set_primary_flags(device, flags, true))
DRIVER_API(cuCtxGetId, (CUcontext handle, unsigned long long* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null context identity output");
    *output = (handle ? lookup(handle) : current())->id)
DRIVER_API(cuDevicePrimaryCtxReset, (CUdevice device), driver_ready(); reset_primary(device))
// The pre-11.0 ABI also releases one usage; v2 preserves every retain.
#undef cuDevicePrimaryCtxReset
DRIVER_API(cuDevicePrimaryCtxReset, (CUdevice device), driver_ready(); reset_primary(device);
    auto it = global().primary_records.find(device);
    if (it != global().primary_records.end() && it->second.retains) release_primary(device))
DRIVER_API(cuCtxGetApiVersion, (CUcontext handle, unsigned* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null context API version output"); auto owner = handle ? lookup(handle) : current();
    *output = context_configuration(owner, [&] { return owner->native->context_api_version(); }))
DRIVER_API(cuCtxGetLimit, (size_t* output, CUlimit limit),
    driver_ready(); require(output && limit >= CU_LIMIT_STACK_SIZE && limit < CU_LIMIT_MAX, CUDA_ERROR_INVALID_VALUE, "invalid context limit query"); auto owner = current();
    *output = context_configuration(owner, [&] { return owner->native->context_limit(limit); }))
DRIVER_API(cuCtxSetLimit, (CUlimit limit, size_t value),
    driver_ready(); require(limit >= CU_LIMIT_STACK_SIZE && limit < CU_LIMIT_MAX, CUDA_ERROR_INVALID_VALUE, "invalid context limit");
    auto owner = current(); native_call(owner, [&] { owner->native->set_context_limit(limit, value); }))
DRIVER_API(cuCtxGetCacheConfig, (CUfunc_cache* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null cache configuration output"); auto owner = current();
    *output = context_configuration(owner, [&] { return owner->native->context_cache_config(); }))
DRIVER_API(cuCtxSetCacheConfig, (CUfunc_cache config),
    driver_ready(); require(config >= CU_FUNC_CACHE_PREFER_NONE && config <= CU_FUNC_CACHE_PREFER_EQUAL, CUDA_ERROR_INVALID_VALUE, "invalid cache preference");
    auto owner = current(); context_configuration(owner, [&] { owner->native->set_context_cache_config(config); }))
DRIVER_API(cuCtxGetSharedMemConfig, (CUsharedconfig* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null shared memory configuration output"); auto owner = current();
    *output = context_configuration(owner, [&] { return owner->native->context_shared_config(); }))
DRIVER_API(cuCtxSetSharedMemConfig, (CUsharedconfig config),
    driver_ready(); require(config >= CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE && config <= CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE,
        CUDA_ERROR_INVALID_VALUE, "invalid shared memory bank preference");
    auto owner = current(); context_configuration(owner, [&] { owner->native->set_context_shared_config(config); }))
