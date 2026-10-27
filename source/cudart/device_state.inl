void runtime_device_flags(unsigned flags) {
    constexpr unsigned allowed = cudaDeviceScheduleMask | cudaDeviceLmemResizeToMax | cudaDeviceSyncMemops;
    require(!(flags & ~allowed), CUDA_ERROR_INVALID_VALUE, "invalid Runtime device flags");
    auto schedule = flags & cudaDeviceScheduleMask;
    require(schedule == cudaDeviceScheduleAuto || schedule == cudaDeviceScheduleSpin ||
            schedule == cudaDeviceScheduleYield || schedule == cudaDeviceScheduleBlockingSync,
            CUDA_ERROR_INVALID_VALUE, "invalid Runtime scheduling combination");
}
cudaDeviceProp runtime_device_properties(int device) {
    CUdevice native = 0; check(cuDeviceGet(&native, device)); cudaDeviceProp value{};
    check(cuDeviceGetName(value.name, sizeof(value.name), native)); check(cuDeviceTotalMem(&value.totalGlobalMem, native));
    auto identity = [&](CUresult status) { if (status != CUDA_ERROR_NOT_SUPPORTED) check(status); };
    CUuuid uuid{}; identity(cuDeviceGetUuid(&uuid, native)); static_assert(sizeof(uuid.bytes) == sizeof(value.uuid.bytes));
    std::memcpy(value.uuid.bytes, uuid.bytes, sizeof(uuid.bytes));
#if defined(_WIN32)
    identity(cuDeviceGetLuid(value.luid, &value.luidDeviceNodeMask, native));
#endif
    auto get = [&](CUdevice_attribute attribute, bool optional = false, int fallback = 0) {
        int result = 0; auto code = cuDeviceGetAttribute(&result, attribute, native);
        if (optional && (code == CUDA_ERROR_INVALID_VALUE || code == CUDA_ERROR_NOT_SUPPORTED)) return fallback;
        check(code); return result;
    };
#define DEVICE_PROPERTY(field, attribute) value.field = get(CU_DEVICE_ATTRIBUTE_##attribute);
#define DEVICE_OPTIONAL_PROPERTY(field, attribute, fallback) value.field = get(CU_DEVICE_ATTRIBUTE_##attribute, true, fallback);
#include "../cuda/device_properties.inl"
#undef DEVICE_PROPERTY
#undef DEVICE_OPTIONAL_PROPERTY
    return value;
}
int runtime_choose_device(const cudaDeviceProp& wanted) {
    int count = 0; check(cuDeviceGetCount(&count)); require(count > 0, CUDA_ERROR_NO_DEVICE, "no devices to choose from");
    int selected = -1, best = -1;
    for (int device = 0; device < count; ++device) {
        auto value = runtime_device_properties(device); int score = 0;
        auto match = [&](auto requested, auto available) { if (requested > 0 && available >= requested) ++score; };
        if ((wanted.major || wanted.minor) && (value.major > wanted.major || (value.major == wanted.major && value.minor >= wanted.minor))) ++score;
        match(wanted.totalGlobalMem, value.totalGlobalMem); match(wanted.sharedMemPerBlock, value.sharedMemPerBlock);
        match(wanted.regsPerBlock, value.regsPerBlock); match(wanted.warpSize, value.warpSize); match(wanted.memPitch, value.memPitch);
        match(wanted.maxThreadsPerBlock, value.maxThreadsPerBlock); match(wanted.totalConstMem, value.totalConstMem);
        match(wanted.multiProcessorCount, value.multiProcessorCount);
        for (int i = 0; i < 3; ++i) { match(wanted.maxThreadsDim[i], value.maxThreadsDim[i]); match(wanted.maxGridSize[i], value.maxGridSize[i]); }
        if (score > best) { best = score; selected = device; }
    }
    return selected;
}
