RUNTIME_API(cudaGetDeviceProperties, (cudaDeviceProp* output, int device),
    require(output, CUDA_ERROR_INVALID_VALUE, "null device properties output"); check(cuInit(0)); *output = runtime_device_properties(device))
RUNTIME_API(cudaChooseDevice, (int* output, const cudaDeviceProp* desired),
    require(output && desired, CUDA_ERROR_INVALID_VALUE, "null device selection arguments"); auto wanted = *desired; check(cuInit(0)); *output = runtime_choose_device(wanted))
RUNTIME_API(cudaInitDevice, (int device, unsigned device_flags, unsigned flags),
    if (flags & cudaInitDeviceFlagsAreValid) runtime_device_flags(device_flags);
    check(cuInit(0)); CUdevice native = 0; check(cuDeviceGet(&native, device));
    if (flags & cudaInitDeviceFlagsAreValid) check(cuDevicePrimaryCtxSetFlags(native, device_flags));
    retain_primary(device))
RUNTIME_API(cudaDeviceGetPCIBusId, (char* output, int length, int device),
    require(output && length > 0, CUDA_ERROR_INVALID_VALUE, "invalid PCI bus output"); check(cuInit(0)); check(cuDeviceGetPCIBusId(output, length, device)))
RUNTIME_API(cudaDeviceGetByPCIBusId, (int* output, const char* text),
    require(output && text && *text, CUDA_ERROR_INVALID_VALUE, "invalid PCI device query"); check(cuInit(0)); CUdevice device = 0; check(cuDeviceGetByPCIBusId(&device, text)); *output = device)
