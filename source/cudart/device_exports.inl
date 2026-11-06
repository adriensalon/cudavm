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
RUNTIME_API(cudaDeviceCanAccessPeer, (int* output, int device, int peer),
    require(output, CUDA_ERROR_INVALID_VALUE, "null peer access output"); check(cuInit(0)); check(cuDeviceCanAccessPeer(output, device, peer)))
RUNTIME_API(cudaDeviceEnablePeerAccess, (int peer, unsigned flags),
    require(!flags, CUDA_ERROR_INVALID_VALUE, "peer access flags must be zero"); runtime_current();
    CUdevice device = 0; check(cuCtxGetDevice(&device)); CUdevice native = 0; check(cuDeviceGet(&native, peer));
    require(device != peer, CUDA_ERROR_INVALID_DEVICE, "peer equals current device");
    check(cuCtxEnablePeerAccess(retain_primary(peer), flags)))
RUNTIME_API(cudaDeviceDisablePeerAccess, (int peer),
    runtime_current(); CUdevice device = 0; check(cuCtxGetDevice(&device)); CUdevice native = 0; check(cuDeviceGet(&native, peer));
    require(device != peer, CUDA_ERROR_INVALID_DEVICE, "peer equals current device");
    check(cuCtxDisablePeerAccess(retain_primary(peer))))
RUNTIME_API(cudaDeviceGetP2PAttribute, (int* output, cudaDeviceP2PAttr attribute, int source, int target), check(cuInit(0)); check(cuDeviceGetP2PAttribute(output, static_cast<CUdevice_P2PAttribute>(attribute), source, target)))
RUNTIME_API(cudaDeviceGetP2PAtomicCapabilities, (unsigned* output, const cudaAtomicOperation* operations, unsigned count, int source, int target),
    require(output && operations && count, CUDA_ERROR_INVALID_VALUE, "missing peer atomic capability array or empty count"); check(cuInit(0));
    std::vector<CUatomicOperation> native; native.reserve(count); for (unsigned i = 0; i < count; ++i) native.push_back(static_cast<CUatomicOperation>(operations[i]));
    check(cuDeviceGetP2PAtomicCapabilities(output, native.data(), count, source, target)))
