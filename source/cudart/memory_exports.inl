RUNTIME_API(cudaMallocManaged, (void** output, size_t bytes, unsigned flags),
    require(output && bytes && (flags == cudaMemAttachGlobal || flags == cudaMemAttachHost), CUDA_ERROR_INVALID_VALUE, "invalid managed allocation arguments");
    runtime_current(); CUdeviceptr pointer = 0; check(runtime_driver_wait([&] { return cuMemAllocManaged(&pointer, bytes, flags); })); *output = reinterpret_cast<void*>(uintptr_t(pointer)))
RUNTIME_API(cudaMemcpy_ptds, (void* to, const void* from, size_t bytes, cudaMemcpyKind kind),
    runtime_current(); require(kind >= cudaMemcpyHostToHost && kind <= cudaMemcpyDefault, CUDA_ERROR_INVALID_VALUE, "invalid copy direction"); if (!bytes) return;
    auto types = runtime_copy_types(to, from, kind);
    if (types.to == CU_MEMORYTYPE_HOST && types.from == CU_MEMORYTYPE_HOST) { require(to && from, CUDA_ERROR_INVALID_VALUE, "null host copy pointer"); std::memcpy(to, from, bytes); return; }
    auto destination = reinterpret_cast<uintptr_t>(to), source = reinterpret_cast<uintptr_t>(from);
    if (types.to == CU_MEMORYTYPE_HOST) check(cuMemcpyDtoHAsync(to, source, bytes, CU_STREAM_PER_THREAD));
    else if (types.from == CU_MEMORYTYPE_HOST) check(cuMemcpyHtoDAsync(destination, from, bytes, CU_STREAM_PER_THREAD));
    else check(cuMemcpyDtoDAsync(destination, source, bytes, CU_STREAM_PER_THREAD));
    check(runtime_driver_wait([&] { return cuStreamSynchronize(CU_STREAM_PER_THREAD); })))
RUNTIME_API(cudaMemGetInfo, (size_t* free, size_t* total),
    require(free && total, CUDA_ERROR_INVALID_VALUE, "null memory info output"); runtime_current();
    check(cuMemGetInfo(free, total)))
RUNTIME_API(cudaMallocPitch, (void** output, size_t* pitch, size_t width, size_t height),
    require(output && pitch, CUDA_ERROR_INVALID_VALUE, "null pitched allocation outputs"); runtime_current();
    CUdeviceptr pointer = 0; size_t value = 0;
    if (width && height) check(runtime_driver_wait([&] { return cuMemAllocPitch(&pointer, &value, width, height, 16); }));
    *output = reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer)); *pitch = value)
RUNTIME_API(cudaMalloc3D, (cudaPitchedPtr* output, cudaExtent extent),
    require(output, CUDA_ERROR_INVALID_VALUE, "null 3D allocation output"); runtime_current();
    CUdeviceptr pointer = 0; size_t pitch = 0;
    if (extent.width && extent.height && extent.depth) check(runtime_driver_wait([&] { return cuMemAllocPitch(&pointer, &pitch, extent.width, memory_mul(extent.height, extent.depth), 16); }));
    *output = {reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer)), pitch, extent.width, extent.height})
RUNTIME_API(cudaMemcpy2D, (void* to, size_t to_pitch, const void* from, size_t from_pitch, size_t width, size_t height, cudaMemcpyKind kind),
    runtime_current(); auto types = runtime_copy_types(to, from, kind);
    if (width && height) { memory_pitch(to_pitch, width); memory_pitch(from_pitch, width); }
    CUDA_MEMCPY2D p{}; p.srcMemoryType = types.from; p.srcHost = from; p.srcDevice = reinterpret_cast<std::uintptr_t>(from); p.srcPitch = from_pitch;
    p.dstMemoryType = types.to; p.dstHost = to; p.dstDevice = reinterpret_cast<std::uintptr_t>(to); p.dstPitch = to_pitch;
    p.WidthInBytes = width; p.Height = height; check(runtime_driver_wait([&] { return cuMemcpy2D(&p); })))
RUNTIME_API(cudaMemcpy3D, (const cudaMemcpy3DParms* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null 3D copy descriptor"); runtime_current(); auto descriptor = runtime_copy_3d(*p); check(runtime_driver_wait([&] { return cuMemcpy3D(&descriptor); })))
RUNTIME_API(cudaMemcpyPeer, (void* to, int to_device, const void* from, int from_device, size_t bytes),
    runtime_current(); CUdevice device = 0; check(cuDeviceGet(&device, to_device)); check(cuDeviceGet(&device, from_device));
    if (!bytes) return;
    auto to_owner = runtime_peer_context(to, to_device), from_owner = runtime_peer_context(from, from_device);
    check(runtime_driver_wait([&] { return cuMemcpyPeer(reinterpret_cast<std::uintptr_t>(to), to_owner, reinterpret_cast<std::uintptr_t>(from), from_owner, bytes); })))
RUNTIME_API(cudaMemcpy3DPeer, (const cudaMemcpy3DPeerParms* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null 3D peer descriptor"); runtime_current(); auto descriptor = runtime_peer_3d(*p);
    if (!p->extent.width || !p->extent.height || !p->extent.depth) return;
    check(runtime_driver_wait([&] { return cuMemcpy3DPeer(&descriptor); })))
RUNTIME_API(cudaMemset2D, (void* pointer, size_t pitch, int value, size_t width, size_t height),
    runtime_current(); if (width && height) memory_pitch(pitch, width);
    check(runtime_driver_wait([&] { return cuMemsetD2D8(reinterpret_cast<std::uintptr_t>(pointer), pitch, static_cast<unsigned char>(value), width, height); })))
RUNTIME_API(cudaMemset3D, (cudaPitchedPtr pointer, int value, cudaExtent extent),
    runtime_current(); auto pitch = extent.height == 1 && extent.depth == 1 ? extent.width : pointer.pitch;
    if (extent.width && extent.height && extent.depth) memory_pitch(pitch, extent.width);
    check(runtime_driver_wait([&] { return cudavmDriverMemset3D(reinterpret_cast<std::uintptr_t>(pointer.ptr), pitch, pointer.ysize, static_cast<unsigned char>(value), extent.width, extent.height, extent.depth); })))
RUNTIME_API(cudaPointerGetAttributes, (cudaPointerAttributes* output, const void* pointer),
    require(output, CUDA_ERROR_INVALID_VALUE, "null pointer attributes output"); auto owner = runtime_current();
    cudaPointerAttributes value{}; value.localityDomainOrdinal = -1;
    auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (cudavmDriverIsManagedPointer(address)) {
        value.type = cudaMemoryTypeManaged; value.hostPointer = const_cast<void*>(pointer); value.devicePointer = const_cast<void*>(pointer);
        check(cuPointerGetAttribute(&value.device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, address));
    } else if (cudavmDriverIsHostPointer(address)) {
        value.type = cudaMemoryTypeHost;
        check(cuPointerGetAttribute(&value.hostPointer, CU_POINTER_ATTRIBUTE_HOST_POINTER, address));
        check(cuPointerGetAttribute(&value.device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, address));
        CUdeviceptr alias = 0; CUpointer_attribute attribute = CU_POINTER_ATTRIBUTE_DEVICE_POINTER; void* view = &alias;
        check(cuPointerGetAttributes(1, &attribute, &view, address));
        value.devicePointer = reinterpret_cast<void*>(static_cast<uintptr_t>(alias));
    } else if (cudavmDriverIsDevicePointer(address)) {
        CUcontext pointer_owner = nullptr; check(cuPointerGetAttribute(&pointer_owner, CU_POINTER_ATTRIBUTE_CONTEXT, address));
        check(cuPointerGetAttribute(&value.device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, address));
        value.type = cudaMemoryTypeDevice;
        int current_device = -1; check(cuCtxGetDevice(&current_device));
        if (owner == pointer_owner || (!pointer_owner && current_device == value.device)) value.devicePointer = const_cast<void*>(pointer);
    } else { value.type = cudaMemoryTypeUnregistered; value.hostPointer = const_cast<void*>(pointer); }
    *output = value)
