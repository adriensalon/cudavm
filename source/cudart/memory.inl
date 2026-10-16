size_t memory_mul(size_t a, size_t b) {
    require(!a || b <= std::numeric_limits<size_t>::max() / a, CUDA_ERROR_INVALID_VALUE, "memory geometry multiplication overflow");
    return a * b;
}
void memory_pitch(size_t pitch, size_t width, size_t x = 0, int device = -1) {
    int maximum = 0;
    if (device < 0) check(cuCtxGetDevice(&device));
    check(cuDeviceGetAttribute(&maximum, CU_DEVICE_ATTRIBUTE_MAX_PITCH, device));
    symbol_require(pitch <= static_cast<size_t>(maximum) && x <= pitch && width <= pitch - x,
                   cudaErrorInvalidPitchValue, "memory width or X offset exceeds pitch");
}
struct copy_types { CUmemorytype to, from; };
copy_types runtime_copy_types(void* to, const void* from, cudaMemcpyKind kind, bool to_array = false, bool from_array = false) {
    symbol_require(kind >= cudaMemcpyHostToHost && kind <= cudaMemcpyDefault, cudaErrorInvalidMemcpyDirection, "invalid copy direction");
    bool device_to = to_array || cudavmDriverIsDevicePointer(reinterpret_cast<std::uintptr_t>(to));
    bool device_from = from_array || cudavmDriverIsDevicePointer(reinterpret_cast<std::uintptr_t>(from));
    // Mapped addresses may represent both endpoints. Explicit directions select
    // the requested view; Default uses the ordinary host view for host aliases.
    bool host_to = cudavmDriverIsHostPointer(reinterpret_cast<std::uintptr_t>(to));
    bool host_from = cudavmDriverIsHostPointer(reinterpret_cast<std::uintptr_t>(from));
    if (host_to) device_to = kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice;
    if (host_from) device_from = kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice;
    // Managed storage has both views. Default selects its unified device
    // address; an explicit direction can also select its host-accessible view.
    if (kind != cudaMemcpyDefault) {
        if (cudavmDriverIsManagedPointer(reinterpret_cast<std::uintptr_t>(to))) device_to = kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice;
        if (cudavmDriverIsManagedPointer(reinterpret_cast<std::uintptr_t>(from))) device_from = kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice;
    }
    auto inferred = device_to ? (device_from ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                              : (device_from ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    symbol_require(kind == cudaMemcpyDefault || kind == inferred, cudaErrorInvalidMemcpyDirection, "copy direction disagrees with pointer ownership");
    return {to_array ? CU_MEMORYTYPE_ARRAY : device_to ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST,
            from_array ? CU_MEMORYTYPE_ARRAY : device_from ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST};
}
CUDA_MEMCPY2D runtime_array_copy_2d(cudaArray_t to_array, void* to, size_t to_pitch, size_t to_x, size_t to_y,
                                  cudaArray_const_t from_array, const void* from, size_t from_pitch, size_t from_x, size_t from_y,
                                  size_t width, size_t height, cudaMemcpyKind kind) {
    auto types = runtime_copy_types(to,from,kind,to_array,from_array);
    if (width && height) {
        if (!to_array) memory_pitch(to_pitch,width,to_x);
        if (!from_array) memory_pitch(from_pitch,width,from_x);
    }
    CUDA_MEMCPY2D p{};
    p.srcMemoryType=types.from; p.srcArray=reinterpret_cast<CUarray>(const_cast<cudaArray*>(from_array));
    p.srcHost=from; p.srcDevice=reinterpret_cast<uintptr_t>(from); p.srcPitch=from_pitch; p.srcXInBytes=from_x; p.srcY=from_y;
    p.dstMemoryType=types.to; p.dstArray=reinterpret_cast<CUarray>(to_array);
    p.dstHost=to; p.dstDevice=reinterpret_cast<uintptr_t>(to); p.dstPitch=to_pitch; p.dstXInBytes=to_x; p.dstY=to_y;
    p.WidthInBytes=width; p.Height=height; return p;
}
CUDA_MEMCPY3D runtime_copy_3d(const cudaMemcpy3DParms& p) {
    require(!(p.srcArray && p.srcPtr.ptr) && !(p.dstArray && p.dstPtr.ptr), CUDA_ERROR_INVALID_VALUE, "multiple 3D copy endpoints");
    auto kind = p.kind;
    // CUDA 3D copies correct host/device implications contradicted by arrays.
    if (kind >= cudaMemcpyHostToHost && kind <= cudaMemcpyDeviceToHost) {
        bool to_device = p.dstArray || kind == cudaMemcpyHostToDevice;
        bool from_device = p.srcArray || kind == cudaMemcpyDeviceToHost;
        kind = to_device ? (from_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) : (from_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    }
    auto types = runtime_copy_types(p.dstPtr.ptr, p.srcPtr.ptr, kind, p.dstArray, p.srcArray);
    auto source_element = p.srcArray ? runtime_array_element(p.srcArray) : size_t(0);
    auto destination_element = p.dstArray ? runtime_array_element(p.dstArray) : size_t(0);
    require(!source_element || !destination_element || source_element == destination_element, CUDA_ERROR_INVALID_VALUE, "3D array elements differ in size");
    auto element = std::max({size_t(1), source_element, destination_element});
    auto width = memory_mul(p.extent.width, element);
    if (p.extent.width && p.extent.height && p.extent.depth) {
        if (!p.srcArray) memory_pitch(p.srcPtr.pitch, width, p.srcPos.x);
        if (!p.dstArray) memory_pitch(p.dstPtr.pitch, width, p.dstPos.x);
    }
    CUDA_MEMCPY3D result{};
    result.srcMemoryType = types.from; result.srcHost = p.srcPtr.ptr; result.srcDevice = reinterpret_cast<std::uintptr_t>(p.srcPtr.ptr);
    result.srcPitch = p.srcPtr.pitch; result.srcHeight = p.srcPtr.ysize;
    result.srcArray = reinterpret_cast<CUarray>(p.srcArray);
    result.srcXInBytes = p.srcArray ? memory_mul(p.srcPos.x, element) : p.srcPos.x; result.srcY = p.srcPos.y; result.srcZ = p.srcPos.z;
    result.dstMemoryType = types.to; result.dstHost = p.dstPtr.ptr; result.dstDevice = reinterpret_cast<std::uintptr_t>(p.dstPtr.ptr);
    result.dstPitch = p.dstPtr.pitch; result.dstHeight = p.dstPtr.ysize;
    result.dstArray = reinterpret_cast<CUarray>(p.dstArray);
    result.dstXInBytes = p.dstArray ? memory_mul(p.dstPos.x, element) : p.dstPos.x; result.dstY = p.dstPos.y; result.dstZ = p.dstPos.z;
    result.WidthInBytes = width; result.Height = p.extent.height; result.Depth = p.extent.depth;
    return result;
}
CUcontext runtime_peer_context(const void* pointer, int device) {
    CUdevice valid = 0; check(cuDeviceGet(&valid, device));
    int ordinal = -1; CUcontext owner = nullptr;
    auto address = reinterpret_cast<std::uintptr_t>(pointer);
    check(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, address));
    require(ordinal == device, CUDA_ERROR_INVALID_DEVICE, "peer pointer belongs to a different device");
    check(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, address));
    if (!owner) {
        CUcontext active = nullptr; int active_device = -1;
        check(cuCtxGetCurrent(&active)); if (active) check(cuCtxGetDevice(&active_device));
        owner = active_device == device ? active : retain_primary(device);
    }
    return owner;
}
CUDA_MEMCPY3D_PEER runtime_peer_3d(const cudaMemcpy3DPeerParms& p) {
    require(!(p.srcArray && p.srcPtr.ptr) && !(p.dstArray && p.dstPtr.ptr), CUDA_ERROR_INVALID_VALUE, "multiple peer endpoints");
    require(!p.srcArray && !p.dstArray, CUDA_ERROR_NOT_SUPPORTED, "CUDA array transfers are not implemented");
    CUdevice device = 0; check(cuDeviceGet(&device, p.srcDevice)); check(cuDeviceGet(&device, p.dstDevice));
    if (!p.extent.width || !p.extent.height || !p.extent.depth) return {};
    auto from_owner = runtime_peer_context(p.srcPtr.ptr, p.srcDevice), to_owner = runtime_peer_context(p.dstPtr.ptr, p.dstDevice);
    memory_pitch(p.srcPtr.pitch, p.extent.width, p.srcPos.x, p.srcDevice);
    memory_pitch(p.dstPtr.pitch, p.extent.width, p.dstPos.x, p.dstDevice);
    CUDA_MEMCPY3D_PEER result{};
    result.srcMemoryType = result.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    result.srcDevice = reinterpret_cast<std::uintptr_t>(p.srcPtr.ptr); result.srcContext = from_owner;
    result.srcPitch = p.srcPtr.pitch; result.srcHeight = p.srcPtr.ysize;
    result.srcXInBytes = p.srcPos.x; result.srcY = p.srcPos.y; result.srcZ = p.srcPos.z;
    result.dstDevice = reinterpret_cast<std::uintptr_t>(p.dstPtr.ptr); result.dstContext = to_owner;
    result.dstPitch = p.dstPtr.pitch; result.dstHeight = p.dstPtr.ysize;
    result.dstXInBytes = p.dstPos.x; result.dstY = p.dstPos.y; result.dstZ = p.dstPos.z;
    result.WidthInBytes = p.extent.width; result.Height = p.extent.height; result.Depth = p.extent.depth;
    return result;
}
