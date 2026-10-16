RUNTIME_API(cudaStreamAttachMemAsync, (cudaStream_t stream, void* pointer, size_t bytes, unsigned flags),
    runtime_current(); check(cuStreamAttachMemAsync(reinterpret_cast<CUstream>(stream), reinterpret_cast<uintptr_t>(pointer), bytes, flags)))
RUNTIME_API(cudaStreamAttachMemAsync_ptsz, (cudaStream_t stream, void* pointer, size_t bytes, unsigned flags),
    runtime_current(); check(cuStreamAttachMemAsync(stream ? reinterpret_cast<CUstream>(stream) : CU_STREAM_PER_THREAD, reinterpret_cast<uintptr_t>(pointer), bytes, flags)))
RUNTIME_API(cudaMemcpyAsync, (void* to, const void* from, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream),
    runtime_current(); auto types = runtime_copy_types(to, from, kind);
    check(cuMemcpyAsync(reinterpret_cast<std::uintptr_t>(to), reinterpret_cast<std::uintptr_t>(from), bytes, reinterpret_cast<CUstream>(stream)));
    if (types.to == CU_MEMORYTYPE_HOST && types.from == CU_MEMORYTYPE_HOST) check(runtime_driver_wait([&] { return cuStreamSynchronize(reinterpret_cast<CUstream>(stream)); })))
RUNTIME_API(cudaMemcpy2DAsync, (void* to, size_t to_pitch, const void* from, size_t from_pitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream),
    runtime_current(); auto types = runtime_copy_types(to, from, kind);
    if (width && height) { memory_pitch(to_pitch, width); memory_pitch(from_pitch, width); }
    CUDA_MEMCPY2D p{}; p.srcMemoryType = types.from; p.srcHost = from; p.srcDevice = reinterpret_cast<std::uintptr_t>(from); p.srcPitch = from_pitch;
    p.dstMemoryType = types.to; p.dstHost = to; p.dstDevice = reinterpret_cast<std::uintptr_t>(to); p.dstPitch = to_pitch;
    p.WidthInBytes = width; p.Height = height; check(cuMemcpy2DAsync(&p, reinterpret_cast<CUstream>(stream)));
    if (types.to == CU_MEMORYTYPE_HOST && types.from == CU_MEMORYTYPE_HOST) check(runtime_driver_wait([&] { return cuStreamSynchronize(reinterpret_cast<CUstream>(stream)); })))
RUNTIME_API(cudaMemcpy3DAsync, (const cudaMemcpy3DParms* p, cudaStream_t stream), require(p, CUDA_ERROR_INVALID_VALUE, "null asynchronous 3D descriptor"); runtime_current(); auto descriptor = runtime_copy_3d(*p); check(cuMemcpy3DAsync(&descriptor, reinterpret_cast<CUstream>(stream))); if (descriptor.srcMemoryType == CU_MEMORYTYPE_HOST && descriptor.dstMemoryType == CU_MEMORYTYPE_HOST) check(runtime_driver_wait([&] { return cuStreamSynchronize(reinterpret_cast<CUstream>(stream)); })))
RUNTIME_API(cudaMemcpyPeerAsync, (void* to, int to_device, const void* from, int from_device, size_t bytes, cudaStream_t stream),
    runtime_current(); CUdevice device = 0; check(cuDeviceGet(&device, to_device)); check(cuDeviceGet(&device, from_device));
    if (!bytes) { unsigned flags = 0; check(cuStreamGetFlags(reinterpret_cast<CUstream>(stream), &flags)); return; }
    auto to_owner = runtime_peer_context(to, to_device), from_owner = runtime_peer_context(from, from_device);
    check(cuMemcpyPeerAsync(reinterpret_cast<std::uintptr_t>(to), to_owner, reinterpret_cast<std::uintptr_t>(from), from_owner, bytes, reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaMemcpy3DPeerAsync, (const cudaMemcpy3DPeerParms* p, cudaStream_t stream),
    require(p, CUDA_ERROR_INVALID_VALUE, "null asynchronous peer descriptor"); runtime_current(); auto descriptor = runtime_peer_3d(*p);
    if (!p->extent.width || !p->extent.height || !p->extent.depth) { unsigned flags = 0; check(cuStreamGetFlags(reinterpret_cast<CUstream>(stream), &flags)); return; }
    check(cuMemcpy3DPeerAsync(&descriptor, reinterpret_cast<CUstream>(stream))))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpyPeerAsync_ptsz(void* to, int to_device, const void* from, int from_device, size_t bytes, cudaStream_t stream) {
    return cudaMemcpyPeerAsync(to, to_device, from, from_device, bytes, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync_ptsz(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) {
    return cudaMemcpy3DPeerAsync(p, stream ? stream : cudaStreamPerThread);
}
RUNTIME_API(cudaMemcpyToSymbolAsync, (const void* symbol, const void* source, size_t bytes, size_t offset, cudaMemcpyKind kind, cudaStream_t stream),
    auto destination = symbol_region(runtime_symbol(symbol), bytes, offset);
    symbol_require(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice || kind == cudaMemcpyDefault, cudaErrorInvalidMemcpyDirection, "invalid copy-to-symbol direction");
    auto pointer = reinterpret_cast<std::uintptr_t>(source); bool device = cudavmDriverIsDevicePointer(pointer);
    if (kind == cudaMemcpyDefault) kind = device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    symbol_require(device == (kind == cudaMemcpyDeviceToDevice), cudaErrorInvalidMemcpyDirection, "symbol copy direction disagrees with source ownership");
    if (device) check(cuMemcpyDtoDAsync(destination, pointer, bytes, reinterpret_cast<CUstream>(stream))); else check(cuMemcpyHtoDAsync(destination, source, bytes, reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaMemcpyFromSymbolAsync, (void* destination, const void* symbol, size_t bytes, size_t offset, cudaMemcpyKind kind, cudaStream_t stream),
    auto source = symbol_region(runtime_symbol(symbol), bytes, offset);
    symbol_require(kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice || kind == cudaMemcpyDefault, cudaErrorInvalidMemcpyDirection, "invalid copy-from-symbol direction");
    auto pointer = reinterpret_cast<std::uintptr_t>(destination); bool device = cudavmDriverIsDevicePointer(pointer);
    if (kind == cudaMemcpyDefault) kind = device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    symbol_require(device == (kind == cudaMemcpyDeviceToDevice), cudaErrorInvalidMemcpyDirection, "symbol copy direction disagrees with destination ownership");
    if (device) check(cuMemcpyDtoDAsync(pointer, source, bytes, reinterpret_cast<CUstream>(stream))); else check(cuMemcpyDtoHAsync(destination, source, bytes, reinterpret_cast<CUstream>(stream))))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync_ptsz(const void* symbol, const void* source, size_t bytes, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpyToSymbolAsync(symbol, source, bytes, offset, kind, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync_ptsz(void* destination, const void* symbol, size_t bytes, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpyFromSymbolAsync(destination, symbol, bytes, offset, kind, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpyToSymbol_ptds(const void* symbol, const void* source, size_t bytes, size_t offset, cudaMemcpyKind kind) {
    auto result = cudaMemcpyToSymbolAsync(symbol, source, bytes, offset, kind, cudaStreamPerThread);
    return result == cudaSuccess && bytes ? cudaStreamSynchronize(cudaStreamPerThread) : result;
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpyFromSymbol_ptds(void* destination, const void* symbol, size_t bytes, size_t offset, cudaMemcpyKind kind) {
    auto result = cudaMemcpyFromSymbolAsync(destination, symbol, bytes, offset, kind, cudaStreamPerThread);
    return result == cudaSuccess && bytes ? cudaStreamSynchronize(cudaStreamPerThread) : result;
}
RUNTIME_API(cudaMemsetAsync, (void* to, int value, size_t bytes, cudaStream_t stream), runtime_current(); check(cuMemsetD8Async(reinterpret_cast<std::uintptr_t>(to), static_cast<unsigned char>(value), bytes, reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaMemset2DAsync, (void* to, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream), runtime_current(); if (width && height) memory_pitch(pitch, width); check(cuMemsetD2D8Async(reinterpret_cast<std::uintptr_t>(to), pitch, static_cast<unsigned char>(value), width, height, reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaMemset3DAsync, (cudaPitchedPtr p, int value, cudaExtent extent, cudaStream_t stream), runtime_current(); auto pitch = extent.height == 1 && extent.depth == 1 ? extent.width : p.pitch; if (extent.width && extent.height && extent.depth) memory_pitch(pitch, extent.width); check(cudavmDriverMemset3DAsync(reinterpret_cast<std::uintptr_t>(p.ptr), pitch, p.ysize, static_cast<unsigned char>(value), extent.width, extent.height, extent.depth, reinterpret_cast<CUstream>(stream))))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void* to, const void* from, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpyAsync(to, from, bytes, kind, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy2DAsync_ptsz(void* to, size_t to_pitch, const void* from, size_t from_pitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpy2DAsync(to, to_pitch, from, from_pitch, width, height, kind, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms* params, cudaStream_t stream) {
    return cudaMemcpy3DAsync(params, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemsetAsync_ptsz(void* to, int value, size_t bytes, cudaStream_t stream) {
    return cudaMemsetAsync(to, value, bytes, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemset2DAsync_ptsz(void* to, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
    return cudaMemset2DAsync(to, pitch, value, width, height, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemset3DAsync_ptsz(cudaPitchedPtr pointer, int value, cudaExtent extent, cudaStream_t stream) {
    return cudaMemset3DAsync(pointer, value, extent, stream ? stream : cudaStreamPerThread);
}
