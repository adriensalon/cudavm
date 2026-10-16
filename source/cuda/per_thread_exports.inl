// NULL in a per-thread entry means CU_STREAM_PER_THREAD. Explicit stream
// handles, including CU_STREAM_LEGACY, retain their original meaning.
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamGetAttribute_ptsz(CUstream stream, CUstreamAttrID attribute, CUstreamAttrValue* output) {
    return cuStreamGetAttribute(stream ? stream : CU_STREAM_PER_THREAD, attribute, output);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamSetAttribute_ptsz(CUstream stream, CUstreamAttrID attribute, const CUstreamAttrValue* value) {
    return cuStreamSetAttribute(stream ? stream : CU_STREAM_PER_THREAD, attribute, value);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamCopyAttributes_ptsz(CUstream destination, CUstream source) {
    return cuStreamCopyAttributes(destination ? destination : CU_STREAM_PER_THREAD, source ? source : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuGraphicsMapResources_ptsz(unsigned count, CUgraphicsResource* handles, CUstream stream) {
    return cuGraphicsMapResources(count, handles, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuGraphicsUnmapResources_ptsz(unsigned count, CUgraphicsResource* handles, CUstream stream) {
    return cuGraphicsUnmapResources(count, handles, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamGetFlags_ptsz(CUstream handle, unsigned* flags) {
    return cuStreamGetFlags(handle ? handle : CU_STREAM_PER_THREAD, flags);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamGetId_ptsz(CUstream handle, unsigned long long* id) {
    return cuStreamGetId(handle ? handle : CU_STREAM_PER_THREAD, id);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamGetDevice_ptsz(CUstream handle, CUdevice* device) {
    return cuStreamGetDevice(handle ? handle : CU_STREAM_PER_THREAD, device);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamGetCtx_ptsz(CUstream handle, CUcontext* output) {
    return cuStreamGetCtx(handle ? handle : CU_STREAM_PER_THREAD, output);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamGetCtx_v2_ptsz(CUstream handle, CUcontext* output, CUgreenCtx* green) {
    return cuStreamGetCtx_v2(handle ? handle : CU_STREAM_PER_THREAD, output, green);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamQuery_ptsz(CUstream handle) {
    return cuStreamQuery(handle ? handle : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream handle) {
    return cuStreamSynchronize(handle ? handle : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream stream, CUevent event, unsigned flags) {
    return cuStreamWaitEvent(stream ? stream : CU_STREAM_PER_THREAD, event, flags);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuEventRecord_ptsz(CUevent event, CUstream stream) {
    return cuEventRecord(event, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuEventRecordWithFlags_ptsz(CUevent event, CUstream stream, unsigned flags) {
    return cuEventRecordWithFlags(event, stream ? stream : CU_STREAM_PER_THREAD, flags);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuLaunchHostFunc_ptsz(CUstream stream, CUhostFn function, void* data) {
    return cuLaunchHostFunc(stream ? stream : CU_STREAM_PER_THREAD, function, data);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr to, CUdeviceptr from, size_t bytes, CUstream stream) {
    return cuMemcpyAsync(to, from, bytes, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr to, const void* from, size_t bytes, CUstream stream) {
    return cuMemcpyHtoDAsync_v2(to, from, bytes, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void* to, CUdeviceptr from, size_t bytes, CUstream stream) {
    return cuMemcpyDtoHAsync_v2(to, from, bytes, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr to, CUdeviceptr from, size_t bytes, CUstream stream) {
    return cuMemcpyDtoDAsync_v2(to, from, bytes, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D* p, CUstream stream) {
    return cuMemcpy2DAsync_v2(p, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D* p, CUstream stream) {
    return cuMemcpy3DAsync_v2(p, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr to, unsigned char value, size_t count, CUstream stream) {
    return cuMemsetD8Async(to, value, count, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemsetD16Async_ptsz(CUdeviceptr to, unsigned short value, size_t count, CUstream stream) {
    return cuMemsetD16Async(to, value, count, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemsetD32Async_ptsz(CUdeviceptr to, unsigned value, size_t count, CUstream stream) {
    return cuMemsetD32Async(to, value, count, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemsetD2D8Async_ptsz(CUdeviceptr to, size_t pitch, unsigned char value, size_t width, size_t height, CUstream stream) {
    return cuMemsetD2D8Async(to, pitch, value, width, height, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemsetD2D16Async_ptsz(CUdeviceptr to, size_t pitch, unsigned short value, size_t width, size_t height, CUstream stream) {
    return cuMemsetD2D16Async(to, pitch, value, width, height, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuMemsetD2D32Async_ptsz(CUdeviceptr to, size_t pitch, unsigned value, size_t width, size_t height, CUstream stream) {
    return cuMemsetD2D32Async(to, pitch, value, width, height, stream ? stream : CU_STREAM_PER_THREAD);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream, void** parameters, void** extra) {
    return cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared, stream ? stream : CU_STREAM_PER_THREAD, parameters, extra);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction function, void** parameters, void** extra) {
    if (!config) return cuLaunchKernelEx(config, function, parameters, extra);
    auto copy = *config; if (!copy.hStream) copy.hStream = CU_STREAM_PER_THREAD;
    return cuLaunchKernelEx(&copy, function, parameters, extra);
}
extern "C" CUDAVM_CUDA_API CUresult CUDAAPI cuLaunchCooperativeKernel_ptsz(CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream, void** parameters) {
    return cuLaunchCooperativeKernel(function, gx, gy, gz, bx, by, bz, shared, stream ? stream : CU_STREAM_PER_THREAD, parameters);
}
DRIVER_API(cuMemcpyHtoD_v2_ptds, (CUdeviceptr to, const void* from, size_t bytes),
    driver_ready(); async_copy_linear(to, nullptr, CU_MEMORYTYPE_DEVICE, 0, from, CU_MEMORYTYPE_HOST, bytes, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemcpyDtoH_v2_ptds, (void* to, CUdeviceptr from, size_t bytes),
    driver_ready(); async_copy_linear(0, to, CU_MEMORYTYPE_HOST, from, nullptr, CU_MEMORYTYPE_DEVICE, bytes, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemcpyDtoD_v2_ptds, (CUdeviceptr to, CUdeviceptr from, size_t bytes),
    driver_ready(); async_copy_linear(to, nullptr, CU_MEMORYTYPE_DEVICE, from, nullptr, CU_MEMORYTYPE_DEVICE, bytes, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemcpy_ptds, (CUdeviceptr to, CUdeviceptr from, size_t bytes),
    driver_ready(); async_copy_linear(to, nullptr, CU_MEMORYTYPE_UNIFIED, from, nullptr, CU_MEMORYTYPE_UNIFIED, bytes, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemcpy2D_v2_ptds, (const CUDA_MEMCPY2D* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null 2D copy descriptor"); async_copy_2d(*p, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemcpy2DUnaligned_v2_ptds, (const CUDA_MEMCPY2D* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null 2D copy descriptor"); async_copy_2d(*p, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemcpy3D_v2_ptds, (const CUDA_MEMCPY3D* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null 3D copy descriptor"); async_copy_3d(*p, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemcpyPeer_ptds, (CUdeviceptr to, CUcontext to_context, CUdeviceptr from, CUcontext from_context, size_t bytes),
    driver_ready(); async_peer_linear(to, to_context, from, from_context, bytes, CU_STREAM_PER_THREAD); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemcpy3DPeer_ptds, (const CUDA_MEMCPY3D_PEER* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null peer copy descriptor"); async_peer_3d(*p, CU_STREAM_PER_THREAD); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemsetD8_v2_ptds, (CUdeviceptr to, unsigned char pattern, size_t count),
    driver_ready(); async_fill_linear(to, pattern, count, CU_STREAM_PER_THREAD); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemsetD2D8_v2_ptds, (CUdeviceptr to, size_t pitch, unsigned char value, size_t width, size_t height),
    driver_ready(); async_fill_box(to, pitch, 0, value, width, height, 1, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemsetD16_v2_ptds, (CUdeviceptr to, unsigned short pattern, size_t count),
    driver_ready(); async_fill_linear(to, pattern, count, CU_STREAM_PER_THREAD); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemsetD2D16_v2_ptds, (CUdeviceptr to, size_t pitch, unsigned short value, size_t width, size_t height),
    driver_ready(); async_fill_box(to, pitch, 0, value, width, height, 1, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemsetD32_v2_ptds, (CUdeviceptr to, unsigned pattern, size_t count),
    driver_ready(); async_fill_linear(to, pattern, count, CU_STREAM_PER_THREAD); stream_status(CU_STREAM_PER_THREAD, true))
DRIVER_API(cuMemsetD2D32_v2_ptds, (CUdeviceptr to, size_t pitch, unsigned value, size_t width, size_t height),
    driver_ready(); async_fill_box(to, pitch, 0, value, width, height, 1, CU_STREAM_PER_THREAD, current()); stream_status(CU_STREAM_PER_THREAD, true))
