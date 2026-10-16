DRIVER_API(cuStreamAttachMemAsync, (CUstream stream, CUdeviceptr pointer, size_t bytes, unsigned flags), driver_ready(); attach_managed(stream, pointer, bytes, flags))
DRIVER_API(cuStreamAttachMemAsync_ptsz, (CUstream stream, CUdeviceptr pointer, size_t bytes, unsigned flags), driver_ready(); attach_managed(stream ? stream : CU_STREAM_PER_THREAD, pointer, bytes, flags))
DRIVER_API(cuMemcpyAsync, (CUdeviceptr to, CUdeviceptr from, size_t bytes, CUstream stream), driver_ready(); async_copy_linear(to, nullptr, CU_MEMORYTYPE_UNIFIED, from, nullptr, CU_MEMORYTYPE_UNIFIED, bytes, stream, current()))
DRIVER_API(cuMemcpyHtoDAsync, (CUdeviceptr to, const void* from, size_t bytes, CUstream stream), driver_ready(); async_copy_linear(to, nullptr, CU_MEMORYTYPE_DEVICE, 0, from, CU_MEMORYTYPE_HOST, bytes, stream, current()))
DRIVER_API(cuMemcpyDtoHAsync, (void* to, CUdeviceptr from, size_t bytes, CUstream stream), driver_ready(); async_copy_linear(0, to, CU_MEMORYTYPE_HOST, from, nullptr, CU_MEMORYTYPE_DEVICE, bytes, stream, current()))
DRIVER_API(cuMemcpyDtoDAsync, (CUdeviceptr to, CUdeviceptr from, size_t bytes, CUstream stream), driver_ready(); async_copy_linear(to, nullptr, CU_MEMORYTYPE_DEVICE, from, nullptr, CU_MEMORYTYPE_DEVICE, bytes, stream, current()))
DRIVER_API(cuMemcpy2DAsync, (const CUDA_MEMCPY2D* p, CUstream stream), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null asynchronous 2D descriptor"); async_copy_2d(*p, stream, current()))
DRIVER_API(cuMemcpy3DAsync, (const CUDA_MEMCPY3D* p, CUstream stream), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null asynchronous 3D descriptor"); async_copy_3d(*p, stream, current()))
DRIVER_API(cuMemcpyPeerAsync, (CUdeviceptr to, CUcontext to_context, CUdeviceptr from, CUcontext from_context, size_t bytes, CUstream stream), driver_ready(); async_peer_linear(to, to_context, from, from_context, bytes, stream))
DRIVER_API(cuMemcpy3DPeerAsync, (const CUDA_MEMCPY3D_PEER* p, CUstream stream), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null asynchronous peer descriptor"); async_peer_3d(*p, stream))
DRIVER_API(cuMemcpyPeerAsync_ptsz, (CUdeviceptr to, CUcontext to_context, CUdeviceptr from, CUcontext from_context, size_t bytes, CUstream stream),
    driver_ready(); async_peer_linear(to, to_context, from, from_context, bytes, stream ? stream : CU_STREAM_PER_THREAD))
DRIVER_API(cuMemcpy3DPeerAsync_ptsz, (const CUDA_MEMCPY3D_PEER* p, CUstream stream),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null asynchronous peer descriptor"); async_peer_3d(*p, stream ? stream : CU_STREAM_PER_THREAD))
DRIVER_API(cuMemsetD8Async, (CUdeviceptr to, unsigned char value, size_t count, CUstream stream), driver_ready(); async_fill_linear(to, value, count, stream))
DRIVER_API(cuMemsetD16Async, (CUdeviceptr to, unsigned short value, size_t count, CUstream stream), driver_ready(); async_fill_linear(to, value, count, stream))
DRIVER_API(cuMemsetD32Async, (CUdeviceptr to, unsigned value, size_t count, CUstream stream), driver_ready(); async_fill_linear(to, value, count, stream))
DRIVER_API(cuMemsetD2D8Async, (CUdeviceptr to, size_t pitch, unsigned char value, size_t width, size_t height, CUstream stream), driver_ready(); async_fill_box(to, pitch, 0, value, width, height, 1, stream, current()))
DRIVER_API(cuMemsetD2D16Async, (CUdeviceptr to, size_t pitch, unsigned short value, size_t width, size_t height, CUstream stream), driver_ready(); async_fill_box(to, pitch, 0, value, width, height, 1, stream, current()))
DRIVER_API(cuMemsetD2D32Async, (CUdeviceptr to, size_t pitch, unsigned value, size_t width, size_t height, CUstream stream), driver_ready(); async_fill_box(to, pitch, 0, value, width, height, 1, stream, current()))
extern "C" CUDAVM_CUDA_API CUresult cudavmDriverMemset3DAsync(CUdeviceptr pointer, size_t pitch, size_t slice_height,
    unsigned char value, size_t width, size_t height, size_t depth, CUstream stream) {
    return driver_call([&] { driver_ready(); async_fill_box(pointer, pitch, slice_height, value, width, height, depth, stream, current()); });
}
