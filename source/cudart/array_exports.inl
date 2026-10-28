RUNTIME_API(cudaMallocArray, (cudaArray_t* output, const cudaChannelFormatDesc* d, size_t width, size_t height, unsigned flags),
    require(output && d, CUDA_ERROR_INVALID_VALUE, "null array allocation descriptor or output"); runtime_current();
    auto descriptor = runtime_array_descriptor(*d, {width,height,0}, flags); CUarray value{};
    check(runtime_driver_wait([&] { return cuArray3DCreate(&value,&descriptor); })); *output = reinterpret_cast<cudaArray_t>(value))
RUNTIME_API(cudaMalloc3DArray, (cudaArray_t* output, const cudaChannelFormatDesc* d, cudaExtent extent, unsigned flags),
    require(output && d, CUDA_ERROR_INVALID_VALUE, "null array allocation descriptor or output"); runtime_current();
    auto descriptor = runtime_array_descriptor(*d,extent,flags); CUarray value{};
    check(runtime_driver_wait([&] { return cuArray3DCreate(&value,&descriptor); })); *output = reinterpret_cast<cudaArray_t>(value))
RUNTIME_API(cudaFreeArray, (cudaArray_t array), runtime_current(); if (array) check(runtime_driver_wait([&] { return cuArrayDestroy(reinterpret_cast<CUarray>(array)); })))
RUNTIME_API(cudaMallocMipmappedArray, (cudaMipmappedArray_t* output, const cudaChannelFormatDesc* d, cudaExtent extent, unsigned levels, unsigned flags),
    require(output && d, CUDA_ERROR_INVALID_VALUE, "null mipmap allocation descriptor or output"); runtime_current();
    auto descriptor = runtime_array_descriptor(*d,extent,flags); CUmipmappedArray value{};
    check(runtime_driver_wait([&] { return cuMipmappedArrayCreate(&value,&descriptor,levels); })); *output = reinterpret_cast<cudaMipmappedArray_t>(value))
RUNTIME_API(cudaGetMipmappedArrayLevel, (cudaArray_t* output, cudaMipmappedArray_const_t array, unsigned level),
    require(output, CUDA_ERROR_INVALID_VALUE, "null mipmap level output"); runtime_current(); CUarray value{};
    check(cuMipmappedArrayGetLevel(&value,reinterpret_cast<CUmipmappedArray>(const_cast<cudaMipmappedArray*>(array)),level)); *output = reinterpret_cast<cudaArray_t>(value))
RUNTIME_API(cudaFreeMipmappedArray, (cudaMipmappedArray_t array), runtime_current(); if (array) check(runtime_driver_wait([&] { return cuMipmappedArrayDestroy(reinterpret_cast<CUmipmappedArray>(array)); })))
RUNTIME_API(cudaArrayGetInfo, (cudaChannelFormatDesc* output, cudaExtent* extent, unsigned* flags, cudaArray_t array),
    runtime_current(); auto d = runtime_array_info(array); auto channel = runtime_channel_descriptor(d);
    if (output) *output = channel; if (extent) *extent = {d.Width,d.Height,d.Depth};
    if (flags) *flags = (d.Flags & CUDA_ARRAY3D_LAYERED ? cudaArrayLayered : 0) |
        (d.Flags & CUDA_ARRAY3D_SURFACE_LDST ? cudaArraySurfaceLoadStore : 0) |
        (d.Flags & CUDA_ARRAY3D_CUBEMAP ? cudaArrayCubemap : 0) | (d.Flags & CUDA_ARRAY3D_TEXTURE_GATHER ? cudaArrayTextureGather : 0))
RUNTIME_API(cudaGetChannelDesc, (cudaChannelFormatDesc* output, cudaArray_const_t array),
    require(output, CUDA_ERROR_INVALID_VALUE, "null channel descriptor output"); runtime_current(); *output = runtime_channel_descriptor(runtime_array_info(array)))
RUNTIME_API(cudaArrayGetMemoryRequirements, (cudaArrayMemoryRequirements* output, cudaArray_t array, int device),
    require(output, CUDA_ERROR_INVALID_VALUE, "null array memory requirements output"); runtime_current();
    CUDA_ARRAY_MEMORY_REQUIREMENTS requirements{};
    check(cuArrayGetMemoryRequirements(&requirements, reinterpret_cast<CUarray>(array), device));
    *output = {}; output->size = requirements.size; output->alignment = requirements.alignment)
extern "C" CUDAVM_CUDART_API cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x,int y,int z,int w,cudaChannelFormatKind kind) { return {x,y,z,w,kind}; }
RUNTIME_API(cudaMemcpy2DToArray, (cudaArray_t to,size_t x,size_t y,const void* from,size_t pitch,size_t width,size_t height,cudaMemcpyKind kind),
    require(to,CUDA_ERROR_INVALID_VALUE,"null destination array"); runtime_current();
    auto p=runtime_array_copy_2d(to,nullptr,0,x,y,nullptr,from,pitch,0,0,width,height,kind); check(runtime_driver_wait([&]{return cuMemcpy2D(&p);})))
RUNTIME_API(cudaMemcpy2DFromArray, (void* to,size_t pitch,cudaArray_const_t from,size_t x,size_t y,size_t width,size_t height,cudaMemcpyKind kind),
    require(from,CUDA_ERROR_INVALID_VALUE,"null source array"); runtime_current();
    auto p=runtime_array_copy_2d(nullptr,to,pitch,0,0,from,nullptr,0,x,y,width,height,kind); check(runtime_driver_wait([&]{return cuMemcpy2D(&p);})))
RUNTIME_API(cudaMemcpy2DArrayToArray, (cudaArray_t to,size_t x,size_t y,cudaArray_const_t from,size_t from_x,size_t from_y,size_t width,size_t height,cudaMemcpyKind kind),
    require(to && from,CUDA_ERROR_INVALID_VALUE,"null array copy endpoint"); runtime_current();
    auto p=runtime_array_copy_2d(to,nullptr,0,x,y,from,nullptr,0,from_x,from_y,width,height,kind); check(runtime_driver_wait([&]{return cuMemcpy2D(&p);})))
RUNTIME_API(cudaMemcpy2DToArrayAsync, (cudaArray_t to,size_t x,size_t y,const void* from,size_t pitch,size_t width,size_t height,cudaMemcpyKind kind,cudaStream_t stream),
    require(to,CUDA_ERROR_INVALID_VALUE,"null destination array"); runtime_current();
    auto p=runtime_array_copy_2d(to,nullptr,0,x,y,nullptr,from,pitch,0,0,width,height,kind); check(cuMemcpy2DAsync(&p,reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaMemcpy2DFromArrayAsync, (void* to,size_t pitch,cudaArray_const_t from,size_t x,size_t y,size_t width,size_t height,cudaMemcpyKind kind,cudaStream_t stream),
    require(from,CUDA_ERROR_INVALID_VALUE,"null source array"); runtime_current();
    auto p=runtime_array_copy_2d(nullptr,to,pitch,0,0,from,nullptr,0,x,y,width,height,kind); check(cuMemcpy2DAsync(&p,reinterpret_cast<CUstream>(stream))))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync_ptsz(cudaArray_t to,size_t x,size_t y,const void* from,size_t pitch,size_t width,size_t height,cudaMemcpyKind kind,cudaStream_t stream) {
    return cudaMemcpy2DToArrayAsync(to,x,y,from,pitch,width,height,kind,stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync_ptsz(void* to,size_t pitch,cudaArray_const_t from,size_t x,size_t y,size_t width,size_t height,cudaMemcpyKind kind,cudaStream_t stream) {
    return cudaMemcpy2DFromArrayAsync(to,pitch,from,x,y,width,height,kind,stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy2DToArray_ptds(cudaArray_t to,size_t x,size_t y,const void* from,size_t pitch,size_t width,size_t height,cudaMemcpyKind kind) {
    return runtime_call([&] { require(to,CUDA_ERROR_INVALID_VALUE,"null destination array"); runtime_current();
        auto p=runtime_array_copy_2d(to,nullptr,0,x,y,nullptr,from,pitch,0,0,width,height,kind);
        check(cuMemcpy2DAsync(&p,CU_STREAM_PER_THREAD)); check(runtime_driver_wait([&]{return cuStreamSynchronize(CU_STREAM_PER_THREAD);})); });
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy2DFromArray_ptds(void* to,size_t pitch,cudaArray_const_t from,size_t x,size_t y,size_t width,size_t height,cudaMemcpyKind kind) {
    return runtime_call([&] { require(from,CUDA_ERROR_INVALID_VALUE,"null source array"); runtime_current();
        auto p=runtime_array_copy_2d(nullptr,to,pitch,0,0,from,nullptr,0,x,y,width,height,kind);
        check(cuMemcpy2DAsync(&p,CU_STREAM_PER_THREAD)); check(runtime_driver_wait([&]{return cuStreamSynchronize(CU_STREAM_PER_THREAD);})); });
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray_ptds(cudaArray_t to,size_t x,size_t y,cudaArray_const_t from,size_t from_x,size_t from_y,size_t width,size_t height,cudaMemcpyKind kind) {
    return runtime_call([&] { require(to && from,CUDA_ERROR_INVALID_VALUE,"null array copy endpoint"); runtime_current();
        auto p=runtime_array_copy_2d(to,nullptr,0,x,y,from,nullptr,0,from_x,from_y,width,height,kind);
        check(cuMemcpy2DAsync(&p,CU_STREAM_PER_THREAD)); check(runtime_driver_wait([&]{return cuStreamSynchronize(CU_STREAM_PER_THREAD);})); });
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpy3D_ptds(const cudaMemcpy3DParms* params) {
    return runtime_call([&] { require(params,CUDA_ERROR_INVALID_VALUE,"null 3D copy descriptor"); runtime_current();
        auto p=runtime_copy_3d(*params); check(cuMemcpy3DAsync(&p,CU_STREAM_PER_THREAD));
        check(runtime_driver_wait([&]{return cuStreamSynchronize(CU_STREAM_PER_THREAD);})); });
}
