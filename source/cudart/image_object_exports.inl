RUNTIME_API(cudaCreateTextureObject, (cudaTextureObject_t* output, const cudaResourceDesc* resource, const cudaTextureDesc* texture, const cudaResourceViewDesc* view),
    require(output && resource && texture, CUDA_ERROR_INVALID_VALUE, "null texture descriptor or output"); runtime_current();
    auto r = runtime_image_resource(*resource); auto t = runtime_texture_descriptor(*texture);
    if (texture->readMode == cudaReadModeElementType && texture->filterMode == cudaFilterModeLinear) {
        CUDA_ARRAY3D_DESCRIPTOR array{}; check(cuArray3DGetDescriptor(&array, r.res.array.hArray));
        symbol_require(array.Format == CU_AD_FORMAT_FLOAT || array.Format == CU_AD_FORMAT_HALF,
            cudaErrorInvalidFilterSetting, "linear filtering requires floating-point reads");
    }
    CUDA_RESOURCE_VIEW_DESC v{}; if (view) v = runtime_image_view(*view);
    check(cuTexObjectCreate(output, &r, &t, view ? &v : nullptr)))
RUNTIME_API(cudaDestroyTextureObject, (cudaTextureObject_t handle), runtime_current(); check(runtime_driver_wait([&] { return cuTexObjectDestroy(handle); })))
RUNTIME_API(cudaGetTextureObjectResourceDesc, (cudaResourceDesc* output, cudaTextureObject_t handle),
    require(output, CUDA_ERROR_INVALID_VALUE, "null texture resource output"); runtime_current(); CUDA_RESOURCE_DESC r{};
    check(cuTexObjectGetResourceDesc(&r, handle)); *output = runtime_image_resource(r))
RUNTIME_API(cudaGetTextureObjectTextureDesc, (cudaTextureDesc* output, cudaTextureObject_t handle),
    require(output, CUDA_ERROR_INVALID_VALUE, "null texture descriptor output"); runtime_current(); CUDA_TEXTURE_DESC t{};
    check(cuTexObjectGetTextureDesc(&t, handle)); *output = runtime_texture_descriptor(t))
RUNTIME_API(cudaGetTextureObjectResourceViewDesc, (cudaResourceViewDesc* output, cudaTextureObject_t handle),
    require(output, CUDA_ERROR_INVALID_VALUE, "null texture view output"); runtime_current(); CUDA_RESOURCE_VIEW_DESC v{};
    check(cuTexObjectGetResourceViewDesc(&v, handle)); *output = runtime_image_view(v))
RUNTIME_API(cudaCreateSurfaceObject, (cudaSurfaceObject_t* output, const cudaResourceDesc* resource),
    require(output && resource, CUDA_ERROR_INVALID_VALUE, "null surface descriptor or output"); runtime_current();
    require(resource->resType == cudaResourceTypeArray, CUDA_ERROR_INVALID_VALUE, "surfaces require an array");
    auto r = runtime_image_resource(*resource); check(cuSurfObjectCreate(output, &r)))
RUNTIME_API(cudaDestroySurfaceObject, (cudaSurfaceObject_t handle), runtime_current(); check(runtime_driver_wait([&] { return cuSurfObjectDestroy(handle); })))
RUNTIME_API(cudaGetSurfaceObjectResourceDesc, (cudaResourceDesc* output, cudaSurfaceObject_t handle),
    require(output, CUDA_ERROR_INVALID_VALUE, "null surface resource output"); runtime_current(); CUDA_RESOURCE_DESC r{};
    check(cuSurfObjectGetResourceDesc(&r, handle)); *output = runtime_image_resource(r))
