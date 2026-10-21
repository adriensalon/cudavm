CUDA_RESOURCE_DESC runtime_image_resource(const cudaResourceDesc& resource) {
    require(resource.resType >= cudaResourceTypeArray && resource.resType <= cudaResourceTypePitch2D, CUDA_ERROR_INVALID_VALUE, "invalid Runtime image resource type");
    require(resource.resType != cudaResourceTypeMipmappedArray, CUDA_ERROR_NOT_SUPPORTED, "mipmapped texture resources are unavailable");
    CUDA_RESOURCE_DESC result{}; result.resType = static_cast<CUresourcetype>(resource.resType);
    if (resource.resType == cudaResourceTypeArray) result.res.array.hArray = reinterpret_cast<CUarray>(resource.res.array.array);
    else if (resource.resType == cudaResourceTypeLinear) {
        auto d = runtime_array_descriptor(resource.res.linear.desc, {1, 0, 0}, 0);
        result.res.linear = {reinterpret_cast<CUdeviceptr>(resource.res.linear.devPtr), d.Format, d.NumChannels, resource.res.linear.sizeInBytes};
    } else {
        auto& r = resource.res.pitch2D; auto d = runtime_array_descriptor(r.desc, {r.width, r.height, 0}, 0);
        result.res.pitch2D = {reinterpret_cast<CUdeviceptr>(r.devPtr), d.Format, d.NumChannels, r.width, r.height, r.pitchInBytes};
    }
    return result;
}
cudaResourceDesc runtime_image_resource(const CUDA_RESOURCE_DESC& resource) {
    cudaResourceDesc result{}; result.resType = static_cast<cudaResourceType>(resource.resType);
    if (resource.resType == CU_RESOURCE_TYPE_ARRAY) result.res.array.array = reinterpret_cast<cudaArray_t>(resource.res.array.hArray);
    else {
        bool linear = resource.resType == CU_RESOURCE_TYPE_LINEAR;
        CUDA_ARRAY3D_DESCRIPTOR d{}; d.Format = linear ? resource.res.linear.format : resource.res.pitch2D.format;
        d.NumChannels = linear ? resource.res.linear.numChannels : resource.res.pitch2D.numChannels;
        auto channel = runtime_channel_descriptor(d);
        if (linear) result.res.linear = {reinterpret_cast<void*>(resource.res.linear.devPtr), channel, resource.res.linear.sizeInBytes};
        else { auto& r = resource.res.pitch2D; result.res.pitch2D = {reinterpret_cast<void*>(r.devPtr), channel, r.width, r.height, r.pitchInBytes}; }
    }
    return result;
}
CUDA_TEXTURE_DESC runtime_texture_descriptor(const cudaTextureDesc& texture) {
    require(texture.readMode == cudaReadModeElementType || texture.readMode == cudaReadModeNormalizedFloat, CUDA_ERROR_INVALID_VALUE, "invalid texture read mode");
    CUDA_TEXTURE_DESC result{};
    for (unsigned i = 0; i < 3; ++i) result.addressMode[i] = static_cast<CUaddress_mode>(texture.addressMode[i]);
    result.filterMode = static_cast<CUfilter_mode>(texture.filterMode);
    result.flags = (texture.readMode == cudaReadModeElementType ? CU_TRSF_READ_AS_INTEGER : 0) |
        (texture.normalizedCoords ? CU_TRSF_NORMALIZED_COORDINATES : 0) | (texture.sRGB ? CU_TRSF_SRGB : 0) |
        (texture.disableTrilinearOptimization ? CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION : 0) |
        (texture.seamlessCubemap ? CU_TRSF_SEAMLESS_CUBEMAP : 0);
    result.maxAnisotropy = texture.maxAnisotropy;
    result.mipmapFilterMode = static_cast<CUfilter_mode>(texture.mipmapFilterMode);
    result.mipmapLevelBias = texture.mipmapLevelBias; result.minMipmapLevelClamp = texture.minMipmapLevelClamp; result.maxMipmapLevelClamp = texture.maxMipmapLevelClamp;
    std::copy(std::begin(texture.borderColor), std::end(texture.borderColor), result.borderColor); return result;
}
cudaTextureDesc runtime_texture_descriptor(const CUDA_TEXTURE_DESC& texture) {
    cudaTextureDesc result{};
    for (unsigned i = 0; i < 3; ++i) result.addressMode[i] = static_cast<cudaTextureAddressMode>(texture.addressMode[i]);
    result.filterMode = static_cast<cudaTextureFilterMode>(texture.filterMode);
    result.readMode = texture.flags & CU_TRSF_READ_AS_INTEGER ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    result.normalizedCoords = bool(texture.flags & CU_TRSF_NORMALIZED_COORDINATES); result.sRGB = bool(texture.flags & CU_TRSF_SRGB);
    result.disableTrilinearOptimization = bool(texture.flags & CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION); result.seamlessCubemap = bool(texture.flags & CU_TRSF_SEAMLESS_CUBEMAP);
    result.maxAnisotropy = texture.maxAnisotropy; result.mipmapFilterMode = static_cast<cudaTextureFilterMode>(texture.mipmapFilterMode);
    result.mipmapLevelBias = texture.mipmapLevelBias; result.minMipmapLevelClamp = texture.minMipmapLevelClamp; result.maxMipmapLevelClamp = texture.maxMipmapLevelClamp;
    std::copy(std::begin(texture.borderColor), std::end(texture.borderColor), result.borderColor); return result;
}
CUDA_RESOURCE_VIEW_DESC runtime_image_view(const cudaResourceViewDesc& view) {
    CUDA_RESOURCE_VIEW_DESC result{}; result.format = static_cast<CUresourceViewFormat>(view.format);
    result.width=view.width; result.height=view.height; result.depth=view.depth;
    result.firstMipmapLevel=view.firstMipmapLevel; result.lastMipmapLevel=view.lastMipmapLevel; result.firstLayer=view.firstLayer; result.lastLayer=view.lastLayer; return result;
}
cudaResourceViewDesc runtime_image_view(const CUDA_RESOURCE_VIEW_DESC& view) {
    cudaResourceViewDesc result{}; result.format = static_cast<cudaResourceViewFormat>(view.format);
    result.width=view.width; result.height=view.height; result.depth=view.depth;
    result.firstMipmapLevel=view.firstMipmapLevel; result.lastMipmapLevel=view.lastMipmapLevel; result.firstLayer=view.firstLayer; result.lastLayer=view.lastLayer; return result;
}
