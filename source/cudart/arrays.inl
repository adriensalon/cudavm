#include "array_formats.inl"
CUDA_ARRAY3D_DESCRIPTOR runtime_array_descriptor(const cudaChannelFormatDesc& d, cudaExtent extent, unsigned flags) {
    constexpr unsigned known_flags = cudaArrayLayered | cudaArraySurfaceLoadStore | cudaArrayCubemap
        | cudaArrayTextureGather | cudaArraySparse | cudaArrayDeferredMapping | cudaArrayColorAttachment;
    require(!(flags & ~known_flags), CUDA_ERROR_INVALID_VALUE, "invalid Runtime array flags");
    const unsigned native_flags = (flags & cudaArrayLayered ? CUDA_ARRAY3D_LAYERED : 0)
        | (flags & cudaArraySurfaceLoadStore ? CUDA_ARRAY3D_SURFACE_LDST : 0)
        | (flags & cudaArrayCubemap ? CUDA_ARRAY3D_CUBEMAP : 0)
        | (flags & cudaArrayTextureGather ? CUDA_ARRAY3D_TEXTURE_GATHER : 0)
        | (flags & cudaArraySparse ? CUDA_ARRAY3D_SPARSE : 0)
        | (flags & cudaArrayDeferredMapping ? CUDA_ARRAY3D_DEFERRED_MAPPING : 0)
        | (flags & cudaArrayColorAttachment ? CUDA_ARRAY3D_COLOR_ATTACHMENT : 0);
    for (const auto& format : runtime_array_formats) if (format.runtime == d.f) {
        const auto properties = array_format_properties(format.driver);
        const auto bits = array_channel_bits(properties, properties.channels);
        symbol_require(d.x == bits[0] && d.y == bits[1] && d.z == bits[2] && d.w == bits[3],
            cudaErrorInvalidChannelDescriptor, "channel bits do not match the selected array format");
        return {extent.width, extent.height, extent.depth, format.driver, properties.channels, native_flags};
    }
    symbol_require(d.x > 0 && (d.x == 8 || d.x == 16 || d.x == 32), cudaErrorInvalidChannelDescriptor, "invalid array component width");
    unsigned channels = 1;
    if (d.y) { symbol_require(d.y == d.x, cudaErrorInvalidChannelDescriptor, "unequal array components"); ++channels; }
    if (d.z || d.w) { symbol_require(d.y == d.x && d.z == d.x && d.w == d.x, cudaErrorInvalidChannelDescriptor, "array requires one, two or four equal components"); channels = 4; }
    CUarray_format format{};
    if (d.f == cudaChannelFormatKindUnsigned) format = d.x == 8 ? CU_AD_FORMAT_UNSIGNED_INT8 : d.x == 16 ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT32;
    else if (d.f == cudaChannelFormatKindSigned) format = d.x == 8 ? CU_AD_FORMAT_SIGNED_INT8 : d.x == 16 ? CU_AD_FORMAT_SIGNED_INT16 : CU_AD_FORMAT_SIGNED_INT32;
    else { symbol_require(d.f == cudaChannelFormatKindFloat && d.x != 8, cudaErrorInvalidChannelDescriptor, "invalid floating array format"); format = d.x == 16 ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_FLOAT; }
    return {extent.width, extent.height, extent.depth, format, channels, native_flags};
}
cudaChannelFormatDesc runtime_channel_descriptor(const CUDA_ARRAY3D_DESCRIPTOR& d) {
    for (const auto& format : runtime_array_formats) if (format.driver == d.Format) {
        const auto properties = array_format_properties(d.Format);
        const auto bits = array_channel_bits(properties, properties.channels);
        return {bits[0], bits[1], bits[2], bits[3], format.runtime};
    }
    int bits = 0; cudaChannelFormatKind kind{};
    switch (d.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8: bits=8; kind=cudaChannelFormatKindUnsigned; break;
    case CU_AD_FORMAT_UNSIGNED_INT16: bits=16; kind=cudaChannelFormatKindUnsigned; break;
    case CU_AD_FORMAT_UNSIGNED_INT32: bits=32; kind=cudaChannelFormatKindUnsigned; break;
    case CU_AD_FORMAT_SIGNED_INT8: bits=8; kind=cudaChannelFormatKindSigned; break;
    case CU_AD_FORMAT_SIGNED_INT16: bits=16; kind=cudaChannelFormatKindSigned; break;
    case CU_AD_FORMAT_SIGNED_INT32: bits=32; kind=cudaChannelFormatKindSigned; break;
    case CU_AD_FORMAT_HALF: bits=16; kind=cudaChannelFormatKindFloat; break;
    case CU_AD_FORMAT_FLOAT: bits=32; kind=cudaChannelFormatKindFloat; break;
    case CU_AD_FORMAT_Y210: case CU_AD_FORMAT_Y216: case CU_AD_FORMAT_Y416:
        bits=16; kind=cudaChannelFormatKindUnsigned; break;
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "array channel format is not implemented");
    }
    return {bits, d.NumChannels >= 2 ? bits : 0, d.NumChannels == 4 ? bits : 0, d.NumChannels == 4 ? bits : 0, kind};
}
CUDA_ARRAY3D_DESCRIPTOR runtime_array_info(cudaArray_const_t array) {
    CUDA_ARRAY3D_DESCRIPTOR d{}; check(cuArray3DGetDescriptor(&d, reinterpret_cast<CUarray>(const_cast<cudaArray*>(array)))); return d;
}
size_t runtime_array_element(cudaArray_const_t array) {
    const auto d = runtime_array_info(array);
    const auto properties = array_format_properties(d.Format);
    require(properties.valid && properties.planes == 1, CUDA_ERROR_INVALID_VALUE,
        "multiplanar transfers require a plane array");
    return properties.block_bytes ? properties.block_bytes : properties.component_bits / 8 * d.NumChannels;
}
