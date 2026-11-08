void validate_tensor_map_output(const CUtensorMap* output) {
    require(output && !(reinterpret_cast<uintptr_t>(output) & 63u), CUDA_ERROR_INVALID_VALUE, "tensor map output must be 64-byte aligned");
}
void validate_tiled_tensor_map(const tensor_map_tiled_options& value) {
    require(unsigned(value.data_type) <= unsigned(CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B), CUDA_ERROR_INVALID_VALUE, "invalid tensor map data type");
    require(value.rank >= 1 && value.rank <= 5, CUDA_ERROR_INVALID_VALUE, "tensor map rank must be between one and five");
    require(value.dimensions && value.box && value.element_strides && (value.rank == 1 || value.strides),
        CUDA_ERROR_INVALID_VALUE, "missing tensor map shape array");
    require(unsigned(value.interleave) <= unsigned(CU_TENSOR_MAP_INTERLEAVE_32B)
        && unsigned(value.swizzle) <= unsigned(CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B)
        && unsigned(value.promotion) <= unsigned(CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
        && unsigned(value.fill) <= unsigned(CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA),
        CUDA_ERROR_INVALID_VALUE, "invalid tensor map layout mode");
    require(value.interleave == CU_TENSOR_MAP_INTERLEAVE_NONE || value.rank >= 3, CUDA_ERROR_INVALID_VALUE, "interleaved tensor maps require at least three dimensions");
    const bool packed_aligned = value.data_type == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B || value.data_type == CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B;
    const unsigned alignment = value.interleave == CU_TENSOR_MAP_INTERLEAVE_32B || packed_aligned ? 32 : 16;
    require(!(reinterpret_cast<uintptr_t>(value.address) & (alignment - 1)), CUDA_ERROR_INVALID_VALUE, "misaligned tensor map global address");
    for (unsigned i = 0; i < value.rank; ++i) {
        require(value.dimensions[i] && value.dimensions[i] <= (uint64_t(1) << 32), CUDA_ERROR_INVALID_VALUE, "invalid tensor map dimension");
        require(value.box[i] && value.box[i] <= 256, CUDA_ERROR_INVALID_VALUE, "invalid tensor map box dimension");
        if (i || value.interleave != CU_TENSOR_MAP_INTERLEAVE_NONE)
            require(value.element_strides[i] && value.element_strides[i] <= 8, CUDA_ERROR_INVALID_VALUE, "invalid tensor map element stride");
        if (i + 1 < value.rank)
            require(value.strides[i] < (uint64_t(1) << 40) && !(value.strides[i] & (alignment - 1)), CUDA_ERROR_INVALID_VALUE, "invalid tensor map byte stride");
    }
    if (packed_aligned)
        require(!(value.dimensions[0] % 128) && value.box[0] == 128, CUDA_ERROR_INVALID_VALUE, "invalid packed tensor map inner dimension");
    if (value.data_type == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B)
        require(!(value.dimensions[0] % 2), CUDA_ERROR_INVALID_VALUE, "packed tensor map inner dimension must be even");
    if (value.interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) {
        constexpr unsigned bytes[]{1, 2, 4, 4, 8, 8, 2, 4, 8, 2, 4, 4, 4};
        if (unsigned(value.data_type) < std::size(bytes))
            require(!((value.box[0] * bytes[unsigned(value.data_type)]) & 15u), CUDA_ERROR_INVALID_VALUE, "tensor map inner box must span a multiple of sixteen bytes");
        else if (value.data_type == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B)
            require(!(value.box[0] % 32), CUDA_ERROR_INVALID_VALUE, "packed tensor map inner box must span a multiple of sixteen bytes");
    }
}
