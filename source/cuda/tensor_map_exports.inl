DRIVER_API(cuTensorMapEncodeTiled, (CUtensorMap* output, CUtensorMapDataType data_type, cuuint32_t rank, void* address,
    const cuuint64_t* dimensions, const cuuint64_t* strides, const cuuint32_t* box, const cuuint32_t* element_strides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promotion, CUtensorMapFloatOOBfill fill),
    driver_ready();
    validate_tensor_map_output(output);
    const tensor_map_tiled_options options{data_type, rank, address, dimensions, strides, box, element_strides, interleave, swizzle, promotion, fill};
    validate_tiled_tensor_map(options);
    auto owner = current();
    CUtensorMap temporary; std::memcpy(&temporary, output, sizeof(temporary));
    context_configuration(owner, [&] { owner->native->encode_tiled_tensor_map(temporary, options); });
    std::memcpy(output, &temporary, sizeof(temporary)))
DRIVER_API(cuTensorMapReplaceAddress, (CUtensorMap* output, void* address),
    driver_ready(); validate_tensor_map_output(output);
    require(!(reinterpret_cast<uintptr_t>(address) & 15u), CUDA_ERROR_INVALID_VALUE, "misaligned tensor map replacement address");
    auto owner = current();
    CUtensorMap temporary; std::memcpy(&temporary, output, sizeof(temporary));
    context_configuration(owner, [&] { owner->native->replace_tensor_map_address(temporary, address); });
    std::memcpy(output, &temporary, sizeof(temporary)))
