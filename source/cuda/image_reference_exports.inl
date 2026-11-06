DRIVER_API(cuModuleGetTexRef, (CUtexref* output, CUmodule module, const char* name),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null texture reference output");
    *output = reinterpret_cast<CUtexref>(find_image_reference(module, name, false)))
DRIVER_API(cuModuleGetSurfRef, (CUsurfref* output, CUmodule module, const char* name),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null surface reference output");
    *output = reinterpret_cast<CUsurfref>(find_image_reference(module, name, true)))
DRIVER_API(cuTexRefSetArray, (CUtexref reference, CUarray array, unsigned flags),
    driver_ready(); bind_reference_array(reinterpret_cast<std::uintptr_t>(reference), false, array, flags))
DRIVER_API(cuSurfRefSetArray, (CUsurfref reference, CUarray array, unsigned flags),
    driver_ready(); bind_reference_array(reinterpret_cast<std::uintptr_t>(reference), true, array, flags))
DRIVER_API(cuTexRefGetArray, (CUarray* output, CUtexref reference),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null reference array output");
    auto value = image_reference_lookup(reinterpret_cast<std::uintptr_t>(reference), false).second->array;
    require(value, CUDA_ERROR_INVALID_VALUE, "texture reference has no array binding"); *output = value)
DRIVER_API(cuSurfRefGetArray, (CUarray* output, CUsurfref reference),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null reference array output");
    auto value = image_reference_lookup(reinterpret_cast<std::uintptr_t>(reference), true).second->array;
    require(value, CUDA_ERROR_INVALID_VALUE, "surface reference has no array binding"); *output = value)
DRIVER_API(cuTexRefSetFormat, (CUtexref reference, CUarray_format format, int channels),
    driver_ready(); require(channels == 1 || channels == 2 || channels == 4, CUDA_ERROR_INVALID_VALUE, "invalid reference channel count");
    image_reference_call(reinterpret_cast<std::uintptr_t>(reference), false, [&](image_reference& value) { value.set_format(format, channels); }))
DRIVER_API(cuTexRefGetFormat, (CUarray_format* format, int* channels, CUtexref reference),
    driver_ready(); image_reference_call(reinterpret_cast<std::uintptr_t>(reference), false, [&](image_reference& value) { value.format(format, channels); }))
DRIVER_API(cuTexRefSetAddressMode, (CUtexref reference, int dimension, CUaddress_mode mode),
    driver_ready(); require(dimension >= 0 && dimension < 3 && mode >= CU_TR_ADDRESS_MODE_WRAP && mode <= CU_TR_ADDRESS_MODE_BORDER,
        CUDA_ERROR_INVALID_VALUE, "invalid reference address mode or dimension");
    image_reference_call(reinterpret_cast<std::uintptr_t>(reference), false, [&](image_reference& value) { value.set_address_mode(dimension, mode); }))
DRIVER_API(cuTexRefGetAddressMode, (CUaddress_mode* output, CUtexref reference, int dimension),
    driver_ready(); require(output && dimension >= 0 && dimension < 3, CUDA_ERROR_INVALID_VALUE, "invalid reference address query");
    *output = image_reference_call(reinterpret_cast<std::uintptr_t>(reference), false, [&](image_reference& value) { return value.address_mode(dimension); }))
DRIVER_API(cuTexRefSetFilterMode, (CUtexref reference, CUfilter_mode mode),
    driver_ready(); require(mode == CU_TR_FILTER_MODE_POINT || mode == CU_TR_FILTER_MODE_LINEAR, CUDA_ERROR_INVALID_VALUE, "invalid reference filter mode");
    image_reference_call(reinterpret_cast<std::uintptr_t>(reference), false, [&](image_reference& value) { value.set_filter_mode(mode); }))
DRIVER_API(cuTexRefGetFilterMode, (CUfilter_mode* output, CUtexref reference),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null reference filter output");
    *output = image_reference_call(reinterpret_cast<std::uintptr_t>(reference), false, [&](image_reference& value) { return value.filter_mode(); }))
DRIVER_API(cuTexRefSetFlags, (CUtexref reference, unsigned flags),
    driver_ready(); require(!(flags & ~(CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB)), CUDA_ERROR_INVALID_VALUE, "invalid texture reference flags");
    image_reference_call(reinterpret_cast<std::uintptr_t>(reference), false, [&](image_reference& value) { value.set_flags(flags); }))
DRIVER_API(cuTexRefGetFlags, (unsigned* output, CUtexref reference),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null reference flags output");
    *output = image_reference_call(reinterpret_cast<std::uintptr_t>(reference), false, [&](image_reference& value) { return value.flags(); }))
