DRIVER_API(cuArrayCreate, (CUarray* output, const CUDA_ARRAY_DESCRIPTOR* d),
    driver_ready(); require(output && d, CUDA_ERROR_INVALID_VALUE, "null array allocation descriptor or output");
    *output = create_array({d->Width, d->Height, 0, d->Format, d->NumChannels, 0}, current()))
DRIVER_API(cuArray3DCreate, (CUarray* output, const CUDA_ARRAY3D_DESCRIPTOR* d),
    driver_ready(); require(output && d, CUDA_ERROR_INVALID_VALUE, "null array allocation descriptor or output"); *output = create_array(*d, current()))
DRIVER_API(cuArrayGetDescriptor, (CUDA_ARRAY_DESCRIPTOR* output, CUarray handle),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null array descriptor output"); auto entry = array_lookup(handle, current());
    require(!entry->descriptor.Depth, CUDA_ERROR_INVALID_VALUE, "array is not one or two dimensional");
    auto& d = entry->descriptor; *output = {d.Width, d.Height, d.Format, d.NumChannels})
DRIVER_API(cuArray3DGetDescriptor, (CUDA_ARRAY3D_DESCRIPTOR* output, CUarray handle),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null array descriptor output"); *output = array_lookup(handle, current())->descriptor)
DRIVER_API(cuArrayGetMemoryRequirements, (CUDA_ARRAY_MEMORY_REQUIREMENTS* output, CUarray handle, CUdevice device),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null array memory requirements output");
    auto entry = array_lookup(handle, current());
    require(entry->descriptor.Flags & CUDA_ARRAY3D_DEFERRED_MAPPING, CUDA_ERROR_INVALID_VALUE, "array memory requirements require deferred mapping");
    (void)device; throw failure(CUDA_ERROR_NOT_SUPPORTED, "deferred array memory requirements are not implemented"))
DRIVER_API(cuArrayDestroy, (CUarray handle), driver_ready(); destroy_array(handle))
DRIVER_API(cuMipmappedArrayCreate, (CUmipmappedArray* output, const CUDA_ARRAY3D_DESCRIPTOR* d, unsigned levels),
    driver_ready(); require(output && d, CUDA_ERROR_INVALID_VALUE, "null mipmap descriptor or output"); *output = create_mipmapped_array(*d, levels, current()))
DRIVER_API(cuMipmappedArrayGetLevel, (CUarray* output, CUmipmappedArray handle, unsigned level),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null mipmap level output"); auto owner = current();
    auto it = global().mipmapped_arrays.find(handle); require(it != global().mipmapped_arrays.end() && !it->second.retiring, CUDA_ERROR_INVALID_HANDLE, "invalid mipmapped array");
    require(it->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "mipmapped array belongs to a different context");
    require(level < it->second.levels.size(), CUDA_ERROR_INVALID_VALUE, "mipmap level exceeds allocation"); *output = it->second.levels[level])
DRIVER_API(cuMipmappedArrayDestroy, (CUmipmappedArray handle), driver_ready(); destroy_mipmapped_array(handle))
