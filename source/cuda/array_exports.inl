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
    require(device == entry->owner->device, CUDA_ERROR_INVALID_VALUE, "array memory requirements require the allocation device");
    *output = native_call(entry->owner, [&] { return entry->native->array_requirements(); }))
DRIVER_API(cuArrayGetPlane, (CUarray* output, CUarray handle, unsigned plane),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null array plane output"); *output = array_plane(handle, plane, current()))
DRIVER_API(cuArrayGetSparseProperties, (CUDA_ARRAY_SPARSE_PROPERTIES* output, CUarray handle),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null sparse properties output");
    auto entry = array_lookup(handle, current());
    require((entry->descriptor.Flags & CUDA_ARRAY3D_SPARSE) && !entry->parent && !entry->parent_array,
        CUDA_ERROR_INVALID_VALUE, "sparse properties require an independently allocated sparse array");
    *output = native_call(entry->owner, [&] { return entry->native->array_sparse_properties(); }))
DRIVER_API(cuArrayDestroy, (CUarray handle), driver_ready(); destroy_array(handle))
DRIVER_API(cuMipmappedArrayCreate, (CUmipmappedArray* output, const CUDA_ARRAY3D_DESCRIPTOR* d, unsigned levels),
    driver_ready(); require(output && d, CUDA_ERROR_INVALID_VALUE, "null mipmap descriptor or output"); *output = create_mipmapped_array(*d, levels, current()))
DRIVER_API(cuMipmappedArrayGetLevel, (CUarray* output, CUmipmappedArray handle, unsigned level),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null mipmap level output"); auto owner = current();
    auto it = global().mipmapped_arrays.find(handle); require(it != global().mipmapped_arrays.end() && !it->second.retiring, CUDA_ERROR_INVALID_HANDLE, "invalid mipmapped array");
    require(it->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "mipmapped array belongs to a different context");
    require(level < it->second.levels.size(), CUDA_ERROR_INVALID_VALUE, "mipmap level exceeds allocation"); *output = it->second.levels[level])
DRIVER_API(cuMipmappedArrayGetMemoryRequirements, (CUDA_ARRAY_MEMORY_REQUIREMENTS* output, CUmipmappedArray handle, CUdevice device),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null mipmap requirements output");
    auto owner = current(); auto found = global().mipmapped_arrays.find(handle);
    require(found != global().mipmapped_arrays.end() && !found->second.retiring, CUDA_ERROR_INVALID_HANDLE, "invalid mipmapped array");
    require(found->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "mipmapped array belongs to a different context");
    auto first = array_lookup(found->second.levels.front(), owner);
    require((first->descriptor.Flags & CUDA_ARRAY3D_DEFERRED_MAPPING) && device == owner->device,
        CUDA_ERROR_INVALID_VALUE, "mipmap requirements need deferred mapping on the allocation device");
    CUDA_ARRAY_MEMORY_REQUIREMENTS result{};
    if (owner->native->kind() == cudavmBackendCuda) {
        result = native_call(owner, [&] { return first->native->mipmapped_requirements(); });
    } else for (auto level : found->second.levels) {
        const auto requirements = global().arrays.at(level)->native->array_requirements();
        require(requirements.size <= SIZE_MAX - result.size, CUDA_ERROR_INVALID_VALUE, "mipmap requirements size overflow");
        result.size += requirements.size; result.alignment = std::max(result.alignment, requirements.alignment);
    }
    *output = result)
DRIVER_API(cuMipmappedArrayDestroy, (CUmipmappedArray handle), driver_ready(); destroy_mipmapped_array(handle))
