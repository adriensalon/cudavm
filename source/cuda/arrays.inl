std::shared_ptr<array_entry> array_lookup(CUarray handle, const std::shared_ptr<context>& owner) {
    auto it = global().arrays.find(handle);
    require(it != global().arrays.end() && !it->second->retiring, CUDA_ERROR_INVALID_HANDLE, "invalid or retired array");
    require(it->second->owner == owner, CUDA_ERROR_INVALID_CONTEXT, "array belongs to a different context");
    return it->second;
}
void validate_array(const CUDA_ARRAY3D_DESCRIPTOR& d) {
    array_storage_bytes(d);
    const auto format = array_format_properties(d.Format);
    constexpr unsigned supported = CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_CUBEMAP
        | CUDA_ARRAY3D_TEXTURE_GATHER | CUDA_ARRAY3D_DEFERRED_MAPPING | CUDA_ARRAY3D_COLOR_ATTACHMENT
        | CUDA_ARRAY3D_DEPTH_TEXTURE | CUDA_ARRAY3D_VIDEO_ENCODE_DECODE | CUDA_ARRAY3D_SPARSE;
    constexpr unsigned known = supported;
    require(!(d.Flags & ~known), CUDA_ERROR_INVALID_VALUE, "invalid array flags");
    require(!(d.Flags & ~supported), CUDA_ERROR_NOT_SUPPORTED, "array mapping or attachment mode is not implemented");
    if (format.encoding == array_storage_encoding::block_compressed)
        require(d.Height && !(d.Flags & CUDA_ARRAY3D_SURFACE_LDST),
            CUDA_ERROR_INVALID_VALUE, "block-compressed arrays require a height and cannot be surface resources");
    if (format.planes > 1)
        require(d.Height && (!d.Depth || (d.Flags & CUDA_ARRAY3D_LAYERED))
            && !(d.Flags & (CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_TEXTURE_GATHER | CUDA_ARRAY3D_DEPTH_TEXTURE)),
            CUDA_ERROR_INVALID_VALUE, "invalid multiplanar array shape or usage");
    require((d.Flags & (CUDA_ARRAY3D_SPARSE | CUDA_ARRAY3D_DEFERRED_MAPPING)) !=
        (CUDA_ARRAY3D_SPARSE | CUDA_ARRAY3D_DEFERRED_MAPPING), CUDA_ERROR_INVALID_VALUE, "sparse and deferred allocation modes are exclusive");
    if (d.Flags & CUDA_ARRAY3D_SPARSE)
        require(d.Height && !(d.Flags & CUDA_ARRAY3D_CUBEMAP), CUDA_ERROR_INVALID_VALUE, "invalid sparse array shape");
    if (d.Flags & CUDA_ARRAY3D_DEPTH_TEXTURE)
        require(d.Height && d.NumChannels == 1 && (d.Format == CU_AD_FORMAT_FLOAT || d.Format == CU_AD_FORMAT_UNSIGNED_INT16),
            CUDA_ERROR_INVALID_VALUE, "invalid depth array format");
    require(!d.Depth || d.Height || (d.Flags & CUDA_ARRAY3D_LAYERED), CUDA_ERROR_INVALID_VALUE, "3D arrays require a height");
    require(!(d.Flags & CUDA_ARRAY3D_LAYERED) || d.Depth, CUDA_ERROR_INVALID_VALUE, "layered arrays require layers");
    if (d.Flags & CUDA_ARRAY3D_CUBEMAP)
        require(d.Width == d.Height && d.Depth && (d.Flags & CUDA_ARRAY3D_LAYERED ? d.Depth % 6 == 0 : d.Depth == 6), CUDA_ERROR_INVALID_VALUE, "invalid cubemap dimensions");
    if (d.Flags & CUDA_ARRAY3D_TEXTURE_GATHER)
        require(d.Height && !d.Depth && !(d.Flags & CUDA_ARRAY3D_LAYERED), CUDA_ERROR_INVALID_VALUE, "gather requires a 2D array");
}
void validate_array_limits(const CUDA_ARRAY3D_DESCRIPTOR& d, const std::shared_ptr<context>& owner) {
    const bool surface = (d.Flags & CUDA_ARRAY3D_SURFACE_LDST) != 0;
    auto fits = [&](size_t extent, CUdevice_attribute attribute) {
        const auto maximum = global().provider->attribute(owner->device, attribute);
        return maximum > 0 && extent <= size_t(maximum);
    };
    auto limit = [&](size_t extent, CUdevice_attribute texture, CUdevice_attribute writable) {
        require(fits(extent, surface ? writable : texture), CUDA_ERROR_INVALID_VALUE, "array extent exceeds device limit");
    };
    if (d.Flags & CUDA_ARRAY3D_CUBEMAP) {
        if (d.Flags & CUDA_ARRAY3D_LAYERED) {
            limit(d.Width, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH);
            limit(d.Depth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS);
        } else limit(d.Width, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH);
    } else if (d.Flags & CUDA_ARRAY3D_LAYERED) {
        if (!d.Height) {
            limit(d.Width, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH);
            limit(d.Depth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS);
        } else {
            limit(d.Width, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH);
            limit(d.Height, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT);
            limit(d.Depth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS);
        }
    } else if (d.Depth) {
        const bool standard = fits(d.Width, surface ? CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH : CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH)
            && fits(d.Height, surface ? CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT : CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT)
            && fits(d.Depth, surface ? CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH : CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH);
        const bool alternate = !surface && fits(d.Width, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE)
            && fits(d.Height, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE)
            && fits(d.Depth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE);
        require(standard || alternate, CUDA_ERROR_INVALID_VALUE, "3D array exceeds standard and alternate limits");
    } else if (d.Height) {
        limit(d.Width, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH);
        limit(d.Height, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT);
    } else limit(d.Width, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH);
    if (d.Flags & CUDA_ARRAY3D_TEXTURE_GATHER)
        require(fits(d.Width, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH)
            && fits(d.Height, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT), CUDA_ERROR_INVALID_VALUE, "array exceeds gather limits");
}
CUarray publish_array(const std::shared_ptr<context>& owner, const CUDA_ARRAY3D_DESCRIPTOR& d,
                      std::shared_ptr<buffer> native, CUmipmappedArray parent = nullptr) {
    auto entry = std::make_shared<array_entry>(); entry->owner = owner; entry->descriptor = d; entry->native = std::move(native); entry->parent = parent;
    if (owner->native->kind() != cudavmBackendCuda) entry->image_address = reserve_device_address(entry->native->size);
    auto handle = reinterpret_cast<CUarray>(global().next_array++);
    global().device_buffers.push_back({owner->device, entry->native});
    global().arrays.emplace(handle, std::move(entry)); return handle;
}
CUarray create_array(const CUDA_ARRAY3D_DESCRIPTOR& d, const std::shared_ptr<context>& owner) {
    validate_array(d); validate_array_limits(d, owner); capture_unsafe(owner);
    if (d.Flags & CUDA_ARRAY3D_SPARSE)
        require(global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED) != 0,
            CUDA_ERROR_NOT_SUPPORTED, "sparse arrays are unavailable on this backend endpoint");
    auto native = native_call(owner, [&] { return owner->native->allocate_array(d); });
    return publish_array(owner, d, std::move(native));
}
CUarray array_plane(CUarray handle, unsigned plane, const std::shared_ptr<context>& owner) {
    auto parent = array_lookup(handle, owner);
    const auto descriptor = array_plane_descriptor(parent->descriptor, plane);
    const auto count = array_format_properties(parent->descriptor.Format).planes;
    if (parent->planes.empty()) parent->planes.resize(count);
    if (parent->planes[plane]) return parent->planes[plane];
    size_t offset = 0;
    const auto layers = std::max(size_t(1), descriptor.Depth);
    for (unsigned i = 0; i < plane; ++i) offset += array_storage_bytes(array_plane_descriptor(parent->descriptor, i)) / layers;
    auto native = std::make_shared<array_plane_storage>(parent->native, offset, array_storage_bytes(descriptor),
        array_storage_bytes(parent->descriptor) / layers);
    auto result = publish_array(owner, descriptor, std::move(native));
    global().arrays.at(result)->parent_array = handle;
    parent->planes[plane] = result;
    return result;
}
CUmipmappedArray create_mipmapped_array(const CUDA_ARRAY3D_DESCRIPTOR& d, unsigned levels, const std::shared_ptr<context>& owner) {
    validate_array(d); validate_array_limits(d, owner); capture_unsafe(owner);
    if (d.Flags & CUDA_ARRAY3D_SPARSE)
        require(global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED) != 0,
            CUDA_ERROR_NOT_SUPPORTED, "sparse arrays are unavailable on this backend endpoint");
    auto extent = std::max({d.Width, d.Height, (d.Flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP)) ? size_t(0) : d.Depth});
    unsigned maximum = 0; for (; extent; extent >>= 1) ++maximum;
    levels = std::clamp(levels, 1u, maximum);
    auto native = native_call(owner, [&] { return owner->native->allocate_mipmapped_array(d, levels); });
    auto handle = reinterpret_cast<CUmipmappedArray>(global().next_mipmapped_array++);
    mipmapped_array_entry entry{owner, {}}; entry.levels.reserve(levels);
    try {
        for (unsigned i = 0; i < levels; ++i) entry.levels.push_back(publish_array(owner, array_level_descriptor(d, i), std::move(native[i]), handle));
        global().mipmapped_arrays.emplace(handle, entry);
    } catch (...) { for (auto level : entry.levels) global().arrays.erase(level); throw; }
    return handle;
}
void destroy_array(CUarray handle) {
    auto owner = current(); auto entry = array_lookup(handle, owner);
    require(!entry->parent && !entry->parent_array, CUDA_ERROR_INVALID_VALUE, "array subresources are owned by their parent");
    capture_unsafe(owner); entry->retiring = true;
    invalidate_native_graphs();
    for (auto plane : entry->planes) if (plane) global().arrays.at(plane)->retiring = true;
    drain_context_work(owner);
    for (auto plane : entry->planes) if (plane) global().arrays.erase(plane);
    global().arrays.erase(handle);
}
void destroy_mipmapped_array(CUmipmappedArray handle) {
    auto owner = current(); auto it = global().mipmapped_arrays.find(handle);
    require(it != global().mipmapped_arrays.end() && !it->second.retiring, CUDA_ERROR_INVALID_HANDLE, "invalid mipmapped array");
    require(it->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "mipmapped array belongs to a different context");
    capture_unsafe(owner);
    auto levels = it->second.levels; it->second.retiring = true;
    invalidate_native_graphs();
    for (auto level : levels) global().arrays.at(level)->retiring = true;
    drain_context_work(owner);
    for (auto level : levels) global().arrays.erase(level);
    global().mipmapped_arrays.erase(handle);
}
