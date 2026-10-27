std::shared_ptr<array_entry> array_lookup(CUarray handle, const std::shared_ptr<context>& owner) {
    auto it = global().arrays.find(handle);
    require(it != global().arrays.end() && !it->second->retiring, CUDA_ERROR_INVALID_HANDLE, "invalid or retired array");
    require(it->second->owner == owner, CUDA_ERROR_INVALID_CONTEXT, "array belongs to a different context");
    return it->second;
}
void validate_array(const CUDA_ARRAY3D_DESCRIPTOR& d) {
    array_storage_bytes(d);
    constexpr unsigned supported = CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_CUBEMAP | CUDA_ARRAY3D_TEXTURE_GATHER;
    constexpr unsigned known = supported | CUDA_ARRAY3D_SPARSE | CUDA_ARRAY3D_DEFERRED_MAPPING | CUDA_ARRAY3D_COLOR_ATTACHMENT;
    require(!(d.Flags & ~known), CUDA_ERROR_INVALID_VALUE, "invalid array flags");
    require(!(d.Flags & ~supported), CUDA_ERROR_NOT_SUPPORTED, "array mapping or attachment mode is not implemented");
    require(!d.Depth || d.Height || (d.Flags & CUDA_ARRAY3D_LAYERED), CUDA_ERROR_INVALID_VALUE, "3D arrays require a height");
    require(!(d.Flags & CUDA_ARRAY3D_LAYERED) || d.Depth, CUDA_ERROR_INVALID_VALUE, "layered arrays require layers");
    if (d.Flags & CUDA_ARRAY3D_CUBEMAP)
        require(d.Width == d.Height && d.Depth && (d.Flags & CUDA_ARRAY3D_LAYERED ? d.Depth % 6 == 0 : d.Depth == 6), CUDA_ERROR_INVALID_VALUE, "invalid cubemap dimensions");
    if (d.Flags & CUDA_ARRAY3D_TEXTURE_GATHER)
        require(d.Height && !d.Depth && !(d.Flags & CUDA_ARRAY3D_LAYERED), CUDA_ERROR_INVALID_VALUE, "gather requires a 2D array");
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
    validate_array(d); capture_unsafe(owner);
    auto native = native_call(owner, [&] { return owner->native->allocate_array(d); });
    return publish_array(owner, d, std::move(native));
}
CUmipmappedArray create_mipmapped_array(const CUDA_ARRAY3D_DESCRIPTOR& d, unsigned levels, const std::shared_ptr<context>& owner) {
    validate_array(d); capture_unsafe(owner);
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
    require(!entry->parent, CUDA_ERROR_INVALID_VALUE, "mipmap levels are owned by their parent");
    capture_unsafe(owner); entry->retiring = true;
    invalidate_native_graphs();
    drain_context_work(owner); global().arrays.erase(handle);
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
