// Own descriptors through map/unmap completion. The adapter retains mapped
// backing after the caller releases its creating handle.
void map_arrays(const CUarrayMapInfo* parameters, unsigned count, CUstream stream) {
    require(parameters && count, CUDA_ERROR_INVALID_VALUE, "missing array mapping descriptors");
    auto target = stream_lookup(stream); auto owner = target->owner.lock();
    capture_check_stream(target);
    require(owner->native->kind() == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "array mapping requires native CUDA");
    require(owner->device < 32, CUDA_ERROR_INVALID_DEVICE, "array mapping device exceeds the public device mask");
    std::vector<array_mapping> mappings;
    mappings.reserve(count);
    for (unsigned i = 0; i < count; ++i) {
        auto p = parameters[i];
        require(!p.flags && !p.reserved[0] && !p.reserved[1] && p.deviceBitMask == (1u << owner->device),
            CUDA_ERROR_INVALID_VALUE, "invalid array mapping flags or device mask");
        require(p.memHandleType == CU_MEM_HANDLE_TYPE_GENERIC &&
            (p.memOperationType == CU_MEM_OPERATION_TYPE_MAP || p.memOperationType == CU_MEM_OPERATION_TYPE_UNMAP),
            CUDA_ERROR_INVALID_VALUE, "invalid array mapping operation or handle type");
        std::shared_ptr<array_entry> resource;
        unsigned levels = 1;
        if (p.resourceType == CU_RESOURCE_TYPE_ARRAY) {
            resource = array_lookup(p.resource.array, owner);
            require(!resource->parent && !resource->parent_array, CUDA_ERROR_INVALID_VALUE, "map the parent mipmapped array");
        } else {
            require(p.resourceType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY, CUDA_ERROR_INVALID_VALUE, "invalid array mapping resource type");
            auto found = global().mipmapped_arrays.find(p.resource.mipmap);
            require(found != global().mipmapped_arrays.end() && !found->second.retiring,
                CUDA_ERROR_INVALID_HANDLE, "invalid mapping mipmapped array");
            require(found->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "mapping mipmap belongs to another context");
            levels = unsigned(found->second.levels.size());
            resource = array_lookup(found->second.levels.front(), owner);
        }
        const auto& d = resource->descriptor;
        require(d.Flags & (CUDA_ARRAY3D_SPARSE | CUDA_ARRAY3D_DEFERRED_MAPPING),
            CUDA_ERROR_INVALID_VALUE, "array mapping needs sparse or deferred storage");
        size_t bytes = 0;
        if (d.Flags & CUDA_ARRAY3D_SPARSE) {
            auto properties = context_configuration(owner, [&] { return p.resourceType == CU_RESOURCE_TYPE_ARRAY
                ? resource->native->array_sparse_properties() : resource->native->mipmapped_sparse_properties(); });
            const size_t tile_bytes = size_t(properties.tileExtent.width) * properties.tileExtent.height
                * properties.tileExtent.depth * array_element_bytes(d);
            require(tile_bytes, CUDA_ERROR_INVALID_VALUE, "invalid sparse tile size");
            if (p.subresourceType == CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL) {
                const auto& s = p.subresource.sparseLevel;
                require(s.level < levels && (p.resourceType == CU_RESOURCE_TYPE_ARRAY || s.level < properties.miptailFirstLevel),
                    CUDA_ERROR_INVALID_VALUE, "mapping level belongs to the mip tail or exceeds the chain");
                auto level = array_level_descriptor(d, s.level);
                const bool layered = (d.Flags & CUDA_ARRAY3D_LAYERED) != 0;
                require(layered ? s.layer < d.Depth : s.layer == 0, CUDA_ERROR_INVALID_VALUE, "invalid sparse layer");
                auto axis = [&](unsigned offset, unsigned extent, size_t bound, unsigned tile) {
                    require(tile && extent && offset < bound && extent <= bound - offset && offset % tile == 0
                        && (extent % tile == 0 || extent == bound - offset), CUDA_ERROR_INVALID_VALUE, "invalid sparse tile extent");
                    return (size_t(extent) + tile - 1) / tile;
                };
                auto x = axis(s.offsetX, s.extentWidth, level.Width, properties.tileExtent.width);
                auto y = axis(s.offsetY, s.extentHeight, level.Height, properties.tileExtent.height);
                auto z = axis(s.offsetZ, s.extentDepth, layered ? 1 : std::max(size_t(1), level.Depth), properties.tileExtent.depth);
                require(x <= SIZE_MAX / y && x * y <= SIZE_MAX / z && x * y * z <= SIZE_MAX / tile_bytes,
                    CUDA_ERROR_INVALID_VALUE, "sparse mapping size overflow");
                bytes = x * y * z * tile_bytes;
            } else {
                require(p.subresourceType == CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL,
                    CUDA_ERROR_INVALID_VALUE, "invalid sparse subresource type");
                const auto& s = p.subresource.miptail;
                const bool per_layer = (d.Flags & CUDA_ARRAY3D_LAYERED) && !(properties.flags & CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL);
                require(per_layer ? s.layer < d.Depth : s.layer == 0, CUDA_ERROR_INVALID_VALUE, "invalid mip-tail layer");
                require(s.size && s.offset < properties.miptailSize && s.size <= properties.miptailSize - s.offset
                    && s.offset % tile_bytes == 0 && s.size % tile_bytes == 0,
                    CUDA_ERROR_INVALID_VALUE, "invalid mip-tail mapping extent");
                bytes = size_t(s.size);
            }
        } else {
            auto requirements = context_configuration(owner, [&] { return p.resourceType == CU_RESOURCE_TYPE_ARRAY
                ? resource->native->array_requirements() : resource->native->mipmapped_requirements(); });
            bytes = requirements.size;
        }
        std::shared_ptr<virtual_allocation> physical;
        if (p.memOperationType == CU_MEM_OPERATION_TYPE_MAP) {
            auto allocation = virtual_handle(p.memHandle.memHandle);
            require(allocation->properties.location.type == CU_MEM_LOCATION_TYPE_DEVICE
                && allocation->properties.location.id == owner->device
                && (allocation->properties.allocFlags.usage & CU_MEM_CREATE_USAGE_TILE_POOL)
                && p.offset % allocation->granularity == 0 && p.offset <= allocation->native->size
                && bytes <= allocation->native->size - size_t(p.offset), CUDA_ERROR_INVALID_VALUE, "invalid tile-pool backing or offset");
            physical = allocation->native;
        } else require(!p.memHandle.memHandle, CUDA_ERROR_INVALID_VALUE, "unmap requires a null physical handle");
        mappings.push_back({p, resource->native, std::move(physical)});
    }
    auto queue = stream_native(owner, target);
    submit_stream(owner, stream, {}, {}, 0, [owner, mappings = std::move(mappings), queue] {
        return owner->native->enqueue_array_mapping(mappings, queue.get());
    });
}
