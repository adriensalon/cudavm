void attach_managed(CUstream stream, CUdeviceptr pointer, size_t bytes, unsigned flags) {
    require(pointer && (flags == CU_MEM_ATTACH_GLOBAL || flags == CU_MEM_ATTACH_HOST || flags == CU_MEM_ATTACH_SINGLE), CUDA_ERROR_INVALID_VALUE, "invalid stream association arguments");
    require(flags != CU_MEM_ATTACH_SINGLE || (stream && stream != CU_STREAM_LEGACY), CUDA_ERROR_INVALID_VALUE, "single association requires a specific stream");
    auto target = stream_lookup(stream); auto owner = target->owner.lock();
    auto it = containing(pointer); std::shared_ptr<buffer> storage;
    if (it != global().allocations.end()) {
        require(it->first == pointer && it->second.native->is_managed() && (!bytes || bytes == it->second.native->size), CUDA_ERROR_INVALID_VALUE, "association requires an entire managed allocation");
        storage = it->second.native;
    } else {
        require(bytes, CUDA_ERROR_INVALID_VALUE, "pageable association requires a nonzero length");
        require(global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS) != 0, CUDA_ERROR_NOT_SUPPORTED, "pageable stream association is unavailable on this device");
    }
    auto queue = stream_native(owner, target);
    submit_stream(owner, stream, {}, {}, 0, [owner, pointer, bytes, flags, storage, queue] {
        return owner->native->enqueue_attach(pointer, bytes, flags, queue.get());
    });
}
std::vector<std::shared_ptr<host_entry>> prepare_async_hosts(const memory_box& to, const memory_box& from) {
    std::vector<std::shared_ptr<host_entry>> result;
    if (to.host) result.push_back(to.host);
    if (from.host && from.host != to.host) result.push_back(from.host);
    for (auto& entry : result) {
        std::erase_if(entry->work, [](auto& item) { return item.expired(); });
        entry->work.reserve(entry->work.size() + 1);
    }
    return result;
}
graph_node capture_copy_boxes(const memory_box& to, const memory_box& from, size_t width, const std::shared_ptr<context>& owner, bool peer = false) {
    graph_node node;
    if (width && to.rows && to.depth) {
        node.type = CU_GRAPH_NODE_TYPE_MEMCPY; node.owner = owner; node.peer_copy = peer;
        if (peer) { node.copy_source = from.owner; node.copy_destination = to.owner; }
        auto& p = node.copy;
        p.srcMemoryType = from.array ? CU_MEMORYTYPE_ARRAY : from.device ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
        p.srcHost = reinterpret_cast<const void*>(uintptr_t(from.base + from.first)); p.srcDevice = from.base + from.first;
        p.srcPitch = from.pitch; p.srcHeight = from.pitch ? from.slice / from.pitch : 0;
        p.dstMemoryType = to.array ? CU_MEMORYTYPE_ARRAY : to.device ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
        p.dstHost = reinterpret_cast<void*>(uintptr_t(to.base + to.first)); p.dstDevice = to.base + to.first;
        p.dstPitch = to.pitch; p.dstHeight = to.pitch ? to.slice / to.pitch : 0;
        if (from.array) { p.srcArray = from.array; p.srcXInBytes = from.first % from.pitch; p.srcY = from.first / from.pitch % p.srcHeight; p.srcZ = from.first / from.slice; }
        if (to.array) { p.dstArray = to.array; p.dstXInBytes = to.first % to.pitch; p.dstY = to.first / to.pitch % p.dstHeight; p.dstZ = to.first / to.slice; }
        p.WidthInBytes = width; p.Height = to.rows; p.Depth = to.depth;
    }
    return node;
}
void async_copy_boxes(const memory_box& to, const memory_box& from, size_t width, CUstream stream) {
    require(to.owner == from.owner, CUDA_ERROR_NOT_SUPPORTED, "asynchronous peer transfers require explicit endpoint contexts");
    if (width && to.rows && to.depth) memory_disjoint(to, from, width);
    if (capture_node(to.owner, stream, capture_copy_boxes(to, from, width, to.owner))) return;
    auto hosts = prepare_async_hosts(to, from);
    auto queue = stream_native(to.owner, stream_lookup(stream, to.owner));
    auto done = submit_stream(to.owner, stream, {}, {}, 0, [to, from, width, queue] { return enqueue_copy_boxes(to, from, width, queue.get()); });
    for (auto& entry : hosts) host_track(entry, done);
}
void async_copy_linear(CUdeviceptr to, const void* to_host, CUmemorytype to_type,
                       CUdeviceptr from, const void* from_host, CUmemorytype from_type,
                       size_t bytes, CUstream stream, const std::shared_ptr<context>& owner) {
    graph_memory_scope scope(bool(stream_lookup(stream, owner)->capture));
    auto source = memory_endpoint(from_type, from_host, from, bytes, 0, 0, 0, 0, bytes, 1, 1, owner, false);
    auto destination = memory_endpoint(to_type, to_host, to, bytes, 0, 0, 0, 0, bytes, 1, 1, owner, false);
    async_copy_boxes(destination, source, bytes, stream);
}
void async_copy_2d(const CUDA_MEMCPY2D& p, CUstream stream, const std::shared_ptr<context>& owner) {
    graph_memory_scope scope(bool(stream_lookup(stream, owner)->capture));
    auto from = memory_endpoint(p.srcMemoryType, p.srcHost, p.srcDevice, p.srcPitch, 0, p.srcXInBytes, p.srcY, 0, p.WidthInBytes, p.Height, 1, owner, true, p.srcArray);
    auto to = memory_endpoint(p.dstMemoryType, p.dstHost, p.dstDevice, p.dstPitch, 0, p.dstXInBytes, p.dstY, 0, p.WidthInBytes, p.Height, 1, owner, true, p.dstArray);
    async_copy_boxes(to, from, p.WidthInBytes, stream);
}
void async_copy_3d(const CUDA_MEMCPY3D& p, CUstream stream, const std::shared_ptr<context>& owner) {
    graph_memory_scope scope(bool(stream_lookup(stream, owner)->capture));
    require(!p.srcLOD && !p.dstLOD && !p.reserved0 && !p.reserved1, CUDA_ERROR_INVALID_VALUE, "invalid 3D copy LOD or reserved fields");
    auto from = memory_endpoint(p.srcMemoryType, p.srcHost, p.srcDevice, p.srcPitch, p.srcHeight, p.srcXInBytes, p.srcY, p.srcZ, p.WidthInBytes, p.Height, p.Depth, owner, true, p.srcArray);
    auto to = memory_endpoint(p.dstMemoryType, p.dstHost, p.dstDevice, p.dstPitch, p.dstHeight, p.dstXInBytes, p.dstY, p.dstZ, p.WidthInBytes, p.Height, p.Depth, owner, true, p.dstArray);
    async_copy_boxes(to, from, p.WidthInBytes, stream);
}
void async_peer_boxes(const memory_box& to, const memory_box& from, size_t width, CUstream stream, const std::shared_ptr<context>& owner) {
    if (width && to.rows && to.depth) memory_disjoint(to, from, width);
    // Peer capture is a cudavm extension; local native CUDA 12.4 rejects it.
    if (capture_node(owner, stream, capture_copy_boxes(to, from, width, owner, true))) return;
    std::vector<std::shared_ptr<context>> foreign;
    if (to.owner != owner) foreign.push_back(to.owner);
    if (from.owner != owner && from.owner != to.owner) foreign.push_back(from.owner);
    // Reserve tracking before enqueueing, so allocation failure cannot leave a
    // submitted command missing from another context's teardown dependencies.
    for (auto& endpoint : foreign) {
        std::erase_if(endpoint->foreign_work, [](auto& item) { return item.expired(); });
        endpoint->foreign_work.reserve(endpoint->foreign_work.size() + 1);
    }
    auto hosts = prepare_async_hosts(to, from);
    auto queue = stream_native(owner, stream_lookup(stream, owner));
    auto done = submit_stream(owner, stream, {}, {}, 0, [to, from, width, queue, owner] {
        return enqueue_copy_boxes(to, from, width, queue.get(), owner);
    });
    for (auto& entry : hosts) host_track(entry, done);
    for (auto& endpoint : foreign) endpoint->foreign_work.push_back(done);
}
void async_peer_linear(CUdeviceptr to, CUcontext to_context, CUdeviceptr from, CUcontext from_context, size_t bytes, CUstream stream) {
    auto to_owner = lookup(to_context), from_owner = lookup(from_context), owner = current();
    auto source = memory_endpoint(CU_MEMORYTYPE_DEVICE, nullptr, from, bytes, 0, 0, 0, 0, bytes, 1, 1, from_owner, false);
    auto destination = memory_endpoint(CU_MEMORYTYPE_DEVICE, nullptr, to, bytes, 0, 0, 0, 0, bytes, 1, 1, to_owner, false);
    async_peer_boxes(destination, source, bytes, stream, owner);
}
void async_peer_3d(const CUDA_MEMCPY3D_PEER& p, CUstream stream) {
    require(!p.srcLOD && !p.dstLOD, CUDA_ERROR_INVALID_VALUE, "copy LOD must be zero");
    auto to_owner = lookup(p.dstContext), from_owner = lookup(p.srcContext), owner = current();
    auto source = memory_endpoint(p.srcMemoryType, p.srcHost, p.srcDevice, p.srcPitch, p.srcHeight, p.srcXInBytes, p.srcY, p.srcZ, p.WidthInBytes, p.Height, p.Depth, from_owner, true, p.srcArray);
    auto destination = memory_endpoint(p.dstMemoryType, p.dstHost, p.dstDevice, p.dstPitch, p.dstHeight, p.dstXInBytes, p.dstY, p.dstZ, p.WidthInBytes, p.Height, p.Depth, to_owner, true, p.dstArray);
    async_peer_boxes(destination, source, p.WidthInBytes, stream, owner);
}
template<class T> void async_fill_box(CUdeviceptr pointer, size_t pitch, size_t slice_height, T value,
                                     size_t width, size_t height, size_t depth, CUstream stream, const std::shared_ptr<context>& owner) {
    graph_memory_scope scope(bool(stream_lookup(stream, owner)->capture));
    require(pointer % alignof(T) == 0 && pitch % alignof(T) == 0, CUDA_ERROR_INVALID_VALUE, "invalid asynchronous fill alignment");
    auto bytes = memory_mul(width, sizeof(T));
    auto to = memory_endpoint(CU_MEMORYTYPE_DEVICE, nullptr, pointer, pitch, slice_height, 0, 0, 0, bytes, height, depth, owner);
    if (stream_lookup(stream, owner)->capture) {
        for (size_t z = 0; z < depth; ++z) {
            graph_node node;
            if (width && height) { node.type = CU_GRAPH_NODE_TYPE_MEMSET; node.owner = owner;
                node.fill = {pointer + z * to.slice, pitch, static_cast<unsigned>(value), sizeof(T), width, height}; }
            capture_node(owner, stream, std::move(node));
        }
        return;
    }
    auto queue = stream_native(owner, stream_lookup(stream, owner));
    auto hosts = prepare_async_hosts(to, {});
    auto done = submit_stream(owner, stream, {}, {}, 0, [to, value, bytes, queue] { return enqueue_fill_box(to, value, bytes, queue.get()); });
    for (auto& host : hosts) host_track(host, done);
}
template<class T> void async_fill_linear(CUdeviceptr pointer, T value, size_t count, CUstream stream) {
    auto bytes = memory_mul(count, sizeof(T));
    // Linear fills have no device maximum-pitch restriction.
    require(pointer % alignof(T) == 0, CUDA_ERROR_INVALID_VALUE, "invalid asynchronous fill alignment");
    auto owner = current();
    graph_memory_scope scope(bool(stream_lookup(stream, owner)->capture));
    auto to = memory_endpoint(CU_MEMORYTYPE_DEVICE, nullptr, pointer, bytes, 0, 0, 0, 0, bytes, 1, 1, owner, false);
    graph_node node;
    if (count) { node.type = CU_GRAPH_NODE_TYPE_MEMSET; node.owner = owner;
        node.fill = {pointer, bytes, static_cast<unsigned>(value), sizeof(T), count, 1}; }
    if (capture_node(owner, stream, std::move(node))) return;
    auto queue = stream_native(owner, stream_lookup(stream, owner));
    auto hosts = prepare_async_hosts(to, {});
    auto done = submit_stream(owner, stream, {}, {}, 0, [to, value, bytes, queue] { return enqueue_fill_box(to, value, bytes, queue.get()); });
    for (auto& host : hosts) host_track(host, done);
}
