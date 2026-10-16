// Included in the Driver registry's private namespace: interop resources use
// the same context ownership, allocation lookup and monotonic handles.
template<class T> bool zero_reserved(const T& values) {
    return std::all_of(std::begin(values), std::end(values), [](auto value) { return value == 0; });
}
template<class Handle> Handle interop_handle() {
    auto result = reinterpret_cast<Handle>(global().next_interop);
    global().next_interop += 0x100;
    return result;
}
template<class Map, class Handle> auto& interop_lookup(Map& values, Handle handle, const std::shared_ptr<context>& owner) {
    auto found = values.find(handle);
    require(found != values.end(), CUDA_ERROR_INVALID_HANDLE, "invalid or destroyed interop handle");
    require(found->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "interop resource belongs to another CUDA context");
    return found->second;
}
void validate_memory(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d) {
    require(d.type >= CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD && d.type <= CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD,
            CUDA_ERROR_INVALID_VALUE, "invalid external memory handle type");
    require(d.size && d.size <= std::numeric_limits<size_t>::max() && !(d.flags & ~CUDA_EXTERNAL_MEMORY_DEDICATED)
            && zero_reserved(d.reserved), CUDA_ERROR_INVALID_VALUE, "invalid external memory descriptor");
    if (d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD || d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD)
        require(d.handle.fd >= 0, CUDA_ERROR_INVALID_VALUE, "invalid external memory file descriptor");
    else if (d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF)
        require(d.handle.nvSciBufObject, CUDA_ERROR_INVALID_VALUE, "null NvSciBuf object");
    else {
        require(bool(d.handle.win32.handle) != bool(d.handle.win32.name), CUDA_ERROR_INVALID_VALUE, "supply exactly one external memory handle or name");
        if (d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT || d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT)
            require(!d.handle.win32.name, CUDA_ERROR_INVALID_VALUE, "KMT handles cannot be named");
    }
    if (d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE || d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
        || d.type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT)
        require(d.flags & CUDA_EXTERNAL_MEMORY_DEDICATED, CUDA_ERROR_INVALID_VALUE, "a dedicated graphics resource requires the dedicated flag");
}
void validate_semaphore(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d) {
    require(d.type >= CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD && d.type <= CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32,
            CUDA_ERROR_INVALID_VALUE, "invalid external semaphore handle type");
    require(!d.flags && zero_reserved(d.reserved), CUDA_ERROR_INVALID_VALUE, "invalid external semaphore descriptor");
    if (d.type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD || d.type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD)
        require(d.handle.fd >= 0, CUDA_ERROR_INVALID_VALUE, "invalid external semaphore file descriptor");
    else if (d.type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC)
        require(d.handle.nvSciSyncObj, CUDA_ERROR_INVALID_VALUE, "null NvSciSync object");
    else {
        require(bool(d.handle.win32.handle) != bool(d.handle.win32.name), CUDA_ERROR_INVALID_VALUE, "supply exactly one external semaphore handle or name");
        if (d.type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT || d.type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT)
            require(!d.handle.win32.name, CUDA_ERROR_INVALID_VALUE, "KMT handles cannot be named");
    }
}
bool valued_semaphore(CUexternalSemaphoreHandleType type) {
    return type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE || type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE
        || type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD || type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32;
}
template<class Params> auto semaphore_entries(const CUexternalSemaphore* handles, const Params* params, unsigned count, const std::shared_ptr<context>& owner) {
    require(!count || (handles && params), CUDA_ERROR_INVALID_VALUE, "null external semaphore arrays");
    std::vector<std::pair<std::shared_ptr<external_semaphore>, uint64_t>> entries;
    entries.reserve(count);
    for (unsigned i = 0; i < count; ++i) {
        auto& entry = interop_lookup(global().semaphores, handles[i], owner);
        require(!params[i].flags && zero_reserved(params[i].reserved) && zero_reserved(params[i].params.reserved)
                && !params[i].params.nvSciSync.reserved && !params[i].params.keyedMutex.key,
                CUDA_ERROR_INVALID_VALUE, "unsupported external semaphore parameters");
        require(valued_semaphore(entry.type) || !params[i].params.fence.value, CUDA_ERROR_INVALID_VALUE, "binary semaphores do not accept fence values");
        entries.emplace_back(entry.native, params[i].params.fence.value);
        if constexpr (requires { params[i].params.keyedMutex.timeoutMs; })
            require(!params[i].params.keyedMutex.timeoutMs, CUDA_ERROR_INVALID_VALUE, "keyed mutex parameters are unavailable");
    }
    return entries;
}
template<class Params> void semaphore_operations(const CUexternalSemaphore* handles, const Params* params, unsigned count, CUstream stream, bool signal) {
    require(count && handles && params, CUDA_ERROR_INVALID_VALUE, "external semaphore submissions require nonempty arrays");
    auto owner = current();
    auto entries = semaphore_entries(handles, params, count, owner);
    graph_node node; node.owner = owner;
    node.type = signal ? CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL : CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT;
    if (count) node.semaphores.assign(handles, handles + count);
    if constexpr (std::is_same_v<Params, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>) {
        if (count) node.signals.assign(params, params + count);
    } else {
        if (count) node.waits.assign(params, params + count);
    }
    if (capture_node(owner, stream, std::move(node))) return;
    // Validate the entire batch before enqueuing any operation.
    submit_stream(owner, stream, [owner, entries = std::move(entries), signal] {
        std::lock_guard lock(owner->execution);
        for (auto& [semaphore, value] : entries)
            if (signal) semaphore->signal(value); else semaphore->wait(value);
        owner->native->synchronize();
    });
}
void gl_flags(unsigned flags) {
    require(flags == CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE || flags == CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY
            || flags == CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD, CUDA_ERROR_INVALID_VALUE, "invalid GL buffer access flags");
}
std::vector<graphics_entry*> graphics_batch(unsigned count, CUgraphicsResource* handles, CUstream stream, bool mapping) {
    require(!stream, CUDA_ERROR_NOT_SUPPORTED, "graphics mapping currently supports only the default stream");
    require(!count || handles, CUDA_ERROR_INVALID_VALUE, "null graphics resource array");
    auto owner = current();
    std::vector<graphics_entry*> entries;
    entries.reserve(count);
    for (unsigned i = 0; i < count; ++i) {
        auto& value = interop_lookup(global().graphics, handles[i], owner);
        require(std::find(entries.begin(), entries.end(), &value) == entries.end(), CUDA_ERROR_INVALID_HANDLE, "duplicate graphics resource in a batch");
        require(mapping ? !value.mapped : bool(value.mapped), mapping ? CUDA_ERROR_ALREADY_MAPPED : CUDA_ERROR_NOT_MAPPED, "invalid graphics mapping state");
        entries.push_back(&value);
    }
    return entries;
}
void map_graphics(unsigned count, CUgraphicsResource* handles, CUstream stream) {
    auto owner = current();
    graphics_batch(count, handles, stream, true);
    synchronize_context(owner);
    auto entries = graphics_batch(count, handles, stream, true);
    std::lock_guard lock(owner->execution);
    std::vector<std::unique_ptr<buffer>> buffers;
    for (auto* entry : entries) {
        auto size = owner->gl->size(entry->object);
        auto native = owner->native->allocate(size);
        if (entry->flags != CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD) {
            std::vector<std::byte> bytes(size); owner->gl->read(entry->object, bytes);
            owner->native->write(*native, 0, bytes);
        } else {
            // Completion is still required even when old contents are discarded.
            owner->gl->synchronize();
        }
        buffers.push_back(std::move(native));
    }
    std::vector<CUdeviceptr> addresses;
    try {
        for (unsigned i = 0; i < count; ++i) addresses.push_back(add_allocation(owner, std::move(buffers[i]), handles[i]));
    } catch (...) { for (auto address : addresses) global().allocations.erase(address); throw; }
    for (unsigned i = 0; i < count; ++i) entries[i]->mapped = addresses[i];
}
void unmap_graphics(unsigned count, CUgraphicsResource* handles, CUstream stream) {
    graphics_batch(count, handles, stream, false);
    auto owner = current(); synchronize_context(owner);
    auto entries = graphics_batch(count, handles, stream, false);
    std::lock_guard lock(owner->execution);
    // Preflight all GL objects before publishing any modified contents.
    for (auto* entry : entries) require(owner->gl->size(entry->object) == global().allocations.at(entry->mapped).native->size,
                                       CUDA_ERROR_INVALID_VALUE, "GL buffer storage changed while mapped");
    for (auto* entry : entries) {
        if (entry->flags != CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY) {
            auto& native = *global().allocations.at(entry->mapped).native;
            std::vector<std::byte> bytes(native.size); owner->native->read(native, 0, bytes);
            owner->gl->write(entry->object, bytes);
        }
    }
    for (auto* entry : entries) { global().allocations.erase(entry->mapped); entry->mapped = 0; }
}
