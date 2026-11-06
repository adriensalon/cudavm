CUipcMemHandle ipc_export(CUdeviceptr pointer) {
    auto owner = current(); auto it = global().allocations.find(pointer);
    require(it != global().allocations.end(), CUDA_ERROR_INVALID_VALUE, "IPC export requires an allocation base");
    auto& entry = it->second;
    require(allocation_visible(entry, owner), CUDA_ERROR_INVALID_CONTEXT, "IPC export allocation is unavailable in this context");
    require(!entry.graphics && !entry.external && !entry.module && !entry.library && entry.host.expired() &&
            !entry.graph_owned && !entry.pool && !entry.virtual_memory && !entry.ipc && !entry.native->is_managed() && !entry.async_free,
            CUDA_ERROR_INVALID_VALUE, "IPC export requires an ordinary device allocation");
    auto native = entry.native; auto storage_owner = entry.owner;
    return native_call(storage_owner, [&] {
        auto found = global().allocations.find(pointer);
        require(found != global().allocations.end() && found->second.native == native && !found->second.async_free,
                CUDA_ERROR_INVALID_VALUE, "IPC allocation changed while waiting for export");
        return native->export_ipc();
    });
}
CUdeviceptr ipc_import(CUipcMemHandle handle, unsigned flags, bool legacy) {
    require(flags == CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS, CUDA_ERROR_INVALID_VALUE, "invalid memory IPC flags");
    auto owner = current(); capture_unsafe(owner);
    std::lock_guard lock(owner->execution);
    std::shared_ptr<buffer> native = owner->native->import_ipc(handle, flags, legacy);
    auto address = native->native_address();
    require(address && native->size, CUDA_ERROR_INVALID_VALUE, "empty native IPC mapping");
    auto it = global().allocations.find(address);
    if (it != global().allocations.end()) {
        require(it->second.owner == owner && it->second.ipc && it->second.native->size == native->size,
                CUDA_ERROR_INVALID_CONTEXT, "native IPC mapping collides with another allocation");
        it->second.ipc->references.push_back(std::move(native));
    } else {
        auto state = std::make_shared<ipc_memory_state>(); state->references.push_back(native);
        add_shared_allocation(owner, std::move(native)); global().allocations.at(address).ipc = std::move(state);
    }
    return address;
}
void ipc_close(CUdeviceptr pointer) {
    auto owner = current(); auto it = global().allocations.find(pointer);
    require(it != global().allocations.end() && it->second.ipc, CUDA_ERROR_INVALID_VALUE, "close requires an imported IPC base");
    require(it->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "IPC import belongs to a different context");
    auto state = it->second.ipc;
    native_call(owner, [&] {
        // Waiting releases the Driver lock. Select the reference only after
        // draining, so concurrent closes cannot select the same native open.
        auto found = global().allocations.find(pointer);
        require(found != global().allocations.end() && found->second.ipc == state && !state->references.empty(),
                CUDA_ERROR_INVALID_VALUE, "IPC mapping changed while waiting for close");
        state->references.back()->close_ipc(); state->references.pop_back();
        if (state->references.empty()) { invalidate_native_graphs(); global().allocations.erase(found); }
    });
}
