void peer_link_devices(int source, int target) {
    valid_device(source); valid_device(target);
    require(source != target, CUDA_ERROR_INVALID_DEVICE, "peer link queries require distinct devices");
}
void peer_attribute(int* output, CUdevice_P2PAttribute attribute, int source, int target) {
    require(output && attribute >= CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK && attribute <= CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED,
        CUDA_ERROR_INVALID_VALUE, "invalid peer link attribute or output");
    peer_link_devices(source, target);
    *output = global().provider->peer_attribute(source, target, attribute);
}
void peer_atomic_capabilities(unsigned* output, const CUatomicOperation* operations, unsigned count, int source, int target) {
    require(output && operations && count, CUDA_ERROR_INVALID_VALUE, "missing peer atomic capability array or empty count");
    for (unsigned i = 0; i < count; ++i)
        require(operations[i] >= CU_ATOMIC_OPERATION_INTEGER_ADD && operations[i] < CU_ATOMIC_OPERATION_MAX, CUDA_ERROR_INVALID_VALUE, "invalid peer atomic operation");
    peer_link_devices(source, target);
    auto result = global().provider->peer_atomic_capabilities(source, target, {operations, count});
    require(result.size() == count, CUDA_ERROR_UNKNOWN, "native peer atomic result extent mismatch");
    std::memcpy(output, result.data(), result.size() * sizeof(unsigned));
}
void peer_enable(CUcontext handle, unsigned flags) {
    auto owner = current(); auto peer = lookup(handle);
    require(!flags, CUDA_ERROR_INVALID_VALUE, "peer access flags must be zero");
    require(owner != peer, CUDA_ERROR_INVALID_CONTEXT, "peer context equals current context");
    require(!peer_access_enabled(owner, peer), CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, "peer access is already enabled");
    require(global().provider->can_access_peer(owner->device, peer->device), CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, "direct access is unavailable for this directed device pair");
    native_call(owner, [&] {
        lookup(handle);
        // Waiting releases the registry mutex; another caller may have enabled
        // the same directed edge while this caller drained earlier work.
        require(!peer_access_enabled(owner, peer), CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, "peer access is already enabled");
        auto [entry, inserted] = owner->peers.try_emplace(handle);
        try { owner->native->enable_peer(*peer->native); }
        catch (...) { if (inserted) owner->peers.erase(entry); throw; }
        entry->second = peer;
    });
    invalidate_native_graphs();
}
void peer_disable(CUcontext handle) {
    auto owner = current(); auto peer = lookup(handle);
    require(owner != peer, CUDA_ERROR_INVALID_CONTEXT, "peer context equals current context");
    require(peer_access_enabled(owner, peer), CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "peer access is not enabled");
    native_call(owner, [&] {
        lookup(handle);
        require(peer_access_enabled(owner, peer), CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "peer access is not enabled");
        owner->native->disable_peer(*peer->native);
    });
    owner->peers.erase(handle); invalidate_native_graphs();
}
