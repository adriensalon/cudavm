void access_policy_resources(const std::shared_ptr<context>& owner, const launch_options& options) {
    if (!options.access_policy || !options.access_policy->bytes) return;
    const auto& window = *options.access_policy; validate_access_policy(window);
    auto address = reinterpret_cast<uintptr_t>(window.base); auto found = containing(address);
    require(found != global().allocations.end(), CUDA_ERROR_INVALID_VALUE, "access policy window does not name registry storage");
    auto host = found->second.host.lock();
    require((host ? (!host->retiring && host_visible(host, owner)) : allocation_visible(found->second, owner)) && !found->second.async_free &&
        window.bytes <= found->second.native->size - (address - found->first),
        CUDA_ERROR_INVALID_VALUE, "access policy window does not name live accessible storage");
}
void stream_attribute_id(CUstreamAttrID attribute) {
    switch (attribute) {
    case CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW:
    case CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY:
    case CU_STREAM_ATTRIBUTE_PRIORITY:
    case CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:
    case CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN:
    case CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING: return;
    default: throw failure(CUDA_ERROR_INVALID_VALUE, "invalid stream attribute");
    }
}
CUstreamAttrValue stream_attribute(const std::shared_ptr<stream_state>& stream, CUstreamAttrID attribute) {
    stream_attribute_id(attribute); auto owner = stream->owner.lock(); CUstreamAttrValue value{};
    const auto& options = stream->attributes;
    switch (attribute) {
    case CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY: value.syncPolicy = static_cast<CUsynchronizationPolicy>(stream->synchronization_policy); break;
    case CU_STREAM_ATTRIBUTE_PRIORITY: value.priority = options.priority.value_or(stream->priority); break;
    case CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW: value.accessPolicyWindow = access_policy_to_native(options.access_policy.value_or(access_policy_window{})); break;
    case CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING:
        value.nvlinkUtilCentricScheduling = options.nvlink.value_or(0);
        // Probe availability even for a stream which has no override.
        if (owner->native->kind() == cudavmBackendCuda) context_configuration(owner, [&] { return owner->native->canonical_stream_attribute(attribute, value, stream->priority); });
        break;
    case CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN:
        require(owner->native->kind() == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "native memory synchronization domains are unavailable");
        value.memSyncDomain = options.domain.value_or(CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT); break;
    case CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP: {
        auto map = options.domain_map ? *options.domain_map : context_configuration(owner, [&] { return owner->native->default_stream_domain_map(); });
        value.memSyncDomainMap = {map[0], map[1]}; break;
    }
    default: break;
    }
    return value;
}
void stream_attribute(const std::shared_ptr<stream_state>& stream, CUstreamAttrID attribute, const CUstreamAttrValue& input) {
    stream_attribute_id(attribute); auto owner = stream->owner.lock(); auto options = stream->attributes;
    auto value = input;
    if (attribute == CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY)
        require(value.syncPolicy >= CU_SYNC_POLICY_AUTO && value.syncPolicy <= CU_SYNC_POLICY_BLOCKING_SYNC, CUDA_ERROR_INVALID_VALUE, "invalid stream synchronization policy");
    else if (attribute == CU_STREAM_ATTRIBUTE_PRIORITY) value.priority = stream_priority(value.priority, owner);
    else {
        launch_attribute(options, attribute, value);
        if (attribute == CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW) {
            require(owner->native->kind() == cudavmBackendCuda || !options.access_policy->bytes, CUDA_ERROR_NOT_SUPPORTED, "nonempty access policy windows require native resource information");
            access_policy_resources(owner, options);
        } else if (attribute != CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING)
            require(owner->native->kind() == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "native memory synchronization domains are unavailable");
    }
    value = context_configuration(owner, [&] { return owner->native->canonical_stream_attribute(attribute, value, stream->priority); });
    if (attribute == CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY) stream->synchronization_policy = int(value.syncPolicy);
    else { launch_attribute(options, attribute, value); stream->attributes = std::move(options); }
}
void copy_stream_attributes(CUstream destination, CUstream source) {
    auto from = stream_lookup(source), to = stream_lookup(destination);
    require(from->owner.lock() == to->owner.lock(), CUDA_ERROR_INVALID_VALUE, "stream attribute contexts differ");
    auto options = from->attributes; options.priority = options.priority.value_or(from->priority);
    to->attributes = std::move(options); to->synchronization_policy = from->synchronization_policy;
}
launch_shape stream_launch_shape(const launch_shape& input, const std::shared_ptr<stream_state>& stream) {
    auto shape = input; auto& options = shape.options; const auto& defaults = stream->attributes;
    if (!options.priority) options.priority = defaults.priority;
    if (!options.access_policy) options.access_policy = defaults.access_policy;
    if (!options.domain_map) options.domain_map = defaults.domain_map;
    if (!options.domain) options.domain = defaults.domain;
    if (!options.nvlink) options.nvlink = defaults.nvlink;
    return shape;
}
