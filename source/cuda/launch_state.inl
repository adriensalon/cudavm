void launch_attribute(launch_options& options, CUlaunchAttributeID id, const CUlaunchAttributeValue& value) {
    switch (id) {
    case CU_LAUNCH_ATTRIBUTE_IGNORE: return;
    case CU_LAUNCH_ATTRIBUTE_COOPERATIVE: options.cooperative = value.cooperative != 0; return;
    case CU_LAUNCH_ATTRIBUTE_PRIORITY: options.priority = value.priority; return;
    case CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION: {
        auto dim = value.clusterDim;
        require((dim.x && dim.y && dim.z) || (!dim.x && !dim.y && !dim.z), CUDA_ERROR_INVALID_CLUSTER_SIZE, "cluster dimensions must be all zero or all positive");
        if (dim.x) options.cluster = {dim.x, dim.y, dim.z}; else options.cluster.reset();
        return;
    }
    case CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
        require(value.clusterSchedulingPolicyPreference >= CU_CLUSTER_SCHEDULING_POLICY_DEFAULT &&
            value.clusterSchedulingPolicyPreference <= CU_CLUSTER_SCHEDULING_POLICY_RUBIN_DSMEM_LOCALITY,
            CUDA_ERROR_INVALID_VALUE, "invalid cluster scheduling policy");
        options.cluster_policy = value.clusterSchedulingPolicyPreference; return;
    case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:
        options.domain_map = {value.memSyncDomainMap.default_, value.memSyncDomainMap.remote}; return;
    case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN:
        require(value.memSyncDomain == CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT || value.memSyncDomain == CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE,
            CUDA_ERROR_INVALID_VALUE, "invalid memory synchronization domain");
        options.domain = value.memSyncDomain; return;
    case CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW:
        options.access_policy = access_policy_from_native(value.accessPolicyWindow); validate_access_policy(*options.access_policy); return;
    case CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE:
        require(value.sharedMemoryMode >= CU_SHARED_MEMORY_MODE_DEFAULT && value.sharedMemoryMode <= CU_SHARED_MEMORY_MODE_PREFER_OVERSIZED_SHARED_MEMORY,
            CUDA_ERROR_INVALID_VALUE, "invalid launch shared-memory mode");
        options.shared_memory_mode = value.sharedMemoryMode; return;
    case CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION: {
        const auto dim = value.preferredClusterDim;
        require((dim.x && dim.y && dim.z) || (!dim.x && !dim.y && !dim.z), CUDA_ERROR_INVALID_CLUSTER_SIZE, "preferred cluster dimensions must be all zero or all positive");
        if (dim.x) options.preferred_cluster = {dim.x, dim.y, dim.z}; else options.preferred_cluster.reset();
        return;
    }
    case CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
        require(value.sharedMemCarveout <= 100, CUDA_ERROR_INVALID_VALUE, "invalid launch shared-memory carveout");
        options.carveout = value.sharedMemCarveout; return;
    case CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING:
        require(value.nvlinkUtilCentricScheduling <= 1, CUDA_ERROR_INVALID_VALUE, "invalid NVLink scheduling hint");
        options.nvlink = value.nvlinkUtilCentricScheduling; return;
    case CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE:
        require(value.portableClusterSizeMode >= CU_LAUNCH_PORTABLE_CLUSTER_MODE_DEFAULT && value.portableClusterSizeMode <= CU_LAUNCH_PORTABLE_CLUSTER_MODE_ALLOW_NON_PORTABLE,
            CUDA_ERROR_INVALID_VALUE, "invalid portable cluster mode");
        options.portable_cluster = value.portableClusterSizeMode; return;
    case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION:
        options.programmatic_serialization = value.programmaticStreamSerializationAllowed != 0; return;
    case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT:
        require(value.programmaticEvent.flags == 0, CUDA_ERROR_INVALID_VALUE, "invalid programmatic event record flags");
        options.programmatic_event = {value.programmaticEvent.event, 0, value.programmaticEvent.triggerAtBlockStart != 0, {}}; return;
    case CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT:
        require(value.launchCompletionEvent.flags == 0, CUDA_ERROR_INVALID_VALUE, "invalid launch completion event record flags");
        options.completion_event = {value.launchCompletionEvent.event, 0, false, {}}; return;
    case CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE:
        require(value.deviceUpdatableKernelNode.deviceUpdatable == 0 || value.deviceUpdatableKernelNode.deviceUpdatable == 1,
            CUDA_ERROR_INVALID_VALUE, "invalid device-updatable kernel node flag");
        return; // cuLaunchKernelEx applies this graphs-only attribute to capture.
    default: throw failure(CUDA_ERROR_INVALID_VALUE, "invalid launch attribute");
    }
}
launch_shape launch_configuration(const CUlaunchConfig* config) {
    require(config && (!config->numAttrs || config->attrs), CUDA_ERROR_INVALID_VALUE, "invalid extended launch descriptor");
    launch_shape shape{{config->gridDimX, config->gridDimY, config->gridDimZ},
        {config->blockDimX, config->blockDimY, config->blockDimZ}, config->sharedMemBytes};
    // Repeated attributes use their final value, as on the native driver.
    for (unsigned i = 0; i < config->numAttrs; ++i) launch_attribute(shape.options, config->attrs[i].id, config->attrs[i].value);
    return shape;
}
int cluster_occupancy(CUfunction handle, const CUlaunchConfig* config, bool potential) {
    auto shape = launch_configuration(config);
    auto& entry = function_resources(handle); auto owner = entry.owner; auto function = entry.native;
    auto target = stream_lookup(config->hStream, owner);
    return context_configuration(owner, [&] {
        auto queue = stream_native(owner, target);
        return function->cluster_occupancy(shape, queue.get(), potential);
    });
}
