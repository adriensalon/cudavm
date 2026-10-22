CUlaunchAttributeValue driver_launch_value(cudaLaunchAttributeID id, const cudaLaunchAttributeValue& value) {
    CUlaunchAttributeValue result{};
    switch (id) {
    case cudaLaunchAttributeAccessPolicyWindow: result.accessPolicyWindow = {value.accessPolicyWindow.base_ptr, value.accessPolicyWindow.num_bytes,
        value.accessPolicyWindow.hitRatio, static_cast<CUaccessProperty>(value.accessPolicyWindow.hitProp), static_cast<CUaccessProperty>(value.accessPolicyWindow.missProp)}; break;
    case cudaLaunchAttributeSynchronizationPolicy: result.syncPolicy = static_cast<CUsynchronizationPolicy>(value.syncPolicy); break;
    case cudaLaunchAttributeCooperative: result.cooperative = value.cooperative; break;
    case cudaLaunchAttributePriority: result.priority = value.priority; break;
    case cudaLaunchAttributeClusterDimension: result.clusterDim = {value.clusterDim.x, value.clusterDim.y, value.clusterDim.z}; break;
    case cudaLaunchAttributeClusterSchedulingPolicyPreference:
        result.clusterSchedulingPolicyPreference = static_cast<CUclusterSchedulingPolicy>(value.clusterSchedulingPolicyPreference); break;
    case cudaLaunchAttributeMemSyncDomainMap: result.memSyncDomainMap = {value.memSyncDomainMap.default_, value.memSyncDomainMap.remote}; break;
    case cudaLaunchAttributeMemSyncDomain: result.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(value.memSyncDomain); break;
    case cudaLaunchAttributeSharedMemoryMode: result.sharedMemoryMode = static_cast<CUsharedMemoryMode>(value.sharedMemoryMode); break;
    case cudaLaunchAttributePreferredClusterDimension: result.preferredClusterDim = {value.preferredClusterDim.x, value.preferredClusterDim.y, value.preferredClusterDim.z}; break;
    case cudaLaunchAttributePreferredSharedMemoryCarveout: result.sharedMemCarveout = value.sharedMemCarveout; break;
    case cudaLaunchAttributeNvlinkUtilCentricScheduling: result.nvlinkUtilCentricScheduling = value.nvlinkUtilCentricScheduling; break;
    case cudaLaunchAttributePortableClusterSizeMode: result.portableClusterSizeMode = static_cast<CUlaunchAttributePortableClusterMode>(value.portableClusterSizeMode); break;
    case cudaLaunchAttributeProgrammaticStreamSerialization: result.programmaticStreamSerializationAllowed = value.programmaticStreamSerializationAllowed; break;
    case cudaLaunchAttributeProgrammaticEvent: result.programmaticEvent = {reinterpret_cast<CUevent>(value.programmaticEvent.event), value.programmaticEvent.flags, value.programmaticEvent.triggerAtBlockStart}; break;
    case cudaLaunchAttributeLaunchCompletionEvent: result.launchCompletionEvent = {reinterpret_cast<CUevent>(value.launchCompletionEvent.event), value.launchCompletionEvent.flags}; break;
    case cudaLaunchAttributeDeviceUpdatableKernelNode:
        result.deviceUpdatableKernelNode.deviceUpdatable = value.deviceUpdatableKernelNode.deviceUpdatable;
        result.deviceUpdatableKernelNode.devNode = reinterpret_cast<CUgraphDeviceNode>(value.deviceUpdatableKernelNode.devNode); break;
    default: break; // The Driver rejects unsupported IDs without reading opaque values.
    }
    return result;
}
CUlaunchConfig driver_launch_config(const cudaLaunchConfig_t* config, std::vector<CUlaunchAttribute>& attributes) {
    require(config && (!config->numAttrs || config->attrs) && config->dynamicSmemBytes <= std::numeric_limits<unsigned>::max(),
        CUDA_ERROR_INVALID_VALUE, "invalid Runtime launch descriptor");
    for (unsigned i = 0; i < config->numAttrs; ++i) {
        auto& input = config->attrs[i]; CUlaunchAttribute value{};
        value.id = static_cast<CUlaunchAttributeID>(input.id); value.value = driver_launch_value(input.id, input.val);
        attributes.push_back(value);
    }
    return {config->gridDim.x, config->gridDim.y, config->gridDim.z, config->blockDim.x, config->blockDim.y, config->blockDim.z,
        unsigned(config->dynamicSmemBytes), reinterpret_cast<CUstream>(config->stream), attributes.data(), unsigned(attributes.size())};
}
cudaLaunchAttributeValue runtime_launch_value(cudaLaunchAttributeID id, const CUlaunchAttributeValue& value) {
    cudaLaunchAttributeValue result{};
    switch (id) {
    case cudaLaunchAttributeAccessPolicyWindow: result.accessPolicyWindow = {value.accessPolicyWindow.base_ptr, value.accessPolicyWindow.num_bytes,
        value.accessPolicyWindow.hitRatio, static_cast<cudaAccessProperty>(value.accessPolicyWindow.hitProp), static_cast<cudaAccessProperty>(value.accessPolicyWindow.missProp)}; break;
    case cudaLaunchAttributeSynchronizationPolicy: result.syncPolicy = static_cast<cudaSynchronizationPolicy>(value.syncPolicy); break;
    case cudaLaunchAttributeCooperative: result.cooperative = value.cooperative; break;
    case cudaLaunchAttributePriority: result.priority = value.priority; break;
    case cudaLaunchAttributeClusterDimension: result.clusterDim = {value.clusterDim.x, value.clusterDim.y, value.clusterDim.z}; break;
    case cudaLaunchAttributeClusterSchedulingPolicyPreference:
        result.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(value.clusterSchedulingPolicyPreference); break;
    case cudaLaunchAttributeMemSyncDomainMap: result.memSyncDomainMap = {value.memSyncDomainMap.default_, value.memSyncDomainMap.remote}; break;
    case cudaLaunchAttributeMemSyncDomain: result.memSyncDomain = static_cast<cudaLaunchMemSyncDomain>(value.memSyncDomain); break;
    case cudaLaunchAttributeSharedMemoryMode: result.sharedMemoryMode = static_cast<cudaSharedMemoryMode>(value.sharedMemoryMode); break;
    case cudaLaunchAttributePreferredClusterDimension: result.preferredClusterDim = {value.preferredClusterDim.x, value.preferredClusterDim.y, value.preferredClusterDim.z}; break;
    case cudaLaunchAttributePreferredSharedMemoryCarveout: result.sharedMemCarveout = value.sharedMemCarveout; break;
    case cudaLaunchAttributeNvlinkUtilCentricScheduling: result.nvlinkUtilCentricScheduling = value.nvlinkUtilCentricScheduling; break;
    case cudaLaunchAttributePortableClusterSizeMode: result.portableClusterSizeMode = static_cast<cudaLaunchAttributePortableClusterMode>(value.portableClusterSizeMode); break;
    case cudaLaunchAttributeProgrammaticStreamSerialization: result.programmaticStreamSerializationAllowed = value.programmaticStreamSerializationAllowed; break;
    case cudaLaunchAttributeProgrammaticEvent: result.programmaticEvent = {reinterpret_cast<cudaEvent_t>(value.programmaticEvent.event), value.programmaticEvent.flags, value.programmaticEvent.triggerAtBlockStart}; break;
    case cudaLaunchAttributeLaunchCompletionEvent: result.launchCompletionEvent = {reinterpret_cast<cudaEvent_t>(value.launchCompletionEvent.event), value.launchCompletionEvent.flags}; break;
    case cudaLaunchAttributeDeviceUpdatableKernelNode:
        result.deviceUpdatableKernelNode.deviceUpdatable = value.deviceUpdatableKernelNode.deviceUpdatable;
        result.deviceUpdatableKernelNode.devNode = reinterpret_cast<cudaGraphDeviceNode_t>(value.deviceUpdatableKernelNode.devNode); break;
    default: break;
    }
    return result;
}
