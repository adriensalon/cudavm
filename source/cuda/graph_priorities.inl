graph_node& graph_priority_node(CUgraphNode handle) {
    auto& node = graph_node_lookup(handle);
    require(node.type == CU_GRAPH_NODE_TYPE_KERNEL, CUDA_ERROR_INVALID_VALUE, "attribute requires a kernel graph node");
    return node;
}
void graph_priority_attribute(CUkernelNodeAttrID attribute) {
    switch (attribute) {
    case CU_KERNEL_NODE_ATTRIBUTE_PRIORITY:
    case CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE:
    case CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION:
    case CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
    case CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:
    case CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN:
    case CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE:
    case CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW:
    case CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION:
    case CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
    case CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING:
    case CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE: return;
    case CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE: return;
    default: throw failure(CUDA_ERROR_INVALID_VALUE, "invalid kernel graph attribute");
    }
}
void graph_launch_attribute(CUgraphNode handle, CUkernelNodeAttrID attribute, const CUkernelNodeAttrValue& value) {
    auto& node = graph_priority_node(handle); graph_priority_attribute(attribute);
    auto owner = node.owner.lock(); require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "kernel graph context was destroyed");
    if (attribute == CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE) {
        require(value.deviceUpdatableKernelNode.deviceUpdatable == 0 || value.deviceUpdatableKernelNode.deviceUpdatable == 1, CUDA_ERROR_INVALID_VALUE, "invalid device-updatable flag");
        require(!node.device_node || value.deviceUpdatableKernelNode.deviceUpdatable, CUDA_ERROR_INVALID_VALUE, "device-updatable nodes cannot opt out");
        if (!node.device_node && !value.deviceUpdatableKernelNode.deviceUpdatable) return;
        auto graph = global().graph_nodes.at(handle); bool materialized = bool(graph_lookup(graph).native);
        try { auto native = graph_native_materialize(graph, owner);
            auto result = native->attribute(node.native_node, attribute, &value);
            node.device_node = result.deviceUpdatableKernelNode.devNode; }
        catch (...) { if (!materialized) { auto& state = graph_lookup(graph); state.native.reset(); state.native_owner.reset(); for (auto& item : state.nodes) item.native_node = nullptr; } throw; }
        return;
    }
    if (attribute == CU_KERNEL_NODE_ATTRIBUTE_PRIORITY) {
        if (node.native_node) graph_lookup(global().graph_nodes.at(handle)).native->attribute(node.native_node, attribute, &value);
        node.priority = value.priority; return;
    }
    auto shape = node.shape; launch_attribute(shape.options, static_cast<CUlaunchAttributeID>(attribute), value);
    access_policy_resources(owner, shape.options);
    auto function = function_lookup(node.function, owner).native;
    context_configuration(owner, [&] { function->validate_configuration(shape); });
    if (node.native_node) graph_lookup(global().graph_nodes.at(handle)).native->attribute(node.native_node, attribute, &value);
    node.shape = std::move(shape);
}
CUkernelNodeAttrValue graph_launch_attribute(CUgraphNode handle, CUkernelNodeAttrID attribute) {
    auto& node = graph_priority_node(handle); graph_priority_attribute(attribute); CUkernelNodeAttrValue value{};
    const auto& options = node.shape.options;
    switch (attribute) {
    case CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE: {
        auto owner = node.owner.lock(); require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "kernel graph context was destroyed");
        value.deviceUpdatableKernelNode.deviceUpdatable = node.device_node != nullptr;
        value.deviceUpdatableKernelNode.devNode = node.device_node; break;
    }
    case CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW: value.accessPolicyWindow = access_policy_to_native(options.access_policy.value_or(access_policy_window{})); break;
    case CU_KERNEL_NODE_ATTRIBUTE_PRIORITY: {
        auto owner = node.owner.lock(); require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "kernel graph context was destroyed");
        value.priority = stream_priority(node.priority, owner); break;
    }
    case CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE: value.cooperative = options.cooperative; break;
    case CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION:
        if (options.cluster) { const auto& dim = *options.cluster; value.clusterDim = {dim[0], dim[1], dim[2]}; } break;
    case CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION:
        if (options.preferred_cluster) { const auto& dim = *options.preferred_cluster; value.preferredClusterDim = {dim[0], dim[1], dim[2]}; } break;
    case CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: value.sharedMemCarveout = options.carveout.value_or(0); break;
    case CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING: value.nvlinkUtilCentricScheduling = options.nvlink.value_or(0); break;
    case CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE: value.portableClusterSizeMode = options.portable_cluster.value_or(CU_LAUNCH_PORTABLE_CLUSTER_MODE_DEFAULT); break;
    case CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
        value.clusterSchedulingPolicyPreference = options.cluster_policy.value_or(CU_CLUSTER_SCHEDULING_POLICY_DEFAULT); break;
    case CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP: {
        auto owner = node.owner.lock(); require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "kernel graph context was destroyed");
        auto function = function_lookup(node.function, owner).native;
        const auto map = options.domain_map ? *options.domain_map : context_configuration(owner, [&] { return function->default_domain_map(); });
        value.memSyncDomainMap = {map[0], map[1]}; break;
    }
    case CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN: value.memSyncDomain = options.domain.value_or(CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT); break;
    case CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE: {
        auto owner = node.owner.lock(); require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "kernel graph context was destroyed");
        auto function = function_lookup(node.function, owner).native;
        context_configuration(owner, [&] { (void)function->attribute(CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE); });
        value.sharedMemoryMode = options.shared_memory_mode.value_or(CU_SHARED_MEMORY_MODE_DEFAULT); break;
    }
    default: break;
    }
    return value;
}
void graph_priority_copy(CUgraphNode destination, CUgraphNode source) {
    auto& from = graph_priority_node(source); auto& to = graph_priority_node(destination);
    require(!from.device_node && !to.device_node, CUDA_ERROR_INVALID_VALUE, "device-updatable node attributes cannot be copied");
    auto owner = from.owner.lock();
    require(owner && !owner->retiring && to.owner.lock() == owner, CUDA_ERROR_INVALID_CONTEXT, "kernel graph attribute contexts differ or were destroyed");
    auto shape = to.shape; shape.options = from.shape.options;
    access_policy_resources(owner, shape.options);
    auto function = function_lookup(to.function, owner).native;
    context_configuration(owner, [&] { function->validate_configuration(shape); });
    if (to.native_node || from.native_node) {
        auto native = graph_native_materialize(global().graph_nodes.at(destination), owner);
        graph_native_materialize(global().graph_nodes.at(source), owner);
        native->copy_attributes(to.native_node, from.native_node);
    }
    to.priority = from.priority; to.shape = std::move(shape);
}
