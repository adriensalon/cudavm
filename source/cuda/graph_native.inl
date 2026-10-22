CUgraphNodeParams graph_node_parameters(graph_node&);
bool graph_has_device_nodes(const std::vector<graph_node>& nodes) {
    for (auto& node : nodes) {
        if (node.device_node) return true;
        if (node.child_nodes ? graph_has_device_nodes(*node.child_nodes) : node.child && graph_has_device_nodes(graph_lookup(node.child).nodes)) return true;
        for (size_t i = 0; i < node.conditional_graphs.size(); ++i)
            if (graph_has_device_nodes(node.conditional_nodes.empty() ? graph_lookup(node.conditional_graphs[i]).nodes : *node.conditional_nodes[i])) return true;
    }
    return false;
}
bool graph_has_conditionals(const std::vector<graph_node>& nodes) {
    for (auto& node : nodes) {
        if (node.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) return true;
        if (node.child_nodes ? graph_has_conditionals(*node.child_nodes) : node.child && graph_has_conditionals(graph_lookup(node.child).nodes)) return true;
    }
    return false;
}
CUgraphNodeParams graph_native_parameters(graph_node& node, const std::shared_ptr<context>& owner) {
    require(node.owner.expired() || node.owner.lock() == owner, CUDA_ERROR_INVALID_CONTEXT, "native graph nodes must share their execution context");
    auto params = graph_node_parameters(node);
    switch (node.type) {
    case CU_GRAPH_NODE_TYPE_KERNEL:
        params.kernel.func = function_lookup(node.function, owner).native->cuda_function();
        require(params.kernel.func, CUDA_ERROR_NOT_SUPPORTED, "graph kernel has no native CUDA function");
        params.kernel.kern = nullptr; break;
    case CU_GRAPH_NODE_TYPE_MEMCPY:
        require(!node.peer_copy, CUDA_ERROR_NOT_SUPPORTED, "native stateful graphs do not support peer capture nodes");
        if (params.memcpy.copyParams.srcMemoryType == CU_MEMORYTYPE_ARRAY)
            params.memcpy.copyParams.srcArray = array_lookup(node.copy.srcArray, owner)->native->cuda_array();
        if (params.memcpy.copyParams.dstMemoryType == CU_MEMORYTYPE_ARRAY)
            params.memcpy.copyParams.dstArray = array_lookup(node.copy.dstArray, owner)->native->cuda_array();
        break;
    case CU_GRAPH_NODE_TYPE_MEMSET:
    case CU_GRAPH_NODE_TYPE_EMPTY:
    case CU_GRAPH_NODE_TYPE_CONDITIONAL: break;
    case CU_GRAPH_NODE_TYPE_GRAPH:
        if (!node.native_child) node.native_child = graph_native_materialize(node.child, owner);
        params.graph.graph = node.native_child->handle();
        params.graph.ownership = CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE; break;
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "node has no stateful native graph representation");
    }
    return params;
}
void graph_native_attributes(const std::shared_ptr<native_graph_definition>& native, graph_node& node) {
    if (node.type != CU_GRAPH_NODE_TYPE_KERNEL) return;
    // Use the same native launch attributes as ordinary enqueue/capture.
    auto owner = node.owner.lock(); auto options = node.shape.options;
    // Event recording attributes are launches-only and have separate graph edges.
    require(!options.programmatic_event && !options.completion_event, CUDA_ERROR_NOT_SUPPORTED, "native stateful graph launch events are unavailable");
    auto set = [&](CUkernelNodeAttrID id, CUkernelNodeAttrValue value) { native->attribute(node.native_node, id, &value); };
    CUkernelNodeAttrValue value{}; value.priority = node.priority; set(CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, value);
    if (options.cooperative) { value = {}; value.cooperative = 1; set(CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, value); }
    if (options.access_policy) { value = {}; value.accessPolicyWindow = access_policy_to_native(*options.access_policy); set(CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, value); }
    if (options.cluster) { auto dim = *options.cluster; value = {}; value.clusterDim = {dim[0],dim[1],dim[2]}; set(CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, value); }
    if (options.cluster_policy) { value = {}; value.clusterSchedulingPolicyPreference = *options.cluster_policy; set(CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, value); }
    if (options.domain_map) { auto map = *options.domain_map; value = {}; value.memSyncDomainMap = {map[0],map[1]}; set(CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, value); }
    if (options.domain) { value = {}; value.memSyncDomain = *options.domain; set(CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, value); }
    if (options.shared_memory_mode) { value = {}; value.sharedMemoryMode = *options.shared_memory_mode; set(CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE, value); }
    if (options.preferred_cluster) { auto dim = *options.preferred_cluster; value = {}; value.preferredClusterDim = {dim[0],dim[1],dim[2]}; set(CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, value); }
    if (options.carveout) { value = {}; value.sharedMemCarveout = *options.carveout; set(CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, value); }
    if (options.nvlink) { value = {}; value.nvlinkUtilCentricScheduling = *options.nvlink; set(CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING, value); }
    if (options.portable_cluster) { value = {}; value.portableClusterSizeMode = *options.portable_cluster; set(CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE, value); }
}
void graph_native_bind_child(CUgraph graph, const std::shared_ptr<native_graph_definition>& native) {
    auto& state = graph_lookup(graph);
    for (auto& node : state.nodes) {
        node.native_node = native->find_in_clone(node.native_node);
        if (node.device_node) node.device_node = native->attribute(node.native_node, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE).deviceUpdatableKernelNode.devNode;
        if (node.child) { node.native_child = native->child(node.native_node); graph_native_bind_child(node.child, node.native_child); }
    }
    state.native = native;
}
void graph_native_bind_clone(CUgraph graph, CUgraph original, const std::shared_ptr<native_graph_definition>& native) {
    auto& state = graph_lookup(graph);
    for (auto& source : graph_lookup(original).nodes) {
        auto& node = graph_node_lookup(state.originals.at(source.handle));
        node.native_node = native->find_in_clone(source.native_node);
        if (node.device_node) node.device_node = native->attribute(node.native_node, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE).deviceUpdatableKernelNode.devNode;
        if (node.child) { node.native_child = native->child(node.native_node); graph_native_bind_clone(node.child, source.child, node.native_child); }
    }
    state.native = native;
}
void graph_native_add(CUgraph graph, graph_node& node) {
    auto native = graph_lookup(graph).native;
    if (!native) return;
    auto owner = graph_lookup(graph).native_owner.lock();
    require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "native graph context was destroyed");
    auto params = graph_native_parameters(node, owner);
    std::vector<CUgraphNode> dependencies;
    for (auto dependency : node.dependencies) dependencies.push_back(graph_node_lookup(dependency).native_node);
    auto handle = native->add(params, dependencies, node.dependency_edges);
    node.native_node = handle;
    try { graph_native_attributes(native, node); }
    catch (...) { native->remove(handle); node.native_node = nullptr; throw; }
    if (node.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) node.conditional.phGraph_out = params.conditional.phGraph_out;
    if (node.child) { node.native_child = native->child(handle); graph_native_bind_child(node.child, node.native_child); }
}
std::shared_ptr<native_graph_definition> graph_native_materialize(CUgraph graph, const std::shared_ptr<context>& owner) {
    auto& state = graph_lookup(graph);
    if (state.native) {
        require(state.native_owner.lock() == owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "native graph context differs or was destroyed");
        return state.native;
    }
    auto native = owner->native->create_graph();
    auto nodes = graph_order(state.nodes);
    std::unordered_map<CUgraphNode, CUgraphNode> handles;
    for (auto& node : nodes) {
        auto params = graph_native_parameters(node, owner);
        std::vector<CUgraphNode> dependencies;
        for (auto dependency : node.dependencies) dependencies.push_back(handles.at(dependency));
        node.native_node = native->add(params, dependencies, node.dependency_edges);
        handles.emplace(node.handle, node.native_node); graph_native_attributes(native, node);
    }
    for (auto& node : state.nodes) {
        node.native_node = handles.at(node.handle);
        if (node.child) { node.native_child = native->child(node.native_node); graph_native_bind_child(node.child, node.native_child); }
    }
    state.native = native; state.native_owner = owner;
    return native;
}
void graph_native_replace(const graph_node& existing, graph_node& replacement) {
    replacement.device_node = existing.device_node;
    replacement.native_node = existing.native_node;
    if (!existing.native_node) return;
    auto graph = global().graph_nodes.at(existing.handle); auto& state = graph_lookup(graph);
    auto params = graph_native_parameters(replacement, state.native_owner.lock());
    state.native->set_params(existing.native_node, params);
    if (replacement.child) {
        replacement.native_child = state.native->child(existing.native_node);
        graph_native_bind_child(replacement.child, replacement.native_child);
    }
}
void graph_native_instantiate(graph_exec& value, CUgraph graph) {
    if (!graph_has_device_nodes(value.nodes) && !graph_has_conditionals(value.nodes)) return;
    auto native = graph_native_materialize(graph, value.owner.lock());
    value.nodes = graph_snapshot(graph); value.native = native->instantiate(value.flags);
    std::function<void(CUgraph)> collect = [&](CUgraph g) {
        auto& state = graph_lookup(g); value.conditions.insert(value.conditions.end(), state.conditions.begin(), state.conditions.end());
        for (auto& node : state.nodes) { if (node.child) collect(node.child); for (auto body : node.conditional_graphs) collect(body); }
    };
    collect(graph);
    for (size_t i = 0; i < value.nodes.size(); ++i) if (!value.enabled[i]) value.native->set_enabled(value.nodes[i].native_node, false);
}
void graph_native_update(graph_exec& value, CUgraph graph, std::vector<graph_node>& nodes) {
    if (!value.native && !graph_has_conditionals(nodes)) return;
    std::vector<graph_state::condition> conditions;
    std::function<void(CUgraph)> collect = [&](CUgraph g) {
        auto& state = graph_lookup(g); conditions.insert(conditions.end(), state.conditions.begin(), state.conditions.end());
        for (auto& node : state.nodes) { if (node.child) collect(node.child); for (auto body : node.conditional_graphs) collect(body); }
    };
    collect(graph);
    require(value.conditions.size() == conditions.size(), CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, "conditional handle creation count changed");
    std::unordered_map<CUgraphConditionalHandle, size_t> old_handles, new_handles;
    for (size_t i = 0; i < conditions.size(); ++i) { old_handles.emplace(value.conditions[i].handle, i); new_handles.emplace(conditions[i].handle, i); }
    std::function<void(const std::vector<graph_node>&, const std::vector<graph_node>&)> validate = [&](const auto& a, const auto& b) {
        for (size_t i = 0; i < a.size(); ++i) if (a[i].type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
            require(old_handles.at(a[i].conditional.handle) == new_handles.at(b[i].conditional.handle), CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, "conditional handle assignment order changed");
            for (size_t j = 0; j < a[i].conditional_nodes.size(); ++j) validate(*a[i].conditional_nodes[j], *b[i].conditional_nodes[j]);
        }
    };
    validate(value.nodes, nodes);
    // A new executable snapshot keeps already queued launches immutable. Device
    // updatable graphs are excluded by graph_update before reaching this path.
    auto native = graph_native_materialize(graph, value.owner.lock())->instantiate(value.flags);
    for (size_t i = 0; i < nodes.size(); ++i) if (!value.enabled[i]) native->set_enabled(nodes[i].native_node, false);
    value.native = std::move(native); value.conditions.swap(conditions); value.native_updates.clear(); value.native_enables.clear();
}
bool graph_native_launch(graph_exec& value, const std::shared_ptr<context>& owner, const std::shared_ptr<stream_state>& target, CUstream stream, bool upload) {
    if (!value.native) return false;
    auto operations = std::make_shared<std::vector<graph_operation>>();
    std::vector<std::shared_ptr<host_entry>> hosts;
    std::function<void(const std::vector<graph_node>&)> retain = [&](const auto& nodes) {
        for (auto& node : nodes) {
            if (node.child_nodes) retain(*node.child_nodes);
            for (auto& body : node.conditional_nodes) retain(*body);
            operations->push_back(graph_prepare(node, bool(value.flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY), false));
            for (auto& host : operations->back().hosts) if (std::find(hosts.begin(), hosts.end(), host) == hosts.end()) hosts.push_back(host);
        }
    };
    retain(value.nodes);
    struct change { graph_node node; CUgraphNodeParams params; };
    auto changes = std::make_shared<std::vector<change>>(); changes->reserve(value.native_updates.size());
    for (size_t i = 0; i < value.native_updates.size(); ++i) {
        auto& node = value.native_updates[i];
        if (std::any_of(value.native_updates.begin() + i + 1, value.native_updates.end(), [&](auto& later) { return later.native_node == node.native_node; })) continue;
        changes->push_back({node, {}}); auto& update = changes->back(); update.params = graph_native_parameters(update.node, owner);
    }
    for (auto& host : hosts) host->work.reserve(host->work.size() + 1);
    auto queue = stream_native(owner, target); auto native = value.native;
    auto done = submit_stream(owner, stream, [] {}, value.tail ? std::vector<completion>{value.tail} : std::vector<completion>{}, 0,
        [owner, queue, native, changes, operations, enables = value.native_enables, nodes = value.nodes, instance = value.memory_instance, objects = value.objects, upload] {
            std::lock_guard lock(owner->execution);
            for (auto& update : *changes) native->set_params(update.node.native_node, update.params);
            for (auto [node, enabled] : enables) native->set_enabled(node, enabled);
            return upload ? native->upload(queue.get()) : native->enqueue(queue.get());
        }, false, {}, queue);
    value.tail = done; value.native_updates.clear(); value.native_enables.clear();
    for (auto& host : hosts) host->work.push_back(done);
    return true;
}
CUgraphConditionalHandle graph_condition_create(CUgraph graph, const std::shared_ptr<context>& owner, unsigned initial, unsigned flags) {
    require(!(flags & ~CU_GRAPH_COND_ASSIGN_DEFAULT), CUDA_ERROR_INVALID_VALUE, "invalid conditional handle flags");
    auto& state = graph_lookup(graph); state.conditions.reserve(state.conditions.size() + 1);
    bool materialized = bool(state.native); CUgraphConditionalHandle result{};
    try { auto native = graph_native_materialize(graph, owner); result = native->condition(initial, flags); }
    catch (...) { if (!materialized) { state.native.reset(); state.native_owner.reset(); for (auto& node : state.nodes) node.native_node = nullptr; } throw; }
    state.conditions.push_back({result, initial, flags}); return result;
}
CUgraphNode graph_conditional_add(CUgraph graph, const CUgraphNode* dependencies, const CUgraphEdgeData* edges, size_t count, CUgraphNodeParams& params) {
    auto p = params.conditional; auto owner = p.ctx ? lookup(p.ctx) : current();
    require((p.type == CU_GRAPH_COND_TYPE_IF && (p.size == 1 || p.size == 2)) ||
        (p.type == CU_GRAPH_COND_TYPE_WHILE && p.size == 1) || (p.type == CU_GRAPH_COND_TYPE_SWITCH && p.size), CUDA_ERROR_INVALID_VALUE, "invalid conditional node type or body count");
    require(!p.phGraph_out, CUDA_ERROR_INVALID_VALUE, "conditional body array is an output");
    bool found = false;
    for (auto ancestor = graph; ancestor; ancestor = graph_lookup(ancestor).parent)
        for (auto condition : graph_lookup(ancestor).conditions) found |= condition.handle == p.handle;
    require(found, CUDA_ERROR_INVALID_VALUE, "conditional handle belongs to another graph tree");
    graph_dependencies_validate(graph, dependencies, edges, count, CU_GRAPH_NODE_TYPE_CONDITIONAL);
    graph_native_materialize(graph, owner);
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_CONDITIONAL; node.owner = owner; node.conditional = p;
    // Body handles and the output array have stable node-owned storage.
    node.conditional_graphs.reserve(p.size);
    auto handle = graph_add(graph, dependencies, count, std::move(node), edges);
    try {
        for (unsigned i = 0; i < p.size; ++i) {
            auto raw = graph_node_lookup(handle).conditional.phGraph_out[i];
            auto body = graph_create(0);
            try {
                auto& state = graph_lookup(body); state.parent = graph; state.conditional_body = true; state.native_owner = owner;
                state.native = graph_lookup(graph).native->body(raw);
                graph_node_lookup(handle).conditional_graphs.push_back(body);
            } catch (...) { graph_destroy(body); throw; }
        }
    } catch (...) { graph_destroy_node(handle); throw; }
    auto& added = graph_node_lookup(handle); added.conditional.phGraph_out = added.conditional_graphs.data();
    params.conditional.phGraph_out = added.conditional.phGraph_out; return handle;
}
