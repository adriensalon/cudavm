// Descriptors own their arrays; only queued actions retain imported resources.
template<class Params> graph_node graph_semaphores(const CUexternalSemaphore* handles, const Params* params, unsigned count, const std::shared_ptr<context>& owner) {
    semaphore_entries(handles, params, count, owner);
    graph_node node; node.owner = owner;
    if (count) node.semaphores.assign(handles, handles + count);
    if constexpr (std::is_same_v<Params, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>) {
        node.type = CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL;
        if (count) node.signals.assign(params, params + count);
    } else {
        node.type = CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT;
        if (count) node.waits.assign(params, params + count);
    }
    return node;
}
std::function<void()> graph_prepare_semaphores(const graph_node& node) {
    auto owner = node.owner.lock();
    bool signal = node.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL;
    auto count = static_cast<unsigned>(node.semaphores.size());
    auto entries = signal ? semaphore_entries(node.semaphores.data(), node.signals.data(), count, owner)
                          : semaphore_entries(node.semaphores.data(), node.waits.data(), count, owner);
    return [owner, entries = std::move(entries), signal] {
        std::lock_guard lock(owner->execution);
        for (auto& [semaphore, value] : entries)
            if (signal) semaphore->signal(value); else semaphore->wait(value);
        owner->native->synchronize();
    };
}
// Materialize executable snapshots as independently editable child graphs.
CUgraph graph_from_snapshot(const std::vector<graph_node>& nodes, const std::vector<bool>* enabled = nullptr, const user_references& objects = {}, bool inherit_priority = false, int priority = 0) {
    auto handle = graph_create(0);
    try {
        graph_lookup(handle).objects = objects;
        std::unordered_map<CUgraphNode, CUgraphNode> mapping;
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto node = nodes[i]; auto original = node.handle; auto dependencies = node.dependencies;
            auto edges = node.dependency_edges;
            node.dependencies.clear(); node.dependency_edges.clear(); node.child = nullptr;
            if (enabled) node.snapshot_disabled = !(*enabled)[i];
            if (inherit_priority && node.type == CU_GRAPH_NODE_TYPE_KERNEL) node.priority = priority;
            if (node.child_nodes) {
                node.child = graph_from_snapshot(*node.child_nodes, nullptr, node.child_objects, inherit_priority, priority);
                graph_lookup(node.child).parent = handle;
                node.child_nodes.reset();
                node.child_objects.clear();
            }
            std::vector<CUgraphNode> translated;
            for (auto dependency : dependencies) translated.push_back(mapping.at(dependency));
            auto child = node.child;
            CUgraphNode added{};
            try { added = graph_add(handle, translated.data(), translated.size(), std::move(node), edges.data()); }
            catch (...) { if (child) graph_destroy(child); throw; }
            mapping.emplace(original, added);
        }
        return handle;
    } catch (...) { graph_destroy(handle); throw; }
}
bool capture_graph_launch(const std::shared_ptr<context>& owner, CUstream stream, const graph_exec& value) {
    auto target = stream_lookup(stream, owner);
    auto child = graph_from_snapshot(value.nodes, &value.enabled, value.objects, !(value.flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY), target->priority);
    try {
        auto node = graph_child(child);
        if (!capture_node(owner, stream, std::move(node))) { graph_destroy(child); return false; }
        graph_lookup(child).parent = target->capture->graph; return true;
    } catch (...) { graph_destroy(child); throw; }
}
