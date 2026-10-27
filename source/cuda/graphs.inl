graph_state& graph_lookup(CUgraph handle) {
    auto it = global().graphs.find(handle);
    require(it != global().graphs.end(), CUDA_ERROR_INVALID_VALUE, "invalid or destroyed graph");
    return it->second;
}
graph_node& graph_node_lookup(CUgraphNode handle) {
    auto it = global().graph_nodes.find(handle);
    require(it != global().graph_nodes.end(), CUDA_ERROR_INVALID_VALUE, "invalid or destroyed graph node");
    auto& nodes = graph_lookup(it->second).nodes;
    return *std::find_if(nodes.begin(), nodes.end(), [&](auto& node) { return node.handle == handle; });
}
graph_exec& graph_exec_lookup(CUgraphExec handle) {
    auto it = global().graph_execs.find(handle);
    require(it != global().graph_execs.end(), CUDA_ERROR_INVALID_VALUE, "invalid or destroyed executable graph");
    auto owner = it->second.owner.lock();
    require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "executable graph context was destroyed");
    return it->second;
}
size_t graph_exec_index(graph_exec& value, CUgraphNode handle) {
    graph_node_lookup(handle); // Executable updates require a live original node.
    auto it = std::find_if(value.nodes.begin(), value.nodes.end(), [&](auto& node) { return node.handle == handle; });
    require(it != value.nodes.end(), CUDA_ERROR_INVALID_VALUE, "node is not from this executable graph");
    return static_cast<size_t>(it - value.nodes.begin());
}
std::vector<graph_node> graph_order(const std::vector<graph_node>& nodes, CUDA_GRAPH_INSTANTIATE_PARAMS* info = nullptr) {
    std::vector<graph_node> result;
    result.reserve(nodes.size());
    std::unordered_map<CUgraphNode, size_t> indices;
    for (size_t i = 0; i < nodes.size(); ++i) indices.emplace(nodes[i].handle, i);
    std::vector<size_t> counts(nodes.size());
    std::vector<std::vector<size_t>> children(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        counts[i] = nodes[i].dependencies.size();
        for (auto dependency : nodes[i].dependencies) {
            auto it = indices.find(dependency);
            require(it != indices.end(), CUDA_ERROR_INVALID_VALUE, "graph dependency is missing");
            children[it->second].push_back(i);
        }
    }
    std::vector<size_t> ready;
    for (size_t i = 0; i < nodes.size(); ++i) if (!counts[i]) ready.push_back(i);
    for (size_t position = 0; position < ready.size(); ++position) {
        auto i = ready[position]; result.push_back(nodes[i]);
        for (auto child : children[i]) if (!--counts[child]) ready.push_back(child);
    }
    if (result.size() != nodes.size() && info) {
        info->result_out = CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE;
        for (size_t i = 0; i < nodes.size(); ++i) if (counts[i]) { info->hErrNode_out = nodes[i].handle; break; }
    }
    require(result.size() == nodes.size(), CUDA_ERROR_INVALID_VALUE, "graph contains a dependency cycle");
    return result;
}
template<class Handle> Handle graph_handle(std::uintptr_t& next) {
    auto value = reinterpret_cast<Handle>(next); next += 0x100; return value;
}
unsigned graph_id() {
    require(global().next_graph_id <= std::numeric_limits<unsigned>::max(), CUDA_ERROR_OUT_OF_MEMORY, "graph identifiers exhausted");
    return static_cast<unsigned>(global().next_graph_id++);
}
CUgraph graph_create(unsigned flags) {
    require(!flags, CUDA_ERROR_INVALID_VALUE, "graph creation flags must be zero");
    auto handle = graph_handle<CUgraph>(global().next_graph);
    graph_state value; value.id = graph_id();
    global().graphs.emplace(handle, std::move(value)); return handle;
}
void graph_capture_guard(CUgraph handle) {
    for (auto& [context_handle, owner] : global().contexts) for (auto& item : owner->streams) if (auto stream = item.lock()) if (stream->capture) {
        for (auto captured = stream->capture->graph; captured; captured = graph_lookup(captured).parent)
            require(captured != handle, CUDA_ERROR_ILLEGAL_STATE, "graph ownership tree contains an active capture");
    }
}
void graph_destroy(CUgraph handle) {
    // Release the owned tree without allocating after an ownership commit.
    auto value = std::move(graph_lookup(handle));
    global().graphs.erase(handle);
    for (auto& node : value.nodes) {
        if (node.child) graph_destroy(node.child);
        for (auto body : node.conditional_graphs) graph_destroy(body);
        global().graph_nodes.erase(node.handle);
    }
}
CUgraphNode graph_add(CUgraph handle, const CUgraphNode* dependencies, size_t count, graph_node node, const CUgraphEdgeData* data) {
    auto& value = graph_lookup(handle);
    require(value.next_local_id <= std::numeric_limits<unsigned>::max(), CUDA_ERROR_OUT_OF_MEMORY, "local node identifiers exhausted");
    if (node.type == CU_GRAPH_NODE_TYPE_KERNEL && node.kernel_name.empty()) {
        auto found = global().functions.find(node.function);
        if (found != global().functions.end()) node.kernel_name = found->second.name;
    }
    if (auto owner = node.owner.lock()) node.parameter_context = owner->handle;
    graph_dependencies_validate(handle, dependencies, data, count, node.type);
    if (count) node.dependencies.assign(dependencies, dependencies + count);
    node.dependency_edges.resize(count);
    if (data) std::copy_n(data, count, node.dependency_edges.begin());
    node.handle = graph_handle<CUgraphNode>(global().next_graph_node);
    node.local_id = static_cast<unsigned>(value.next_local_id);
    auto output = node.handle;
    value.nodes.push_back(std::move(node));
    try { global().graph_nodes.emplace(output, handle); graph_native_add(handle, value.nodes.back()); }
    catch (...) { global().graph_nodes.erase(output); value.nodes.pop_back(); throw; }
    ++value.next_local_id;
    return output;
}
graph_node graph_host(const CUDA_HOST_NODE_PARAMS& p) {
    require(p.fn, CUDA_ERROR_INVALID_VALUE, "null graph host function");
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_HOST; node.host = p; return node;
}
graph_node graph_copy(const CUDA_MEMCPY3D& p, const std::shared_ptr<context>& owner) {
    require(p.WidthInBytes && p.Height && p.Depth, CUDA_ERROR_INVALID_VALUE, "graph copy dimensions must be nonzero");
    require(!p.srcLOD && !p.dstLOD && !p.reserved0 && !p.reserved1, CUDA_ERROR_INVALID_VALUE, "invalid graph copy LOD or reserved fields");
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_MEMCPY; node.owner = owner; node.copy = p; return node;
}
graph_node graph_fill(const CUDA_MEMSET_NODE_PARAMS& p, const std::shared_ptr<context>& owner) {
    require(p.width && p.height && (p.elementSize == 1 || p.elementSize == 2 || p.elementSize == 4), CUDA_ERROR_INVALID_VALUE, "invalid graph fill dimensions or element size");
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_MEMSET; node.owner = owner; node.fill = p; return node;
}
graph_node graph_kernel(const CUDA_KERNEL_NODE_PARAMS& p, const std::shared_ptr<context>& owner, const launch_options& options = {}) {
    graph_memory_scope memory_scope;
    auto context = !p.func && p.ctx ? lookup(p.ctx) : owner;
    require(p.func || p.kern, CUDA_ERROR_INVALID_VALUE, "missing graph kernel");
    auto function = resolve_library_function(p.func ? p.func : reinterpret_cast<CUfunction>(p.kern), context);
    launch_shape shape{{p.gridDimX, p.gridDimY, p.gridDimZ}, {p.blockDimX, p.blockDimY, p.blockDimZ}, p.sharedMemBytes};
    shape.options = options;
    auto command = prepare_launch(function, shape, p.kernelParams, p.extra, context);
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_KERNEL; node.owner = context;
    node.function = function; node.library_kernel = p.func ? nullptr : p.kern; node.shape = shape; node.arguments = std::move(command->arguments);
    node.kernel_name = global().functions.at(function).name;
    node.kernel_layout = std::move(command->parameters); node.kernel_extra = p.extra != nullptr; return node;
}
struct graph_operation {
    std::function<void()> action;
    std::function<device_work(native_queue*)> start;
    std::vector<std::shared_ptr<host_entry>> hosts;
    std::vector<std::shared_ptr<context>> contexts;
    bool native_compatible = false;
    native_graph_key native_key;
};
graph_operation graph_prepare(const graph_node& node, bool use_node_priority = false, bool use_snapshot_enabled = true) {
    graph_memory_scope memory_scope;
    graph_operation result;
    if (use_snapshot_enabled && node.snapshot_disabled) { result.action = [] {}; return result; }
    for (auto& item : node.host_regions) {
        auto host = item.lock(); require(host && !host->retiring, CUDA_ERROR_INVALID_VALUE, "graph host storage was released");
    }
    if (node.type == CU_GRAPH_NODE_TYPE_GRAPH) {
        auto nodes = node.child_nodes ? *node.child_nodes : graph_order(graph_lookup(node.child).nodes);
        for (auto& child : nodes) {
            auto operation = graph_prepare(child, use_node_priority);
            result.hosts.insert(result.hosts.end(), operation.hosts.begin(), operation.hosts.end());
            result.contexts.insert(result.contexts.end(), operation.contexts.begin(), operation.contexts.end());
        }
        result.action = [] {}; return result; // Replay expands the child DAG.
    }
    if (node.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
        for (size_t i = 0; i < node.conditional_graphs.size(); ++i) {
            const auto nodes = node.conditional_nodes.empty() ? graph_order(graph_lookup(node.conditional_graphs[i]).nodes) : *node.conditional_nodes[i];
            for (auto& child : nodes) {
                auto operation = graph_prepare(child, use_node_priority);
                result.hosts.insert(result.hosts.end(), operation.hosts.begin(), operation.hosts.end());
                result.contexts.insert(result.contexts.end(), operation.contexts.begin(), operation.contexts.end());
            }
        }
        result.action = [] {}; return result;
    }
    if (node.type == CU_GRAPH_NODE_TYPE_EVENT_RECORD || node.type == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
        auto& event = event_lookup(node.event); auto owner = node.owner.lock();
        require(owner && !owner->retiring && event.owner.lock() == owner, CUDA_ERROR_INVALID_HANDLE, "graph event context was destroyed or changed");
        result.contexts.push_back(owner);
        result.action = [] {}; return result;
    }
    if (node.type == CU_GRAPH_NODE_TYPE_EMPTY) { result.action = [] {}; return result; }
    if (node.type == CU_GRAPH_NODE_TYPE_HOST) {
        if (node.host_context) {
            auto owner = node.owner.lock();
            require(owner && !owner->retiring && global().contexts.contains(node.host_context), CUDA_ERROR_INVALID_CONTEXT, "graph host context was destroyed");
            result.contexts.push_back(owner);
        }
        auto p = node.host;
        result.action = [p] {
            struct guard { guard() { executing_host_function = true; } ~guard() { executing_host_function = false; } } scope;
            p.fn(p.userData);
        };
        return result;
    }
    auto owner = node.owner.lock();
    require(owner && !owner->retiring && global().contexts.contains(owner->handle), CUDA_ERROR_INVALID_CONTEXT, "graph node context was destroyed");
    result.contexts.push_back(owner);
    if (node.memory) {
        auto memory = node.memory;
        require(global().allocations.contains(memory->address), CUDA_ERROR_INVALID_VALUE, "graph allocation context was released");
        bool allocate = node.type == CU_GRAPH_NODE_TYPE_MEM_ALLOC;
        result.action = [memory, allocate] {
            require(memory->live.load() != allocate, CUDA_ERROR_INVALID_VALUE, "graph memory transition is not valid");
            memory->live = allocate;
        };
    } else if (node.type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
        auto task = prepare_memops(node, result.hosts);
        result.start = [task](native_queue*) -> device_work {
            std::lock_guard lock(task->owner->execution);
            task->queue = task->owner->native->create_queue(0); return task;
        };
    } else if (node.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL || node.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT) {
        result.action = graph_prepare_semaphores(node);
    } else if (node.type == CU_GRAPH_NODE_TYPE_MEMCPY) {
        for (auto& item : node.host_regions) {
            auto host = item.lock();
            require(host && !host->retiring, CUDA_ERROR_INVALID_VALUE, "graph host storage was released");
        }
        auto& p = node.copy;
        auto from_owner = node.peer_copy ? node.copy_source.lock() : owner;
        auto to_owner = node.peer_copy ? node.copy_destination.lock() : owner;
        require(from_owner && to_owner && !from_owner->retiring && !to_owner->retiring &&
                global().contexts.contains(from_owner->handle) && global().contexts.contains(to_owner->handle),
                CUDA_ERROR_INVALID_CONTEXT, "graph copy endpoint context was destroyed");
        result.contexts.push_back(from_owner); result.contexts.push_back(to_owner);
        auto from = memory_endpoint(p.srcMemoryType, p.srcHost, p.srcDevice, p.srcPitch, p.srcHeight, p.srcXInBytes, p.srcY, p.srcZ, p.WidthInBytes, p.Height, p.Depth, from_owner, true, p.srcArray);
        auto to = memory_endpoint(p.dstMemoryType, p.dstHost, p.dstDevice, p.dstPitch, p.dstHeight, p.dstXInBytes, p.dstY, p.dstZ, p.WidthInBytes, p.Height, p.Depth, to_owner, true, p.dstArray);
        memory_disjoint(to, from, p.WidthInBytes);
        if (to.host) result.hosts.push_back(to.host);
        if (from.host && from.host != to.host) result.hosts.push_back(from.host);
        result.action = [to, from, width = p.WidthInBytes] { execute_copy_boxes(to, from, width); };
        result.start = [to, from, owner, width = p.WidthInBytes](native_queue* queue) { return enqueue_copy_boxes(to, from, width, queue, owner); };
        result.native_compatible = to.owner == owner && from.owner == owner &&
            to.device_storage() && from.device_storage() && !to.array && !from.array;
        if (result.native_compatible && owner->native->kind() != cudavmBackendCuda)
            for (size_t i = 0; i < to.rows * to.depth; ++i)
                result.native_compatible &= owner->native->can_record_copy(*to.storage, to.allocation_offset + to.offset(i),
                    *from.storage, from.allocation_offset + from.offset(i), p.WidthInBytes);
        if (result.native_compatible) result.native_key = {owner->id, owner->native->graph_generation(),
            reinterpret_cast<uintptr_t>(to.storage.get()), reinterpret_cast<uintptr_t>(from.storage.get())};
    } else if (node.type == CU_GRAPH_NODE_TYPE_MEMSET) {
        auto& p = node.fill; auto bytes = memory_mul(p.width, p.elementSize);
        auto pitch = p.height == 1 ? bytes : p.pitch;
        require(p.dst % p.elementSize == 0 && pitch % p.elementSize == 0, CUDA_ERROR_INVALID_VALUE, "invalid graph fill alignment");
        auto to = memory_endpoint(CU_MEMORYTYPE_DEVICE, nullptr, p.dst, pitch, 0, 0, 0, 0, bytes, p.height, 1, owner);
        if (to.host) result.hosts.push_back(to.host);
        result.action = [to, p, bytes] {
            if (p.elementSize == 1) execute_fill_box(to, static_cast<unsigned char>(p.value), bytes);
            else if (p.elementSize == 2) execute_fill_box(to, static_cast<unsigned short>(p.value), bytes);
            else execute_fill_box(to, p.value, bytes);
        };
        result.start = [to, p, bytes](native_queue* queue) {
            if (p.elementSize == 1) return enqueue_fill_box(to, static_cast<unsigned char>(p.value), bytes, queue);
            if (p.elementSize == 2) return enqueue_fill_box(to, static_cast<unsigned short>(p.value), bytes, queue);
            return enqueue_fill_box(to, p.value, bytes, queue);
        };
        result.native_compatible = to.device_storage();
        if (result.native_compatible && owner->native->kind() != cudavmBackendCuda) {
            result.native_compatible = !to.storage->pool_session || to.storage->pool_session.get() == owner->native.get();
            for (size_t i = 0; i < to.rows * to.depth; ++i)
                result.native_compatible &= owner->native->can_record_fill(*to.storage, to.allocation_offset + to.offset(i), bytes);
        }
        if (result.native_compatible) result.native_key = {owner->id, owner->native->graph_generation(), reinterpret_cast<uintptr_t>(to.storage.get())};
    } else {
        auto bytes = node.arguments.size();
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, const_cast<std::byte*>(node.arguments.data()), CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
        auto shape = node.shape;
        shape.options.priority = use_node_priority ? std::optional<int>(stream_priority(node.priority, owner)) : std::nullopt;
        auto command = prepare_launch(node.function, shape, nullptr, extra, owner);
        result.hosts = command->hosts;
        result.action = [command] { command->execute(); };
        result.start = [command](native_queue* queue) { return command->enqueue(queue); };
        result.native_compatible = owner->native->kind() == cudavmBackendCuda || command->function->can_record_launch(command->bindings);
        if (result.native_compatible) {
            result.native_key = {owner->id, owner->native->graph_generation(), reinterpret_cast<uintptr_t>(command->function.get()), command->function->graph_generation()};
            for (auto& binding : command->bindings) {
                result.native_key.push_back(binding.address);
                result.native_key.push_back(reinterpret_cast<uintptr_t>(binding.native));
                result.native_key.push_back(binding.surface_descriptor);
            }
        }
    }
    return result;
}
void graph_validate(graph_node& node) {
    auto operation = graph_prepare(node);
    node.host_regions.clear();
    for (auto& host : operation.hosts) node.host_regions.push_back(host);
}
void graph_replace(graph_node& existing, graph_node replacement) {
    require(existing.type == replacement.type, CUDA_ERROR_INVALID_VALUE, "wrong graph node type");
    if (existing.type == CU_GRAPH_NODE_TYPE_GRAPH)
        require(!graph_has_memory(graph_lookup(existing.child).nodes) && !graph_has_memory(graph_lookup(replacement.child).nodes),
                CUDA_ERROR_NOT_SUPPORTED, "memory child graph replacement is unavailable");
    if (existing.type == CU_GRAPH_NODE_TYPE_KERNEL) {
        replacement.priority = existing.priority; replacement.shape.options = existing.shape.options;
    }
    if (existing.peer_copy && !replacement.peer_copy) {
        replacement.peer_copy = true; replacement.copy_source = existing.copy_source; replacement.copy_destination = existing.copy_destination;
    }
    graph_validate(replacement);
    replacement.handle = existing.handle; replacement.dependencies = existing.dependencies;
    replacement.dependency_edges = existing.dependency_edges;
    replacement.snapshot_disabled = existing.snapshot_disabled;
    replacement.local_id = existing.local_id;
    if (auto owner = replacement.owner.lock()) replacement.parameter_context = owner->handle;
    replacement.parameter_history = existing.parameter_history;
    graph_native_replace(existing, replacement);
    existing = std::move(replacement);
}
void graph_edit_dependencies(CUgraph handle, const CUgraphNode* from, const CUgraphNode* to, const CUgraphEdgeData* data, size_t count, bool adding) {
    auto& value = graph_lookup(handle);
    require(adding || !count || !graph_has_memory(value.nodes), CUDA_ERROR_INVALID_VALUE, "memory graph edges cannot be deleted");
    require(!count || (from && to), CUDA_ERROR_INVALID_VALUE, "null graph edge arrays");
    auto candidate = value.nodes;
    for (size_t i = 0; i < count; ++i) {
        auto source = std::find_if(candidate.begin(), candidate.end(), [&](auto& node) { return node.handle == from[i]; });
        auto destination = std::find_if(candidate.begin(), candidate.end(), [&](auto& node) { return node.handle == to[i]; });
        require(source != candidate.end() && destination != candidate.end() && source != destination, CUDA_ERROR_INVALID_VALUE, "invalid graph edge endpoints");
        auto metadata = data ? data[i] : CUgraphEdgeData{};
        graph_edge_validate(metadata, source->type, destination->type);
        auto& dependencies = destination->dependencies;
        auto& edges = destination->dependency_edges;
        size_t index = 0;
        while (index < dependencies.size() && (dependencies[index] != from[i] || !graph_edge_equal(edges[index], metadata))) ++index;
        require(adding ? index == dependencies.size() : index != dependencies.size(), CUDA_ERROR_INVALID_VALUE, "duplicate or missing graph edge");
        if (adding) { dependencies.push_back(from[i]); edges.push_back(metadata); }
        else { dependencies.erase(dependencies.begin() + index); edges.erase(edges.begin() + index); }
    }
    // CUDA permits cycles while editing, but rejects them on instantiation.
    if (value.native && count) {
        std::vector<CUgraphNode> sources, destinations; std::vector<CUgraphEdgeData> edges(count);
        for (size_t i = 0; i < count; ++i) { sources.push_back(graph_node_lookup(from[i]).native_node); destinations.push_back(graph_node_lookup(to[i]).native_node); if (data) edges[i] = data[i]; }
        value.native->edges(sources, destinations, edges, adding);
    }
    value.nodes.swap(candidate);
}
void graph_destroy_node(CUgraphNode handle) {
    auto& removing = graph_node_lookup(handle); auto child = removing.child; auto bodies = removing.conditional_graphs;
    require(!removing.device_node, CUDA_ERROR_INVALID_VALUE, "device-updatable kernel nodes cannot be removed");
    if (child) graph_capture_guard(child);
    auto graph = global().graph_nodes.at(handle); auto& value = graph_lookup(graph);
    require(!graph_has_memory(value.nodes), CUDA_ERROR_INVALID_VALUE, "memory graph nodes cannot be deleted");
    if (value.native) value.native->remove(removing.native_node);
    for (auto& node : value.nodes) for (size_t i = node.dependencies.size(); i-- > 0;) if (node.dependencies[i] == handle) {
        node.dependencies.erase(node.dependencies.begin() + i); node.dependency_edges.erase(node.dependency_edges.begin() + i);
    }
    std::erase_if(value.nodes, [&](auto& node) { return node.handle == handle; });
    global().graph_nodes.erase(handle);
    std::erase_if(value.originals, [&](auto& pair) { return pair.second == handle; });
    if (child) graph_destroy(child);
    for (auto body : bodies) graph_destroy(body);
}
CUgraph graph_clone(CUgraph original) {
    require(!graph_has_memory(graph_lookup(original).nodes), CUDA_ERROR_NOT_SUPPORTED, "memory graphs cannot be cloned");
    require(!graph_has_conditionals(graph_lookup(original).nodes), CUDA_ERROR_NOT_SUPPORTED, "conditional graphs cannot be cloned");
    auto source = graph_lookup(original);
    graph_state copy; copy.nodes = std::move(source.nodes); copy.objects = std::move(source.objects);
    if (source.native) {
        auto owner = source.native_owner.lock(); require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "native graph context was destroyed");
        copy.native = source.native->clone(); copy.native_owner = source.native_owner;
    }
    copy.id = graph_id(); copy.next_local_id = source.next_local_id;
    for (auto& node : copy.nodes) copy.originals.emplace(node.handle, graph_handle<CUgraphNode>(global().next_graph_node));
    for (auto& node : copy.nodes) {
        node.handle = copy.originals.at(node.handle);
        if (copy.native) {
            node.native_node = copy.native->find_in_clone(node.native_node);
            if (node.device_node) node.device_node = copy.native->attribute(node.native_node, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE).deviceUpdatableKernelNode.devNode;
        }
        for (auto& dependency : node.dependencies) dependency = copy.originals.at(dependency);
    }
    auto handle = graph_handle<CUgraph>(global().next_graph);
    global().graphs.emplace(handle, std::move(copy));
    // Nested graphs are independent editable clones owned by this new graph.
    for (auto& node : graph_lookup(handle).nodes) node.child = nullptr;
    try {
        for (size_t i = 0; i < graph_lookup(handle).nodes.size(); ++i) {
            auto child = graph_lookup(original).nodes[i].child;
            if (child) { auto cloned = graph_clone(child); graph_lookup(cloned).parent = handle; graph_lookup(handle).nodes[i].child = cloned; }
            global().graph_nodes.emplace(graph_lookup(handle).nodes[i].handle, handle);
        }
        if (graph_lookup(handle).native) graph_native_bind_clone(handle, original, graph_lookup(handle).native);
    }
    catch (...) { graph_destroy(handle); throw; }
    return handle;
}
template<class T> void graph_query_list(const std::vector<T>& values, T* output, size_t* count) {
    require(count, CUDA_ERROR_INVALID_VALUE, "null graph query count");
    if (!output) { *count = values.size(); return; }
    auto capacity = *count; auto written = std::min(capacity, values.size());
    std::copy_n(values.begin(), written, output);
    std::fill(output + written, output + capacity, T{});
    *count = written;
}
std::vector<CUgraphNode> graph_query_nodes(CUgraph handle, bool roots) {
    std::vector<CUgraphNode> result;
    for (auto& node : graph_lookup(handle).nodes) if (!roots || node.dependencies.empty()) result.push_back(node.handle);
    return result;
}
void graph_query_edges(CUgraph handle, CUgraphNode* from, CUgraphNode* to, CUgraphEdgeData* data, size_t* count) {
    require(count && ((!from && !to && !data) || (from && to)), CUDA_ERROR_INVALID_VALUE, "invalid graph edge outputs");
    std::vector<CUgraphNode> sources, targets;
    std::vector<CUgraphEdgeData> edges;
    for (auto& node : graph_lookup(handle).nodes) for (size_t i = 0; i < node.dependencies.size(); ++i) {
        sources.push_back(node.dependencies[i]); targets.push_back(node.handle); edges.push_back(node.dependency_edges[i]);
    }
    if (!from) { *count = sources.size(); return; }
    auto capacity = *count, written = std::min(capacity, sources.size());
    require(data || std::all_of(edges.begin(), edges.end(), graph_edge_default), CUDA_ERROR_LOSSY_QUERY, "graph query would discard dependency data");
    graph_query_list(sources, from, count); auto other_count = capacity; graph_query_list(targets, to, &other_count);
    if (data) { auto size = capacity; graph_query_list(edges, data, &size); }
    *count = written;
}
void graph_query_neighbors(CUgraphNode handle, CUgraphNode* output, CUgraphEdgeData* data, size_t* count, bool dependents) {
    auto& node = graph_node_lookup(handle);
    require(!data || output, CUDA_ERROR_INVALID_VALUE, "edge data requires graph node outputs");
    std::vector<CUgraphNode> values;
    std::vector<CUgraphEdgeData> edges;
    if (!dependents) { values = node.dependencies; edges = node.dependency_edges; }
    else for (auto& other : graph_lookup(global().graph_nodes.at(handle)).nodes)
        for (size_t i = 0; i < other.dependencies.size(); ++i) if (other.dependencies[i] == handle) {
            values.push_back(other.handle); edges.push_back(other.dependency_edges[i]);
        }
    auto capacity = count && output ? *count : 0;
    require(!output || data || std::all_of(edges.begin(), edges.end(), graph_edge_default), CUDA_ERROR_LOSSY_QUERY, "graph query would discard dependency data");
    graph_query_list(values, output, count);
    if (data) { auto size = capacity; graph_query_list(edges, data, &size); }
}
std::vector<graph_node> graph_snapshot(CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS* info = nullptr) {
    auto nodes = graph_order(graph_lookup(graph).nodes, info);
    for (auto& node : nodes) if (node.child) {
        node.child_nodes = std::make_shared<const std::vector<graph_node>>(graph_snapshot(node.child, info));
        node.child_objects = graph_lookup(node.child).objects;
    }
    for (auto& node : nodes) for (auto body : node.conditional_graphs)
        node.conditional_nodes.push_back(std::make_shared<const std::vector<graph_node>>(graph_snapshot(body, info)));
    return nodes;
}
graph_node graph_child(CUgraph graph) {
    require(!graph_lookup(graph).conditional_body && !graph_has_conditionals(graph_lookup(graph).nodes), CUDA_ERROR_NOT_SUPPORTED, "conditional graphs cannot be child graph inputs");
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_GRAPH; node.child = graph;
    return node;
}
CUgraphNode graph_add_child(CUgraph parent, const CUgraphNode* dependencies, size_t count, CUgraph child, const CUgraphEdgeData* data = nullptr) {
    graph_lookup(parent); graph_lookup(child);
    require(!graph_lookup(child).moved || graph_lookup(child).parent == parent, CUDA_ERROR_INVALID_VALUE, "moved graph cannot be added to another parent");
    auto clone = graph_clone(child);
    try {
        auto result = graph_add(parent, dependencies, count, graph_child(clone), data);
        graph_lookup(clone).parent = parent; return result;
    } catch (...) { graph_destroy(clone); throw; }
}
void graph_move_validate(CUgraph parent, CUgraph child) {
    graph_lookup(parent);
    require(!graph_lookup(child).parent, CUDA_ERROR_INVALID_VALUE, "graph is already owned by a parent");
    for (auto ancestor = parent; ancestor; ancestor = graph_lookup(ancestor).parent)
        require(ancestor != child, CUDA_ERROR_INVALID_VALUE, "child graph transfer would create an ownership cycle");
    graph_capture_guard(child);
    graph_memory_move(parent, child, false);
}
CUgraphNode graph_move_child(CUgraph parent, const CUgraphNode* dependencies, size_t count, CUgraph child, const CUgraphEdgeData* data = nullptr) {
    graph_move_validate(parent, child);
    auto result = graph_add(parent, dependencies, count, graph_child(child), data);
    auto& transferred = graph_lookup(child); transferred.parent = parent; transferred.moved = true;
    graph_memory_move(parent, child, true);
    return result;
}
graph_node graph_event(CUevent event, CUgraphNodeType type) {
    auto& value = event_lookup(event); graph_node node; node.type = type; node.event = event; node.owner = value.owner;
    return node;
}
CUgraphExec graph_instantiate(CUgraph graph, unsigned long long flags, CUDA_GRAPH_INSTANTIATE_PARAMS* info = nullptr) {
    require(!graph_lookup(graph).conditional_body, CUDA_ERROR_INVALID_VALUE, "conditional body graphs cannot be independently instantiated");
    require(!graph_lookup(graph).moved, CUDA_ERROR_INVALID_VALUE, "moved child graphs cannot be independently instantiated");
    require(!(flags & ~15ull), CUDA_ERROR_INVALID_VALUE, "unknown graph instantiation flags");
    require(!(flags & ~static_cast<unsigned long long>(CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY | CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)), CUDA_ERROR_NOT_SUPPORTED, "graph instantiation flag is not implemented");
    auto owner = current(); auto nodes = graph_snapshot(graph, info);
    bool memory_graph = graph_has_memory(nodes);
    // An executable also instantiates the owned child tree. Moving a child
    // retains its earlier leases, including already queued replay snapshots.
    std::vector<CUgraph> descriptions;
    std::function<void(CUgraph)> retain_tree = [&](CUgraph description) {
        auto& state = graph_lookup(description);
        std::erase_if(state.instances, [](auto& item) { return item.expired(); });
        require(!(memory_graph || graph_has_device_nodes(nodes)) || state.instances.empty(), CUDA_ERROR_NOT_SUPPORTED, "graph already has an executable");
        descriptions.push_back(description);
        for (auto& node : state.nodes) if (node.child) retain_tree(node.child);
        for (auto& node : state.nodes) for (auto body : node.conditional_graphs) retain_tree(body);
    };
    retain_tree(graph);
    graph_memory_validate(nodes);
    for (auto& node : nodes) graph_prepare(node);
    graph_exec value; value.id = graph_id(); value.source_id = graph_lookup(graph).id;
    value.owner = owner; value.nodes = std::move(nodes);
    for (auto& node : value.nodes) value.enabled.push_back(!node.snapshot_disabled);
    value.flags = flags;
    value.memory_instance = std::make_shared<graph_memory_instance>();
    value.memory_instance->executable_id = value.id;
    value.objects = graph_lookup(graph).objects;
    graph_native_instantiate(value, graph);
    auto handle = graph_handle<CUgraphExec>(global().next_graph_exec);
    auto instance = value.memory_instance;
    for (auto description : descriptions) {
        auto& instances = graph_lookup(description).instances;
        instances.reserve(instances.size() + 1);
    }
    global().graph_execs.emplace(handle, std::move(value));
    for (auto description : descriptions) graph_lookup(description).instances.push_back(instance);
    return handle;
}
void graph_exec_replace(CUgraphExec handle, CUgraphNode node, graph_node replacement) {
    auto& value = graph_exec_lookup(handle); auto index = graph_exec_index(value, node);
    auto& existing = value.nodes[index];
    if (value.native) value.native_updates.reserve(value.native_updates.size() + 1);
    require(existing.type == replacement.type, CUDA_ERROR_INVALID_VALUE, "wrong executable graph node type");
    require(existing.owner.lock() == replacement.owner.lock(), CUDA_ERROR_INVALID_VALUE, "executable graph node context cannot change");
    if (replacement.type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP)
        require(memops_compatible(existing, replacement), CUDA_ERROR_INVALID_VALUE, "executable memory operation layout or flags cannot change");
    if (replacement.type == CU_GRAPH_NODE_TYPE_MEMCPY) {
        require(existing.copy.Height == 1 && existing.copy.Depth == 1 && replacement.copy.Height == 1 && replacement.copy.Depth == 1,
                CUDA_ERROR_NOT_SUPPORTED, "executable copy updates require one-dimensional nodes");
        require(existing.copy.srcMemoryType == replacement.copy.srcMemoryType && existing.copy.dstMemoryType == replacement.copy.dstMemoryType,
                CUDA_ERROR_INVALID_VALUE, "executable copy endpoint memory types cannot change");
    }
    if (replacement.type == CU_GRAPH_NODE_TYPE_MEMSET) {
        require(existing.fill.elementSize == replacement.fill.elementSize, CUDA_ERROR_INVALID_VALUE, "executable fill element size cannot change");
        require(existing.fill.height == replacement.fill.height && (existing.fill.height == 1 ||
                (existing.fill.width == replacement.fill.width && existing.fill.pitch == replacement.fill.pitch)), CUDA_ERROR_INVALID_VALUE, "executable two-dimensional fill geometry cannot change");
    }
    if (replacement.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL || replacement.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT)
        require(existing.semaphores.size() == replacement.semaphores.size(), CUDA_ERROR_INVALID_VALUE, "executable semaphore operation count cannot change");
    if (graph_has_memory(value.nodes)) {
        auto candidate = value.nodes; candidate[index] = replacement;
        candidate[index].handle = existing.handle; candidate[index].dependencies = existing.dependencies;
        candidate[index].dependency_edges = existing.dependency_edges;
        graph_memory_validate(candidate);
    }
    // Executable snapshots never edit the source graph's native description.
    auto native_node = existing.native_node; auto device_node = existing.device_node;
    existing.native_node = nullptr;
    try { graph_replace(existing, std::move(replacement)); }
    catch (...) { existing.native_node = native_node; throw; }
    existing.native_node = native_node; existing.device_node = device_node;
    if (value.native) value.native_updates.push_back(existing);
    value.native_regions.clear();
}
bool graph_enable_type(CUgraphNodeType type) { return type == CU_GRAPH_NODE_TYPE_KERNEL || type == CU_GRAPH_NODE_TYPE_MEMCPY || type == CU_GRAPH_NODE_TYPE_MEMSET; }
bool graph_update_nodes(const std::vector<graph_node>& existing, const std::vector<graph_node>& candidate, CUgraphExecUpdateResultInfo& info, bool node_priority) {
    auto fail = [&](CUgraphExecUpdateResult result, CUgraphNode node = nullptr, CUgraphNode from = nullptr) {
        info = {result, node, from}; return false;
    };
    if (existing.size() != candidate.size()) return fail(CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED);
    std::unordered_map<CUgraphNode, size_t> old_indices, new_indices;
    for (size_t i = 0; i < existing.size(); ++i) { old_indices.emplace(existing[i].handle, i); new_indices.emplace(candidate[i].handle, i); }
    for (size_t i = 0; i < existing.size(); ++i) {
        auto& a = existing[i]; auto& b = candidate[i];
        if (a.dependencies.size() != b.dependencies.size()) return fail(CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED, b.handle);
        for (size_t j = 0; j < a.dependencies.size(); ++j)
            if (old_indices.at(a.dependencies[j]) != new_indices.at(b.dependencies[j]) || !graph_edge_equal(a.dependency_edges[j], b.dependency_edges[j]))
                return fail(CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED, b.handle, b.dependencies[j]);
        if (a.type != b.type) return fail(CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED, b.handle);
        if (a.memory != b.memory) return fail(CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED, b.handle);
        if (a.owner.lock() != b.owner.lock()) return fail(CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED, b.handle);
        if (a.type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP && !memops_compatible(a, b))
            return fail(CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED, b.handle);
        if (node_priority && a.type == CU_GRAPH_NODE_TYPE_KERNEL && a.priority != b.priority)
            return fail(CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED, b.handle);
        if (a.type == CU_GRAPH_NODE_TYPE_KERNEL && a.shape.options != b.shape.options)
            return fail(CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED, b.handle);
        if ((a.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL || a.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT) && a.semaphores.size() != b.semaphores.size())
            return fail(CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED, b.handle);
        if (a.type == CU_GRAPH_NODE_TYPE_MEMCPY) {
            if (a.peer_copy != b.peer_copy || a.copy_source.lock() != b.copy_source.lock() || a.copy_destination.lock() != b.copy_destination.lock())
                return fail(CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED, b.handle);
            if (a.copy.srcMemoryType != b.copy.srcMemoryType || a.copy.dstMemoryType != b.copy.dstMemoryType)
                return fail(CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED, b.handle);
            if (a.copy.Height != 1 || a.copy.Depth != 1 || b.copy.Height != 1 || b.copy.Depth != 1)
                return fail(CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED, b.handle);
        }
        if (a.type == CU_GRAPH_NODE_TYPE_MEMSET && (a.fill.elementSize != b.fill.elementSize ||
            a.fill.height != b.fill.height || (a.fill.height != 1 && (a.fill.width != b.fill.width || a.fill.pitch != b.fill.pitch))))
            return fail(CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED, b.handle);
        if (a.type == CU_GRAPH_NODE_TYPE_GRAPH && !graph_update_nodes(*a.child_nodes, *b.child_nodes, info, node_priority)) return false;
        if (a.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
            if (a.conditional.type != b.conditional.type || a.conditional.size != b.conditional.size)
                return fail(CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED, b.handle);
            for (size_t j = 0; j < a.conditional_nodes.size(); ++j)
                if (!graph_update_nodes(*a.conditional_nodes[j], *b.conditional_nodes[j], info, node_priority)) return false;
        }
    }
    return true;
}
void graph_rebind_nodes(std::vector<graph_node>& candidate, const std::vector<graph_node>& existing) {
    for (size_t i = 0; i < candidate.size(); ++i) {
        candidate[i].handle = existing[i].handle; candidate[i].dependencies = existing[i].dependencies;
        candidate[i].local_id = existing[i].local_id;
        if (candidate[i].child_nodes) {
            auto nested = *candidate[i].child_nodes; graph_rebind_nodes(nested, *existing[i].child_nodes);
            candidate[i].child_nodes = std::make_shared<const std::vector<graph_node>>(std::move(nested));
        }
    }
}
void graph_update(CUgraphExec handle, CUgraph graph, CUgraphExecUpdateResultInfo* info) {
    require(info, CUDA_ERROR_INVALID_VALUE, "null graph update result"); *info = {CU_GRAPH_EXEC_UPDATE_ERROR, nullptr, nullptr};
    require(!graph_lookup(graph).moved, CUDA_ERROR_INVALID_VALUE, "moved child graphs cannot be executable update sources");
    auto& value = graph_exec_lookup(handle); auto nodes = graph_snapshot(graph);
    if (graph_has_device_nodes(value.nodes) || graph_has_device_nodes(nodes)) {
        info->result = CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED;
        throw failure(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, "device-updatable graphs cannot be whole-graph updated");
    }
    std::vector<CUgraph> descriptions;
    const bool memory_graph = graph_has_memory(nodes);
    std::function<void(CUgraph)> retain_tree = [&](CUgraph description) {
        auto& state = graph_lookup(description);
        std::erase_if(state.instances, [](auto& item) { return item.expired(); });
        if (memory_graph) for (auto& item : state.instances) if (auto instance = item.lock())
            require(instance->executable_id == value.id, CUDA_ERROR_NOT_SUPPORTED, "memory update source has another executable");
        state.instances.reserve(state.instances.size() + 1); descriptions.push_back(description);
        for (auto& node : state.nodes) { if (node.child) retain_tree(node.child); for (auto body : node.conditional_graphs) retain_tree(body); }
    };
    retain_tree(graph);
    require(graph_update_nodes(value.nodes, nodes, *info, (value.flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY) != 0), CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, "executable graph update is incompatible");
    for (auto& node : nodes) graph_prepare(node);
    graph_memory_validate(nodes);
    auto objects = graph_lookup(graph).objects;
    auto instance = std::make_shared<graph_memory_instance>();
    instance->executable_id = value.id;
    graph_rebind_nodes(nodes, value.nodes);
    graph_native_update(value, graph, nodes);
    for (auto description : descriptions) graph_lookup(description).instances.push_back(instance);
    value.nodes.swap(nodes); value.objects.swap(objects);
    value.native_regions.clear();
    value.memory_instance = std::move(instance);
    *info = {CU_GRAPH_EXEC_UPDATE_SUCCESS, nullptr, nullptr};
}
void graph_exec_child_replace(CUgraphExec handle, CUgraphNode node, CUgraph child) {
    auto& value = graph_exec_lookup(handle); auto index = graph_exec_index(value, node); auto& existing = value.nodes[index];
    require(existing.type == CU_GRAPH_NODE_TYPE_GRAPH, CUDA_ERROR_INVALID_VALUE, "not a child graph node");
    require(!graph_has_memory(*existing.child_nodes) && !graph_has_memory(graph_lookup(child).nodes), CUDA_ERROR_NOT_SUPPORTED, "executable memory child replacement is unavailable");
    auto nodes = graph_snapshot(child); CUgraphExecUpdateResultInfo info{};
    require(graph_update_nodes(*existing.child_nodes, nodes, info, (value.flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY) != 0), CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, "replacement child graph is incompatible");
    for (auto& item : nodes) graph_prepare(item);
    graph_rebind_nodes(nodes, *existing.child_nodes);
    auto objects = graph_lookup(child).objects;
    auto snapshot = std::make_shared<const std::vector<graph_node>>(std::move(nodes));
    auto replacement = existing; replacement.child_nodes = snapshot;
    if (graph_has_memory(value.nodes)) { auto candidate = value.nodes; candidate[index] = replacement; graph_memory_validate(candidate); }
    if (value.native) {
        require(!graph_has_device_nodes(*existing.child_nodes) && !graph_has_device_nodes(*snapshot), CUDA_ERROR_NOT_SUPPORTED, "device-updatable child replacement is unavailable");
        replacement.native_child = graph_native_materialize(child, value.owner.lock())->clone();
        value.native_updates.reserve(value.native_updates.size() + 1); value.native_updates.push_back(std::move(replacement));
    }
    existing.child_nodes = std::move(snapshot);
    existing.child_objects.swap(objects);
    value.native_regions.clear();
}
void graph_launch(CUgraphExec handle, CUstream stream, bool upload) {
    auto& value = graph_exec_lookup(handle); auto owner = value.owner.lock();
    require(current() == owner, CUDA_ERROR_INVALID_CONTEXT, "executable graph belongs to another context");
    owner->native->check_device_fault();
    auto target = stream_lookup(stream, owner); capture_check_stream(target);
    if (target->capture) {
        if (graph_has_memory(value.nodes) || value.native) {
            target->capture->invalidated = true;
            throw failure(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "stateful graph launch capture is unavailable");
        }
        if (upload) {
            target->capture->invalidated = true;
            throw failure(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "graph upload during capture is unavailable");
        }
        capture_graph_launch(owner, stream, value); return;
    }
    if (graph_native_launch(value, owner, target, stream, upload)) return;
    std::vector<command_queue::submission> replay;
    std::vector<native_graph_command> recorders;
    std::vector<native_graph_key> native_keys;
    std::vector<std::vector<CUgraphEdgeData>> native_edges;
    bool auto_free = (value.flags & CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH) != 0;
    auto memory_plan = graph_memory_validate(value.nodes, !upload, auto_free, &value.enabled);
    auto memory_resources = graph_memory_prepare(value.nodes);
    auto queue = stream_native(owner, target);
    // A queued replay owns its complete user-object snapshot, including empty
    // child graphs, until every branch has completed or propagated its error.
    std::vector<std::shared_ptr<user_object>> objects;
    for (auto& [object, reference] : value.objects) objects.push_back(reference.object);
    auto initial = stream_dependencies(owner, target);
    command_order initial_order;
    for (auto& dependency : initial) inherit_order(initial_order, dependency);
    if (owner->transport_tail) initial.push_back(owner->transport_tail);
    if (value.tail) initial.push_back(value.tail);
    if (!upload && auto_free && !memory_plan.allocations.empty()) {
        auto reset = std::make_shared<command_completion>();
        replay.push_back({reset, initial, [allocations = memory_plan.allocations] { for (auto& memory : allocations) memory->live = false; }, 0, {}});
        initial = {reset};
        recorders.emplace_back();
        native_keys.emplace_back();
        native_edges.emplace_back(replay.back().dependencies.size());
    }
    std::vector<std::shared_ptr<host_entry>> hosts;
    std::vector<std::shared_ptr<context>> contexts;
    std::unordered_map<CUevent, completion> records;
    completion transport;
    std::unordered_map<graph_memory_region*, completion> memory_transitions;
    std::vector<completion> submitted;
    std::vector<std::shared_ptr<context>> transport_contexts;
    std::unordered_map<CUevent,std::shared_ptr<device_timestamp>> timestamps;
    std::function<std::vector<completion>(const std::vector<graph_node>&, const std::vector<completion>&, const command_order&, bool)> expand;
    expand = [&](const std::vector<graph_node>& nodes, const std::vector<completion>& roots, const command_order& root_order, bool top) {
        std::unordered_map<CUgraphNode, completion> indices; std::vector<completion> all;
        std::unordered_set<CUgraphNode> parents;
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto& node = nodes[i]; auto done = std::make_shared<command_completion>();
            if (!upload && node.memory) memory_transitions[node.memory.get()] = done;
            for (auto& [object, reference] : node.child_objects) objects.push_back(reference.object);
            auto dependencies = node.dependencies.empty() ? roots : std::vector<completion>{};
            std::vector<CUgraphEdgeData> dependency_edges(dependencies.size());
            done->order = node.dependencies.empty() ? root_order : command_order{};
            // Native regions preserve ports; scheduler boundaries conservatively
            // wait for full completion and visibility.
            for (size_t j = 0; j < node.dependencies.size(); ++j) {
                auto dependency = node.dependencies[j];
                parents.insert(dependency);
                dependencies.push_back(indices.at(dependency)); inherit_order(done->order, indices.at(dependency));
                dependency_edges.push_back(node.dependency_edges[j]);
            }
            if (transport) dependencies.push_back(transport);
            std::function<void()> action = [] {};
            std::function<device_work()> start;
            std::shared_ptr<native_queue> native_lane;
            native_graph_command recorder;
            native_graph_key native_key;
            int priority = 0;
            if (top ? value.enabled[i] : !node.snapshot_disabled) {
                auto operation = graph_prepare(node, bool(value.flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY), !top);
                for (auto& context : operation.contexts)
                    if (context != owner && std::find(contexts.begin(), contexts.end(), context) == contexts.end()) contexts.push_back(context);
                if (!upload) {
                    if (node.type == CU_GRAPH_NODE_TYPE_KERNEL && owner->native->kind() != cudavmBackendCuda && node.owner.lock() != owner) {
                        dependencies.insert(dependencies.end(), submitted.begin(), submitted.end());
                        auto node_owner = node.owner.lock();
                        if (std::find(transport_contexts.begin(), transport_contexts.end(), node_owner) == transport_contexts.end()) {
                            transport_contexts.push_back(node_owner);
                            if (node_owner->transport_tail) dependencies.push_back(node_owner->transport_tail);
                            for (auto& item : node_owner->streams) if (auto stream = item.lock()) if (stream->tail) dependencies.push_back(stream->tail);
                        }
                        transport = done;
                    }
                    if (node.type == CU_GRAPH_NODE_TYPE_KERNEL)
                        priority = (value.flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY) ? stream_priority(node.priority, node.owner.lock()) : target->priority;
                    action = std::move(operation.action);
                    if (operation.start) {
                        auto node_owner = node.owner.lock();
                        // A native lane belongs to its execution context. Mixed
                        // graphs keep their existing synchronous foreign path.
                        if (node_owner == owner || node.type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP)
                            start = [operation = operation.start, queue] { return operation(queue.get()); };
                        if (node_owner == owner && operation.native_compatible) {
                            recorder = std::move(operation.start); native_key = std::move(operation.native_key);
                        }
                        if (recorder) native_lane = queue;
                    }
                    for (auto& host : operation.hosts) if (std::find(hosts.begin(), hosts.end(), host) == hosts.end()) hosts.push_back(host);
                    if (node.type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
                        records[node.event] = done;
                        auto timestamp=(event_lookup(node.event).flags & CU_EVENT_DISABLE_TIMING) ? std::shared_ptr<device_timestamp>{} : std::make_shared<device_timestamp>();
                        timestamps[node.event]=timestamp;
                        auto node_owner=node.owner.lock();auto lane=node_owner==owner ? queue : node_owner->native->create_queue(0);
                        native_lane = lane;
                        start=[node_owner,lane,timestamp] {
                            std::lock_guard lock(node_owner->execution);
                            return timestamp ? node_owner->native->enqueue_timestamp(timestamp,lane.get()) : node_owner->native->enqueue_marker(lane.get());
                        };
                    }
                    if (node.type == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
                        auto it = records.find(node.event); auto record = it == records.end() ? event_lookup(node.event).record : it->second;
                        if (record) { dependencies.push_back(record); inherit_order(done->order, record); }
                        native_lane = queue;
                        start = [owner, queue] {
                            std::lock_guard lock(owner->execution);
                            return owner->native->enqueue_marker(queue.get());
                        };
                    }
                }
                if (node.type == CU_GRAPH_NODE_TYPE_GRAPH) {
                    dependencies = expand(*node.child_nodes, dependencies, done->order, false);
                    dependency_edges.assign(dependencies.size(), {});
                    for (auto& dependency : dependencies) inherit_order(done->order, dependency);
                }
            }
            bool exclusive = node.type == CU_GRAPH_NODE_TYPE_KERNEL && owner->native->kind() != cudavmBackendCuda;
            auto node_owner = node.owner.lock();
            auto domain = node_owner ? node_owner->command_domain : owner->command_domain;
            replay.push_back({done, std::move(dependencies), std::move(action), priority, std::move(start), exclusive, std::move(domain), {}, std::move(native_lane)}); indices.emplace(node.handle, done);
            recorders.push_back(std::move(recorder));
            native_keys.push_back(std::move(native_key));
            dependency_edges.resize(replay.back().dependencies.size());
            native_edges.push_back(std::move(dependency_edges));
            submitted.push_back(done);
        }
        // Joining the leaves also waits for every ancestor, including failures.
        // Intermediate completions can then stay inside a native region.
        for (auto& node : nodes) if (!parents.contains(node.handle)) all.push_back(indices.at(node.handle));
        return all.empty() ? roots : all;
    };
    auto final_dependencies = expand(value.nodes, initial, initial_order, true);
    for (auto& host : hosts) { std::erase_if(host->work, [](auto& item) { return item.expired(); }); host->work.reserve(host->work.size() + 1); }
    // Publish foreign-context lifetime tracking only after the atomic enqueue.
    // Reserve first, so neither allocation failure nor validation can publish a
    // partial replay or an untracked command.
    for (auto& context : contexts) {
        std::erase_if(context->foreign_work, [](auto& item) { return item.expired(); });
        context->foreign_work.reserve(context->foreign_work.size() + 1);
    }
    auto done = std::make_shared<command_completion>();
    done->order = initial_order;
    for (auto& dependency : final_dependencies) inherit_order(done->order, dependency);
    done->order[target->id] = target->sequence + 1;
    replay.push_back({done, std::move(final_dependencies), [target, objects = std::move(objects), contexts,
        memory_resources = std::move(memory_resources), instance = value.memory_instance] {}, 0, {}});
    recorders.emplace_back();
    native_keys.emplace_back();
    native_edges.emplace_back(replay.back().dependencies.size());
    if (!upload && owner->native->kind() == cudavmBackendCuda) {
        // Retain the replay snapshot through its GPU join, while allowing a
        // subsequent native stream command to consume that join's signal.
        auto retained = std::move(replay.back().action);
        replay.back().start = [owner, queue, retained = std::move(retained)] {
            std::lock_guard lock(owner->execution);
            return owner->native->enqueue_marker(queue.get());
        };
        replay.back().lane = queue;
    }
    if (!upload && (owner->native->kind() == cudavmBackendCuda || owner->native->command_regions())) lower_graph_regions(replay, recorders,
        [&](auto nodes) { return owner->native->compile_graph(nodes); }, queue, &value.native_regions, native_keys, native_edges,
        owner->native->command_regions());
    if (!owner->queue) owner->queue = std::make_unique<command_queue>(owner->command_domain);
    std::vector<CUdeviceptr> inserted;
    inserted.reserve(memory_plan.regions.size());
    try {
        if (!upload) for (auto& memory : memory_plan.regions) {
            if (global().graph_live.emplace(memory->address, memory).second) inserted.push_back(memory->address);
        }
        owner->queue->submit_batch(std::move(replay));
    } catch (...) { for (auto address : inserted) global().graph_live.erase(address); throw; }
    if (!upload) for (auto& memory : memory_plan.regions) {
        memory->planned_live = memory_plan.final_live.at(memory.get());
        memory->transition = memory_transitions.at(memory.get());
    }
    ++target->sequence; value.tail = done; target->tail = done;
    for (auto& context : transport_contexts) context->transport_tail = done;
    for (auto& [event, record] : records) {
        auto& state=event_lookup(event);state.record = std::move(record);state.timestamp=std::move(timestamps.at(event));
    }
    for (auto& host : hosts) host_track(host, done);
    for (auto& context : contexts) context->foreign_work.push_back(done);
}
