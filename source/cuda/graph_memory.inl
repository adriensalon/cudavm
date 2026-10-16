graph_state& graph_lookup(CUgraph);
CUgraphNode graph_add(CUgraph, const CUgraphNode*, size_t, graph_node, const CUgraphEdgeData* = nullptr);
std::shared_ptr<graph_pool_stats> graph_pool(int device) {
    valid_device(device);
    auto& pool = global().graph_pools[device];
    if (!pool) pool = std::make_shared<graph_pool_stats>();
    return pool;
}
void graph_memory_resident(allocation& entry) {
    if (entry.native->graph_charge) return;
    auto pool = graph_pool(entry.owner->device);
    std::lock_guard lock(entry.owner->execution);
    if (entry.native->native_address()) {
        auto bytes = entry.native->restore_graph();
        try { entry.native->graph_charge = std::make_shared<graph_pool_charge>(pool, bytes); }
        catch (...) { entry.native->trim_graph(); throw; }
    } else {
        // Reuse detached shader storage only within its native session. Native
        // CUDA mappings keep their own reserved virtual addresses.
        auto& cache = global().graph_cache;
        auto found = std::find_if(cache.begin(), cache.end(), [&](auto& cached) {
            return cached.owner == entry.owner && cached.native.use_count() == 1 &&
                   !cached.native->native_address() && cached.native->size == entry.native->size;
        });
        if (found != cache.end()) {
            entry.native = std::move(found->native); cache.erase(found);
            entry.native->graph_charge->associate(true);
        } else {
            std::shared_ptr<buffer> native = entry.owner->native->allocate(entry.native->size);
            native->graph_charge = std::make_shared<graph_pool_charge>(pool, native->graph_backing_size());
            std::erase_if(global().device_buffers, [](auto& value) { return value.native.expired(); });
            global().device_buffers.push_back({entry.owner->device, native});
            entry.native = std::move(native);
        }
    }
}
std::vector<std::shared_ptr<buffer>> graph_memory_prepare(const std::vector<graph_node>& nodes) {
    std::vector<std::shared_ptr<context>> kernel_owners;
    std::vector<CUdeviceptr> addresses;
    auto use = [&](CUdeviceptr pointer) {
        auto found = containing(pointer);
        if (found != global().allocations.end() && found->second.graph_owned) addresses.push_back(found->first);
    };
    std::function<void(const std::vector<graph_node>&)> visit = [&](const auto& nested) {
        for (auto& node : nested) {
            if (auto owner = node.owner.lock()) {
                require(!owner->retiring && global().contexts.contains(owner->handle), CUDA_ERROR_INVALID_CONTEXT, "graph node context was destroyed");
                // Kernel transport snapshots all context allocations to support
                // indirect pointers. Copy/fill/memory nodes need only their ranges.
                if (node.type == CU_GRAPH_NODE_TYPE_KERNEL && std::find(kernel_owners.begin(), kernel_owners.end(), owner) == kernel_owners.end())
                    kernel_owners.push_back(owner);
            }
            if (node.memory) addresses.push_back(node.memory->address);
            if (node.type == CU_GRAPH_NODE_TYPE_MEMSET) use(node.fill.dst);
            if (node.type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP) for (auto& op : node.memops) use(memop_address(op));
            if (node.type == CU_GRAPH_NODE_TYPE_MEMCPY) {
                if (node.copy.srcMemoryType == CU_MEMORYTYPE_DEVICE || node.copy.srcMemoryType == CU_MEMORYTYPE_UNIFIED) use(node.copy.srcDevice);
                if (node.copy.dstMemoryType == CU_MEMORYTYPE_DEVICE || node.copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED) use(node.copy.dstDevice);
            }
            if (node.child_nodes) visit(*node.child_nodes);
        }
    };
    visit(nodes);
    std::vector<std::shared_ptr<buffer>> resources;
    for (auto& [address, entry] : global().allocations) if (entry.graph_owned && !entry.graph_memory.expired() &&
        (std::find(addresses.begin(), addresses.end(), address) != addresses.end() ||
         std::find(kernel_owners.begin(), kernel_owners.end(), entry.owner) != kernel_owners.end())) {
        graph_memory_resident(entry); resources.push_back(entry.native);
    }
    return resources;
}
void graph_memory_trim(int device) {
    valid_device(device);
    // Workers own storage directly. Do not wait for them, or unmap buffers
    // protected by a scheduled replay, upload, transfer or kernel snapshot.
    for (auto& [address, entry] : global().allocations) if (entry.graph_owned && entry.owner->device == device &&
        entry.native->graph_charge && entry.native.use_count() == 1) {
        auto memory = entry.graph_memory.lock();
        if (!memory || memory->live.load() || memory->planned_live || memory->transition) continue;
        std::lock_guard lock(entry.owner->execution);
        if (entry.native->native_address()) { entry.native->trim_graph(); entry.native->graph_charge.reset(); }
        else entry.native = std::make_shared<graph_address_buffer>(entry.native->size);
    }
    std::erase_if(global().graph_cache, [&](auto& entry) { return entry.owner->device == device && entry.native.use_count() == 1; });
}
uint64_t graph_memory_attribute(int device, CUgraphMem_attribute attr) {
    auto pool = graph_pool(device); std::lock_guard lock(pool->mutex);
    switch (attr) {
    case CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT: return pool->used;
    case CU_GRAPH_MEM_ATTR_USED_MEM_HIGH: return pool->used_high;
    case CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT: return pool->reserved;
    case CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH: return pool->reserved_high;
    default: throw failure(CUDA_ERROR_INVALID_VALUE, "invalid graph memory attribute");
    }
}
void graph_memory_set_attribute(int device, CUgraphMem_attribute attr, uint64_t value) {
    auto pool = graph_pool(device);
    require(!value && (attr == CU_GRAPH_MEM_ATTR_USED_MEM_HIGH || attr == CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH),
            CUDA_ERROR_INVALID_VALUE, "only graph memory high watermarks can be reset to zero");
    std::lock_guard lock(pool->mutex);
    // A zero reset starts a new observation period at the current footprint.
    if (attr == CU_GRAPH_MEM_ATTR_USED_MEM_HIGH) pool->used_high = pool->used;
    else pool->reserved_high = pool->reserved;
}
CUgraph graph_memory_root(CUgraph graph) {
    while (graph_lookup(graph).parent) graph = graph_lookup(graph).parent;
    return graph;
}
void graph_memory_move(CUgraph parent, CUgraph child, bool commit) {
    auto parent_root = graph_memory_root(parent);
    auto root_after_move = [&](CUgraph graph) {
        auto root = graph_memory_root(graph); return root == child ? parent_root : root;
    };
    for (auto& [address, allocation] : global().allocations) if (allocation.graph_owned) if (auto memory = allocation.graph_memory.lock()) {
        bool seen = false, previous = false; unsigned internal_count = 0;
        for (auto& free : memory->frees) if (!free.token.expired()) {
            bool internal = free.internal;
            if (global().graphs.contains(memory->origin) && global().graphs.contains(free.graph))
                internal = root_after_move(memory->origin) == root_after_move(free.graph);
            if (commit) free.internal = internal;
            else {
                require(!seen || previous == internal, CUDA_ERROR_INVALID_VALUE, "child transfer would mix owning and outside graph frees");
                require(!internal || !internal_count, CUDA_ERROR_INVALID_VALUE, "child transfer would duplicate an owning graph free");
                if (internal) ++internal_count;
                seen = true; previous = internal;
            }
        }
    }
}
bool graph_has_memory(const std::vector<graph_node>& nodes) {
    for (auto& node : nodes) {
        if (node.memory) return true;
        if (node.child_nodes ? graph_has_memory(*node.child_nodes) : node.child && graph_has_memory(graph_lookup(node.child).nodes)) return true;
    }
    return false;
}
CUDA_MEM_ALLOC_NODE_PARAMS graph_memory_parameters(const graph_node& node) {
    require(node.type == CU_GRAPH_NODE_TYPE_MEM_ALLOC && node.memory, CUDA_ERROR_INVALID_VALUE, "not a graph allocation node");
    auto result = node.memory->params;
    result.accessDescs = node.memory->access.empty() ? nullptr : node.memory->access.data();
    return result;
}
void graph_memory_dependencies(CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUgraphEdgeData* data, CUgraphNodeType type) {
    graph_dependencies_validate(graph, dependencies, data, count, type);
}
CUgraphNode graph_memory_add_alloc(CUgraph graph, const CUgraphNode* dependencies, size_t count, CUDA_MEM_ALLOC_NODE_PARAMS& params, bool stream_ordered = false, std::shared_ptr<context> capture_owner = {}, const CUgraphEdgeData* data = nullptr) {
    graph_lookup(graph); auto owner = capture_owner ? capture_owner : current();
    graph_memory_dependencies(graph, dependencies, count, data, CU_GRAPH_NODE_TYPE_MEM_ALLOC);
    auto& pool = params.poolProps;
    require(params.bytesize && (!params.accessDescCount || params.accessDescs) && params.accessDescCount <= global().devices.size(),
            CUDA_ERROR_INVALID_VALUE, "invalid graph allocation size or access descriptors");
    require(pool.allocType == CU_MEM_ALLOCATION_TYPE_PINNED && pool.location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
            pool.location.id >= 0 && static_cast<size_t>(pool.location.id) < global().devices.size(), CUDA_ERROR_INVALID_VALUE, "graph allocation requires a pinned device location");
    require(pool.location.id == owner->device, CUDA_ERROR_NOT_SUPPORTED, "foreign-device graph allocation is unavailable");
    require(pool.handleTypes == CU_MEM_HANDLE_TYPE_NONE && !pool.win32SecurityAttributes,
            CUDA_ERROR_INVALID_VALUE, "graph allocations cannot export memory handles");
    require(std::all_of(std::begin(pool.reserved), std::end(pool.reserved), [](auto v) { return v == 0; }),
            CUDA_ERROR_INVALID_VALUE, "nonzero graph allocation reserved fields");
    require(!pool.maxSize && !pool.usage && !pool.gpuDirectRDMACapable, CUDA_ERROR_NOT_SUPPORTED, "graph allocation pool hints are unavailable");
    auto memory = std::make_shared<graph_memory_region>(); memory->owner = owner; memory->origin = graph; memory->params = params;
    for (size_t i = 0; i < params.accessDescCount; ++i) {
        auto descriptor = params.accessDescs[i];
        require(descriptor.location.type == CU_MEM_LOCATION_TYPE_DEVICE && descriptor.location.id >= 0 &&
                static_cast<size_t>(descriptor.location.id) < global().devices.size(), CUDA_ERROR_INVALID_VALUE, "invalid graph memory access location");
        require(descriptor.location.id == owner->device && descriptor.flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                CUDA_ERROR_NOT_SUPPORTED, "graph memory peer or read-only access is unavailable");
        require(std::none_of(memory->access.begin(), memory->access.end(), [&](auto& value) { return value.location.id == descriptor.location.id; }),
                CUDA_ERROR_INVALID_VALUE, "duplicate graph memory access location");
        memory->access.push_back(descriptor);
    }
    auto native = stream_ordered ? [&] { std::lock_guard lock(owner->execution); return owner->native->reserve_graph(params.bytesize); }() :
        native_call(owner, [&] { return owner->native->reserve_graph(params.bytesize); }, true);
    auto address = add_allocation(owner, std::move(native)); memory->address = address; memory->params.dptr = address;
    auto& entry = global().allocations.at(address); entry.graph_owned = true; entry.graph_memory = memory;
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_MEM_ALLOC; node.owner = owner; node.memory = memory;
    try {
        auto result = graph_add(graph, dependencies, count, std::move(node), data);
        params.dptr = address; return result;
    } catch (...) { global().allocations.erase(address); throw; }
}
CUgraphNode graph_memory_add_free(CUgraph graph, const CUgraphNode* dependencies, size_t count, CUdeviceptr pointer, const CUgraphEdgeData* data = nullptr) {
    graph_lookup(graph); auto found = global().allocations.find(pointer);
    graph_memory_dependencies(graph, dependencies, count, data, CU_GRAPH_NODE_TYPE_MEM_FREE);
    require(found != global().allocations.end() && found->second.graph_owned, CUDA_ERROR_INVALID_VALUE, "graph free requires a graph allocation base");
    auto memory = found->second.graph_memory.lock();
    require(bool(memory), CUDA_ERROR_INVALID_VALUE, "graph allocation description was released");
    for (auto& node : graph_lookup(graph).nodes)
        require(node.type != CU_GRAPH_NODE_TYPE_MEM_FREE || node.memory != memory, CUDA_ERROR_INVALID_VALUE, "duplicate graph memory free");
    bool internal = global().graphs.contains(memory->origin) && graph_memory_root(memory->origin) == graph_memory_root(graph);
    for (auto& entry : memory->frees) if (!entry.token.expired()) {
        require(entry.internal == internal, CUDA_ERROR_INVALID_VALUE, "graph memory cannot be freed both inside and outside its owning graph");
        require(!internal, CUDA_ERROR_INVALID_VALUE, "duplicate owning graph memory free");
    }
    auto token = std::make_shared<unsigned>(0); memory->frees.reserve(memory->frees.size() + 1);
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_MEM_FREE; node.owner = memory->owner; node.memory = memory; node.memory_free_token = token;
    auto result = graph_add(graph, dependencies, count, std::move(node), data); memory->frees.push_back({token, graph, internal}); return result;
}
// Flatten nested joins just as replay does, then require CUDA-visible lifetime
// edges for direct copy/fill/kernel arguments. Indirect device-pointer accesses
// remain the application's responsibility.
struct graph_memory_plan {
    std::vector<std::shared_ptr<graph_memory_region>> regions;
    std::vector<std::shared_ptr<graph_memory_region>> allocations;
    std::unordered_map<graph_memory_region*, bool> final_live;
};
graph_memory_plan graph_memory_validate(const std::vector<graph_node>& nodes, bool launching = false, bool auto_free = false,
                                      const std::vector<bool>* enabled = nullptr) {
    struct item { const graph_node* node; std::vector<size_t> dependencies; bool enabled; };
    std::vector<item> items;
    std::function<std::vector<size_t>(const std::vector<graph_node>&, const std::vector<size_t>&, bool)> expand;
    expand = [&](const auto& nested, const auto& roots, bool top) {
        std::unordered_map<CUgraphNode, size_t> indices; std::vector<size_t> all;
        for (size_t i = 0; i < nested.size(); ++i) {
            auto& node = nested[i];
            auto dependencies = node.dependencies.empty() ? roots : std::vector<size_t>{};
            for (auto dependency : node.dependencies) dependencies.push_back(indices.at(dependency));
            if (node.child_nodes) dependencies = expand(*node.child_nodes, dependencies, false);
            auto index = items.size(); items.push_back({&node, std::move(dependencies), top && enabled ? (*enabled)[i] : !node.snapshot_disabled}); indices.emplace(node.handle, index); all.push_back(index);
        }
        return all.empty() ? roots : all;
    };
    expand(nodes, {}, true);
    graph_memory_plan plan;
    std::unordered_map<graph_memory_region*, size_t> allocated, freed;
    for (size_t i = 0; i < items.size(); ++i) {
        auto& node = *items[i].node;
        if (!node.memory) continue;
        auto pointer = node.memory.get();
        if (!plan.final_live.contains(pointer)) { plan.regions.push_back(node.memory); plan.final_live.emplace(pointer, node.memory->planned_live); }
        if (node.type == CU_GRAPH_NODE_TYPE_MEM_ALLOC) {
            require(allocated.emplace(pointer, i).second, CUDA_ERROR_INVALID_VALUE, "duplicate allocation in graph replay");
            plan.allocations.push_back(node.memory);
        } else require(freed.emplace(pointer, i).second, CUDA_ERROR_INVALID_VALUE, "duplicate free in graph replay");
    }
    auto precedes = [&](size_t first, size_t last) {
        std::vector<size_t> work{last}; std::vector<bool> visited(items.size());
        while (!work.empty()) {
            auto index = work.back(); work.pop_back();
            if (index == first) return true;
            if (!visited[index]) { visited[index] = true; work.insert(work.end(), items[index].dependencies.begin(), items[index].dependencies.end()); }
        }
        return false;
    };
    for (auto& [memory, index] : freed) if (auto allocation = allocated.find(memory); allocation != allocated.end())
        require(precedes(allocation->second, index), CUDA_ERROR_INVALID_VALUE, "graph free must depend on its allocation");
    for (size_t i = 0; i < items.size(); ++i) {
        if (!items[i].enabled) continue;
        auto& node = *items[i].node; std::vector<graph_memory_region*> uses;
        auto use = [&](CUdeviceptr pointer) {
            auto found = containing(pointer);
            if (found != global().allocations.end() && found->second.graph_owned) {
                auto memory = found->second.graph_memory.lock();
                require(bool(memory), CUDA_ERROR_INVALID_VALUE, "graph memory description was released"); uses.push_back(memory.get());
            }
        };
        if (node.type == CU_GRAPH_NODE_TYPE_MEMSET) use(node.fill.dst);
        else if (node.type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP) for (auto& op : node.memops) use(memop_address(op));
        else if (node.type == CU_GRAPH_NODE_TYPE_MEMCPY) {
            if (node.copy.srcMemoryType == CU_MEMORYTYPE_DEVICE || node.copy.srcMemoryType == CU_MEMORYTYPE_UNIFIED) use(node.copy.srcDevice);
            if (node.copy.dstMemoryType == CU_MEMORYTYPE_DEVICE || node.copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED) use(node.copy.dstDevice);
        } else if (node.type == CU_GRAPH_NODE_TYPE_KERNEL) for (auto parameter : node.kernel_layout) if (parameter.size == sizeof(CUdeviceptr)) {
            CUdeviceptr pointer{}; std::memcpy(&pointer, node.arguments.data() + parameter.offset, sizeof(pointer)); use(pointer);
        }
        for (auto memory : uses) {
            if (auto allocation = allocated.find(memory); allocation != allocated.end())
                require(precedes(allocation->second, i), CUDA_ERROR_INVALID_VALUE, "graph memory use must depend on allocation");
            else if (launching) require(memory->planned_live, CUDA_ERROR_INVALID_VALUE, "graph memory use is not live");
            if (auto free = freed.find(memory); free != freed.end())
                require(precedes(i, free->second), CUDA_ERROR_INVALID_VALUE, "graph memory free must depend on every use");
        }
    }
    if (launching) {
        for (auto& memory : plan.allocations) {
            require(auto_free || !memory->planned_live, CUDA_ERROR_INVALID_VALUE, "graph allocation remains live from a previous launch");
            plan.final_live.at(memory.get()) = false;
        }
        for (auto& item : items) if (auto memory = item.node->memory) {
            auto& live = plan.final_live.at(memory.get());
            if (item.node->type == CU_GRAPH_NODE_TYPE_MEM_ALLOC) { require(!live, CUDA_ERROR_INVALID_VALUE, "graph allocation is already live"); live = true; }
            else { require(live, CUDA_ERROR_INVALID_VALUE, "graph free allocation is not live"); live = false; }
        }
    }
    return plan;
}
