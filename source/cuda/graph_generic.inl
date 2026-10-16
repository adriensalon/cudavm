template<class Params, class Member> void graph_union_validate(const Params& p, const Member&) {
    require(zero_reserved(p.reserved0) && !p.reserved2, CUDA_ERROR_INVALID_VALUE, "nonzero generic graph reserved fields");
    auto bytes = reinterpret_cast<const unsigned char*>(&p.reserved1);
    require(std::all_of(bytes + sizeof(Member), bytes + sizeof(p.reserved1), [](auto byte) { return byte == 0; }),
            CUDA_ERROR_INVALID_VALUE, "nonzero unused graph union bytes");
}
graph_node graph_generic(const CUgraphNodeParams& p, const launch_options& options = {}) {
    switch (p.type) {
    case CU_GRAPH_NODE_TYPE_EMPTY:
        require(zero_reserved(p.reserved0) && !p.reserved2 && zero_reserved(p.reserved1), CUDA_ERROR_INVALID_VALUE, "nonzero empty node parameters");
        return {};
    case CU_GRAPH_NODE_TYPE_KERNEL: {
        graph_union_validate(p, p.kernel);
        auto owner = p.kernel.ctx ? lookup(p.kernel.ctx) : current();
        if (p.kernel.func) {
            auto found = global().functions.find(p.kernel.func);
            require(found != global().functions.end(), CUDA_ERROR_INVALID_HANDLE, "invalid graph kernel function");
            require(!p.kernel.ctx || found->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "kernel context mismatch");
            owner = found->second.owner;
        }
        CUDA_KERNEL_NODE_PARAMS params{};
        static_assert(sizeof(params) <= sizeof(p.kernel));
        std::memcpy(&params, &p.kernel, sizeof(params));
        return graph_kernel(params, owner, options);
    }
    case CU_GRAPH_NODE_TYPE_MEMCPY:
        graph_union_validate(p, p.memcpy);
        require(!p.memcpy.flags && !p.memcpy.reserved, CUDA_ERROR_INVALID_VALUE, "nonzero generic copy flags");
        return graph_copy(p.memcpy.copyParams, p.memcpy.copyCtx ? lookup(p.memcpy.copyCtx) : current());
    case CU_GRAPH_NODE_TYPE_MEMSET: {
        graph_union_validate(p, p.memset);
        CUDA_MEMSET_NODE_PARAMS params{}; std::memcpy(&params, &p.memset, sizeof(params));
        return graph_fill(params, p.memset.ctx ? lookup(p.memset.ctx) : current());
    }
    case CU_GRAPH_NODE_TYPE_HOST: {
        graph_union_validate(p, p.host);
        require(!p.host.syncMode, CUDA_ERROR_NOT_SUPPORTED, "non-default host node synchronization is unavailable");
        auto node = graph_host({p.host.fn, p.host.userData});
        if (p.host.ctx) { node.owner = lookup(p.host.ctx); node.host_context = p.host.ctx; }
        return node;
    }
    case CU_GRAPH_NODE_TYPE_GRAPH:
        graph_union_validate(p, p.graph);
        require(p.graph.ownership == CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE || p.graph.ownership == CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE,
                CUDA_ERROR_INVALID_VALUE, "invalid child graph ownership");
        return graph_child(p.graph.graph);
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
        graph_union_validate(p, p.eventRecord); return graph_event(p.eventRecord.event, p.type);
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT:
        graph_union_validate(p, p.eventWait); return graph_event(p.eventWait.event, p.type);
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL:
        graph_union_validate(p, p.extSemSignal);
        return graph_semaphores(p.extSemSignal.extSemArray, p.extSemSignal.paramsArray, p.extSemSignal.numExtSems,
                               p.extSemSignal.ctx ? lookup(p.extSemSignal.ctx) : current());
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT:
        graph_union_validate(p, p.extSemWait);
        return graph_semaphores(p.extSemWait.extSemArray, p.extSemWait.paramsArray, p.extSemWait.numExtSems,
                               p.extSemWait.ctx ? lookup(p.extSemWait.ctx) : current());
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: {
        graph_union_validate(p, p.memOp);
        CUDA_BATCH_MEM_OP_NODE_PARAMS params{}; std::memcpy(&params, &p.memOp, sizeof(params));
        return graph_memops(params);
    }
    default:
        require(p.type >= CU_GRAPH_NODE_TYPE_KERNEL && p.type <= CU_GRAPH_NODE_TYPE_CONDITIONAL,
                CUDA_ERROR_INVALID_VALUE, "invalid generic graph node type");
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "generic graph node type is not implemented");
    }
}
CUgraphNode graph_generic_add(CUgraph graph, const CUgraphNode* dependencies, const CUgraphEdgeData* data, size_t count, CUgraphNodeParams& params) {
    if (params.type == CU_GRAPH_NODE_TYPE_MEM_ALLOC) {
        graph_union_validate(params, params.alloc);
        CUDA_MEM_ALLOC_NODE_PARAMS allocation{}; std::memcpy(&allocation, &params.alloc, sizeof(allocation));
        auto result = graph_memory_add_alloc(graph, dependencies, count, allocation, false, {}, data); params.alloc.dptr = allocation.dptr; return result;
    }
    if (params.type == CU_GRAPH_NODE_TYPE_MEM_FREE) {
        graph_union_validate(params, params.free); return graph_memory_add_free(graph, dependencies, count, params.free.dptr, data);
    }
    auto node = graph_generic(params); graph_validate(node);
    if (node.type == CU_GRAPH_NODE_TYPE_GRAPH) return params.graph.ownership == CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE
        ? graph_move_child(graph, dependencies, count, node.child, data) : graph_add_child(graph, dependencies, count, node.child, data);
    return graph_add(graph, dependencies, count, std::move(node), data);
}
void graph_generic_replace(CUgraphNode handle, const CUgraphNodeParams& params) {
    auto& existing = graph_node_lookup(handle); auto replacement = graph_generic(params, existing.shape.options);
    if (replacement.type != CU_GRAPH_NODE_TYPE_GRAPH) { graph_replace(existing, std::move(replacement)); return; }
    require(existing.type == replacement.type, CUDA_ERROR_INVALID_VALUE, "wrong graph node type");
    graph_capture_guard(existing.child);
    auto parent = global().graph_nodes.at(handle);
    if (params.graph.ownership == CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE) {
        graph_move_validate(parent, replacement.child); auto child = replacement.child, old = existing.child;
        graph_replace(existing, std::move(replacement));
        auto& transferred = graph_lookup(child); transferred.parent = parent; transferred.moved = true;
        graph_destroy(old); graph_memory_move(parent, child, true); return;
    }
    require(!graph_lookup(replacement.child).moved || graph_lookup(replacement.child).parent == parent,
            CUDA_ERROR_INVALID_VALUE, "moved graph cannot be added to another parent");
    auto clone = graph_clone(replacement.child); auto old = existing.child;
    try { replacement.child = clone; graph_replace(existing, std::move(replacement)); }
    catch (...) { graph_destroy(clone); throw; }
    graph_lookup(clone).parent = global().graph_nodes.at(handle); graph_destroy(old);
}
void graph_generic_exec_replace(CUgraphExec handle, CUgraphNode node, const CUgraphNodeParams& params) {
    auto& value = graph_exec_lookup(handle); auto replacement = graph_generic(params, value.nodes[graph_exec_index(value, node)].shape.options);
    if (replacement.type == CU_GRAPH_NODE_TYPE_GRAPH) {
        require(params.graph.ownership == CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE, CUDA_ERROR_NOT_SUPPORTED, "executable child updates do not transfer graph ownership");
        graph_exec_child_replace(handle, node, replacement.child);
    }
    else graph_exec_replace(handle, node, std::move(replacement));
}
