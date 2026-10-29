DRIVER_API(cuGraphCreate, (CUgraph* output, unsigned flags), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph output"); *output = graph_create(flags))
DRIVER_API(cuGraphDestroy, (CUgraph handle), driver_ready();
    require(!graph_lookup(handle).parent, CUDA_ERROR_INVALID_VALUE, "child graph is owned by its parent");
    graph_capture_guard(handle);
    graph_destroy(handle))
DRIVER_API(cuGraphClone, (CUgraph* output, CUgraph handle), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph clone output"); *output = graph_clone(handle))
DRIVER_API(cuGraphNodeFindInClone, (CUgraphNode* output, CUgraphNode original, CUgraph clone),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null cloned node output"); graph_node_lookup(original);
    auto& mapping = graph_lookup(clone).originals; auto it = mapping.find(original);
    require(it != mapping.end(), CUDA_ERROR_INVALID_VALUE, "node was not cloned into this graph"); *output = it->second)
DRIVER_API(cuGraphAddEmptyNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph node output"); *output = graph_add(graph, dependencies, count, graph_node{}))
DRIVER_API(cuGraphAddHostNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUDA_HOST_NODE_PARAMS* p),
    driver_ready(); require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph host parameters"); *output = graph_add(graph, dependencies, count, graph_host(*p)))
DRIVER_API(cuGraphAddMemcpyNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUDA_MEMCPY3D* p, CUcontext owner),
    driver_ready(); require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph copy parameters"); auto node = graph_copy(*p, lookup(owner)); graph_validate(node); *output = graph_add(graph, dependencies, count, std::move(node)))
DRIVER_API(cuGraphAddMemsetNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUDA_MEMSET_NODE_PARAMS* p, CUcontext owner),
    driver_ready(); require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph fill parameters"); auto node = graph_fill(*p, lookup(owner)); graph_validate(node); *output = graph_add(graph, dependencies, count, std::move(node)))
DRIVER_API(cuGraphAddKernelNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUDA_KERNEL_NODE_PARAMS* p),
    driver_ready(); require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); *output = graph_add(graph, dependencies, count, graph_kernel(*p, !p->func && p->ctx ? lookup(p->ctx) : current())))
DRIVER_API(cuGraphKernelNodeSetParams, (CUgraphNode handle, const CUDA_KERNEL_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); auto& node = graph_node_lookup(handle); auto owner = node.owner.lock();
    require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "graph kernel context was destroyed"); graph_replace(node, graph_kernel(*p, owner, node.shape.options)))
DRIVER_API(cuGraphHostNodeGetParams, (CUgraphNode handle, CUDA_HOST_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph host output"); auto& node = graph_node_lookup(handle); require(node.type == CU_GRAPH_NODE_TYPE_HOST, CUDA_ERROR_INVALID_VALUE, "not a host node"); *p = node.host)
DRIVER_API(cuGraphHostNodeSetParams, (CUgraphNode handle, const CUDA_HOST_NODE_PARAMS* p), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph host parameters"); graph_replace(graph_node_lookup(handle), graph_host(*p)))
DRIVER_API(cuGraphMemcpyNodeGetParams, (CUgraphNode handle, CUDA_MEMCPY3D* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph copy output"); auto& node = graph_node_lookup(handle); require(node.type == CU_GRAPH_NODE_TYPE_MEMCPY, CUDA_ERROR_INVALID_VALUE, "not a copy node"); *p = node.copy)
DRIVER_API(cuGraphMemcpyNodeSetParams, (CUgraphNode handle, const CUDA_MEMCPY3D* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph copy parameters"); auto& node = graph_node_lookup(handle); auto owner = node.owner.lock(); require(owner != nullptr, CUDA_ERROR_INVALID_CONTEXT, "graph copy context was destroyed"); graph_replace(node, graph_copy(*p, owner)))
DRIVER_API(cuGraphMemsetNodeGetParams, (CUgraphNode handle, CUDA_MEMSET_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph fill output"); auto& node = graph_node_lookup(handle); require(node.type == CU_GRAPH_NODE_TYPE_MEMSET, CUDA_ERROR_INVALID_VALUE, "not a fill node"); *p = node.fill)
DRIVER_API(cuGraphMemsetNodeSetParams, (CUgraphNode handle, const CUDA_MEMSET_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph fill parameters"); auto& node = graph_node_lookup(handle); auto owner = node.owner.lock(); require(owner != nullptr, CUDA_ERROR_INVALID_CONTEXT, "graph fill context was destroyed"); graph_replace(node, graph_fill(*p, owner)))
DRIVER_API(cuGraphDestroyNode, (CUgraphNode handle), driver_ready(); graph_destroy_node(handle))
DRIVER_API(cuGraphNodeGetType, (CUgraphNode handle, CUgraphNodeType* output), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph node type output"); *output = graph_node_lookup(handle).type)
DRIVER_API(cuGraphGetNodes, (CUgraph handle, CUgraphNode* output, size_t* count), driver_ready(); graph_query_list(graph_query_nodes(handle, false), output, count))
DRIVER_API(cuGraphGetRootNodes, (CUgraph handle, CUgraphNode* output, size_t* count), driver_ready(); graph_query_list(graph_query_nodes(handle, true), output, count))
DRIVER_API(cuGraphGetEdges, (CUgraph handle, CUgraphNode* from, CUgraphNode* to, CUgraphEdgeData* data, size_t* count), driver_ready(); graph_query_edges(handle, from, to, data, count))
DRIVER_API(cuGraphNodeGetDependencies, (CUgraphNode handle, CUgraphNode* output, CUgraphEdgeData* data, size_t* count), driver_ready(); graph_query_neighbors(handle, output, data, count, false))
DRIVER_API(cuGraphNodeGetDependentNodes, (CUgraphNode handle, CUgraphNode* output, CUgraphEdgeData* data, size_t* count), driver_ready(); graph_query_neighbors(handle, output, data, count, true))
DRIVER_API(cuGraphAddDependencies, (CUgraph handle, const CUgraphNode* from, const CUgraphNode* to, const CUgraphEdgeData* data, size_t count), driver_ready(); graph_edit_dependencies(handle, from, to, data, count, true))
DRIVER_API(cuGraphRemoveDependencies, (CUgraph handle, const CUgraphNode* from, const CUgraphNode* to, const CUgraphEdgeData* data, size_t count), driver_ready(); graph_edit_dependencies(handle, from, to, data, count, false))
DRIVER_API(cuGraphInstantiate, (CUgraphExec* output, CUgraph handle, unsigned long long flags), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null executable graph output"); *output = graph_instantiate(handle, flags))
DRIVER_API(cuGraphExecDestroy, (CUgraphExec handle), driver_ready(); graph_exec_lookup(handle); global().graph_execs.erase(handle))
DRIVER_API(cuGraphLaunch, (CUgraphExec handle, CUstream stream), driver_ready(); graph_launch(handle, stream, false))
DRIVER_API(cuGraphUpload, (CUgraphExec handle, CUstream stream), driver_ready(); graph_launch(handle, stream, true))
DRIVER_API(cuGraphLaunch_ptsz, (CUgraphExec handle, CUstream stream), driver_ready(); graph_launch(handle, stream ? stream : CU_STREAM_PER_THREAD, false))
DRIVER_API(cuGraphUpload_ptsz, (CUgraphExec handle, CUstream stream), driver_ready(); graph_launch(handle, stream ? stream : CU_STREAM_PER_THREAD, true))
DRIVER_API(cuGraphNodeSetEnabled, (CUgraphExec handle, CUgraphNode node, unsigned enabled),
    driver_ready(); auto& value = graph_exec_lookup(handle); auto index = graph_exec_index(value, node);
    require(graph_enable_type(value.nodes[index].type), CUDA_ERROR_INVALID_VALUE, "this graph node type cannot be disabled");
    if (value.native) value.native_enables.emplace_back(value.nodes[index].native_node, enabled != 0);
    if (value.enabled[index] != (enabled != 0)) { value.enabled[index] = enabled != 0; value.native_regions.clear(); })
DRIVER_API(cuGraphNodeGetEnabled, (CUgraphExec handle, CUgraphNode node, unsigned* enabled),
    driver_ready(); require(enabled, CUDA_ERROR_INVALID_VALUE, "null graph enabled output"); auto& value = graph_exec_lookup(handle); auto index = graph_exec_index(value, node);
    require(graph_enable_type(value.nodes[index].type), CUDA_ERROR_INVALID_VALUE, "this graph node type cannot be disabled"); *enabled = value.enabled[index] ? 1 : 0)
DRIVER_API(cuGraphExecHostNodeSetParams, (CUgraphExec handle, CUgraphNode node, const CUDA_HOST_NODE_PARAMS* p), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph host parameters"); graph_exec_replace(handle, node, graph_host(*p)))
DRIVER_API(cuGraphExecMemcpyNodeSetParams, (CUgraphExec handle, CUgraphNode node, const CUDA_MEMCPY3D* p, CUcontext owner), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph copy parameters"); graph_exec_replace(handle, node, graph_copy(*p, lookup(owner))))
DRIVER_API(cuGraphExecMemsetNodeSetParams, (CUgraphExec handle, CUgraphNode node, const CUDA_MEMSET_NODE_PARAMS* p, CUcontext owner), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph fill parameters"); graph_exec_replace(handle, node, graph_fill(*p, lookup(owner))))
DRIVER_API(cuGraphExecKernelNodeSetParams, (CUgraphExec handle, CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); auto& value = graph_exec_lookup(handle); auto owner = value.nodes[graph_exec_index(value, node)].owner.lock();
    require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "graph kernel context was destroyed"); graph_exec_replace(handle, node, graph_kernel(*p, owner, value.nodes[graph_exec_index(value, node)].shape.options)))

// Preserve the pre-edge-data Driver ABI alongside the pinned header's aliases.
#undef cuGraphGetEdges
#undef cuGraphNodeGetDependencies
#undef cuGraphNodeGetDependentNodes
#undef cuGraphAddDependencies
#undef cuGraphRemoveDependencies
DRIVER_API(cuGraphGetEdges, (CUgraph handle, CUgraphNode* from, CUgraphNode* to, size_t* count), driver_ready(); graph_query_edges(handle, from, to, nullptr, count))
DRIVER_API(cuGraphNodeGetDependencies, (CUgraphNode handle, CUgraphNode* output, size_t* count), driver_ready(); graph_query_neighbors(handle, output, nullptr, count, false))
DRIVER_API(cuGraphNodeGetDependentNodes, (CUgraphNode handle, CUgraphNode* output, size_t* count), driver_ready(); graph_query_neighbors(handle, output, nullptr, count, true))
DRIVER_API(cuGraphAddDependencies, (CUgraph handle, const CUgraphNode* from, const CUgraphNode* to, size_t count), driver_ready(); graph_edit_dependencies(handle, from, to, nullptr, count, true))
DRIVER_API(cuGraphRemoveDependencies, (CUgraph handle, const CUgraphNode* from, const CUgraphNode* to, size_t count), driver_ready(); graph_edit_dependencies(handle, from, to, nullptr, count, false))
#undef cuGraphAddKernelNode
#undef cuGraphKernelNodeSetParams
#undef cuGraphExecKernelNodeSetParams
CUDA_KERNEL_NODE_PARAMS graph_legacy_kernel(const CUDA_KERNEL_NODE_PARAMS_v1& p) {
    CUDA_KERNEL_NODE_PARAMS value{};
    value.func = p.func; value.gridDimX = p.gridDimX; value.gridDimY = p.gridDimY; value.gridDimZ = p.gridDimZ;
    value.blockDimX = p.blockDimX; value.blockDimY = p.blockDimY; value.blockDimZ = p.blockDimZ;
    value.sharedMemBytes = p.sharedMemBytes; value.kernelParams = p.kernelParams; value.extra = p.extra; return value;
}
DRIVER_API(cuGraphAddKernelNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUDA_KERNEL_NODE_PARAMS_v1* p),
    driver_ready(); require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); *output = graph_add(graph, dependencies, count, graph_kernel(graph_legacy_kernel(*p), current())))
DRIVER_API(cuGraphKernelNodeSetParams, (CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v1* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); auto& value = graph_node_lookup(node); auto owner = value.owner.lock(); require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "graph kernel context was destroyed"); graph_replace(value, graph_kernel(graph_legacy_kernel(*p), owner, value.shape.options)))
DRIVER_API(cuGraphExecKernelNodeSetParams, (CUgraphExec handle, CUgraphNode node, const CUDA_KERNEL_NODE_PARAMS_v1* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); auto& value = graph_exec_lookup(handle); auto owner = value.nodes[graph_exec_index(value, node)].owner.lock();
    require(owner && !owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "graph kernel context was destroyed"); graph_exec_replace(handle, node, graph_kernel(graph_legacy_kernel(*p), owner, value.nodes[graph_exec_index(value, node)].shape.options)))
#undef cuGraphInstantiate
DRIVER_API(cuGraphInstantiate, (CUgraphExec* output, CUgraph handle, CUgraphNode* error_node, char* log, size_t capacity),
    driver_ready(); require(output && (!capacity || log), CUDA_ERROR_INVALID_VALUE, "invalid legacy graph instantiation output"); if (error_node) *error_node = nullptr; if (capacity) log[0] = 0; *output = graph_instantiate(handle, 0))
DRIVER_API(cuGraphInstantiate_v2, (CUgraphExec* output, CUgraph handle, CUgraphNode* error_node, char* log, size_t capacity),
    driver_ready(); require(output && (!capacity || log), CUDA_ERROR_INVALID_VALUE, "invalid legacy graph instantiation output"); if (error_node) *error_node = nullptr; if (capacity) log[0] = 0; *output = graph_instantiate(handle, 0))
