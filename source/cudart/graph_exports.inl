RUNTIME_API(cudaGraphCreate, (cudaGraph_t* output, unsigned flags), require(output, CUDA_ERROR_INVALID_VALUE, "null graph output"); runtime_current(); CUgraph value{}; check(cuGraphCreate(&value, flags)); *output = graph_cast<cudaGraph_t>(value))
RUNTIME_API(cudaGraphDestroy, (cudaGraph_t handle), runtime_current(); check(cuGraphDestroy(graph_cast<CUgraph>(handle))))
RUNTIME_API(cudaGraphClone, (cudaGraph_t* output, cudaGraph_t handle), require(output, CUDA_ERROR_INVALID_VALUE, "null graph clone output"); runtime_current(); CUgraph value{}; check(cuGraphClone(&value, graph_cast<CUgraph>(handle))); *output = graph_cast<cudaGraph_t>(value))
RUNTIME_API(cudaGraphNodeFindInClone, (cudaGraphNode_t* output, cudaGraphNode_t original, cudaGraph_t clone),
    require(output, CUDA_ERROR_INVALID_VALUE, "null cloned node output"); runtime_current(); CUgraphNode value{}; check(cuGraphNodeFindInClone(&value, graph_cast<CUgraphNode>(original), graph_cast<CUgraph>(clone))); *output = graph_cast<cudaGraphNode_t>(value))
RUNTIME_API(cudaGraphAddEmptyNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph node output"); runtime_current(); auto nodes = graph_dependencies(dependencies, count); CUgraphNode value{};
    check(cuGraphAddEmptyNode(&value, graph_cast<CUgraph>(graph), nodes.data(), count)); *output = graph_cast<cudaGraphNode_t>(value))
RUNTIME_API(cudaGraphAddHostNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, const cudaHostNodeParams* p),
    require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph host parameters"); runtime_current(); auto nodes = graph_dependencies(dependencies, count); CUDA_HOST_NODE_PARAMS params{p->fn, p->userData}; CUgraphNode value{};
    check(cuGraphAddHostNode(&value, graph_cast<CUgraph>(graph), nodes.data(), count, &params)); *output = graph_cast<cudaGraphNode_t>(value))
RUNTIME_API(cudaGraphAddMemcpyNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, const cudaMemcpy3DParms* p),
    require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph copy parameters"); auto owner = runtime_current(); auto nodes = graph_dependencies(dependencies, count); auto params = runtime_copy_3d(*p); CUgraphNode value{};
    check(cuGraphAddMemcpyNode(&value, graph_cast<CUgraph>(graph), nodes.data(), count, &params, owner)); *output = graph_cast<cudaGraphNode_t>(value))
RUNTIME_API(cudaGraphAddMemcpyNode1D, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, void* to, const void* from, size_t bytes, cudaMemcpyKind kind),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph node output"); auto owner = runtime_current(); auto nodes = graph_dependencies(dependencies, count); auto params = runtime_graph_copy_1d(to, from, bytes, kind); CUgraphNode value{};
    check(cuGraphAddMemcpyNode(&value, graph_cast<CUgraph>(graph), nodes.data(), count, &params, owner)); *output = graph_cast<cudaGraphNode_t>(value))
RUNTIME_API(cudaGraphAddMemsetNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, const cudaMemsetParams* p),
    require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph fill parameters"); auto owner = runtime_current(); auto nodes = graph_dependencies(dependencies, count); auto params = runtime_graph_fill(*p); CUgraphNode value{};
    check(cuGraphAddMemsetNode(&value, graph_cast<CUgraph>(graph), nodes.data(), count, &params, owner)); *output = graph_cast<cudaGraphNode_t>(value))
RUNTIME_API(cudaGraphAddKernelNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, const cudaKernelNodeParams* p),
    require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); auto owner = runtime_current(); auto nodes = graph_dependencies(dependencies, count); auto params = runtime_graph_kernel(*p, owner); CUgraphNode value{};
    check(cuGraphAddKernelNode(&value, graph_cast<CUgraph>(graph), nodes.data(), count, &params)); *output = graph_cast<cudaGraphNode_t>(value))
RUNTIME_API(cudaGraphKernelNodeSetParams, (cudaGraphNode_t handle, const cudaKernelNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); auto params = runtime_graph_kernel(*p, runtime_current()); check(cuGraphKernelNodeSetParams(graph_cast<CUgraphNode>(handle), &params)))
RUNTIME_API(cudaGraphHostNodeGetParams, (cudaGraphNode_t handle, cudaHostNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph host output"); runtime_current(); CUDA_HOST_NODE_PARAMS params{}; check(cuGraphHostNodeGetParams(graph_cast<CUgraphNode>(handle), &params)); *p = {params.fn, params.userData})
RUNTIME_API(cudaGraphHostNodeSetParams, (cudaGraphNode_t handle, const cudaHostNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph host parameters"); runtime_current(); CUDA_HOST_NODE_PARAMS params{p->fn, p->userData}; check(cuGraphHostNodeSetParams(graph_cast<CUgraphNode>(handle), &params)))
RUNTIME_API(cudaGraphMemcpyNodeGetParams, (cudaGraphNode_t handle, cudaMemcpy3DParms* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph copy output"); runtime_current(); CUDA_MEMCPY3D params{}; check(cuGraphMemcpyNodeGetParams(graph_cast<CUgraphNode>(handle), &params)); *p = runtime_graph_copy_params(params))
RUNTIME_API(cudaGraphMemcpyNodeSetParams, (cudaGraphNode_t handle, const cudaMemcpy3DParms* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph copy parameters"); runtime_current(); auto params = runtime_copy_3d(*p); check(cuGraphMemcpyNodeSetParams(graph_cast<CUgraphNode>(handle), &params)))
RUNTIME_API(cudaGraphMemcpyNodeSetParams1D, (cudaGraphNode_t handle, void* to, const void* from, size_t bytes, cudaMemcpyKind kind),
    runtime_current(); auto params = runtime_graph_copy_1d(to, from, bytes, kind); check(cuGraphMemcpyNodeSetParams(graph_cast<CUgraphNode>(handle), &params)))
RUNTIME_API(cudaGraphMemsetNodeGetParams, (cudaGraphNode_t handle, cudaMemsetParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph fill output"); runtime_current(); CUDA_MEMSET_NODE_PARAMS params{}; check(cuGraphMemsetNodeGetParams(graph_cast<CUgraphNode>(handle), &params));
    *p = {reinterpret_cast<void*>(uintptr_t(params.dst)), params.pitch, params.value, params.elementSize, params.width, params.height})
RUNTIME_API(cudaGraphMemsetNodeSetParams, (cudaGraphNode_t handle, const cudaMemsetParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph fill parameters"); runtime_current(); auto params = runtime_graph_fill(*p); check(cuGraphMemsetNodeSetParams(graph_cast<CUgraphNode>(handle), &params)))
RUNTIME_API(cudaGraphDestroyNode, (cudaGraphNode_t handle), runtime_current(); check(cuGraphDestroyNode(graph_cast<CUgraphNode>(handle))))
RUNTIME_API(cudaGraphNodeGetType, (cudaGraphNode_t handle, cudaGraphNodeType* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph node type output"); runtime_current(); CUgraphNodeType type{}; check(cuGraphNodeGetType(graph_cast<CUgraphNode>(handle), &type)); *output = static_cast<cudaGraphNodeType>(type))
RUNTIME_API(cudaGraphGetNodes, (cudaGraph_t handle, cudaGraphNode_t* output, size_t* count),
    runtime_current(); runtime_graph_query(output, count, [&](CUgraphNode* nodes, size_t* size) { return cuGraphGetNodes(graph_cast<CUgraph>(handle), nodes, size); }))
RUNTIME_API(cudaGraphGetRootNodes, (cudaGraph_t handle, cudaGraphNode_t* output, size_t* count),
    runtime_current(); runtime_graph_query(output, count, [&](CUgraphNode* nodes, size_t* size) { return cuGraphGetRootNodes(graph_cast<CUgraph>(handle), nodes, size); }))
RUNTIME_API(cudaGraphGetEdges, (cudaGraph_t handle, cudaGraphNode_t* from, cudaGraphNode_t* to, cudaGraphEdgeData* data, size_t* count),
    runtime_current(); require(count && ((!from && !to && !data) || (from && to)), CUDA_ERROR_INVALID_VALUE, "invalid graph edge outputs");
    if (!from) { check(cuGraphGetEdges(graph_cast<CUgraph>(handle), nullptr, nullptr, nullptr, count)); return; }
    auto capacity = *count; std::vector<CUgraphNode> sources(capacity), targets(capacity); std::vector<CUgraphEdgeData> edges(data ? capacity : 0); CUgraphNode source_sentinel{}, target_sentinel{}; CUgraphEdgeData edge_sentinel{}; auto written = capacity;
    check(cuGraphGetEdges(graph_cast<CUgraph>(handle), capacity ? sources.data() : &source_sentinel, capacity ? targets.data() : &target_sentinel, data ? (capacity ? edges.data() : &edge_sentinel) : nullptr, &written));
    for (size_t i = 0; i < capacity; ++i) { from[i] = graph_cast<cudaGraphNode_t>(sources[i]); to[i] = graph_cast<cudaGraphNode_t>(targets[i]); } if (data && capacity) std::memcpy(data, edges.data(), capacity * sizeof(CUgraphEdgeData)); *count = written)
RUNTIME_API(cudaGraphNodeGetDependencies, (cudaGraphNode_t handle, cudaGraphNode_t* output, cudaGraphEdgeData* data, size_t* count),
    runtime_current(); require(!data || output, CUDA_ERROR_INVALID_VALUE, "edge data requires graph node outputs"); std::vector<CUgraphEdgeData> edges(data && count ? *count : 0); CUgraphEdgeData sentinel{};
    runtime_graph_query(output, count, [&](CUgraphNode* nodes, size_t* size) { return cuGraphNodeGetDependencies(graph_cast<CUgraphNode>(handle), nodes, data ? (edges.empty() ? &sentinel : edges.data()) : nullptr, size); }); if (data && !edges.empty()) std::memcpy(data, edges.data(), edges.size() * sizeof(CUgraphEdgeData)))
RUNTIME_API(cudaGraphNodeGetDependentNodes, (cudaGraphNode_t handle, cudaGraphNode_t* output, cudaGraphEdgeData* data, size_t* count),
    runtime_current(); require(!data || output, CUDA_ERROR_INVALID_VALUE, "edge data requires graph node outputs"); std::vector<CUgraphEdgeData> edges(data && count ? *count : 0); CUgraphEdgeData sentinel{};
    runtime_graph_query(output, count, [&](CUgraphNode* nodes, size_t* size) { return cuGraphNodeGetDependentNodes(graph_cast<CUgraphNode>(handle), nodes, data ? (edges.empty() ? &sentinel : edges.data()) : nullptr, size); }); if (data && !edges.empty()) std::memcpy(data, edges.data(), edges.size() * sizeof(CUgraphEdgeData)))
RUNTIME_API(cudaGraphAddDependencies, (cudaGraph_t handle, const cudaGraphNode_t* from, const cudaGraphNode_t* to, const cudaGraphEdgeData* data, size_t count),
    runtime_current(); auto sources = graph_dependencies(from, count), targets = graph_dependencies(to, count); auto edges = graph_edge_data(data, count);
    check(cuGraphAddDependencies(graph_cast<CUgraph>(handle), sources.data(), targets.data(), data ? edges.data() : nullptr, count)))
RUNTIME_API(cudaGraphRemoveDependencies, (cudaGraph_t handle, const cudaGraphNode_t* from, const cudaGraphNode_t* to, const cudaGraphEdgeData* data, size_t count),
    runtime_current(); auto sources = graph_dependencies(from, count), targets = graph_dependencies(to, count); auto edges = graph_edge_data(data, count);
    check(cuGraphRemoveDependencies(graph_cast<CUgraph>(handle), sources.data(), targets.data(), data ? edges.data() : nullptr, count)))
RUNTIME_API(cudaGraphInstantiate, (cudaGraphExec_t* output, cudaGraph_t handle, unsigned long long flags),
    require(output, CUDA_ERROR_INVALID_VALUE, "null executable graph output"); runtime_current(); CUgraphExec value{}; check(cuGraphInstantiate(&value, graph_cast<CUgraph>(handle), flags)); *output = graph_cast<cudaGraphExec_t>(value))
RUNTIME_API(cudaGraphInstantiateWithFlags, (cudaGraphExec_t* output, cudaGraph_t handle, unsigned long long flags),
    require(output, CUDA_ERROR_INVALID_VALUE, "null executable graph output"); runtime_current(); CUgraphExec value{}; check(cuGraphInstantiate(&value, graph_cast<CUgraph>(handle), flags)); *output = graph_cast<cudaGraphExec_t>(value))
RUNTIME_API(cudaGraphLaunch, (cudaGraphExec_t handle, cudaStream_t stream), runtime_current(); check(cuGraphLaunch(graph_cast<CUgraphExec>(handle), graph_cast<CUstream>(stream))))
RUNTIME_API(cudaGraphUpload, (cudaGraphExec_t handle, cudaStream_t stream), runtime_current(); check(cuGraphUpload(graph_cast<CUgraphExec>(handle), graph_cast<CUstream>(stream))))
RUNTIME_API(cudaGraphLaunch_ptsz, (cudaGraphExec_t handle, cudaStream_t stream), runtime_current(); check(cuGraphLaunch(graph_cast<CUgraphExec>(handle), stream ? graph_cast<CUstream>(stream) : CU_STREAM_PER_THREAD)))
RUNTIME_API(cudaGraphUpload_ptsz, (cudaGraphExec_t handle, cudaStream_t stream), runtime_current(); check(cuGraphUpload(graph_cast<CUgraphExec>(handle), stream ? graph_cast<CUstream>(stream) : CU_STREAM_PER_THREAD)))
RUNTIME_API(cudaGraphExecDestroy, (cudaGraphExec_t handle), runtime_current(); check(cuGraphExecDestroy(graph_cast<CUgraphExec>(handle))))
RUNTIME_API(cudaGraphNodeSetEnabled, (cudaGraphExec_t handle, cudaGraphNode_t node, unsigned enabled), runtime_current(); check(cuGraphNodeSetEnabled(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), enabled)))
RUNTIME_API(cudaGraphNodeGetEnabled, (cudaGraphExec_t handle, cudaGraphNode_t node, unsigned* enabled), runtime_current(); check(cuGraphNodeGetEnabled(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), enabled)))
RUNTIME_API(cudaGraphExecHostNodeSetParams, (cudaGraphExec_t handle, cudaGraphNode_t node, const cudaHostNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph host parameters"); runtime_current(); CUDA_HOST_NODE_PARAMS params{p->fn, p->userData}; check(cuGraphExecHostNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), &params)))
RUNTIME_API(cudaGraphExecMemcpyNodeSetParams, (cudaGraphExec_t handle, cudaGraphNode_t node, const cudaMemcpy3DParms* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph copy parameters"); auto owner = runtime_current(); auto params = runtime_copy_3d(*p); check(cuGraphExecMemcpyNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), &params, owner)))
RUNTIME_API(cudaGraphExecMemcpyNodeSetParams1D, (cudaGraphExec_t handle, cudaGraphNode_t node, void* to, const void* from, size_t bytes, cudaMemcpyKind kind),
    auto owner = runtime_current(); auto params = runtime_graph_copy_1d(to, from, bytes, kind); check(cuGraphExecMemcpyNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), &params, owner)))
RUNTIME_API(cudaGraphExecMemsetNodeSetParams, (cudaGraphExec_t handle, cudaGraphNode_t node, const cudaMemsetParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph fill parameters"); auto owner = runtime_current(); auto params = runtime_graph_fill(*p); check(cuGraphExecMemsetNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), &params, owner)))
RUNTIME_API(cudaGraphExecKernelNodeSetParams, (cudaGraphExec_t handle, cudaGraphNode_t node, const cudaKernelNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph kernel parameters"); auto params = runtime_graph_kernel(*p, runtime_current()); check(cuGraphExecKernelNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), &params)))
