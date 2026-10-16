DRIVER_API(cuGraphAddExternalSemaphoresSignalNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* p),
    driver_ready(); require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore arguments");
    *output = graph_add(graph, dependencies, count, graph_semaphores(p->extSemArray, p->paramsArray, p->numExtSems, current())))
DRIVER_API(cuGraphExternalSemaphoresSignalNodeGetParams, (CUgraphNode handle, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore output"); auto& node = graph_node_lookup(handle);
    require(node.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL, CUDA_ERROR_INVALID_VALUE, "not a semaphore signal node");
    auto& view = graph_parameter_view(node); *p = {view.semaphores.data(), view.signals.data(), static_cast<unsigned>(view.semaphores.size())})
DRIVER_API(cuGraphExternalSemaphoresSignalNodeSetParams, (CUgraphNode handle, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore parameters"); auto& node = graph_node_lookup(handle);
    graph_replace(node, graph_semaphores(p->extSemArray, p->paramsArray, p->numExtSems, node.owner.lock())))
DRIVER_API(cuGraphExecExternalSemaphoresSignalNodeSetParams, (CUgraphExec handle, CUgraphNode node, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore parameters"); auto& value = graph_exec_lookup(handle);
    graph_exec_replace(handle, node, graph_semaphores(p->extSemArray, p->paramsArray, p->numExtSems, value.owner.lock())))
DRIVER_API(cuGraphAddExternalSemaphoresWaitNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* p),
    driver_ready(); require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore arguments");
    *output = graph_add(graph, dependencies, count, graph_semaphores(p->extSemArray, p->paramsArray, p->numExtSems, current())))
DRIVER_API(cuGraphExternalSemaphoresWaitNodeGetParams, (CUgraphNode handle, CUDA_EXT_SEM_WAIT_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore output"); auto& node = graph_node_lookup(handle);
    require(node.type == CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT, CUDA_ERROR_INVALID_VALUE, "not a semaphore wait node");
    auto& view = graph_parameter_view(node); *p = {view.semaphores.data(), view.waits.data(), static_cast<unsigned>(view.semaphores.size())})
DRIVER_API(cuGraphExternalSemaphoresWaitNodeSetParams, (CUgraphNode handle, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore parameters"); auto& node = graph_node_lookup(handle);
    graph_replace(node, graph_semaphores(p->extSemArray, p->paramsArray, p->numExtSems, node.owner.lock())))
DRIVER_API(cuGraphExecExternalSemaphoresWaitNodeSetParams, (CUgraphExec handle, CUgraphNode node, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore parameters"); auto& value = graph_exec_lookup(handle);
    graph_exec_replace(handle, node, graph_semaphores(p->extSemArray, p->paramsArray, p->numExtSems, value.owner.lock())))
DRIVER_API(cuSignalExternalSemaphoresAsync_ptsz, (const CUexternalSemaphore* handles, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* params, unsigned count, CUstream stream),
    driver_ready(); semaphore_operations(handles, params, count, stream ? stream : CU_STREAM_PER_THREAD, true))
DRIVER_API(cuWaitExternalSemaphoresAsync_ptsz, (const CUexternalSemaphore* handles, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* params, unsigned count, CUstream stream),
    driver_ready(); semaphore_operations(handles, params, count, stream ? stream : CU_STREAM_PER_THREAD, false))
