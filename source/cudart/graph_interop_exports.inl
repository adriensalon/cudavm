namespace {
template<class Node, class Params> struct runtime_semaphore_node {
    std::vector<CUexternalSemaphore> handles;
    std::vector<Params> params;
    template<class Runtime> explicit runtime_semaphore_node(const Runtime& p) {
        require(!p.numExtSems || (p.extSemArray && p.paramsArray), CUDA_ERROR_INVALID_VALUE, "null graph semaphore arrays");
        handles.reserve(p.numExtSems); params.reserve(p.numExtSems);
        for (unsigned i = 0; i < p.numExtSems; ++i) {
            handles.push_back(graph_cast<CUexternalSemaphore>(p.extSemArray[i]));
            params.push_back(interop_descriptor<Params>(p.paramsArray[i]));
        }
    }
    Node descriptor() { return {handles.data(), params.data(), static_cast<unsigned>(handles.size())}; }
};
}
RUNTIME_API(cudaGraphAddExternalSemaphoresSignalNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, const cudaExternalSemaphoreSignalNodeParams* p),
    require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore arguments"); runtime_current(); auto nodes = graph_dependencies(dependencies, count);
    runtime_semaphore_node<CUDA_EXT_SEM_SIGNAL_NODE_PARAMS, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> arrays(*p); auto params = arrays.descriptor(); CUgraphNode node{};
    check(cuGraphAddExternalSemaphoresSignalNode(&node, graph_cast<CUgraph>(graph), nodes.data(), count, &params)); *output = graph_cast<cudaGraphNode_t>(node))
RUNTIME_API(cudaGraphExternalSemaphoresSignalNodeGetParams, (cudaGraphNode_t node, cudaExternalSemaphoreSignalNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore output"); runtime_current(); CUDA_EXT_SEM_SIGNAL_NODE_PARAMS params{};
    check(cuGraphExternalSemaphoresSignalNodeGetParams(graph_cast<CUgraphNode>(node), &params)); *p = interop_descriptor<cudaExternalSemaphoreSignalNodeParams>(params))
RUNTIME_API(cudaGraphExternalSemaphoresSignalNodeSetParams, (cudaGraphNode_t node, const cudaExternalSemaphoreSignalNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore parameters"); runtime_current();
    runtime_semaphore_node<CUDA_EXT_SEM_SIGNAL_NODE_PARAMS, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> arrays(*p); auto params = arrays.descriptor();
    check(cuGraphExternalSemaphoresSignalNodeSetParams(graph_cast<CUgraphNode>(node), &params)))
RUNTIME_API(cudaGraphExecExternalSemaphoresSignalNodeSetParams, (cudaGraphExec_t handle, cudaGraphNode_t node, const cudaExternalSemaphoreSignalNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore parameters"); runtime_current();
    runtime_semaphore_node<CUDA_EXT_SEM_SIGNAL_NODE_PARAMS, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> arrays(*p); auto params = arrays.descriptor();
    check(cuGraphExecExternalSemaphoresSignalNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), &params)))
RUNTIME_API(cudaGraphAddExternalSemaphoresWaitNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, const cudaExternalSemaphoreWaitNodeParams* p),
    require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore arguments"); runtime_current(); auto nodes = graph_dependencies(dependencies, count);
    runtime_semaphore_node<CUDA_EXT_SEM_WAIT_NODE_PARAMS, CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> arrays(*p); auto params = arrays.descriptor(); CUgraphNode node{};
    check(cuGraphAddExternalSemaphoresWaitNode(&node, graph_cast<CUgraph>(graph), nodes.data(), count, &params)); *output = graph_cast<cudaGraphNode_t>(node))
RUNTIME_API(cudaGraphExternalSemaphoresWaitNodeGetParams, (cudaGraphNode_t node, cudaExternalSemaphoreWaitNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore output"); runtime_current(); CUDA_EXT_SEM_WAIT_NODE_PARAMS params{};
    check(cuGraphExternalSemaphoresWaitNodeGetParams(graph_cast<CUgraphNode>(node), &params)); *p = interop_descriptor<cudaExternalSemaphoreWaitNodeParams>(params))
RUNTIME_API(cudaGraphExternalSemaphoresWaitNodeSetParams, (cudaGraphNode_t node, const cudaExternalSemaphoreWaitNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore parameters"); runtime_current();
    runtime_semaphore_node<CUDA_EXT_SEM_WAIT_NODE_PARAMS, CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> arrays(*p); auto params = arrays.descriptor();
    check(cuGraphExternalSemaphoresWaitNodeSetParams(graph_cast<CUgraphNode>(node), &params)))
RUNTIME_API(cudaGraphExecExternalSemaphoresWaitNodeSetParams, (cudaGraphExec_t handle, cudaGraphNode_t node, const cudaExternalSemaphoreWaitNodeParams* p),
    require(p, CUDA_ERROR_INVALID_VALUE, "null graph semaphore parameters"); runtime_current();
    runtime_semaphore_node<CUDA_EXT_SEM_WAIT_NODE_PARAMS, CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> arrays(*p); auto params = arrays.descriptor();
    check(cuGraphExecExternalSemaphoresWaitNodeSetParams(graph_cast<CUgraphExec>(handle), graph_cast<CUgraphNode>(node), &params)))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync_ptsz(const cudaExternalSemaphore_t* handles, const cudaExternalSemaphoreSignalParams* params, unsigned count, cudaStream_t stream) {
    return cudaSignalExternalSemaphoresAsync(handles, params, count, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync_ptsz(const cudaExternalSemaphore_t* handles, const cudaExternalSemaphoreWaitParams* params, unsigned count, cudaStream_t stream) {
    return cudaWaitExternalSemaphoresAsync(handles, params, count, stream ? stream : cudaStreamPerThread);
}
