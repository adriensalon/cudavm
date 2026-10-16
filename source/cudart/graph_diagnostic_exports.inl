RUNTIME_API(cudaGraphNodeGetContainingGraph, (cudaGraphNode_t node, cudaGraph_t* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null containing graph output"); runtime_current(); CUgraph graph{};
    check(cuGraphNodeGetContainingGraph(graph_cast<CUgraphNode>(node), &graph)); *output = graph_cast<cudaGraph_t>(graph))
RUNTIME_API(cudaGraphNodeGetLocalId, (cudaGraphNode_t node, unsigned* output),
    runtime_current(); check(cuGraphNodeGetLocalId(graph_cast<CUgraphNode>(node), output)))
RUNTIME_API(cudaGraphNodeGetToolsId, (cudaGraphNode_t node, unsigned long long* output),
    runtime_current(); check(cuGraphNodeGetToolsId(graph_cast<CUgraphNode>(node), output)))
RUNTIME_API(cudaGraphGetId, (cudaGraph_t graph, unsigned* output),
    runtime_current(); check(cuGraphGetId(graph_cast<CUgraph>(graph), output)))
RUNTIME_API(cudaGraphExecGetId, (cudaGraphExec_t graph, unsigned* output),
    runtime_current(); check(cuGraphExecGetId(graph_cast<CUgraphExec>(graph), output)))
RUNTIME_API(cudaGraphDebugDotPrint, (cudaGraph_t graph, const char* path, unsigned flags),
    runtime_current(); check(cuGraphDebugDotPrint(graph_cast<CUgraph>(graph), path, flags | CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES)))
