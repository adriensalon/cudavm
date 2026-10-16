DRIVER_API(cuGraphAddBatchMemOpNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, const CUDA_BATCH_MEM_OP_NODE_PARAMS* p),
    driver_ready(); require(output && p, CUDA_ERROR_INVALID_VALUE, "null graph memory operation arguments");
    auto node = graph_memops(*p); graph_validate(node); *output = graph_add(graph, dependencies, count, std::move(node)))
DRIVER_API(cuGraphBatchMemOpNodeGetParams, (CUgraphNode handle, CUDA_BATCH_MEM_OP_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph memory operation output"); auto& node = graph_node_lookup(handle);
    require(node.type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP, CUDA_ERROR_INVALID_VALUE, "not a batch memory operation node");
    auto& view = graph_parameter_view(node); *p = {node.parameter_context, static_cast<unsigned>(view.memops.size()), view.memops.data(), 0})
DRIVER_API(cuGraphBatchMemOpNodeSetParams, (CUgraphNode handle, const CUDA_BATCH_MEM_OP_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph memory operation parameters");
    graph_replace(graph_node_lookup(handle), graph_memops(*p)))
DRIVER_API(cuGraphExecBatchMemOpNodeSetParams, (CUgraphExec handle, CUgraphNode node, const CUDA_BATCH_MEM_OP_NODE_PARAMS* p),
    driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null graph memory operation parameters");
    graph_exec_replace(handle, node, graph_memops(*p)))
