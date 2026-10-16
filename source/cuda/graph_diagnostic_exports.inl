DRIVER_API(cuGraphNodeGetContainingGraph, (CUgraphNode node, CUgraph* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null containing graph output"); graph_node_lookup(node); *output = global().graph_nodes.at(node))
DRIVER_API(cuGraphNodeGetLocalId, (CUgraphNode node, unsigned* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null local node ID output"); *output = graph_node_lookup(node).local_id)
DRIVER_API(cuGraphNodeGetToolsId, (CUgraphNode node, unsigned long long* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null tools node ID output"); graph_node_lookup(node); *output = reinterpret_cast<uintptr_t>(node))
DRIVER_API(cuGraphGetId, (CUgraph graph, unsigned* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph ID output"); *output = graph_lookup(graph).id)
DRIVER_API(cuGraphExecGetId, (CUgraphExec graph, unsigned* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null executable graph ID output"); *output = graph_exec_lookup(graph).id)
DRIVER_API(cuGraphDebugDotPrint, (CUgraph graph, const char* path, unsigned flags),
    driver_ready(); graph_dot_print(graph, path, flags))
