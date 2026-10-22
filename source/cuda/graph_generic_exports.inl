DRIVER_API(cuGraphAddNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, const CUgraphEdgeData* data, size_t count, CUgraphNodeParams* params),
    driver_ready(); require(output && params, CUDA_ERROR_INVALID_VALUE, "null generic graph node arguments");
    *output = graph_generic_add(graph, dependencies, data, count, *params))
DRIVER_API(cuGraphAddNode_v3, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, const CUgraphEdgeData* data, size_t count, CUgraphNodeParams* params),
    driver_ready(); require(output && params, CUDA_ERROR_INVALID_VALUE, "null generic graph node arguments");
    *output = graph_generic_add(graph, dependencies, data, count, *params))
#undef cuGraphAddNode
DRIVER_API(cuGraphAddNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, CUgraphNodeParams* params),
    driver_ready(); require(output && params, CUDA_ERROR_INVALID_VALUE, "null generic graph node arguments");
    *output = graph_generic_add(graph, dependencies, nullptr, count, *params))
DRIVER_API(cuGraphNodeSetParams, (CUgraphNode node, CUgraphNodeParams* params),
    driver_ready(); require(params, CUDA_ERROR_INVALID_VALUE, "null generic graph node parameters"); graph_generic_replace(node, *params))
DRIVER_API(cuGraphNodeSetParams_v2, (CUgraphNode node, CUgraphNodeParams* params),
    driver_ready(); require(params, CUDA_ERROR_INVALID_VALUE, "null generic graph node parameters"); graph_generic_replace(node, *params))
DRIVER_API(cuGraphExecNodeSetParams, (CUgraphExec handle, CUgraphNode node, CUgraphNodeParams* params),
    driver_ready(); require(params, CUDA_ERROR_INVALID_VALUE, "null generic graph node parameters"); graph_generic_exec_replace(handle, node, *params))
DRIVER_API(cuGraphExecGetFlags, (CUgraphExec handle, cuuint64_t* flags),
    driver_ready(); require(flags, CUDA_ERROR_INVALID_VALUE, "null executable graph flags output"); *flags = graph_exec_lookup(handle).flags)
DRIVER_API(cuGraphConditionalHandleCreate, (CUgraphConditionalHandle* output, CUgraph graph, CUcontext owner, unsigned initial, unsigned flags),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null conditional handle output"); *output = graph_condition_create(graph, lookup(owner), initial, flags))
