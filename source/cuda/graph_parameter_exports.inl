DRIVER_API(cuGraphKernelNodeGetParams, (CUgraphNode handle, CUDA_KERNEL_NODE_PARAMS* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph kernel output"); *output = graph_kernel_parameters(graph_node_lookup(handle)))
#undef cuGraphKernelNodeGetParams
DRIVER_API(cuGraphKernelNodeGetParams, (CUgraphNode handle, CUDA_KERNEL_NODE_PARAMS_v1* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph kernel output");
    auto& node = graph_node_lookup(handle); auto params = graph_kernel_parameters(node);
    if (!params.func) params.func = node.function; std::memcpy(output, &params, sizeof(*output)))
DRIVER_API(cuGraphNodeGetParams, (CUgraphNode handle, CUgraphNodeParams* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph node output"); *output = graph_node_parameters(graph_node_lookup(handle)))
