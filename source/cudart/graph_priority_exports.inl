RUNTIME_API(cudaGraphKernelNodeGetAttribute, (cudaGraphNode_t handle, cudaKernelNodeAttrID attribute, cudaKernelNodeAttrValue* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null kernel graph attribute output"); runtime_current(); CUkernelNodeAttrValue value{};
    check(cuGraphKernelNodeGetAttribute(graph_cast<CUgraphNode>(handle), static_cast<CUkernelNodeAttrID>(attribute), &value));
    *output = runtime_launch_value(attribute, value))
RUNTIME_API(cudaGraphKernelNodeSetAttribute, (cudaGraphNode_t handle, cudaKernelNodeAttrID attribute, const cudaKernelNodeAttrValue* value),
    require(value, CUDA_ERROR_INVALID_VALUE, "null kernel graph attribute value"); runtime_current(); CUkernelNodeAttrValue converted{};
    converted = driver_launch_value(attribute, *value);
    check(cuGraphKernelNodeSetAttribute(graph_cast<CUgraphNode>(handle), static_cast<CUkernelNodeAttrID>(attribute), &converted)))
RUNTIME_API(cudaGraphKernelNodeCopyAttributes, (cudaGraphNode_t destination, cudaGraphNode_t source),
    runtime_current(); check(cuGraphKernelNodeCopyAttributes(graph_cast<CUgraphNode>(destination), graph_cast<CUgraphNode>(source))))
