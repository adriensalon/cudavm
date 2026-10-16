DRIVER_API(cuGraphKernelNodeGetAttribute, (CUgraphNode handle, CUkernelNodeAttrID attribute, CUkernelNodeAttrValue* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null kernel graph attribute output");
    *output = graph_launch_attribute(handle, attribute))
DRIVER_API(cuGraphKernelNodeSetAttribute, (CUgraphNode handle, CUkernelNodeAttrID attribute, const CUkernelNodeAttrValue* value),
    driver_ready(); require(value, CUDA_ERROR_INVALID_VALUE, "null kernel graph attribute value");
    graph_launch_attribute(handle, attribute, *value))
DRIVER_API(cuGraphKernelNodeCopyAttributes, (CUgraphNode destination, CUgraphNode source),
    driver_ready(); graph_priority_copy(destination, source))
