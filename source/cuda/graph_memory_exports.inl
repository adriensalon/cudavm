DRIVER_API(cuGraphAddMemAllocNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, CUDA_MEM_ALLOC_NODE_PARAMS* params),
    driver_ready(); require(output && params, CUDA_ERROR_INVALID_VALUE, "null graph allocation arguments"); *output = graph_memory_add_alloc(graph, dependencies, count, *params))
DRIVER_API(cuGraphAddMemFreeNode, (CUgraphNode* output, CUgraph graph, const CUgraphNode* dependencies, size_t count, CUdeviceptr pointer),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph free output"); *output = graph_memory_add_free(graph, dependencies, count, pointer))
DRIVER_API(cuGraphMemAllocNodeGetParams, (CUgraphNode handle, CUDA_MEM_ALLOC_NODE_PARAMS* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph allocation parameters output"); *output = graph_memory_parameters(graph_node_lookup(handle)))
DRIVER_API(cuGraphMemFreeNodeGetParams, (CUgraphNode handle, CUdeviceptr* output),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graph free parameters output"); auto& node = graph_node_lookup(handle);
    require(node.type == CU_GRAPH_NODE_TYPE_MEM_FREE, CUDA_ERROR_INVALID_VALUE, "not a graph free node"); *output = node.memory->address)
DRIVER_API(cuDeviceGraphMemTrim, (CUdevice device), driver_ready(); graph_memory_trim(device))
DRIVER_API(cuDeviceGetGraphMemAttribute, (CUdevice device, CUgraphMem_attribute attr, void* output),
    driver_ready(); valid_device(device); require(output, CUDA_ERROR_INVALID_VALUE, "null graph memory attribute output");
    auto value = graph_memory_attribute(device, attr); std::memcpy(output, &value, sizeof(value)))
DRIVER_API(cuDeviceSetGraphMemAttribute, (CUdevice device, CUgraphMem_attribute attr, void* input),
    driver_ready(); valid_device(device); require(input, CUDA_ERROR_INVALID_VALUE, "null graph memory attribute input");
    uint64_t value; std::memcpy(&value, input, sizeof(value)); graph_memory_set_attribute(device, attr, value))
