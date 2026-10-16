RUNTIME_API(cudaGraphAddMemAllocNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, cudaMemAllocNodeParams* params),
    require(output && params, CUDA_ERROR_INVALID_VALUE, "null graph allocation arguments"); runtime_current(); runtime_graph_allocation converted(*params);
    auto nodes = graph_dependencies(dependencies, count); CUgraphNode node{};
    check(cuGraphAddMemAllocNode(&node, graph_cast<CUgraph>(graph), nodes.data(), count, &converted.params));
    params->dptr = reinterpret_cast<void*>(uintptr_t(converted.params.dptr)); *output = graph_cast<cudaGraphNode_t>(node))
RUNTIME_API(cudaGraphAddMemFreeNode, (cudaGraphNode_t* output, cudaGraph_t graph, const cudaGraphNode_t* dependencies, size_t count, void* pointer),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph free output"); runtime_current(); auto nodes = graph_dependencies(dependencies, count); CUgraphNode node{};
    check(cuGraphAddMemFreeNode(&node, graph_cast<CUgraph>(graph), nodes.data(), count, reinterpret_cast<uintptr_t>(pointer))); *output = graph_cast<cudaGraphNode_t>(node))
RUNTIME_API(cudaGraphMemAllocNodeGetParams, (cudaGraphNode_t node, cudaMemAllocNodeParams* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph allocation parameters output"); runtime_current(); CUDA_MEM_ALLOC_NODE_PARAMS params{};
    check(cuGraphMemAllocNodeGetParams(graph_cast<CUgraphNode>(node), &params)); *output = runtime_graph_allocation_parameters<cudaMemAllocNodeParams>(params))
RUNTIME_API(cudaGraphMemFreeNodeGetParams, (cudaGraphNode_t node, void* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph free parameters output"); runtime_current(); CUdeviceptr pointer{};
    check(cuGraphMemFreeNodeGetParams(graph_cast<CUgraphNode>(node), &pointer)); *static_cast<void**>(output) = reinterpret_cast<void*>(uintptr_t(pointer)))
RUNTIME_API(cudaDeviceGraphMemTrim, (int device), check(cuInit(0)); check(cuDeviceGraphMemTrim(device)))
RUNTIME_API(cudaDeviceGetGraphMemAttribute, (int device, cudaGraphMemAttributeType attr, void* output),
    check(cuInit(0)); check(cuDeviceGetGraphMemAttribute(device, static_cast<CUgraphMem_attribute>(attr), output)))
RUNTIME_API(cudaDeviceSetGraphMemAttribute, (int device, cudaGraphMemAttributeType attr, void* input),
    check(cuInit(0)); check(cuDeviceSetGraphMemAttribute(device, static_cast<CUgraphMem_attribute>(attr), input)))
