namespace {
void* runtime_graph_entry(CUfunction function) {
    auto found = global().function_entries.find(function);
    if (found == global().function_entries.end() || found->second.generation != cudavmDriverContextGeneration(found->second.owner)) return nullptr;
    auto& entry = found->second; auto active = global().bindings.find(entry.token);
    // A weak host token may now select another image. Keep an existing graph's
    // original kernel identity when its parameters are queried and reused.
    return active != global().bindings.end() && active->second.identity == entry.identity
        ? const_cast<void*>(entry.token) : reinterpret_cast<void*>(entry.kernel);
}
cudaKernelNodeParams runtime_graph_kernel_parameters(const CUDA_KERNEL_NODE_PARAMS& p) {
    auto entry = p.func ? runtime_graph_entry(p.func) : reinterpret_cast<void*>(p.kern);
    symbol_require(entry != nullptr, cudaErrorInvalidDeviceFunction, "graph kernel has no registered Runtime device entry");
    return {entry, dim3(p.gridDimX, p.gridDimY, p.gridDimZ), dim3(p.blockDimX, p.blockDimY, p.blockDimZ),
            p.sharedMemBytes, p.kernelParams, p.extra};
}
cudaGraphNodeParams runtime_graph_node_parameters(const CUgraphNodeParams& p, CUcontext owner) {
    cudaGraphNodeParams result{}; result.type = static_cast<cudaGraphNodeType>(p.type);
    auto context = [&](CUcontext value) { require(!value || value == owner, CUDA_ERROR_NOT_SUPPORTED, "foreign Runtime execution context queries are unavailable"); };
    switch (p.type) {
    case CU_GRAPH_NODE_TYPE_CONDITIONAL:
        result.conditional.handle = p.conditional.handle; result.conditional.type = static_cast<cudaGraphConditionalNodeType>(p.conditional.type);
        result.conditional.size = p.conditional.size; result.conditional.phGraph_out = graph_cast<cudaGraph_t*>(p.conditional.phGraph_out); break;
    case CU_GRAPH_NODE_TYPE_EMPTY: break;
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC: {
        CUDA_MEM_ALLOC_NODE_PARAMS allocation{}; std::memcpy(&allocation, &p.alloc, sizeof(allocation));
        result.alloc = runtime_graph_allocation_parameters<cudaMemAllocNodeParamsV2>(allocation); break;
    }
    case CU_GRAPH_NODE_TYPE_MEM_FREE: result.free.dptr = reinterpret_cast<void*>(uintptr_t(p.free.dptr)); break;
    case CU_GRAPH_NODE_TYPE_KERNEL:
        if (!p.kernel.func && p.kernel.kern) {
            result.kernel.kern = graph_cast<cudaKernel_t>(p.kernel.kern); result.kernel.functionType = cudaKernelFunctionTypeKernel;
        } else if (auto entry = runtime_graph_entry(p.kernel.func); entry && (!p.kernel.ctx || p.kernel.ctx == owner)) {
            if (cudavmDriverIsLibraryKernel(entry)) {
                result.kernel.kern = graph_cast<cudaKernel_t>(entry); result.kernel.functionType = cudaKernelFunctionTypeKernel;
            } else { result.kernel.func = entry; result.kernel.functionType = cudaKernelFunctionTypeDeviceEntry; }
        } else {
            result.kernel.cuFunc = graph_cast<cudaFunction_t>(p.kernel.func); result.kernel.functionType = cudaKernelFunctionTypeFunction;
        }
        result.kernel.gridDim = dim3(p.kernel.gridDimX,p.kernel.gridDimY,p.kernel.gridDimZ);
        result.kernel.blockDim = dim3(p.kernel.blockDimX,p.kernel.blockDimY,p.kernel.blockDimZ);
        result.kernel.sharedMemBytes = p.kernel.sharedMemBytes;
        result.kernel.kernelParams = p.kernel.kernelParams; result.kernel.extra = p.kernel.extra; break;
    case CU_GRAPH_NODE_TYPE_MEMCPY:
        context(p.memcpy.copyCtx); result.memcpy.copyParams = runtime_graph_copy_params(p.memcpy.copyParams); break;
    case CU_GRAPH_NODE_TYPE_MEMSET:
        context(p.memset.ctx); result.memset.dst = reinterpret_cast<void*>(uintptr_t(p.memset.dst));
        result.memset.pitch = p.memset.pitch; result.memset.value = p.memset.value; result.memset.elementSize = p.memset.elementSize;
        result.memset.width = p.memset.width; result.memset.height = p.memset.height; break;
    case CU_GRAPH_NODE_TYPE_HOST:
        require(!p.host.ctx, CUDA_ERROR_NOT_SUPPORTED, "explicit Runtime host execution context queries are unavailable");
        result.host.fn = p.host.fn; result.host.userData = p.host.userData; break;
    case CU_GRAPH_NODE_TYPE_GRAPH:
        result.graph.graph = graph_cast<cudaGraph_t>(p.graph.graph);
        result.graph.ownership = cudaGraphChildGraphOwnershipInvalid; break;
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD: result.eventRecord.event = graph_cast<cudaEvent_t>(p.eventRecord.event); break;
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: result.eventWait.event = graph_cast<cudaEvent_t>(p.eventWait.event); break;
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL:
        context(p.extSemSignal.ctx);
        static_assert(sizeof(cudaExternalSemaphoreSignalParams) == sizeof(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS));
        result.extSemSignal.extSemArray = graph_cast<cudaExternalSemaphore_t*>(p.extSemSignal.extSemArray);
        result.extSemSignal.paramsArray = reinterpret_cast<const cudaExternalSemaphoreSignalParams*>(p.extSemSignal.paramsArray);
        result.extSemSignal.numExtSems = p.extSemSignal.numExtSems; break;
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT:
        context(p.extSemWait.ctx);
        static_assert(sizeof(cudaExternalSemaphoreWaitParams) == sizeof(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS));
        result.extSemWait.extSemArray = graph_cast<cudaExternalSemaphore_t*>(p.extSemWait.extSemArray);
        result.extSemWait.paramsArray = reinterpret_cast<const cudaExternalSemaphoreWaitParams*>(p.extSemWait.paramsArray);
        result.extSemWait.numExtSems = p.extSemWait.numExtSems; break;
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "Runtime graph parameter query for this node type is not implemented");
    }
    return result;
}
}
RUNTIME_API(cudaGraphKernelNodeGetParams, (cudaGraphNode_t handle, cudaKernelNodeParams* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph kernel output"); runtime_current(); CUDA_KERNEL_NODE_PARAMS params{};
    check(cuGraphKernelNodeGetParams_v2(graph_cast<CUgraphNode>(handle), &params)); *output = runtime_graph_kernel_parameters(params))
RUNTIME_API(cudaGraphNodeGetParams, (cudaGraphNode_t handle, cudaGraphNodeParams* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graph node output"); auto owner = runtime_current(); CUgraphNodeParams params{};
    check(cuGraphNodeGetParams(graph_cast<CUgraphNode>(handle), &params)); *output = runtime_graph_node_parameters(params, owner))
