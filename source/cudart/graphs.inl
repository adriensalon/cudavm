template<class To, class From> To graph_cast(From handle) { return reinterpret_cast<To>(handle); }
std::vector<CUgraphNode> graph_dependencies(const cudaGraphNode_t* nodes, size_t count) {
    require(!count || nodes, CUDA_ERROR_INVALID_VALUE, "null graph dependency array");
    std::vector<CUgraphNode> result; result.reserve(count);
    for (size_t i = 0; i < count; ++i) result.push_back(graph_cast<CUgraphNode>(nodes[i]));
    return result;
}
std::vector<CUgraphEdgeData> graph_edge_data(const cudaGraphEdgeData* data, size_t count) {
    static_assert(sizeof(cudaGraphEdgeData) == sizeof(CUgraphEdgeData));
    std::vector<CUgraphEdgeData> result;
    if (data) { result.resize(count); if (count) std::memcpy(result.data(), data, count * sizeof(CUgraphEdgeData)); }
    return result;
}
template<class F> void runtime_graph_query(cudaGraphNode_t* output, size_t* count, F function) {
    require(count, CUDA_ERROR_INVALID_VALUE, "null graph query count");
    if (!output) { check(function(nullptr, count)); return; }
    auto capacity = *count; std::vector<CUgraphNode> nodes(capacity);
    // data() can be null for zero capacity; use a non-null sentinel to retain
    // the distinction between count-only queries and zero-capacity queries.
    CUgraphNode sentinel{}; auto* data = capacity ? nodes.data() : &sentinel;
    auto written = capacity; check(function(data, &written));
    for (size_t i = 0; i < capacity; ++i) output[i] = graph_cast<cudaGraphNode_t>(nodes[i]);
    *count = written;
}
CUDA_MEMCPY3D runtime_graph_copy_1d(void* to, const void* from, size_t bytes, cudaMemcpyKind kind) {
    auto types = runtime_copy_types(to, from, kind);
    CUDA_MEMCPY3D p{};
    p.srcMemoryType = types.from; p.srcHost = from; p.srcDevice = reinterpret_cast<uintptr_t>(from); p.srcPitch = bytes;
    p.dstMemoryType = types.to; p.dstHost = to; p.dstDevice = reinterpret_cast<uintptr_t>(to); p.dstPitch = bytes;
    p.WidthInBytes = bytes; p.Height = p.Depth = 1; return p;
}
cudaMemcpy3DParms runtime_graph_copy_params(const CUDA_MEMCPY3D& p) {
    cudaMemcpy3DParms result{};
    result.srcPtr = {p.srcMemoryType == CU_MEMORYTYPE_HOST ? const_cast<void*>(p.srcHost) : reinterpret_cast<void*>(uintptr_t(p.srcDevice)), p.srcPitch, p.srcPitch, p.srcHeight ? p.srcHeight : p.Height};
    result.dstPtr = {p.dstMemoryType == CU_MEMORYTYPE_HOST ? p.dstHost : reinterpret_cast<void*>(uintptr_t(p.dstDevice)), p.dstPitch, p.dstPitch, p.dstHeight ? p.dstHeight : p.Height};
    result.srcPos = {p.srcXInBytes, p.srcY, p.srcZ}; result.dstPos = {p.dstXInBytes, p.dstY, p.dstZ};
    result.extent = {p.WidthInBytes, p.Height, p.Depth};
    size_t element = 1;
    if (p.srcMemoryType == CU_MEMORYTYPE_ARRAY) { result.srcArray = reinterpret_cast<cudaArray_t>(p.srcArray); result.srcPtr = {}; element = runtime_array_element(result.srcArray); result.srcPos.x /= element; }
    if (p.dstMemoryType == CU_MEMORYTYPE_ARRAY) { result.dstArray = reinterpret_cast<cudaArray_t>(p.dstArray); result.dstPtr = {}; element = runtime_array_element(result.dstArray); result.dstPos.x /= element; }
    result.extent.width /= element;
    // Query the description, including after storage is released. Inferring
    // from today's allocation registry would misclassify an old device address.
    bool to_device = p.dstMemoryType == CU_MEMORYTYPE_ARRAY || p.dstMemoryType == CU_MEMORYTYPE_DEVICE || (p.dstMemoryType == CU_MEMORYTYPE_UNIFIED && cudavmDriverIsDevicePointer(p.dstDevice));
    bool from_device = p.srcMemoryType == CU_MEMORYTYPE_ARRAY || p.srcMemoryType == CU_MEMORYTYPE_DEVICE || (p.srcMemoryType == CU_MEMORYTYPE_UNIFIED && cudavmDriverIsDevicePointer(p.srcDevice));
    result.kind = to_device ? (from_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                            : (from_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    return result;
}
CUDA_MEMSET_NODE_PARAMS runtime_graph_fill(const cudaMemsetParams& p) {
    CUDA_MEMSET_NODE_PARAMS result{}; result.dst = reinterpret_cast<uintptr_t>(p.dst);
    result.pitch = p.pitch; result.value = p.value; result.elementSize = p.elementSize; result.width = p.width; result.height = p.height; return result;
}
CUDA_KERNEL_NODE_PARAMS runtime_graph_kernel(const cudaKernelNodeParams& p, CUcontext owner) {
    if (cudavmDriverIsLibraryKernel(p.func)) {
        CUDA_KERNEL_NODE_PARAMS result{}; result.kern = reinterpret_cast<CUkernel>(p.func); result.ctx = owner;
        result.gridDimX = p.gridDim.x; result.gridDimY = p.gridDim.y; result.gridDimZ = p.gridDim.z;
        result.blockDimX = p.blockDim.x; result.blockDimY = p.blockDim.y; result.blockDimZ = p.blockDim.z;
        result.sharedMemBytes = p.sharedMemBytes; result.kernelParams = p.kernelParams; result.extra = p.extra; return result;
    }
    auto it = global().bindings.find(p.func);
    symbol_require(p.func && it != global().bindings.end(), cudaErrorInvalidDeviceFunction, "graph kernel is not registered with this cudavm runtime");
    CUDA_KERNEL_NODE_PARAMS result{}; result.func = runtime_function(it->second, owner);
    result.gridDimX = p.gridDim.x; result.gridDimY = p.gridDim.y; result.gridDimZ = p.gridDim.z;
    result.blockDimX = p.blockDim.x; result.blockDimY = p.blockDim.y; result.blockDimZ = p.blockDim.z;
    result.sharedMemBytes = p.sharedMemBytes; result.kernelParams = p.kernelParams; result.extra = p.extra; return result;
}
