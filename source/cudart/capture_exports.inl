RUNTIME_API(cudaStreamBeginCapture, (cudaStream_t stream, cudaStreamCaptureMode mode), runtime_current(); check(cuStreamBeginCapture(graph_cast<CUstream>(stream), static_cast<CUstreamCaptureMode>(mode))))
RUNTIME_API(cudaStreamBeginCaptureToGraph, (cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* data, size_t count, cudaStreamCaptureMode mode),
    runtime_current(); auto nodes = graph_dependencies(dependencies, count); auto edges = graph_edge_data(data, count);
    check(cuStreamBeginCaptureToGraph(graph_cast<CUstream>(stream), graph_cast<CUgraph>(graph), nodes.data(), data ? edges.data() : nullptr, count, static_cast<CUstreamCaptureMode>(mode))))
RUNTIME_API(cudaStreamEndCapture, (cudaStream_t stream, cudaGraph_t* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null captured graph output"); runtime_current(); *output = nullptr; CUgraph graph{};
    check(cuStreamEndCapture(graph_cast<CUstream>(stream), &graph)); *output = graph_cast<cudaGraph_t>(graph))
RUNTIME_API(cudaStreamIsCapturing, (cudaStream_t stream, cudaStreamCaptureStatus* status),
    require(status, CUDA_ERROR_INVALID_VALUE, "null capture status output"); runtime_current(); CUstreamCaptureStatus value{};
    check(cuStreamIsCapturing(graph_cast<CUstream>(stream), &value)); *status = static_cast<cudaStreamCaptureStatus>(value))
RUNTIME_API(cudaStreamGetCaptureInfo, (cudaStream_t stream, cudaStreamCaptureStatus* status, unsigned long long* id, cudaGraph_t* graph, const cudaGraphNode_t** dependencies, const cudaGraphEdgeData** data, size_t* count),
    require(status && (!data || dependencies), CUDA_ERROR_INVALID_VALUE, "invalid capture info outputs"); runtime_current(); CUstreamCaptureStatus value{}; CUgraph captured{};
    const CUgraphNode* nodes{}; const CUgraphEdgeData* edges{}; cuuint64_t identity{};
    check(cuStreamGetCaptureInfo(graph_cast<CUstream>(stream), &value, &identity, &captured, dependencies ? &nodes : nullptr, data ? &edges : nullptr, count)); *status = static_cast<cudaStreamCaptureStatus>(value);
    if (value == CU_STREAM_CAPTURE_STATUS_ACTIVE) { if (id) *id = identity; if (graph) *graph = graph_cast<cudaGraph_t>(captured);
        if (dependencies) *dependencies = reinterpret_cast<const cudaGraphNode_t*>(nodes); if (data) *data = reinterpret_cast<const cudaGraphEdgeData*>(edges); })
RUNTIME_API(cudaStreamUpdateCaptureDependencies, (cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* data, size_t count, unsigned flags),
    runtime_current(); auto nodes = graph_dependencies(dependencies, count); auto edges = graph_edge_data(data, count);
    check(cuStreamUpdateCaptureDependencies(graph_cast<CUstream>(stream), nodes.data(), data ? edges.data() : nullptr, count, flags)))
RUNTIME_API(cudaThreadExchangeStreamCaptureMode, (cudaStreamCaptureMode* mode),
    require(mode, CUDA_ERROR_INVALID_VALUE, "null capture mode"); auto value = static_cast<CUstreamCaptureMode>(*mode);
    check(cuThreadExchangeStreamCaptureMode(&value)); *mode = static_cast<cudaStreamCaptureMode>(value))
RUNTIME_API(cudaStreamBeginCapture_ptsz, (cudaStream_t stream, cudaStreamCaptureMode mode), runtime_current(); check(cuStreamBeginCapture(graph_cast<CUstream>(stream ? stream : cudaStreamPerThread), static_cast<CUstreamCaptureMode>(mode))))
RUNTIME_API(cudaStreamEndCapture_ptsz, (cudaStream_t stream, cudaGraph_t* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null captured graph output"); runtime_current(); *output = nullptr; CUgraph graph{};
    check(cuStreamEndCapture(graph_cast<CUstream>(stream ? stream : cudaStreamPerThread), &graph)); *output = graph_cast<cudaGraph_t>(graph))
RUNTIME_API(cudaStreamIsCapturing_ptsz, (cudaStream_t stream, cudaStreamCaptureStatus* status),
    require(status, CUDA_ERROR_INVALID_VALUE, "null capture status output"); runtime_current(); CUstreamCaptureStatus value{};
    check(cuStreamIsCapturing(graph_cast<CUstream>(stream ? stream : cudaStreamPerThread), &value)); *status = static_cast<cudaStreamCaptureStatus>(value))
RUNTIME_API(cudaStreamBeginCaptureToGraph_ptsz, (cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* data, size_t count, cudaStreamCaptureMode mode),
    runtime_current(); auto nodes = graph_dependencies(dependencies, count); auto edges = graph_edge_data(data, count);
    check(cuStreamBeginCaptureToGraph(graph_cast<CUstream>(stream ? stream : cudaStreamPerThread), graph_cast<CUgraph>(graph), nodes.data(), data ? edges.data() : nullptr, count, static_cast<CUstreamCaptureMode>(mode))))
RUNTIME_API(cudaStreamGetCaptureInfo_ptsz, (cudaStream_t stream, cudaStreamCaptureStatus* status, unsigned long long* id, cudaGraph_t* graph, const cudaGraphNode_t** dependencies, const cudaGraphEdgeData** data, size_t* count),
    require(status && (!data || dependencies), CUDA_ERROR_INVALID_VALUE, "invalid capture info outputs"); runtime_current(); CUstreamCaptureStatus value{}; CUgraph captured{};
    const CUgraphNode* nodes{}; const CUgraphEdgeData* edges{}; cuuint64_t identity{};
    check(cuStreamGetCaptureInfo(graph_cast<CUstream>(stream ? stream : cudaStreamPerThread), &value, &identity, &captured, dependencies ? &nodes : nullptr, data ? &edges : nullptr, count)); *status = static_cast<cudaStreamCaptureStatus>(value);
    if (value == CU_STREAM_CAPTURE_STATUS_ACTIVE) { if (id) *id = identity; if (graph) *graph = graph_cast<cudaGraph_t>(captured);
        if (dependencies) *dependencies = reinterpret_cast<const cudaGraphNode_t*>(nodes); if (data) *data = reinterpret_cast<const cudaGraphEdgeData*>(edges); })
RUNTIME_API(cudaStreamUpdateCaptureDependencies_ptsz, (cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* data, size_t count, unsigned flags),
    runtime_current(); auto nodes = graph_dependencies(dependencies, count); auto edges = graph_edge_data(data, count);
    check(cuStreamUpdateCaptureDependencies(graph_cast<CUstream>(stream ? stream : cudaStreamPerThread), nodes.data(), data ? edges.data() : nullptr, count, flags)))
