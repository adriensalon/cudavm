DRIVER_API(cuStreamBeginCapture, (CUstream stream, CUstreamCaptureMode mode), driver_ready(); capture_begin(stream, mode))
DRIVER_API(cuStreamBeginCaptureToGraph, (CUstream stream, CUgraph graph, const CUgraphNode* dependencies, const CUgraphEdgeData* data, size_t count, CUstreamCaptureMode mode),
    driver_ready(); require(graph, CUDA_ERROR_INVALID_VALUE, "null capture graph"); capture_begin(stream, mode, graph, dependencies, data, count))
DRIVER_API(cuStreamEndCapture, (CUstream stream, CUgraph* output), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null captured graph output"); *output = nullptr; *output = capture_end(stream))
DRIVER_API(cuStreamIsCapturing, (CUstream stream, CUstreamCaptureStatus* status), driver_ready(); capture_info(stream, status, nullptr, nullptr, nullptr, nullptr, nullptr))
DRIVER_API(cuStreamGetCaptureInfo, (CUstream stream, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph, const CUgraphNode** dependencies, const CUgraphEdgeData** data, size_t* count),
    driver_ready(); capture_info(stream, status, id, graph, dependencies, data, count))
DRIVER_API(cuStreamUpdateCaptureDependencies, (CUstream stream, CUgraphNode* dependencies, const CUgraphEdgeData* data, size_t count, unsigned flags),
    driver_ready(); capture_update_dependencies(stream, dependencies, data, count, flags))
DRIVER_API(cuThreadExchangeStreamCaptureMode, (CUstreamCaptureMode* mode),
    require(mode && *mode >= CU_STREAM_CAPTURE_MODE_GLOBAL && *mode <= CU_STREAM_CAPTURE_MODE_RELAXED, CUDA_ERROR_INVALID_VALUE, "invalid thread capture mode"); std::swap(*mode, thread_capture_mode))
DRIVER_API(cuStreamBeginCapture_v2_ptsz, (CUstream stream, CUstreamCaptureMode mode), driver_ready(); capture_begin(stream ? stream : CU_STREAM_PER_THREAD, mode))
DRIVER_API(cuStreamBeginCaptureToGraph_ptsz, (CUstream stream, CUgraph graph, const CUgraphNode* dependencies, const CUgraphEdgeData* data, size_t count, CUstreamCaptureMode mode),
    driver_ready(); require(graph, CUDA_ERROR_INVALID_VALUE, "null capture graph"); capture_begin(stream ? stream : CU_STREAM_PER_THREAD, mode, graph, dependencies, data, count))
DRIVER_API(cuStreamEndCapture_ptsz, (CUstream stream, CUgraph* output), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null captured graph output"); *output = nullptr; *output = capture_end(stream ? stream : CU_STREAM_PER_THREAD))
DRIVER_API(cuStreamIsCapturing_ptsz, (CUstream stream, CUstreamCaptureStatus* status), driver_ready(); capture_info(stream ? stream : CU_STREAM_PER_THREAD, status, nullptr, nullptr, nullptr, nullptr, nullptr))
DRIVER_API(cuStreamGetCaptureInfo_v3_ptsz, (CUstream stream, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph, const CUgraphNode** dependencies, const CUgraphEdgeData** data, size_t* count),
    driver_ready(); capture_info(stream ? stream : CU_STREAM_PER_THREAD, status, id, graph, dependencies, data, count))
DRIVER_API(cuStreamUpdateCaptureDependencies_v2_ptsz, (CUstream stream, CUgraphNode* dependencies, const CUgraphEdgeData* data, size_t count, unsigned flags),
    driver_ready(); capture_update_dependencies(stream ? stream : CU_STREAM_PER_THREAD, dependencies, data, count, flags))
#undef cuStreamBeginCapture
#undef cuStreamGetCaptureInfo
#undef cuStreamUpdateCaptureDependencies
DRIVER_API(cuStreamBeginCapture, (CUstream stream), driver_ready(); capture_begin(stream, CU_STREAM_CAPTURE_MODE_GLOBAL))
DRIVER_API(cuStreamBeginCapture_ptsz, (CUstream stream), driver_ready(); capture_begin(stream ? stream : CU_STREAM_PER_THREAD, CU_STREAM_CAPTURE_MODE_GLOBAL))
DRIVER_API(cuStreamGetCaptureInfo, (CUstream stream, CUstreamCaptureStatus* status, cuuint64_t* id), driver_ready(); capture_info(stream, status, id, nullptr, nullptr, nullptr, nullptr))
DRIVER_API(cuStreamGetCaptureInfo_v2, (CUstream stream, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph, const CUgraphNode** dependencies, size_t* count),
    driver_ready(); capture_info(stream, status, id, graph, dependencies, nullptr, count))
DRIVER_API(cuStreamGetCaptureInfo_ptsz, (CUstream stream, CUstreamCaptureStatus* status, cuuint64_t* id), driver_ready(); capture_info(stream ? stream : CU_STREAM_PER_THREAD, status, id, nullptr, nullptr, nullptr, nullptr))
DRIVER_API(cuStreamGetCaptureInfo_v2_ptsz, (CUstream stream, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph, const CUgraphNode** dependencies, size_t* count),
    driver_ready(); capture_info(stream ? stream : CU_STREAM_PER_THREAD, status, id, graph, dependencies, nullptr, count))
DRIVER_API(cuStreamUpdateCaptureDependencies, (CUstream stream, CUgraphNode* dependencies, size_t count, unsigned flags), driver_ready(); capture_update_dependencies(stream, dependencies, nullptr, count, flags))
DRIVER_API(cuStreamUpdateCaptureDependencies_ptsz, (CUstream stream, CUgraphNode* dependencies, size_t count, unsigned flags), driver_ready(); capture_update_dependencies(stream ? stream : CU_STREAM_PER_THREAD, dependencies, nullptr, count, flags))
