RUNTIME_API(cudaStreamCreate, (cudaStream_t* output), require(output, CUDA_ERROR_INVALID_VALUE, "null stream output"); runtime_current(); CUstream stream = nullptr; check(cuStreamCreate(&stream, 0)); *output = reinterpret_cast<cudaStream_t>(stream))
RUNTIME_API(cudaStreamCreateWithFlags, (cudaStream_t* output, unsigned flags), require(output, CUDA_ERROR_INVALID_VALUE, "null stream output"); runtime_current(); CUstream stream = nullptr; check(cuStreamCreate(&stream, flags)); *output = reinterpret_cast<cudaStream_t>(stream))
RUNTIME_API(cudaStreamCreateWithPriority, (cudaStream_t* output, unsigned flags, int priority), require(output, CUDA_ERROR_INVALID_VALUE, "null stream output"); runtime_current(); CUstream stream = nullptr; check(cuStreamCreateWithPriority(&stream, flags, priority)); *output = reinterpret_cast<cudaStream_t>(stream))
RUNTIME_API(cudaDeviceGetStreamPriorityRange, (int* least, int* greatest), runtime_current(); check(cuCtxGetStreamPriorityRange(least, greatest)))
RUNTIME_API(cudaStreamGetPriority, (cudaStream_t stream, int* priority), runtime_current(); check(cuStreamGetPriority(reinterpret_cast<CUstream>(stream), priority)))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaStreamGetPriority_ptsz(cudaStream_t stream, int* priority) {
    return cudaStreamGetPriority(stream ? stream : cudaStreamPerThread, priority);
}
RUNTIME_API(cudaStreamGetFlags, (cudaStream_t stream, unsigned* flags), runtime_current(); check(cuStreamGetFlags(reinterpret_cast<CUstream>(stream), flags)))
RUNTIME_API(cudaStreamGetAttribute, (cudaStream_t stream, cudaStreamAttrID attribute, cudaStreamAttrValue* output),
    require(output, CUDA_ERROR_INVALID_VALUE, "null stream attribute output"); runtime_current(); CUstreamAttrValue value{};
    check(cuStreamGetAttribute(reinterpret_cast<CUstream>(stream), static_cast<CUstreamAttrID>(attribute), &value)); *output = runtime_launch_value(attribute, value))
RUNTIME_API(cudaStreamSetAttribute, (cudaStream_t stream, cudaStreamAttrID attribute, const cudaStreamAttrValue* input),
    require(input, CUDA_ERROR_INVALID_VALUE, "null stream attribute value"); runtime_current(); auto value = driver_launch_value(attribute, *input);
    check(cuStreamSetAttribute(reinterpret_cast<CUstream>(stream), static_cast<CUstreamAttrID>(attribute), &value)))
RUNTIME_API(cudaStreamCopyAttributes, (cudaStream_t destination, cudaStream_t source), runtime_current(); check(cuStreamCopyAttributes(reinterpret_cast<CUstream>(destination), reinterpret_cast<CUstream>(source))))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaStreamGetAttribute_ptsz(cudaStream_t stream, cudaStreamAttrID attribute, cudaStreamAttrValue* output) {
    return cudaStreamGetAttribute(stream ? stream : cudaStreamPerThread, attribute, output);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaStreamSetAttribute_ptsz(cudaStream_t stream, cudaStreamAttrID attribute, const cudaStreamAttrValue* value) {
    return cudaStreamSetAttribute(stream ? stream : cudaStreamPerThread, attribute, value);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaStreamCopyAttributes_ptsz(cudaStream_t destination, cudaStream_t source) {
    return cudaStreamCopyAttributes(destination ? destination : cudaStreamPerThread, source ? source : cudaStreamPerThread);
}
RUNTIME_API(cudaStreamGetId, (cudaStream_t stream, unsigned long long* id), runtime_current(); check(cuStreamGetId(reinterpret_cast<CUstream>(stream), id)))
RUNTIME_API(cudaStreamGetDevice, (cudaStream_t stream, int* device), runtime_current(); check(cuStreamGetDevice(reinterpret_cast<CUstream>(stream), device)))
RUNTIME_API(cudaStreamQuery, (cudaStream_t stream), runtime_current(); check(cuStreamQuery(reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaStreamSynchronize, (cudaStream_t stream), runtime_current(); check(runtime_driver_wait([&] { return cuStreamSynchronize(reinterpret_cast<CUstream>(stream)); })))
RUNTIME_API(cudaStreamDestroy, (cudaStream_t stream), runtime_current(); check(cuStreamDestroy(reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaStreamWaitEvent, (cudaStream_t stream, cudaEvent_t event, unsigned flags), runtime_current(); check(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream), reinterpret_cast<CUevent>(event), flags)))
RUNTIME_API(cudaEventCreate, (cudaEvent_t* output), require(output, CUDA_ERROR_INVALID_VALUE, "null event output"); runtime_current(); CUevent event = nullptr; check(cuEventCreate(&event, 0)); *output = reinterpret_cast<cudaEvent_t>(event))
RUNTIME_API(cudaEventCreateWithFlags, (cudaEvent_t* output, unsigned flags), require(output, CUDA_ERROR_INVALID_VALUE, "null event output"); runtime_current(); CUevent event = nullptr; check(cuEventCreate(&event, flags)); *output = reinterpret_cast<cudaEvent_t>(event))
RUNTIME_API(cudaEventRecord, (cudaEvent_t event, cudaStream_t stream), runtime_current(); check(cuEventRecord(reinterpret_cast<CUevent>(event), reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaEventRecordWithFlags, (cudaEvent_t event, cudaStream_t stream, unsigned flags), runtime_current(); check(cuEventRecordWithFlags(reinterpret_cast<CUevent>(event), reinterpret_cast<CUstream>(stream), flags)))
RUNTIME_API(cudaEventQuery, (cudaEvent_t event), runtime_current(); check(cuEventQuery(reinterpret_cast<CUevent>(event))))
RUNTIME_API(cudaEventSynchronize, (cudaEvent_t event), runtime_current(); check(runtime_driver_wait([&] { return cuEventSynchronize(reinterpret_cast<CUevent>(event)); })))
RUNTIME_API(cudaEventDestroy, (cudaEvent_t event), runtime_current(); check(cuEventDestroy(reinterpret_cast<CUevent>(event))))
RUNTIME_API(cudaEventElapsedTime, (float* output, cudaEvent_t start, cudaEvent_t end), runtime_current(); check(cuEventElapsedTime(output, reinterpret_cast<CUevent>(start), reinterpret_cast<CUevent>(end))))
RUNTIME_API(cudaLaunchHostFunc, (cudaStream_t stream, cudaHostFn_t function, void* data), runtime_current(); check(cuLaunchHostFunc(reinterpret_cast<CUstream>(stream), function, data)))
RUNTIME_API(cudaStreamAddCallback, (cudaStream_t stream, cudaStreamCallback_t callback, void* data, unsigned flags),
    runtime_stream_callback(stream, stream, callback, data, flags))
RUNTIME_API(cudaStreamAddCallback_ptsz, (cudaStream_t stream, cudaStreamCallback_t callback, void* data, unsigned flags),
    runtime_stream_callback(stream ? stream : cudaStreamPerThread, stream, callback, data, flags))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream) {
    return cudaStreamQuery(stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream) {
    return cudaStreamSynchronize(stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags) {
    return cudaStreamWaitEvent(stream ? stream : cudaStreamPerThread, event, flags);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream) {
    return cudaEventRecord(event, stream ? stream : cudaStreamPerThread);
}
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaLaunchHostFunc_ptsz(cudaStream_t stream, cudaHostFn_t function, void* data) {
    return cudaLaunchHostFunc(stream ? stream : cudaStreamPerThread, function, data);
}
