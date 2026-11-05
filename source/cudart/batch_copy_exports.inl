RUNTIME_API(cudaMemcpyBatchAsync, (void* const* destinations, const void* const* sources, const size_t* sizes, size_t count,
    cudaMemcpyAttributes* attributes, size_t* indices, size_t attribute_count, cudaStream_t stream),
    runtime_current();
    require(!count || (destinations && sources && sizes), CUDA_ERROR_INVALID_VALUE, "missing batch copy arrays");
    require(attribute_count <= count && (!count || attribute_count) && (!attribute_count || (attributes && indices)),
            CUDA_ERROR_INVALID_VALUE, "missing batch copy attributes");
    std::vector<CUdeviceptr> to(count), from(count); std::vector<size_t> bytes(count);
    for (size_t i = 0; i < count; ++i) { to[i] = reinterpret_cast<uintptr_t>(destinations[i]); from[i] = reinterpret_cast<uintptr_t>(sources[i]); bytes[i] = sizes[i]; }
    std::vector<CUmemcpyAttributes> values(attribute_count);
    for (size_t i = 0; i < attribute_count; ++i) values[i] = runtime_memcpy_attributes(attributes[i]);
    check(runtime_driver_wait([&] { return cuMemcpyBatchAsync(to.data(), from.data(), bytes.data(), count, values.data(), indices, attribute_count, reinterpret_cast<CUstream>(stream)); })))

extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpyBatchAsync_ptsz(void* const* destinations, const void* const* sources,
    const size_t* sizes, size_t count, cudaMemcpyAttributes* attributes, size_t* indices, size_t attribute_count, cudaStream_t stream) {
    return cudaMemcpyBatchAsync(destinations, sources, sizes, count, attributes, indices, attribute_count, stream ? stream : cudaStreamPerThread);
}
RUNTIME_API(cudaMemcpyWithAttributesAsync, (void* destination, const void* source, size_t bytes, cudaMemcpyAttributes* attributes, cudaStream_t stream),
    runtime_current(); require(attributes, CUDA_ERROR_INVALID_VALUE, "missing copy attributes");
    auto native = runtime_memcpy_attributes(*attributes);
    check(runtime_driver_wait([&] { return cuMemcpyWithAttributesAsync(reinterpret_cast<uintptr_t>(destination), reinterpret_cast<uintptr_t>(source), bytes, &native, reinterpret_cast<CUstream>(stream)); })))
extern "C" CUDAVM_CUDART_API cudaError_t CUDARTAPI cudaMemcpyWithAttributesAsync_ptsz(void* destination, const void* source,
    size_t bytes, cudaMemcpyAttributes* attributes, cudaStream_t stream) {
    return cudaMemcpyWithAttributesAsync(destination, source, bytes, attributes, stream ? stream : cudaStreamPerThread);
}
