RUNTIME_API(cudaMemAdvise, (const void* pointer, size_t bytes, cudaMemoryAdvise advice, cudaMemLocation location),
    runtime_current(); CUmemLocation native{}; native.type = static_cast<CUmemLocationType>(location.type); native.id = location.id;
    check(runtime_driver_wait([&] { return cuMemAdvise(reinterpret_cast<uintptr_t>(pointer), bytes, static_cast<CUmem_advise>(advice), native); })))
RUNTIME_API(cudaMemPrefetchAsync, (const void* pointer, size_t bytes, cudaMemLocation location, unsigned flags, cudaStream_t stream),
    runtime_current(); CUmemLocation native{}; native.type = static_cast<CUmemLocationType>(location.type); native.id = location.id;
    check(cuMemPrefetchAsync(reinterpret_cast<uintptr_t>(pointer), bytes, native, flags, reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaMemPrefetchAsync_ptsz, (const void* pointer, size_t bytes, cudaMemLocation location, unsigned flags, cudaStream_t stream),
    runtime_current(); CUmemLocation native{}; native.type = static_cast<CUmemLocationType>(location.type); native.id = location.id;
    check(cuMemPrefetchAsync(reinterpret_cast<uintptr_t>(pointer), bytes, native, flags, stream ? reinterpret_cast<CUstream>(stream) : CU_STREAM_PER_THREAD)))
RUNTIME_API(cudaMemRangeGetAttribute, (void* output, size_t output_size, cudaMemRangeAttribute attribute, const void* pointer, size_t bytes),
    runtime_current(); check(cuMemRangeGetAttribute(output, output_size, static_cast<CUmem_range_attribute>(attribute), reinterpret_cast<uintptr_t>(pointer), bytes)))
RUNTIME_API(cudaMemRangeGetAttributes, (void** outputs, size_t* sizes, cudaMemRangeAttribute* attributes, size_t count, const void* pointer, size_t bytes),
    runtime_current(); require(!count || attributes, CUDA_ERROR_INVALID_VALUE, "missing managed range query attributes");
    std::vector<CUmem_range_attribute> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = static_cast<CUmem_range_attribute>(attributes[i]);
    check(cuMemRangeGetAttributes(outputs, sizes, values.data(), count, reinterpret_cast<uintptr_t>(pointer), bytes)))
