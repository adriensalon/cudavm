DRIVER_API(cuMemAdvise, (CUdeviceptr pointer, size_t bytes, CUmem_advise advice, CUmemLocation location),
    driver_ready(); managed_advise(pointer, bytes, advice, location))
DRIVER_API(cuMemPrefetchAsync_v2, (CUdeviceptr pointer, size_t bytes, CUmemLocation location, unsigned flags, CUstream stream),
    driver_ready(); managed_prefetch(pointer, bytes, location, flags, stream))
DRIVER_API(cuMemPrefetchAsync_v2_ptsz, (CUdeviceptr pointer, size_t bytes, CUmemLocation location, unsigned flags, CUstream stream),
    driver_ready(); managed_prefetch(pointer, bytes, location, flags, stream ? stream : CU_STREAM_PER_THREAD))
#pragma push_macro("cuMemPrefetchAsync")
#undef cuMemPrefetchAsync
DRIVER_API(cuMemPrefetchAsync, (CUdeviceptr pointer, size_t bytes, CUdevice device, CUstream stream),
    driver_ready(); CUmemLocation location{}; location.type = device == CU_DEVICE_CPU ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE;
    location.id = device; managed_prefetch(pointer, bytes, location, 0, stream))
DRIVER_API(cuMemPrefetchAsync_ptsz, (CUdeviceptr pointer, size_t bytes, CUdevice device, CUstream stream),
    driver_ready(); CUmemLocation location{}; location.type = device == CU_DEVICE_CPU ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE;
    location.id = device; managed_prefetch(pointer, bytes, location, 0, stream ? stream : CU_STREAM_PER_THREAD))
#pragma pop_macro("cuMemPrefetchAsync")
DRIVER_API(cuMemRangeGetAttribute, (void* output, size_t output_size, CUmem_range_attribute attribute, CUdeviceptr pointer, size_t bytes),
    driver_ready(); managed_attribute(output, output_size, attribute, pointer, bytes))
DRIVER_API(cuMemRangeGetAttributes, (void** outputs, size_t* sizes, CUmem_range_attribute* attributes, size_t count, CUdeviceptr pointer, size_t bytes),
    driver_ready(); require(!count || (outputs && sizes && attributes), CUDA_ERROR_INVALID_VALUE, "missing managed range query arrays");
    managed_advice_range(pointer, bytes);
    for (size_t i = 0; i < count; ++i) validate_managed_attribute(outputs[i], sizes[i], attributes[i]);
    for (size_t i = 0; i < count; ++i) managed_attribute(outputs[i], sizes[i], attributes[i], pointer, bytes))
