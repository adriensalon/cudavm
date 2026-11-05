DRIVER_API(cuMemcpyBatchAsync, (CUdeviceptr* destinations, CUdeviceptr* sources, size_t* sizes, size_t count,
    CUmemcpyAttributes* attributes, size_t* indices, size_t attribute_count, CUstream stream),
    driver_ready(); async_copy_batch(destinations, sources, sizes, count, attributes, indices, attribute_count, stream))
DRIVER_API(cuMemcpyBatchAsync_v2_ptsz, (CUdeviceptr* destinations, CUdeviceptr* sources, size_t* sizes, size_t count,
    CUmemcpyAttributes* attributes, size_t* indices, size_t attribute_count, CUstream stream),
    driver_ready(); async_copy_batch(destinations, sources, sizes, count, attributes, indices, attribute_count, stream ? stream : CU_STREAM_PER_THREAD))
DRIVER_API(cuMemcpyWithAttributesAsync, (CUdeviceptr destination, CUdeviceptr source, size_t bytes, CUmemcpyAttributes* attributes, CUstream stream),
    driver_ready(); size_t index = 0; async_copy_batch(&destination, &source, &bytes, 1, attributes, &index, 1, stream, true))
DRIVER_API(cuMemcpyWithAttributesAsync_ptsz, (CUdeviceptr destination, CUdeviceptr source, size_t bytes, CUmemcpyAttributes* attributes, CUstream stream),
    driver_ready(); size_t index = 0; async_copy_batch(&destination, &source, &bytes, 1, attributes, &index, 1, stream ? stream : CU_STREAM_PER_THREAD, true))
