DRIVER_API(cuMemBatchDecompressAsync, (CUmemDecompressParams* records, size_t count, unsigned flags, size_t* error_index, CUstream stream),
    decompression_ready(error_index); async_decompress(records, count, flags, error_index, stream))
DRIVER_API(cuMemBatchDecompressAsync_ptsz, (CUmemDecompressParams* records, size_t count, unsigned flags, size_t* error_index, CUstream stream),
    decompression_ready(error_index); async_decompress(records, count, flags, error_index, stream ? stream : CU_STREAM_PER_THREAD))
