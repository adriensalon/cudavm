RUNTIME_API(cudaMallocHost, (void** output, size_t bytes),
    require(output, CUDA_ERROR_INVALID_VALUE, "null host allocation output"); runtime_current();
    void* pointer = nullptr; if (bytes) check(cuMemAllocHost(&pointer, bytes)); *output = pointer)
RUNTIME_API(cudaHostAlloc, (void** output, size_t bytes, unsigned flags),
    require(output, CUDA_ERROR_INVALID_VALUE, "null host allocation output"); runtime_current();
    void* pointer = nullptr; check(cuMemHostAlloc(&pointer, bytes, flags)); *output = pointer)
RUNTIME_API(cudaFreeHost, (void* pointer), runtime_current(); check(runtime_driver_wait([&] { return cuMemFreeHost(pointer); })))
RUNTIME_API(cudaHostRegister, (void* pointer, size_t bytes, unsigned flags), runtime_current(); check(cuMemHostRegister(pointer, bytes, flags)))
RUNTIME_API(cudaHostUnregister, (void* pointer), runtime_current(); check(runtime_driver_wait([&] { return cuMemHostUnregister(pointer); })))
RUNTIME_API(cudaHostGetFlags, (unsigned* output, void* pointer), runtime_current(); check(cuMemHostGetFlags(output, pointer)))
RUNTIME_API(cudaHostGetDevicePointer, (void** output, void* pointer, unsigned flags), require(output, CUDA_ERROR_INVALID_VALUE, "null host device pointer output"); runtime_current(); CUdeviceptr value = 0; check(cuMemHostGetDevicePointer(&value, pointer, flags)); *output = reinterpret_cast<void*>(static_cast<uintptr_t>(value)))
