// Runtime composes the public Driver interop API; handles share one registry.
extern "C" CUresult CUDAAPI cuGraphicsGLRegisterBuffer(CUgraphicsResource*, unsigned, unsigned);
extern "C" CUresult CUDAAPI cuGraphicsGLRegisterImage(CUgraphicsResource*, unsigned, unsigned, unsigned);
namespace {
template<class Driver, class Runtime> Driver interop_descriptor(const Runtime& input) {
    static_assert(sizeof(Driver) == sizeof(Runtime));
    static_assert(alignof(Driver) == alignof(Runtime));
    Driver result{}; std::memcpy(&result, &input, sizeof(result)); return result;
}
std::vector<CUgraphicsResource> graphics_handles(int count, cudaGraphicsResource_t* resources) {
    require(count >= 0 && (!count || resources), CUDA_ERROR_INVALID_VALUE, "invalid graphics resource array");
    std::vector<CUgraphicsResource> result;
    result.reserve(size_t(count));
    for (int i = 0; i < count; ++i) result.push_back(reinterpret_cast<CUgraphicsResource>(resources[i]));
    return result;
}
}
RUNTIME_API(cudaImportExternalMemory, (cudaExternalMemory_t* output, const cudaExternalMemoryHandleDesc* descriptor),
    require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external memory arguments"); runtime_current();
    auto d = interop_descriptor<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>(*descriptor); CUexternalMemory value{};
    check(cuImportExternalMemory(&value, &d)); *output = reinterpret_cast<cudaExternalMemory_t>(value))
RUNTIME_API(cudaExternalMemoryGetMappedBuffer, (void** output, cudaExternalMemory_t handle, const cudaExternalMemoryBufferDesc* descriptor),
    require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external buffer arguments"); runtime_current();
    auto d = interop_descriptor<CUDA_EXTERNAL_MEMORY_BUFFER_DESC>(*descriptor); CUdeviceptr value = 0;
    check(cuExternalMemoryGetMappedBuffer(&value, reinterpret_cast<CUexternalMemory>(handle), &d)); *output = reinterpret_cast<void*>(uintptr_t(value)))
RUNTIME_API(cudaDestroyExternalMemory, (cudaExternalMemory_t handle), runtime_current(); check(runtime_driver_wait([&] { return cuDestroyExternalMemory(reinterpret_cast<CUexternalMemory>(handle)); })))
RUNTIME_API(cudaExternalMemoryGetMappedMipmappedArray, (cudaMipmappedArray_t* output, cudaExternalMemory_t handle, const cudaExternalMemoryMipmappedArrayDesc* descriptor),
    require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external image arguments"); runtime_current();
    throw failure(CUDA_ERROR_NOT_SUPPORTED, "external images are not implemented"); (void)handle)
RUNTIME_API(cudaImportExternalSemaphore, (cudaExternalSemaphore_t* output, const cudaExternalSemaphoreHandleDesc* descriptor),
    require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external semaphore arguments"); runtime_current();
    auto d = interop_descriptor<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>(*descriptor); CUexternalSemaphore value{};
    check(cuImportExternalSemaphore(&value, &d)); *output = reinterpret_cast<cudaExternalSemaphore_t>(value))
RUNTIME_API(cudaDestroyExternalSemaphore, (cudaExternalSemaphore_t handle), runtime_current(); check(runtime_driver_wait([&] { return cuDestroyExternalSemaphore(reinterpret_cast<CUexternalSemaphore>(handle)); })))
RUNTIME_API(cudaSignalExternalSemaphoresAsync, (const cudaExternalSemaphore_t* handles, const cudaExternalSemaphoreSignalParams* params, unsigned count, cudaStream_t stream),
    require(count && handles && params, CUDA_ERROR_INVALID_VALUE, "external semaphore submissions require nonempty arrays"); runtime_current();
    std::vector<CUexternalSemaphore> values; std::vector<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> descriptors;
    for (unsigned i = 0; i < count; ++i) { values.push_back(reinterpret_cast<CUexternalSemaphore>(handles[i])); descriptors.push_back(interop_descriptor<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>(params[i])); }
    check(cuSignalExternalSemaphoresAsync(values.data(), descriptors.data(), count, reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaWaitExternalSemaphoresAsync, (const cudaExternalSemaphore_t* handles, const cudaExternalSemaphoreWaitParams* params, unsigned count, cudaStream_t stream),
    require(count && handles && params, CUDA_ERROR_INVALID_VALUE, "external semaphore submissions require nonempty arrays"); runtime_current();
    std::vector<CUexternalSemaphore> values; std::vector<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> descriptors;
    for (unsigned i = 0; i < count; ++i) { values.push_back(reinterpret_cast<CUexternalSemaphore>(handles[i])); descriptors.push_back(interop_descriptor<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS>(params[i])); }
    check(cuWaitExternalSemaphoresAsync(values.data(), descriptors.data(), count, reinterpret_cast<CUstream>(stream))))
RUNTIME_API(cudaGraphicsGLRegisterBuffer, (cudaGraphicsResource_t* output, unsigned object, unsigned flags),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graphics resource output"); runtime_current(); CUgraphicsResource value{};
    check(cuGraphicsGLRegisterBuffer(&value, object, flags)); *output = reinterpret_cast<cudaGraphicsResource_t>(value))
RUNTIME_API(cudaGraphicsGLRegisterImage, (cudaGraphicsResource_t* output, unsigned object, unsigned target, unsigned flags),
    require(output, CUDA_ERROR_INVALID_VALUE, "null graphics resource output"); runtime_current(); CUgraphicsResource value{};
    check(cuGraphicsGLRegisterImage(&value, object, target, flags)); *output = reinterpret_cast<cudaGraphicsResource_t>(value))
RUNTIME_API(cudaGraphicsUnregisterResource, (cudaGraphicsResource_t handle), runtime_current(); check(cuGraphicsUnregisterResource(reinterpret_cast<CUgraphicsResource>(handle))))
RUNTIME_API(cudaGraphicsResourceSetMapFlags, (cudaGraphicsResource_t handle, unsigned flags), runtime_current(); check(cuGraphicsResourceSetMapFlags(reinterpret_cast<CUgraphicsResource>(handle), flags)))
RUNTIME_API(cudaGraphicsMapResources, (int count, cudaGraphicsResource_t* resources, cudaStream_t stream),
    runtime_current(); auto values = graphics_handles(count, resources); check(runtime_driver_wait([&] { return cuGraphicsMapResources(unsigned(count), values.data(), reinterpret_cast<CUstream>(stream)); })))
RUNTIME_API(cudaGraphicsUnmapResources, (int count, cudaGraphicsResource_t* resources, cudaStream_t stream),
    runtime_current(); auto values = graphics_handles(count, resources); check(runtime_driver_wait([&] { return cuGraphicsUnmapResources(unsigned(count), values.data(), reinterpret_cast<CUstream>(stream)); })))
RUNTIME_API(cudaGraphicsResourceGetMappedPointer, (void** output, size_t* bytes, cudaGraphicsResource_t handle),
    require(output && bytes, CUDA_ERROR_INVALID_VALUE, "null mapped graphics pointer outputs"); runtime_current(); CUdeviceptr value = 0; size_t size = 0;
    check(cuGraphicsResourceGetMappedPointer(&value, &size, reinterpret_cast<CUgraphicsResource>(handle))); *output = reinterpret_cast<void*>(uintptr_t(value)); *bytes = size)
