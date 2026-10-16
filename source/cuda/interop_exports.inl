extern "C" CUDAVM_API int cudavmGraphicsGL(const cudavmGraphicsGLDesc* descriptor) {
    return static_cast<int>(driver_call([&] {
        driver_ready(); auto owner = current();
        require(std::none_of(global().graphics.begin(), global().graphics.end(), [&](auto& value) { return value.second.owner == owner; }),
                CUDA_ERROR_NOT_PERMITTED, "unregister GL resources before replacing or detaching the bridge");
        auto bridge = descriptor ? std::make_shared<opengl_graphics>(*descriptor) : nullptr;
        owner->gl = std::move(bridge);
    }));
}
extern "C" CUDAVM_API int cudavmGetInteropCapabilities(cudavmInteropCapabilities* output) {
    return static_cast<int>(driver_call([&] {
        driver_ready(); require(output && output->abi_version == 1 && output->struct_size == sizeof(*output), CUDA_ERROR_INVALID_VALUE, "invalid interop capabilities ABI");
        auto owner = current(); auto caps = owner->native->interop();
        output->memory_handle_types = caps.memory; output->semaphore_handle_types = caps.semaphore;
        output->memory_metadata_required_types = caps.metadata;
        output->graphics_gl_buffers = owner->gl ? 1u : 0u;
    }));
}
extern "C" CUDAVM_API int cudavmImportExternalMemory(CUexternalMemory* output, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* descriptor, const cudavmExternalMemoryMetadata* metadata) {
    return static_cast<int>(driver_call([&] {
        driver_ready(); require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external memory arguments"); validate_memory(*descriptor);
        require(!metadata || (metadata->abi_version == 1 && metadata->struct_size == sizeof(*metadata)), CUDA_ERROR_INVALID_VALUE, "invalid external memory metadata ABI");
        auto owner = current(); auto native = owner->native->import_memory(*descriptor, metadata); auto handle = interop_handle<CUexternalMemory>();
        global().memories.emplace(handle, memory_entry{owner, std::move(native), descriptor->size}); *output = handle;
    }));
}
DRIVER_API(cuImportExternalMemory, (CUexternalMemory* output, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* descriptor),
    driver_ready(); require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external memory arguments"); validate_memory(*descriptor);
    auto owner = current(); auto native = owner->native->import_memory(*descriptor); auto handle = interop_handle<CUexternalMemory>();
    global().memories.emplace(handle, memory_entry{owner, std::move(native), descriptor->size}); *output = handle)
DRIVER_API(cuExternalMemoryGetMappedBuffer, (CUdeviceptr* output, CUexternalMemory handle, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* descriptor),
    driver_ready(); require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external buffer arguments"); auto owner = current();
    auto& memory = interop_lookup(global().memories, handle, owner); auto& d = *descriptor;
    require(!d.flags && zero_reserved(d.reserved) && d.size && d.offset <= memory.size && d.size <= memory.size - d.offset,
            CUDA_ERROR_INVALID_VALUE, "invalid external buffer range or flags");
    *output = add_allocation(owner, memory.native->map(d), nullptr, handle))
DRIVER_API(cuDestroyExternalMemory, (CUexternalMemory handle),
    driver_ready(); auto owner = current(); interop_lookup(global().memories, handle, owner);
    require(std::none_of(global().allocations.begin(), global().allocations.end(), [&](auto& value) { return value.second.external == handle; }),
            CUDA_ERROR_INVALID_HANDLE, "free mapped external buffers before destroying their memory object");
    synchronize_context(owner); global().memories.erase(handle))
DRIVER_API(cuExternalMemoryGetMappedMipmappedArray, (CUmipmappedArray* output, CUexternalMemory handle, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* descriptor),
    driver_ready(); require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external image arguments"); interop_lookup(global().memories, handle, current());
    throw failure(CUDA_ERROR_NOT_SUPPORTED, "external images are not implemented"))
DRIVER_API(cuImportExternalSemaphore, (CUexternalSemaphore* output, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* descriptor),
    driver_ready(); require(output && descriptor, CUDA_ERROR_INVALID_VALUE, "null external semaphore arguments"); validate_semaphore(*descriptor);
    auto owner = current(); auto native = owner->native->import_semaphore(*descriptor); auto handle = interop_handle<CUexternalSemaphore>();
    global().semaphores.emplace(handle, semaphore_entry{owner, std::move(native), descriptor->type}); *output = handle)
DRIVER_API(cuDestroyExternalSemaphore, (CUexternalSemaphore handle),
    driver_ready(); auto owner = current(); interop_lookup(global().semaphores, handle, owner); synchronize_context(owner); global().semaphores.erase(handle))
DRIVER_API(cuSignalExternalSemaphoresAsync, (const CUexternalSemaphore* handles, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* params, unsigned count, CUstream stream),
    driver_ready(); semaphore_operations(handles, params, count, stream, true))
DRIVER_API(cuWaitExternalSemaphoresAsync, (const CUexternalSemaphore* handles, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* params, unsigned count, CUstream stream),
    driver_ready(); semaphore_operations(handles, params, count, stream, false))
// GLenum/GLuint are unsigned integers; keep GL headers out of Driver builds.
DRIVER_API(cuGraphicsGLRegisterBuffer, (CUgraphicsResource* output, unsigned object, unsigned flags),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null graphics resource output"); gl_flags(flags); auto owner = current();
    require(owner->gl != nullptr, CUDA_ERROR_INVALID_GRAPHICS_CONTEXT, "attach the application GL context with cudavmGraphicsGL first");
    require(std::none_of(global().graphics.begin(), global().graphics.end(), [&](auto& item) { return item.second.owner == owner && item.second.object == object; }),
            CUDA_ERROR_INVALID_HANDLE, "GL buffer is already registered"); owner->gl->size(object);
    auto handle = interop_handle<CUgraphicsResource>(); global().graphics.emplace(handle, graphics_entry{owner, object, flags}); *output = handle)
DRIVER_API(cuGraphicsGLRegisterImage, (CUgraphicsResource* output, unsigned object, unsigned target, unsigned flags),
    driver_ready(); (void)object; (void)target; (void)flags; require(output, CUDA_ERROR_INVALID_VALUE, "null graphics resource output"); current();
    throw failure(CUDA_ERROR_NOT_SUPPORTED, "GL image registration is not implemented"))
DRIVER_API(cuGraphicsUnregisterResource, (CUgraphicsResource handle),
    driver_ready(); auto owner = current(); auto& entry = interop_lookup(global().graphics, handle, owner);
    require(!entry.mapped, CUDA_ERROR_ALREADY_MAPPED, "unmap graphics resource before unregistering it"); global().graphics.erase(handle))
DRIVER_API(cuGraphicsResourceSetMapFlags, (CUgraphicsResource handle, unsigned flags),
    driver_ready(); gl_flags(flags); auto& entry = interop_lookup(global().graphics, handle, current());
    require(!entry.mapped, CUDA_ERROR_ALREADY_MAPPED, "cannot change flags while mapped"); entry.flags = flags)
DRIVER_API(cuGraphicsMapResources, (unsigned count, CUgraphicsResource* handles, CUstream stream), driver_ready(); map_graphics(count, handles, stream))
DRIVER_API(cuGraphicsUnmapResources, (unsigned count, CUgraphicsResource* handles, CUstream stream), driver_ready(); unmap_graphics(count, handles, stream))
DRIVER_API(cuGraphicsResourceGetMappedPointer, (CUdeviceptr* output, size_t* bytes, CUgraphicsResource handle),
    driver_ready(); require(output && bytes, CUDA_ERROR_INVALID_VALUE, "null mapped graphics pointer outputs");
    auto& entry = interop_lookup(global().graphics, handle, current()); require(entry.mapped, CUDA_ERROR_NOT_MAPPED, "graphics resource is not mapped");
    *output = entry.mapped; *bytes = global().allocations.at(entry.mapped).native->size)
