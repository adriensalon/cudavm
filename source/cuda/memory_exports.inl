DRIVER_API(cuMemAllocManaged, (CUdeviceptr* output, size_t bytes, unsigned flags),
    driver_ready(); require(output && bytes && (flags == CU_MEM_ATTACH_GLOBAL || flags == CU_MEM_ATTACH_HOST), CUDA_ERROR_INVALID_VALUE, "invalid managed allocation arguments");
    auto owner = current(); *output = add_allocation(owner, native_call(owner, [&] { return owner->native->allocate_managed(bytes, flags); })))
DRIVER_API(cuMemGetInfo, (size_t* free, size_t* total),
    driver_ready(); require(free && total, CUDA_ERROR_INVALID_VALUE, "null memory info output");
    auto owner = current(); size_t available = 0, capacity = 0;
    if (global().kind == cudavmBackendCuda) {
        std::lock_guard lock(owner->execution); owner->native->memory_info(available, capacity);
    } else {
        capacity = global().devices[owner->device].memory_bytes;
        require(capacity, CUDA_ERROR_NOT_SUPPORTED, "backend memory capacity query is unavailable");
        available = capacity;
        std::erase_if(global().device_buffers, [](auto& entry) { return entry.native.expired(); });
        for (auto& entry : global().device_buffers) if (entry.device == owner->device) if (auto native = entry.native.lock()) {
            auto bytes = native->physical_size(); available -= std::min(available, bytes);
        }
    }
    *free = available; *total = capacity)
DRIVER_API(cuMemAllocPitch, (CUdeviceptr* output, size_t* pitch, size_t width, size_t height, unsigned element),
    driver_ready(); require(output && pitch, CUDA_ERROR_INVALID_VALUE, "null pitched allocation outputs");
    size_t value = 0; auto pointer = allocate_pitch(current(), width, height, element, value); *output = pointer; *pitch = value)
DRIVER_API(cuMemcpy, (CUdeviceptr to, CUdeviceptr from, size_t bytes), driver_ready(); copy_unified(to, from, bytes, current()))
DRIVER_API(cuMemcpy2D, (const CUDA_MEMCPY2D* p), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null 2D copy descriptor"); copy_2d(*p, current()))
DRIVER_API(cuMemcpy2DUnaligned, (const CUDA_MEMCPY2D* p), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null 2D copy descriptor"); copy_2d(*p, current()))
DRIVER_API(cuMemcpy3D, (const CUDA_MEMCPY3D* p), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null 3D copy descriptor"); auto owner = current(); copy_3d(*p, owner, owner))
DRIVER_API(cuMemcpyPeer, (CUdeviceptr to, CUcontext to_context, CUdeviceptr from, CUcontext from_context, size_t bytes),
    driver_ready(); auto to_owner = lookup(to_context), from_owner = lookup(from_context); current();
    auto source = memory_endpoint(CU_MEMORYTYPE_DEVICE, nullptr, from, bytes, 0, 0, 0, 0, bytes, 1, 1, from_owner, false);
    auto destination = memory_endpoint(CU_MEMORYTYPE_DEVICE, nullptr, to, bytes, 0, 0, 0, 0, bytes, 1, 1, to_owner, false);
    copy_boxes(destination, source, bytes))
DRIVER_API(cuMemcpy3DPeer, (const CUDA_MEMCPY3D_PEER* p), driver_ready(); require(p, CUDA_ERROR_INVALID_VALUE, "null peer copy descriptor"); current(); copy_3d(*p, lookup(p->dstContext), lookup(p->srcContext)))
DRIVER_API(cuMemsetD2D8, (CUdeviceptr to, size_t pitch, unsigned char value, size_t width, size_t height), driver_ready(); fill_box(to, pitch, 0, value, width, height, 1, current()))
DRIVER_API(cuMemsetD2D16, (CUdeviceptr to, size_t pitch, unsigned short value, size_t width, size_t height), driver_ready(); fill_box(to, pitch, 0, value, width, height, 1, current()))
DRIVER_API(cuMemsetD2D32, (CUdeviceptr to, size_t pitch, unsigned value, size_t width, size_t height), driver_ready(); fill_box(to, pitch, 0, value, width, height, 1, current()))
DRIVER_API(cuPointerGetAttributes, (unsigned count, CUpointer_attribute* attributes, void** outputs, CUdeviceptr pointer),
    driver_ready(); require(!count || (attributes && outputs), CUDA_ERROR_INVALID_VALUE, "null pointer attribute arrays");
    for (unsigned i = 0; i < count; ++i) { require(outputs[i], CUDA_ERROR_INVALID_VALUE, "null pointer attribute output"); pointer_attribute_size(attributes[i]); }
    for (unsigned i = 0; i < count; ++i) pointer_attribute(outputs[i], attributes[i], pointer, true))
DRIVER_API(cuPointerSetAttribute, (const void* input, CUpointer_attribute attribute, CUdeviceptr pointer),
    driver_ready(); require(input && attribute == CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, CUDA_ERROR_INVALID_VALUE, "invalid settable pointer attribute");
    auto it = containing(pointer); require(it != global().allocations.end(), CUDA_ERROR_INVALID_VALUE, "unknown device pointer");
    auto value = *static_cast<const unsigned*>(input); require(value <= 1, CUDA_ERROR_INVALID_VALUE, "sync-memops value must be boolean");
    auto native = it->second.native; auto owner = it->second.owner; if (value) synchronize_context(owner); it = containing(pointer); require(it != global().allocations.end() && it->second.native == native, CUDA_ERROR_INVALID_VALUE, "pointer changed while synchronizing"); it->second.sync_memops = value)
extern "C" CUDAVM_CUDA_API CUresult cudavmDriverMemset3D(CUdeviceptr pointer, size_t pitch, size_t slice_height,
                                                        unsigned char value, size_t width, size_t height, size_t depth) {
    return driver_call([&] { driver_ready(); fill_box(pointer, pitch, slice_height, value, width, height, depth, current()); });
}
