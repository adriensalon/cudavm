region managed_advice_range(CUdeviceptr pointer, size_t bytes) {
    require(bytes, CUDA_ERROR_INVALID_VALUE, "managed range must not be empty");
    auto range = resolve(pointer, bytes, current());
    require(range.value->native->is_managed(), CUDA_ERROR_INVALID_VALUE, "range must belong to managed memory");
    return range;
}
void managed_prefetch(CUdeviceptr pointer, size_t bytes, CUmemLocation location, unsigned flags, CUstream stream) {
    require(!flags, CUDA_ERROR_INVALID_VALUE, "prefetch flags must be zero");
    require(location.type == CU_MEM_LOCATION_TYPE_DEVICE || location.type == CU_MEM_LOCATION_TYPE_HOST ||
            location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA || location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT,
            CUDA_ERROR_INVALID_VALUE, "invalid prefetch location");
    auto target = stream_lookup(stream, current()); auto owner = target->owner.lock();
    auto range = managed_advice_range(pointer, bytes);
    if (location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        valid_device(location.id);
        require(device_kind(location.id) == cudavmBackendCuda &&
                global().provider->attribute(location.id, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS),
                CUDA_ERROR_INVALID_DEVICE, "prefetch destination requires concurrent managed access");
        location.id = native_device(location.id);
    }
    require(global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS),
            CUDA_ERROR_INVALID_DEVICE, "prefetch stream requires concurrent managed access");
    auto storage = range.value->native;
    auto queue = stream_native(owner, target);
    submit_stream(owner, stream, {}, {}, 0, [owner, pointer, bytes, location, storage, queue] {
        return owner->native->enqueue_prefetch(pointer, bytes, location, queue.get());
    });
}
int managed_logical_device(int ordinal) {
    if (ordinal < 0) return ordinal;
    for (size_t i = 0; i < global().devices.size(); ++i)
        if (device_kind(int(i)) == cudavmBackendCuda && native_device(int(i)) == ordinal) return int(i);
    return CU_DEVICE_INVALID;
}
void managed_advise(CUdeviceptr pointer, size_t bytes, CUmem_advise advice, CUmemLocation location) {
    require(advice >= CU_MEM_ADVISE_SET_READ_MOSTLY && advice <= CU_MEM_ADVISE_UNSET_ACCESSED_BY,
            CUDA_ERROR_INVALID_VALUE, "invalid managed memory advice");
    auto range = managed_advice_range(pointer, bytes);
    bool ignored = advice == CU_MEM_ADVISE_SET_READ_MOSTLY || advice == CU_MEM_ADVISE_UNSET_READ_MOSTLY || advice == CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION;
    if (!ignored) {
        require(location.type == CU_MEM_LOCATION_TYPE_DEVICE || location.type == CU_MEM_LOCATION_TYPE_HOST ||
                (advice == CU_MEM_ADVISE_SET_PREFERRED_LOCATION &&
                 (location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA || location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT)),
                CUDA_ERROR_INVALID_VALUE, "invalid managed advice location");
        if (location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
            valid_device(location.id);
            require(device_kind(location.id) == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "managed advice requires a native managed device");
            location.id = native_device(location.id);
        }
    }
    auto owner = range.value->owner; auto storage = range.value->native;
    native_call(owner, [&] { storage->advise(range.offset, bytes, advice, location); });
}
void validate_managed_attribute(void* output, size_t size, CUmem_range_attribute attribute) {
    require(output && attribute >= CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY && attribute <= CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID,
            CUDA_ERROR_INVALID_VALUE, "invalid managed range query");
    require(attribute == CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY ? size && size % sizeof(int) == 0 : size == sizeof(int),
            CUDA_ERROR_INVALID_VALUE, "invalid managed range output size");
}
void managed_attribute(void* output, size_t size, CUmem_range_attribute attribute, CUdeviceptr pointer, size_t bytes) {
    validate_managed_attribute(output, size, attribute);
    auto range = managed_advice_range(pointer, bytes);
    std::vector<int> values(size / sizeof(int));
    auto owner = range.value->owner;
    capture_unsafe(owner);
    std::lock_guard lock(owner->execution);
    range.value->native->range_attribute(values.data(), size, attribute, range.offset, bytes);
    if (attribute == CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY || attribute == CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION ||
        attribute == CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION)
        for (auto& value : values) value = managed_logical_device(value);
    if (attribute == CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID || attribute == CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID) {
        int type = 0;
        auto type_attribute = attribute == CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID ?
            CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE : CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE;
        range.value->native->range_attribute(&type, sizeof(type), type_attribute, range.offset, bytes);
        if (type == CU_MEM_LOCATION_TYPE_DEVICE) values[0] = managed_logical_device(values[0]);
    }
    std::memcpy(output, values.data(), size);
}
