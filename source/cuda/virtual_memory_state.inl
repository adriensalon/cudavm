std::shared_ptr<context> virtual_memory_owner(int device = -1);
std::shared_ptr<context> virtual_control_owner(const std::shared_ptr<context>& storage);
CUmemLocation virtual_native_location(CUmemLocation location, const std::shared_ptr<context>& owner) {
    if (location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        valid_device(location.id);
        require(device_kind(location.id) == owner->native->kind(), CUDA_ERROR_NOT_SUPPORTED, "virtual memory location belongs to another adapter");
        location.id = native_device(location.id);
    }
    return location;
}
std::shared_ptr<virtual_allocation_state> virtual_handle(CUmemGenericAllocationHandle handle) {
    auto found = global().virtual_allocations.find(handle);
    require(found != global().virtual_allocations.end(), CUDA_ERROR_INVALID_VALUE, "unknown physical allocation handle");
    return found->second;
}
bool virtual_os_handle_type(CUmemAllocationHandleType type) {
    return type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR || type == CU_MEM_HANDLE_TYPE_WIN32 ||
        type == CU_MEM_HANDLE_TYPE_WIN32_KMT || type == CU_MEM_HANDLE_TYPE_FABRIC;
}
void virtual_import(CUmemGenericAllocationHandle* output, void* handle, CUmemAllocationHandleType type) {
    require(output && virtual_os_handle_type(type), CUDA_ERROR_INVALID_VALUE, "invalid physical handle import arguments");
    auto owner = virtual_memory_owner();
    auto physical = native_call(owner, [&] { return owner->native->import_virtual_allocation(handle, type); });
    auto properties = physical->properties();
    if (properties.location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        properties.location.id = managed_logical_device(properties.location.id);
        require(properties.location.id >= 0, CUDA_ERROR_INVALID_DEVICE, "imported physical device is not visible");
        owner = virtual_memory_owner(properties.location.id);
    }
    auto native = properties; native.location = virtual_native_location(native.location, owner);
    auto granularity = native_call(owner, [&] { return owner->native->allocation_granularity(native, CU_MEM_ALLOC_GRANULARITY_MINIMUM); });
    require(granularity, CUDA_ERROR_UNKNOWN, "invalid imported allocation granularity");
    auto entry = std::make_shared<virtual_allocation_state>();
    entry->native = std::move(physical); entry->properties = properties; entry->granularity = granularity;
    require(global().next_virtual_allocation != std::numeric_limits<CUmemGenericAllocationHandle>::max(), CUDA_ERROR_OUT_OF_MEMORY, "physical handle space exhausted");
    entry->handle = global().next_virtual_allocation++;
    global().virtual_allocations.emplace(entry->handle, entry); *output = entry->handle;
}
void virtual_export(void* output, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType type, unsigned long long flags) {
    require(output && !flags && virtual_os_handle_type(type), CUDA_ERROR_INVALID_VALUE, "invalid physical handle export arguments");
    auto entry = virtual_handle(handle);
    require((entry->properties.requestedHandleTypes & type) != 0, CUDA_ERROR_INVALID_VALUE, "physical allocation is not exportable with this handle type");
    entry->native->export_handle(output, type);
}
void virtual_create(CUmemGenericAllocationHandle* output, size_t bytes, const CUmemAllocationProp* properties, unsigned long long flags) {
    require(output && properties && bytes && !flags, CUDA_ERROR_INVALID_VALUE, "invalid physical allocation arguments");
    auto owner = virtual_memory_owner(properties->location.type == CU_MEM_LOCATION_TYPE_DEVICE ? properties->location.id : -1);
    auto native = *properties;
    native.location = virtual_native_location(native.location, owner);
    auto granularity = native_call(owner, [&] { return owner->native->allocation_granularity(native, CU_MEM_ALLOC_GRANULARITY_MINIMUM); });
    require(granularity && bytes % granularity == 0, CUDA_ERROR_INVALID_VALUE, "physical allocation size is not a granularity multiple");
    auto physical = native_call(owner, [&] { return owner->native->create_virtual_allocation(bytes, native); });
    auto entry = std::make_shared<virtual_allocation_state>();
    entry->native = std::move(physical); entry->properties = *properties; entry->granularity = granularity;
    require(global().next_virtual_allocation != std::numeric_limits<CUmemGenericAllocationHandle>::max(), CUDA_ERROR_OUT_OF_MEMORY, "physical handle space exhausted");
    entry->handle = global().next_virtual_allocation++;
    global().virtual_allocations.emplace(entry->handle, entry); *output = entry->handle;
}
void virtual_map(CUdeviceptr pointer, size_t bytes, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    require(pointer && bytes && !flags && !offset, CUDA_ERROR_INVALID_VALUE, "invalid virtual mapping arguments or physical offset");
    auto physical = virtual_handle(handle); const auto granularity = physical->granularity;
    require(pointer % granularity == 0 && bytes % granularity == 0 && bytes <= physical->native->size,
            CUDA_ERROR_INVALID_VALUE, "virtual mapping exceeds physical storage or granularity");
    auto reserved = global().address_reservations.upper_bound(pointer);
    require(reserved != global().address_reservations.begin(), CUDA_ERROR_INVALID_VALUE, "mapping has no address reservation");
    --reserved;
    const auto& reservation = reserved->second;
    require(pointer >= reservation->address && pointer - reservation->address < reservation->size &&
            bytes <= reservation->size - size_t(pointer - reservation->address), CUDA_ERROR_INVALID_VALUE, "mapping exceeds its reservation");
    auto next = global().allocations.lower_bound(pointer);
    require(next == global().allocations.end() || bytes <= next->first - pointer, CUDA_ERROR_INVALID_VALUE, "virtual mapping overlaps existing storage");
    if (next != global().allocations.begin()) {
        auto previous = std::prev(next);
        require(pointer - previous->first >= previous->second.native->size, CUDA_ERROR_INVALID_VALUE, "virtual mapping overlaps existing storage");
    }
    auto owner = virtual_memory_owner(physical->properties.location.type == CU_MEM_LOCATION_TYPE_DEVICE ? physical->properties.location.id : -1);
    auto native = native_call(virtual_control_owner(owner), [&] { return owner->native->map_virtual(pointer, bytes, physical->native); });
    auto address = add_allocation(owner, std::move(native));
    global().allocations.at(address).virtual_memory = std::move(physical);
}
void virtual_unmap(CUdeviceptr pointer, size_t bytes) {
    auto found = global().allocations.find(pointer);
    require(found != global().allocations.end() && found->second.virtual_memory && bytes == found->second.native->size,
            CUDA_ERROR_INVALID_VALUE, "unmap requires one entire virtual mapping");
    auto owner = found->second.owner; auto storage = found->second.native;
    native_call(virtual_control_owner(owner), [&] { owner->native->unmap_virtual(*storage); });
    found = global().allocations.find(pointer);
    require(found != global().allocations.end() && found->second.native == storage, CUDA_ERROR_INVALID_VALUE, "mapping changed while waiting for unmap");
    invalidate_native_graphs(); global().allocations.erase(found);
}
void virtual_set_access(CUdeviceptr pointer, size_t bytes, const CUmemAccessDesc* descriptors, size_t count) {
    require(bytes && (!count || descriptors), CUDA_ERROR_INVALID_VALUE, "invalid virtual access descriptor array");
    auto first = global().allocations.find(pointer);
    require(first != global().allocations.end() && first->second.virtual_memory, CUDA_ERROR_INVALID_VALUE, "access range requires mapped virtual memory");
    auto owner = first->second.owner; std::vector<CUmemAccessDesc> native;
    if (count) native.assign(descriptors, descriptors + count);
    // Validate the entire descriptor and mapping inventory before changing access.
    for (auto& access : native) {
        require(access.flags == CU_MEM_ACCESS_FLAGS_PROT_NONE || access.flags == CU_MEM_ACCESS_FLAGS_PROT_READ ||
                access.flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CUDA_ERROR_INVALID_VALUE, "invalid virtual access flags");
        access.location = virtual_native_location(access.location, owner);
    }
    auto cursor = pointer; size_t remaining = bytes;
    while (remaining) {
        auto found = global().allocations.find(cursor);
        require(found != global().allocations.end() && found->second.virtual_memory && found->second.native->size <= remaining,
                CUDA_ERROR_INVALID_VALUE, "access range requires complete contiguous virtual mappings");
        auto size = found->second.native->size;
        require(size <= std::numeric_limits<CUdeviceptr>::max() - cursor, CUDA_ERROR_INVALID_VALUE, "access range overflow");
        cursor += size; remaining -= size;
    }
    native_call(virtual_control_owner(owner), [&] { owner->native->set_virtual_access(pointer, bytes, native); });
}
void virtual_get_access(unsigned long long* output, const CUmemLocation* location, CUdeviceptr pointer) {
    require(output && location, CUDA_ERROR_INVALID_VALUE, "missing virtual access query output or location");
    auto found = containing(pointer);
    require(found != global().allocations.end() && found->second.virtual_memory, CUDA_ERROR_INVALID_VALUE, "access query requires mapped virtual memory");
    auto owner = found->second.owner;
    auto native = virtual_native_location(*location, owner);
    *output = native_call(virtual_control_owner(owner), [&] { return owner->native->virtual_access(pointer, native); });
}
void virtual_retain(CUmemGenericAllocationHandle* output, void* pointer) {
    require(output, CUDA_ERROR_INVALID_VALUE, "missing physical handle output");
    auto found = containing(reinterpret_cast<uintptr_t>(pointer));
    require(found != global().allocations.end(), CUDA_ERROR_INVALID_VALUE, "retain requires mapped virtual memory");
    auto physical = found->second.virtual_memory;
    require(physical && physical->references < std::numeric_limits<size_t>::max(), CUDA_ERROR_INVALID_VALUE, "retain requires a virtual mapping");
    global().virtual_allocations.emplace(physical->handle, physical);
    ++physical->references; *output = physical->handle;
}
std::shared_ptr<context> virtual_memory_owner(int device) {
    if (device < 0) device = cuda_device();
    valid_device(device);
    require(device_kind(device) == cudavmBackendCuda, CUDA_ERROR_NOT_SUPPORTED, "virtual memory requires a native CUDA device");
    auto& owner = global().virtual_storage[device];
    if (!owner) {
        auto created = std::make_shared<context>(); created->device = device;
        created->native = global().provider->open(device, 0);
        created->flags = created->native->get_context_flags(0);
        owner = std::move(created);
    }
    return owner;
}
std::shared_ptr<context> virtual_control_owner(const std::shared_ptr<context>& storage) {
    if (!context_stack().empty()) {
        auto owner = current();
        if (owner->native->kind() == cudavmBackendCuda) return owner;
    }
    return storage;
}
