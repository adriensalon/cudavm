DRIVER_API(cuMemGetAllocationGranularity, (size_t* output, const CUmemAllocationProp* properties, CUmemAllocationGranularity_flags option),
    driver_ready(); require(output && properties && (option == CU_MEM_ALLOC_GRANULARITY_MINIMUM || option == CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                            CUDA_ERROR_INVALID_VALUE, "invalid allocation granularity query");
    auto owner = virtual_memory_owner(properties->location.type == CU_MEM_LOCATION_TYPE_DEVICE ? properties->location.id : -1); auto native = *properties;
    if (native.location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        valid_device(native.location.id);
        require(device_kind(native.location.id) == owner->native->kind(), CUDA_ERROR_NOT_SUPPORTED, "allocation location belongs to another adapter");
        native.location.id = native_device(native.location.id);
    }
    auto result = native_call(owner, [&] { return owner->native->allocation_granularity(native, option); }); *output = result)
DRIVER_API(cuMemAddressReserve, (CUdeviceptr* output, size_t bytes, size_t alignment, CUdeviceptr hint, unsigned long long flags),
    driver_ready(); require(output && bytes && !flags && (!alignment || !(alignment & (alignment - 1))),
                            CUDA_ERROR_INVALID_VALUE, "invalid virtual address reservation");
    auto owner = virtual_memory_owner();
    auto reservation = native_call(owner, [&] { return owner->native->reserve_virtual_address(bytes, alignment, hint); });
    auto address = reservation->address;
    require(address && bytes <= std::numeric_limits<CUdeviceptr>::max() - address, CUDA_ERROR_UNKNOWN, "invalid native reservation extent");
    require(!global().address_reservations.contains(address), CUDA_ERROR_UNKNOWN, "native reservation address collision");
    global().address_reservations.emplace(address, std::move(reservation)); *output = address)
DRIVER_API(cuMemAddressFree, (CUdeviceptr pointer, size_t bytes),
    driver_ready(); auto found = global().address_reservations.find(pointer);
    require(found != global().address_reservations.end() && found->second->size == bytes, CUDA_ERROR_INVALID_VALUE, "virtual free requires the exact reservation");
    auto mapped = global().allocations.lower_bound(pointer);
    require(mapped == global().allocations.end() || mapped->first - pointer >= bytes, CUDA_ERROR_INVALID_VALUE, "virtual reservation still contains mappings");
    found->second->release(); global().address_reservations.erase(found))
DRIVER_API(cuMemCreate, (CUmemGenericAllocationHandle* output, size_t bytes, const CUmemAllocationProp* properties, unsigned long long flags),
    driver_ready(); virtual_create(output, bytes, properties, flags))
DRIVER_API(cuMemRelease, (CUmemGenericAllocationHandle handle),
    driver_ready(); auto entry = virtual_handle(handle); if (--entry->references == 0) global().virtual_allocations.erase(handle))
DRIVER_API(cuMemMap, (CUdeviceptr pointer, size_t bytes, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags),
    driver_ready(); virtual_map(pointer, bytes, offset, handle, flags))
DRIVER_API(cuMemUnmap, (CUdeviceptr pointer, size_t bytes), driver_ready(); virtual_unmap(pointer, bytes))
DRIVER_API(cuMemSetAccess, (CUdeviceptr pointer, size_t bytes, const CUmemAccessDesc* descriptors, size_t count),
    driver_ready(); virtual_set_access(pointer, bytes, descriptors, count))
DRIVER_API(cuMemGetAccess, (unsigned long long* output, const CUmemLocation* location, CUdeviceptr pointer),
    driver_ready(); virtual_get_access(output, location, pointer))
DRIVER_API(cuMemGetAllocationPropertiesFromHandle, (CUmemAllocationProp* output, CUmemGenericAllocationHandle handle),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "missing physical properties output"); auto entry = virtual_handle(handle);
    auto properties = entry->native->properties();
    if (properties.location.type == CU_MEM_LOCATION_TYPE_DEVICE) properties.location.id = managed_logical_device(properties.location.id);
    *output = properties)
DRIVER_API(cuMemRetainAllocationHandle, (CUmemGenericAllocationHandle* output, void* pointer), driver_ready(); virtual_retain(output, pointer))
DRIVER_API(cuMemExportToShareableHandle, (void* output, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType type, unsigned long long flags),
    driver_ready(); virtual_export(output, handle, type, flags))
DRIVER_API(cuMemImportFromShareableHandle, (CUmemGenericAllocationHandle* output, void* handle, CUmemAllocationHandleType type),
    driver_ready(); virtual_import(output, handle, type))
