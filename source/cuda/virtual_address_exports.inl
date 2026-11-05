DRIVER_API(cuMemGetAllocationGranularity, (size_t* output, const CUmemAllocationProp* properties, CUmemAllocationGranularity_flags option),
    driver_ready(); require(output && properties && (option == CU_MEM_ALLOC_GRANULARITY_MINIMUM || option == CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                            CUDA_ERROR_INVALID_VALUE, "invalid allocation granularity query");
    auto owner = current(); auto native = *properties;
    if (native.location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
        valid_device(native.location.id);
        require(device_kind(native.location.id) == owner->native->kind(), CUDA_ERROR_NOT_SUPPORTED, "allocation location belongs to another adapter");
        native.location.id = native_device(native.location.id);
    }
    auto result = native_call(owner, [&] { return owner->native->allocation_granularity(native, option); }); *output = result)
DRIVER_API(cuMemAddressReserve, (CUdeviceptr* output, size_t bytes, size_t alignment, CUdeviceptr hint, unsigned long long flags),
    driver_ready(); require(output && bytes && !flags && (!alignment || !(alignment & (alignment - 1))),
                            CUDA_ERROR_INVALID_VALUE, "invalid virtual address reservation");
    auto owner = current();
    auto reservation = native_call(owner, [&] { return owner->native->reserve_virtual_address(bytes, alignment, hint); });
    auto address = reservation->address;
    require(address && bytes <= std::numeric_limits<CUdeviceptr>::max() - address, CUDA_ERROR_UNKNOWN, "invalid native reservation extent");
    require(!global().address_reservations.contains(address), CUDA_ERROR_UNKNOWN, "native reservation address collision");
    global().address_reservations.emplace(address, std::move(reservation)); *output = address)
DRIVER_API(cuMemAddressFree, (CUdeviceptr pointer, size_t bytes),
    driver_ready(); auto found = global().address_reservations.find(pointer);
    require(found != global().address_reservations.end() && found->second->size == bytes, CUDA_ERROR_INVALID_VALUE, "virtual free requires the exact reservation");
    found->second->release(); global().address_reservations.erase(found))
