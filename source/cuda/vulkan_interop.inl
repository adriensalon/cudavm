VkExternalMemoryHandleTypeFlagBits vk_memory_type(unsigned type) {
#if defined(_WIN32)
    switch (type) {
    case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32: return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    case CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT: return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
    case CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP: return VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
    case CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE: return VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    }
#else
    if (type == CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD) return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    throw failure(CUDA_ERROR_NOT_SUPPORTED, "unsupported Vulkan memory handle type");
}
VkExternalSemaphoreHandleTypeFlagBits vk_semaphore_type(unsigned type) {
#if defined(_WIN32)
    switch (type) {
    case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32: return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT: return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
    case CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE: return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    }
#else
    if (type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD) return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    throw failure(CUDA_ERROR_NOT_SUPPORTED, "unsupported Vulkan semaphore handle type");
}
constexpr VkBufferUsageFlags interop_buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
void vulkan_device::negotiate_interop(bool memory_enabled, bool semaphore_enabled) {
#if defined(_WIN32)
    const unsigned memory_types[] = {2, 3, 4, 5}; const unsigned semaphore_types[] = {2, 3, 4};
#else
    const unsigned memory_types[] = {1}; const unsigned semaphore_types[] = {1};
#endif
    if (memory_enabled) for (auto type : memory_types) {
        VkPhysicalDeviceExternalBufferInfo info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
        info.usage = interop_buffer_usage; info.handleType = vk_memory_type(type);
        VkExternalBufferProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        instance->load<PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR>("vkGetPhysicalDeviceExternalBufferPropertiesKHR")(physical, &info, &properties);
        if (properties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) caps.memory |= handle_bit(type);
    }
    caps.metadata = caps.memory & (handle_bit(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD) | handle_bit(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32) | handle_bit(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT));
    if (semaphore_enabled) for (auto type : semaphore_types) {
        VkPhysicalDeviceExternalSemaphoreInfo info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO}; info.handleType = vk_semaphore_type(type);
        VkExternalSemaphoreProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
        instance->load<PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR>("vkGetPhysicalDeviceExternalSemaphorePropertiesKHR")(physical, &info, &properties);
        if (properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) caps.semaphore |= handle_bit(type);
    }
}
struct vulkan_memory : external_memory, std::enable_shared_from_this<vulkan_memory> {
    std::shared_ptr<vulkan_device> owner;
    VkBuffer value = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t bytes;
    vulkan_memory(std::shared_ptr<vulkan_device> input, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d, const cudavmExternalMemoryMetadata* metadata)
        : owner(std::move(input)), bytes(metadata ? metadata->buffer_size : d.size) {
        require(!(owner->caps.metadata & handle_bit(d.type)) || metadata, CUDA_ERROR_NOT_SUPPORTED, "opaque Vulkan memory requires cudavmImportExternalMemory allocation metadata");
        require(bytes && bytes <= d.size, CUDA_ERROR_INVALID_VALUE, "invalid external Vulkan buffer size");
        if (metadata) {
            require(std::memcmp(metadata->device_uuid, owner->device_uuid, VK_UUID_SIZE) == 0 && std::memcmp(metadata->driver_uuid, owner->driver_uuid, VK_UUID_SIZE) == 0,
                    CUDA_ERROR_INVALID_DEVICE, "external Vulkan allocation belongs to another device or driver");
            require(metadata->buffer_usage == interop_buffer_usage, CUDA_ERROR_NOT_SUPPORTED, "opaque Vulkan buffers require exactly storage and transfer usage");
        }
        auto type = vk_memory_type(d.type);
        VkPhysicalDeviceExternalBufferInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO}; query.usage = interop_buffer_usage; query.handleType = type;
        VkExternalBufferProperties external{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        owner->instance->load<PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR>("vkGetPhysicalDeviceExternalBufferPropertiesKHR")(owner->physical, &query, &external);
        bool dedicated = (d.flags & CUDA_EXTERNAL_MEMORY_DEDICATED) != 0;
        require(dedicated || !(external.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT), CUDA_ERROR_INVALID_VALUE, "external Vulkan memory requires a dedicated allocation");
        VkExternalMemoryBufferCreateInfo external_info{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO}; external_info.handleTypes = type;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; info.pNext = &external_info; info.size = bytes; info.usage = interop_buffer_usage;
        check(owner->vkCreateBuffer(owner->value, &info, nullptr, &value), "create imported Vulkan buffer");
        try {
            VkMemoryDedicatedRequirements dedicated_requirements{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
            VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2}; requirements.pNext = &dedicated_requirements;
            VkBufferMemoryRequirementsInfo2 requirements_info{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2}; requirements_info.buffer = value;
            owner->load<PFN_vkGetBufferMemoryRequirements2KHR>("vkGetBufferMemoryRequirements2KHR")(owner->value, &requirements_info, &requirements);
            require(dedicated || !dedicated_requirements.requiresDedicatedAllocation, CUDA_ERROR_INVALID_VALUE, "Vulkan requires a dedicated allocation");
            require(requirements.memoryRequirements.size <= d.size, CUDA_ERROR_INVALID_VALUE, "external allocation is smaller than Vulkan buffer requirements");
            uint32_t bits = requirements.memoryRequirements.memoryTypeBits;
#if defined(_WIN32)
            VkImportMemoryWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
            import.handleType = type; import.handle = static_cast<HANDLE>(d.handle.win32.handle); import.name = static_cast<LPCWSTR>(d.handle.win32.name);
            require(import.handle || (owner->caps.metadata & handle_bit(d.type)), CUDA_ERROR_NOT_SUPPORTED, "named nonopaque Vulkan allocations require a handle");
            if (!(owner->caps.metadata & handle_bit(d.type))) {
                VkMemoryWin32HandlePropertiesKHR properties{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
                check(owner->load<PFN_vkGetMemoryWin32HandlePropertiesKHR>("vkGetMemoryWin32HandlePropertiesKHR")(owner->value, type, import.handle, &properties), "query imported Vulkan memory type");
                bits &= properties.memoryTypeBits;
            }
#else
            VkImportMemoryFdInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR}; import.handleType = type;
#endif
            uint32_t index = metadata ? metadata->memory_type_index : 0;
            if (!metadata) while (index < owner->memory.memoryTypeCount && !(bits & (1u << index))) ++index;
            require(index < owner->memory.memoryTypeCount && (bits & (1u << index)), CUDA_ERROR_INVALID_VALUE, "external Vulkan memory type is incompatible with the buffer");
            VkMemoryDedicatedAllocateInfo dedicated_info{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO}; dedicated_info.buffer = value;
            if (dedicated) import.pNext = &dedicated_info;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; allocation.pNext = &import; allocation.allocationSize = d.size; allocation.memoryTypeIndex = index;
#if !defined(_WIN32)
            import.fd = dup(d.handle.fd); require(import.fd >= 0, CUDA_ERROR_INVALID_HANDLE, "duplicate external memory fd");
#endif
            auto status = owner->vkAllocateMemory(owner->value, &allocation, nullptr, &memory);
#if !defined(_WIN32)
            if (status != VK_SUCCESS) close(import.fd);
#endif
            check(status, "import Vulkan memory");
            check(owner->vkBindBufferMemory(owner->value, value, memory, 0), "bind imported Vulkan memory");
#if !defined(_WIN32)
            close(d.handle.fd);
#endif
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() { if (value) owner->vkDestroyBuffer(owner->value, value, nullptr); if (memory) owner->vkFreeMemory(owner->value, memory, nullptr); value = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; }
    ~vulkan_memory() { cleanup(); }
    std::unique_ptr<buffer> map(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC& d) override {
        require(d.offset <= bytes && d.size <= bytes - d.offset && !((d.offset | d.size) & 3), CUDA_ERROR_INVALID_VALUE, "external Vulkan view exceeds the buffer or is not four-byte aligned");
        return std::make_unique<vulkan_buffer>(owner, size_t(d.size), value, size_t(d.offset), shared_from_this());
    }
};
struct vulkan_semaphore : external_semaphore {
    std::shared_ptr<vulkan_device> owner;
    VkSemaphore value = VK_NULL_HANDLE;
    bool fence;
    vulkan_semaphore(std::shared_ptr<vulkan_device> input, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d)
        : owner(std::move(input)), fence(d.type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE) {
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        check(owner->load<PFN_vkCreateSemaphore>("vkCreateSemaphore")(owner->value, &info, nullptr, &value), "create external Vulkan semaphore");
        try {
#if defined(_WIN32)
            VkImportSemaphoreWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
            import.semaphore = value; import.handleType = vk_semaphore_type(d.type);
            import.handle = static_cast<HANDLE>(d.handle.win32.handle); import.name = static_cast<LPCWSTR>(d.handle.win32.name);
            check(owner->load<PFN_vkImportSemaphoreWin32HandleKHR>("vkImportSemaphoreWin32HandleKHR")(owner->value, &import), "import Vulkan semaphore");
#else
            VkImportSemaphoreFdInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
            import.semaphore = value; import.handleType = vk_semaphore_type(d.type); import.fd = dup(d.handle.fd);
            require(import.fd >= 0, CUDA_ERROR_INVALID_HANDLE, "duplicate external semaphore fd");
            auto status = owner->load<PFN_vkImportSemaphoreFdKHR>("vkImportSemaphoreFdKHR")(owner->value, &import);
            if (status != VK_SUCCESS) close(import.fd);
            check(status, "import Vulkan semaphore"); close(d.handle.fd);
#endif
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() { if (value) owner->load<PFN_vkDestroySemaphore>("vkDestroySemaphore")(owner->value, value, nullptr); value = VK_NULL_HANDLE; }
    ~vulkan_semaphore() { cleanup(); }
    void submit(bool signal, uint64_t epoch) {
        std::lock_guard lock(owner->execution);
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        if (signal) { info.signalSemaphoreCount = 1; info.pSignalSemaphores = &value; }
        else { info.waitSemaphoreCount = 1; info.pWaitSemaphores = &value; info.pWaitDstStageMask = &stage; }
#if defined(_WIN32)
        VkD3D12FenceSubmitInfoKHR values{VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR};
        values.waitSemaphoreValuesCount = info.waitSemaphoreCount; values.signalSemaphoreValuesCount = info.signalSemaphoreCount;
        if (signal) values.pSignalSemaphoreValues = &epoch; else values.pWaitSemaphoreValues = &epoch;
        if (fence) info.pNext = &values;
#else
        (void)epoch;
#endif
        check(owner->vkQueueSubmit(owner->queue, 1, &info, VK_NULL_HANDLE), "submit external Vulkan synchronization");
    }
    void signal(uint64_t epoch) override { submit(true, epoch); }
    void wait(uint64_t epoch) override { submit(false, epoch); }
};
