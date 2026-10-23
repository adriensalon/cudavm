#include "dynamic_library.hpp"
#include "command_regions.hpp"
#include "device_resources.hpp"
#define VK_NO_PROTOTYPES
#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#else
#include <unistd.h>
#endif
#include <vulkan/vulkan.h>
namespace cudavm {
namespace {
void check(VkResult value, const char* action) {
    if (value == VK_SUCCESS) return;
    CUresult error = (value == VK_ERROR_OUT_OF_DEVICE_MEMORY || value == VK_ERROR_OUT_OF_HOST_MEMORY) ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
    throw failure(error, std::string(action) + " failed: " + std::to_string(value));
}
struct vulkan_instance {
#if defined(_WIN32)
    dynamic_library loader{{"vulkan-1.dll"}};
#else
    dynamic_library loader{{"libvulkan.so.1", "libvulkan.so"}};
#endif
    PFN_vkGetInstanceProcAddr get = loader.get<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VkInstance value = VK_NULL_HANDLE;
    uint32_t api_version = VK_API_VERSION_1_0;
    bool external_memory = false, external_semaphore = false;
    template<class T> T load(const char* name) const {
        auto result = reinterpret_cast<T>(get(value, name));
        require(result != nullptr, CUDA_ERROR_NOT_SUPPORTED, name);
        return result;
    }
    PFN_vkDestroyInstance destroy = nullptr;
    vulkan_instance() {
        auto create = load<PFN_vkCreateInstance>("vkCreateInstance");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "cudavm";
        if (auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(get(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"))) {
            uint32_t available = VK_API_VERSION_1_0; check(enumerate_version(&available), "vkEnumerateInstanceVersion");
            api_version = std::min(available, uint32_t(VK_API_VERSION_1_2));
        }
        app.apiVersion = api_version;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &app;
        auto enumerate = load<PFN_vkEnumerateInstanceExtensionProperties>("vkEnumerateInstanceExtensionProperties");
        uint32_t count = 0; check(enumerate(nullptr, &count, nullptr), "enumerate Vulkan instance extensions");
        std::vector<VkExtensionProperties> available(count);
        check(enumerate(nullptr, &count, available.data()), "enumerate Vulkan instance extensions");
        auto has = [&](const char* name) { return std::any_of(available.begin(), available.end(), [&](auto& e) { return std::strcmp(e.extensionName, name) == 0; }); };
        std::vector<const char*> extensions;
        if (has(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
            external_memory = has(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
            external_semaphore = has(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
            if (external_memory) extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
            if (external_semaphore) extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        }
        info.enabledExtensionCount = uint32_t(extensions.size()); info.ppEnabledExtensionNames = extensions.data();
        check(create(&info, nullptr, &value), "vkCreateInstance");
        destroy = load<PFN_vkDestroyInstance>("vkDestroyInstance");
    }
    ~vulkan_instance() { if (value && destroy) destroy(value, nullptr); }
};
struct vulkan_device {
    // Vulkan requires external host synchronization for its queue and command
    // pool, including command-buffer retirement on another context's worker.
    std::recursive_mutex execution;
    std::shared_ptr<vulkan_instance> instance;
    VkPhysicalDevice physical;
    VkDevice value = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceLimits limits{};
    bool native_spirv = false;
    VkPhysicalDeviceFeatures native_features{};
    bool native_int8 = false;
    shader_device_resources resources;
    PFN_vkGetDeviceProcAddr get = nullptr;
    interop_capabilities caps;
    uint32_t family = 0;
    unsigned timestamp_bits = 0;
    unsigned char device_uuid[VK_UUID_SIZE]{}, driver_uuid[VK_UUID_SIZE]{};
    template<class T> T load(const char* name) const {
        auto result = reinterpret_cast<T>(get(value, name));
        require(result != nullptr, CUDA_ERROR_NOT_SUPPORTED, name);
        return result;
    }
#define ENTRY(name) PFN_##name name = nullptr
    ENTRY(vkDestroyDevice); ENTRY(vkGetDeviceQueue); ENTRY(vkCreateCommandPool); ENTRY(vkDestroyCommandPool);
    ENTRY(vkCreateBuffer); ENTRY(vkDestroyBuffer); ENTRY(vkGetBufferMemoryRequirements); ENTRY(vkAllocateMemory); ENTRY(vkFreeMemory);
    ENTRY(vkBindBufferMemory); ENTRY(vkMapMemory); ENTRY(vkUnmapMemory); ENTRY(vkDeviceWaitIdle);
    ENTRY(vkAllocateCommandBuffers); ENTRY(vkFreeCommandBuffers); ENTRY(vkBeginCommandBuffer); ENTRY(vkEndCommandBuffer);
    ENTRY(vkCmdCopyBuffer); ENTRY(vkCmdFillBuffer); ENTRY(vkCmdPipelineBarrier); ENTRY(vkQueueSubmit); ENTRY(vkQueueWaitIdle);
    ENTRY(vkCreateFence); ENTRY(vkDestroyFence); ENTRY(vkGetFenceStatus); ENTRY(vkWaitForFences);
    ENTRY(vkCreateQueryPool); ENTRY(vkDestroyQueryPool); ENTRY(vkCmdResetQueryPool); ENTRY(vkCmdWriteTimestamp); ENTRY(vkGetQueryPoolResults);
    ENTRY(vkCreateShaderModule);ENTRY(vkDestroyShaderModule);ENTRY(vkCreateDescriptorSetLayout);ENTRY(vkDestroyDescriptorSetLayout);
    ENTRY(vkCreatePipelineLayout);ENTRY(vkDestroyPipelineLayout);ENTRY(vkCreateComputePipelines);ENTRY(vkDestroyPipeline);
    ENTRY(vkCreateDescriptorPool);ENTRY(vkDestroyDescriptorPool);ENTRY(vkAllocateDescriptorSets);ENTRY(vkUpdateDescriptorSets);
    ENTRY(vkCmdBindPipeline);ENTRY(vkCmdBindDescriptorSets);ENTRY(vkCmdDispatch);
#undef ENTRY
    vulkan_device(std::shared_ptr<vulkan_instance> input, VkPhysicalDevice device) : instance(std::move(input)), physical(device) {
        VkPhysicalDeviceProperties properties{};
        instance->load<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(physical,&properties);limits=properties.limits;
        native_spirv = instance->api_version >= VK_API_VERSION_1_2 && properties.apiVersion >= VK_API_VERSION_1_2;
        auto families = instance->load<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
        uint32_t count = 0;
        families(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        families(physical, &count, queues.data());
        family = 0;
        while (family < count && !(queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
        require(family < count, CUDA_ERROR_NOT_SUPPORTED, "Vulkan compute queue unavailable");
        timestamp_bits = queues[family].timestampValidBits;
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        info.queueCreateInfoCount = 1;
        info.pQueueCreateInfos = &queue_info;
        auto enumerate = instance->load<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        uint32_t extension_count = 0; check(enumerate(physical, nullptr, &extension_count, nullptr), "enumerate Vulkan device extensions");
        std::vector<VkExtensionProperties> available(extension_count);
        check(enumerate(physical, nullptr, &extension_count, available.data()), "enumerate Vulkan device extensions");
        auto has = [&](const char* name) { return std::any_of(available.begin(), available.end(), [&](auto& e) { return std::strcmp(e.extensionName, name) == 0; }); };
        std::vector<const char*> extensions;
#if defined(_WIN32)
        const char* memory_platform = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
        const char* semaphore_platform = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
#else
        const char* memory_platform = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
        const char* semaphore_platform = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif
        bool memory_enabled = instance->external_memory && has(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) && has(memory_platform)
            && has(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME) && has(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        bool semaphore_enabled = instance->external_semaphore && has(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME) && has(semaphore_platform);
        if (memory_enabled) {
            extensions.insert(extensions.end(), {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, memory_platform,
                VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME});
            VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; properties.pNext = &ids;
            instance->load<PFN_vkGetPhysicalDeviceProperties2KHR>("vkGetPhysicalDeviceProperties2KHR")(physical, &properties);
            std::memcpy(device_uuid, ids.deviceUUID, VK_UUID_SIZE); std::memcpy(driver_uuid, ids.driverUUID, VK_UUID_SIZE);
        }
        if (semaphore_enabled) extensions.insert(extensions.end(), {VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, semaphore_platform});
        info.enabledExtensionCount = uint32_t(extensions.size()); info.ppEnabledExtensionNames = extensions.data();
        VkPhysicalDeviceShaderFloat16Int8Features narrow{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        if (native_spirv) {
            VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; properties.pNext = &subgroup;
            instance->load<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(physical, &properties);
            if (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) {
                resources.subgroup_min = subgroup.subgroupSize; resources.subgroup_max = subgroup.subgroupSize;
            }
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; features.pNext = &narrow;
            instance->load<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &features);
            native_features.shaderInt64 = features.features.shaderInt64;
            native_features.shaderInt16 = features.features.shaderInt16;
            native_features.shaderFloat64 = features.features.shaderFloat64;
            native_int8 = narrow.shaderInt8;
            narrow.shaderFloat16 = VK_FALSE; narrow.shaderInt8 = native_int8;
            info.pEnabledFeatures = &native_features; info.pNext = &narrow;
        }
        check(instance->load<PFN_vkCreateDevice>("vkCreateDevice")(physical, &info, nullptr, &value), "vkCreateDevice");
        get = instance->load<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        try {
#define LOAD(name) name = load<PFN_##name>(#name)
            LOAD(vkDestroyDevice); LOAD(vkGetDeviceQueue); LOAD(vkCreateCommandPool); LOAD(vkDestroyCommandPool);
            LOAD(vkCreateBuffer); LOAD(vkDestroyBuffer); LOAD(vkGetBufferMemoryRequirements); LOAD(vkAllocateMemory); LOAD(vkFreeMemory);
            LOAD(vkBindBufferMemory); LOAD(vkMapMemory); LOAD(vkUnmapMemory); LOAD(vkDeviceWaitIdle);
            LOAD(vkAllocateCommandBuffers); LOAD(vkFreeCommandBuffers); LOAD(vkBeginCommandBuffer); LOAD(vkEndCommandBuffer);
            LOAD(vkCmdCopyBuffer); LOAD(vkCmdFillBuffer); LOAD(vkCmdPipelineBarrier); LOAD(vkQueueSubmit); LOAD(vkQueueWaitIdle);
            LOAD(vkCreateFence); LOAD(vkDestroyFence); LOAD(vkGetFenceStatus); LOAD(vkWaitForFences);
            LOAD(vkCreateQueryPool); LOAD(vkDestroyQueryPool); LOAD(vkCmdResetQueryPool); LOAD(vkCmdWriteTimestamp); LOAD(vkGetQueryPoolResults);
            LOAD(vkCreateShaderModule);LOAD(vkDestroyShaderModule);LOAD(vkCreateDescriptorSetLayout);LOAD(vkDestroyDescriptorSetLayout);
            LOAD(vkCreatePipelineLayout);LOAD(vkDestroyPipelineLayout);LOAD(vkCreateComputePipelines);LOAD(vkDestroyPipeline);
            LOAD(vkCreateDescriptorPool);LOAD(vkDestroyDescriptorPool);LOAD(vkAllocateDescriptorSets);LOAD(vkUpdateDescriptorSets);
            LOAD(vkCmdBindPipeline);LOAD(vkCmdBindDescriptorSets);LOAD(vkCmdDispatch);
#undef LOAD
            vkGetDeviceQueue(value, family, 0, &queue);
            VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool_info.queueFamilyIndex = family;
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            check(vkCreateCommandPool(value, &pool_info, nullptr, &pool), "vkCreateCommandPool");
            instance->load<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(physical, &memory);
            negotiate_interop(memory_enabled, semaphore_enabled);
        } catch (...) {
            if (vkDestroyDevice) vkDestroyDevice(value, nullptr);
            value = VK_NULL_HANDLE;
            throw;
        }
    }
    void negotiate_interop(bool memory_enabled, bool semaphore_enabled);
    ~vulkan_device() {
        if (value) {
            vkDeviceWaitIdle(value);
            if (pool) vkDestroyCommandPool(value, pool, nullptr);
            vkDestroyDevice(value, nullptr);
        }
    }
};
struct vulkan_buffer : buffer {
    std::shared_ptr<vulkan_device> owner;
    VkBuffer value = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t base_offset = 0;
    size_t backing_size = 0;
    std::shared_ptr<external_memory> imported;
    vulkan_buffer(std::shared_ptr<vulkan_device> input, size_t bytes, VkBuffer native, size_t offset, std::shared_ptr<external_memory> memory)
        : buffer(bytes), owner(std::move(input)), value(native), base_offset(offset), imported(std::move(memory)) {}
    vulkan_buffer(std::shared_ptr<vulkan_device> input, size_t bytes) : buffer(bytes), owner(std::move(input)) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(owner->vkCreateBuffer(owner->value, &info, nullptr, &value), "vkCreateBuffer");
        try {
            VkMemoryRequirements requirements{};
            owner->vkGetBufferMemoryRequirements(owner->value, value, &requirements);
            uint32_t type = 0;
            for (; type < owner->memory.memoryTypeCount; ++type) {
                auto flags = owner->memory.memoryTypes[type].propertyFlags;
                if ((requirements.memoryTypeBits & (1u << type)) && (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
            }
            require(type < owner->memory.memoryTypeCount, CUDA_ERROR_NOT_SUPPORTED, "host coherent Vulkan storage memory unavailable");
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            backing_size = static_cast<size_t>(requirements.size);
            allocation.memoryTypeIndex = type;
            check(owner->vkAllocateMemory(owner->value, &allocation, nullptr, &memory), "vkAllocateMemory");
            check(owner->vkBindBufferMemory(owner->value, value, memory, 0), "vkBindBufferMemory");
            check(owner->vkMapMemory(owner->value, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() {
        if (mapped) owner->vkUnmapMemory(owner->value, memory);
        if (value && !imported) owner->vkDestroyBuffer(owner->value, value, nullptr);
        if (memory) owner->vkFreeMemory(owner->value, memory, nullptr);
        value = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; mapped = nullptr;
    }
    ~vulkan_buffer() { cleanup(); }
    size_t graph_backing_size() const override { return backing_size; }
    size_t physical_size() const override { return imported ? 0 : backing_size; }
};
#include "vulkan_interop.inl"
struct vulkan_pipeline : compute_pipeline {
    std::shared_ptr<vulkan_device> owner;
    VkShaderModule shader=VK_NULL_HANDLE;
    VkDescriptorSetLayout slots=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;
    explicit vulkan_pipeline(std::shared_ptr<vulkan_device> device) : owner(std::move(device)) {}
    ~vulkan_pipeline() {
        auto device=owner->value;
        if(pipeline) owner->vkDestroyPipeline(device,pipeline,nullptr);
        if(layout) owner->vkDestroyPipelineLayout(device,layout,nullptr);
        if(slots) owner->vkDestroyDescriptorSetLayout(device,slots,nullptr);
        if(shader) owner->vkDestroyShaderModule(device,shader,nullptr);
    }
};
struct vulkan_task : device_task {
    std::shared_ptr<vulkan_device> owner;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool owns_command = true;
    bool submitted = false;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    std::shared_ptr<device_timestamp> timestamp;
    std::shared_ptr<compute_profile> marker_profile,dispatch_profile;
    bool measured = false;
    explicit vulkan_task(std::shared_ptr<vulkan_device> input) : owner(std::move(input)) {
        VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(owner->vkCreateFence(owner->value, &info, nullptr, &fence), "vkCreateFence");
    }
    bool query() override {
        auto result = owner->vkGetFenceStatus(owner->value, fence);
        if (result == VK_NOT_READY) return false;
        check(result, "vkGetFenceStatus");
        if(timestamp) {
            uint64_t ticks{};check(owner->vkGetQueryPoolResults(owner->value,query_pool,0,1,sizeof(ticks),&ticks,sizeof(ticks),VK_QUERY_RESULT_64_BIT),"vkGetQueryPoolResults");
            timestamp->ticks=ticks;timestamp->valid_bits=owner->timestamp_bits;timestamp->nanoseconds_per_tick=owner->limits.timestampPeriod;
            if(marker_profile) marker_profile->snapshot(*timestamp);
        }
        if(dispatch_profile && !measured) {
            uint64_t ticks[2]{};check(owner->vkGetQueryPoolResults(owner->value,query_pool,0,2,sizeof(ticks),ticks,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT),"vkGetQueryPoolResults dispatch");
            auto delta=ticks[1]-ticks[0];if(owner->timestamp_bits<64)delta&=(uint64_t(1)<<owner->timestamp_bits)-1;
            dispatch_profile->nanoseconds.fetch_add(uint64_t(double(delta)*owner->limits.timestampPeriod));measured=true;
        }
        return true;
    }
    ~vulkan_task() {
        std::lock_guard lock(owner->execution);
        if (submitted) owner->vkWaitForFences(owner->value, 1, &fence, VK_TRUE, UINT64_MAX);
        if (command && owns_command) owner->vkFreeCommandBuffers(owner->value, owner->pool, 1, &command);
        if (fence) owner->vkDestroyFence(owner->value, fence, nullptr);
        if (descriptors) owner->vkDestroyDescriptorPool(owner->value, descriptors, nullptr);
        if (query_pool) owner->vkDestroyQueryPool(owner->value, query_pool, nullptr);
    }
};
struct vulkan_graph : native_graph, command_region_queue, std::enable_shared_from_this<vulkan_graph> {
    std::shared_ptr<vulkan_task> commands;
    explicit vulkan_graph(std::shared_ptr<vulkan_device> owner) : commands(std::make_shared<vulkan_task>(owner)) {
        std::lock_guard lock(owner->execution);
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = owner->pool; allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocation.commandBufferCount = 1;
        check(owner->vkAllocateCommandBuffers(owner->value, &allocation, &commands->command), "vkAllocateCommandBuffers region");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        check(owner->vkBeginCommandBuffer(commands->command, &begin), "vkBeginCommandBuffer region");
    }
    void barrier() {
        auto owner = commands->owner;
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        owner->vkCmdPipelineBarrier(commands->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    device_work enqueue(native_queue*) override {
        auto owner = commands->owner;
        std::lock_guard lock(owner->execution);
        auto retained = std::make_shared<device_continuation>(device_work{}, [self = shared_from_this()] {});
        auto task = std::make_shared<vulkan_task>(owner);
        task->command = commands->command; task->owns_command = false;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &task->command;
        check(owner->vkQueueSubmit(owner->queue, 1, &submit, task->fence), "vkQueueSubmit region"); task->submitted = true;
        retained->task = task; return retained;
    }
};
struct vulkan_session : session {
    std::shared_ptr<vulkan_device> owner;
    // External waits need an independent queue: a wait on the shared resource
    // queue could prevent another logical context from submitting its signal.
    // Create this device lazily; pool kernels and transfers stay on owner.
    std::shared_ptr<vulkan_device> synchronization;
    explicit vulkan_session(std::shared_ptr<vulkan_device> input) : owner(std::move(input)) {}
    bool command_regions() const override { return true; }
    bool can_record_copy(const buffer& to, size_t destination, const buffer& from, size_t source, size_t bytes) const override {
        return can_copy_direct(to) && can_copy_direct(from) &&
            !((destination + to.storage_offset()) & 3) && !((source + from.storage_offset()) & 3) && !(bytes & 3);
    }
    bool can_record_fill(const buffer& value, size_t offset, size_t bytes) const override {
        return can_copy_direct(value) && !(((offset + value.storage_offset()) | bytes) & 3);
    }
    std::shared_ptr<native_graph> compile_graph(std::span<const native_graph_node> nodes) override {
        if (!command_region_compatible(nodes)) return {};
        auto graph = std::make_shared<vulkan_graph>(owner);
        try {
            for (auto& node : nodes) if (auto resource = node.command(graph.get())) graph->resources.push_back(std::move(resource));
        } catch (const failure& error) { if (error.code == CUDA_ERROR_NOT_SUPPORTED) return {}; throw; }
        std::lock_guard lock(owner->execution);
        graph->barrier();
        check(owner->vkEndCommandBuffer(graph->commands->command), "vkEndCommandBuffer region");
        return graph;
    }
    interop_capabilities interop() override { return owner->caps; }
    std::shared_ptr<external_memory> import_memory(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d, const cudavmExternalMemoryMetadata* metadata = nullptr) override {
        require(owner->caps.memory & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "Vulkan external memory handle type is unavailable");
        return std::make_shared<vulkan_memory>(owner, d, metadata);
    }
    std::shared_ptr<external_semaphore> import_semaphore(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d) override {
        require(owner->caps.semaphore & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "Vulkan external semaphore handle type is unavailable");
        if (!synchronization) synchronization = std::make_shared<vulkan_device>(owner->instance, owner->physical);
        return std::make_shared<vulkan_semaphore>(synchronization, d);
    }
    std::unique_ptr<buffer> allocate(size_t bytes) override { return std::make_unique<vulkan_buffer>(owner, bytes); }
    void write_device(buffer& value, size_t offset, std::span<const std::byte> data) override {
        if (&value.storage() != &value) { write_device(value.storage(), offset + value.storage_offset(), data); return; }
        std::lock_guard lock(owner->execution);
        if (static_cast<vulkan_buffer&>(value).imported) {
            auto begin = offset & ~size_t(3), end = (offset + data.size() + 3) & ~size_t(3);
            auto staging = allocate(end - begin);
            if ((offset | data.size()) & 3) copy(*staging, 0, value, begin, end - begin);
            std::memcpy(static_cast<std::byte*>(static_cast<vulkan_buffer&>(*staging).mapped) + offset - begin, data.data(), data.size());
            copy(value, begin, *staging, 0, end - begin); return;
        }
        synchronize();
        std::memcpy(static_cast<std::byte*>(static_cast<vulkan_buffer&>(value).mapped) + offset, data.data(), data.size());
    }
    void read_device(buffer& value, size_t offset, std::span<std::byte> data) override {
        if (&value.storage() != &value) { read_device(value.storage(), offset + value.storage_offset(), data); return; }
        std::lock_guard lock(owner->execution);
        if (static_cast<vulkan_buffer&>(value).imported) {
            auto begin = offset & ~size_t(3), end = (offset + data.size() + 3) & ~size_t(3);
            auto staging = allocate(end - begin);
            copy(*staging, 0, value, begin, end - begin);
            std::memcpy(data.data(), static_cast<std::byte*>(static_cast<vulkan_buffer&>(*staging).mapped) + offset - begin, data.size()); return;
        }
        synchronize();
        std::memcpy(data.data(), static_cast<std::byte*>(static_cast<vulkan_buffer&>(value).mapped) + offset, data.size());
    }
    device_work transfer(buffer& to, size_t to_offset, buffer& from, size_t from_offset, size_t size) {
        if (&to.storage() != &to || &from.storage() != &from) return transfer(to.storage(), to_offset + to.storage_offset(), from.storage(), from_offset + from.storage_offset(), size);
        std::lock_guard lock(owner->execution);
        auto& destination = static_cast<vulkan_buffer&>(to);
        auto& source = static_cast<vulkan_buffer&>(from);
        if (source.value == destination.value && source.imported) {
            std::shared_ptr<buffer> temporary = allocate((size + 3) & ~size_t(3));
            auto tasks = std::make_shared<device_group>(); tasks->tasks.reserve(2);
            auto retained = std::make_shared<device_continuation>(tasks, [temporary] {});
            tasks->tasks.push_back(transfer(*temporary, 0, from, from_offset, size));
            tasks->tasks.push_back(transfer(to, to_offset, *temporary, 0, size)); return retained;
        }
        // The initial 1.0 adapter stages byte-unaligned regions through actual
        // coherent GPU mappings. Aligned regions use a GPU transfer command.
        to_offset += destination.base_offset; from_offset += source.base_offset;
        if ((to_offset | from_offset | size) & 3) {
            if (destination.imported || source.imported) {
                std::vector<std::byte> staging(size); read(from, from_offset - source.base_offset, staging);
                write(to, to_offset - destination.base_offset, staging); return {};
            }
            synchronize();
            std::memcpy(static_cast<std::byte*>(destination.mapped) + to_offset, static_cast<std::byte*>(source.mapped) + from_offset, size);
            return {};
        }
        auto task = std::make_shared<vulkan_task>(owner);
        auto& command = task->command;
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = owner->pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        check(owner->vkAllocateCommandBuffers(owner->value, &allocation, &command), "vkAllocateCommandBuffers");
        try {
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(owner->vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            owner->vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            auto external_barrier = [&](vulkan_buffer& buffer, bool acquire, VkAccessFlags access) {
                if (!buffer.imported) return;
                VkBufferMemoryBarrier transition{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                transition.buffer = buffer.value; transition.offset = 0; transition.size = VK_WHOLE_SIZE;
                transition.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : owner->family;
                transition.dstQueueFamilyIndex = acquire ? owner->family : VK_QUEUE_FAMILY_EXTERNAL;
                transition.srcAccessMask = acquire ? 0 : access; transition.dstAccessMask = acquire ? access : 0;
                owner->vkCmdPipelineBarrier(command, acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                    acquire ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &transition, 0, nullptr);
            };
            external_barrier(source, true, VK_ACCESS_TRANSFER_READ_BIT); external_barrier(destination, true, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferCopy region{from_offset, to_offset, size};
            owner->vkCmdCopyBuffer(command, source.value, destination.value, 1, &region);
            external_barrier(source, false, VK_ACCESS_TRANSFER_READ_BIT); external_barrier(destination, false, VK_ACCESS_TRANSFER_WRITE_BIT);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
            owner->vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            check(owner->vkEndCommandBuffer(command), "vkEndCommandBuffer");
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            check(owner->vkQueueSubmit(owner->queue, 1, &submit, task->fence), "vkQueueSubmit");
            task->submitted = true;
        } catch (...) { throw; }
        return task;
    }
    void copy_device(buffer& to, size_t destination, buffer& from, size_t source, size_t bytes) override {
        if (&to.storage() != &to || &from.storage() != &from) { copy_device(to.storage(), destination + to.storage_offset(), from.storage(), source + from.storage_offset(), bytes); return; }
        wait_device(transfer(to, destination, from, source, bytes));
    }
    device_work enqueue_copy(buffer& to, size_t destination, buffer& from, size_t source, size_t bytes, native_queue* queue) override {
        if (queue && queue->recording()) {
            require(can_record_copy(to, destination, from, source, bytes), CUDA_ERROR_NOT_SUPPORTED, "Vulkan copy cannot be recorded");
            if (!bytes) return {};
            auto& graph = dynamic_cast<vulkan_graph&>(*queue);
            require(graph.commands->owner == owner, CUDA_ERROR_NOT_SUPPORTED, "foreign Vulkan recording lane");
            std::lock_guard lock(owner->execution);
            graph.barrier();
            VkBufferCopy copy{source + from.storage_offset(), destination + to.storage_offset(), bytes};
            owner->vkCmdCopyBuffer(graph.commands->command, static_cast<vulkan_buffer&>(from.storage()).value,
                static_cast<vulkan_buffer&>(to.storage()).value, 1, &copy);
            return {};
        }
        if (&to.storage() != &to || &from.storage() != &from) return enqueue_copy(to.storage(), destination + to.storage_offset(), from.storage(), source + from.storage_offset(), bytes, queue);
        if (to.host_address() || from.host_address()) return session::enqueue_copy(to, destination, from, source, bytes, queue);
        return transfer(to, destination, from, source, bytes);
    }
    device_work enqueue_write(buffer& value, size_t offset, std::span<const std::byte> bytes, native_queue* queue) override {
        require(!queue || !queue->recording() || can_record_fill(value, offset, bytes.size()),
            CUDA_ERROR_NOT_SUPPORTED, "Vulkan upload cannot be recorded");
        if (&value.storage() != &value) return enqueue_write(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (value.host_address() || ((offset | bytes.size()) & 3)) return session::enqueue_write(value, offset, bytes, queue);
        std::shared_ptr<buffer> staging = allocate(bytes.size());
        std::memcpy(static_cast<vulkan_buffer&>(*staging).mapped, bytes.data(), bytes.size());
        auto continuation = std::make_shared<device_continuation>(device_work{}, [staging] {});
        if (queue && queue->recording()) dynamic_cast<vulkan_graph&>(*queue).resources.push_back(continuation);
        continuation->task = enqueue_copy(value, offset, *staging, 0, bytes.size(), queue);
        return queue && queue->recording() ? device_work{} : continuation;
    }
    device_work enqueue_fill(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes, native_queue* queue) override {
        if (queue && queue->recording()) {
            require(can_record_fill(value, offset, bytes), CUDA_ERROR_NOT_SUPPORTED, "Vulkan fill cannot be recorded");
            if (!bytes) return {};
            auto& graph = dynamic_cast<vulkan_graph&>(*queue);
            require(graph.commands->owner == owner, CUDA_ERROR_NOT_SUPPORTED, "foreign Vulkan recording lane");
            std::lock_guard lock(owner->execution);
            graph.barrier();
            owner->vkCmdFillBuffer(graph.commands->command, static_cast<vulkan_buffer&>(value.storage()).value,
                offset + value.storage_offset(), bytes, repeated_fill_word(pattern));
            return {};
        }
        if (!bytes) return {};
        if (&value.storage() != &value) return enqueue_fill(value.storage(), offset + value.storage_offset(), pattern, bytes, queue);
        if (value.host_address()) return session::enqueue_fill(value, offset, pattern, bytes, queue);
        auto& target = static_cast<vulkan_buffer&>(value);
        if (((offset + target.base_offset) | bytes) & 3) return session::enqueue_fill(value, offset, pattern, bytes, queue);
        auto word = repeated_fill_word(pattern);
        std::lock_guard lock(owner->execution);
        auto task = std::make_shared<vulkan_task>(owner);
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = owner->pool; allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocation.commandBufferCount = 1;
        check(owner->vkAllocateCommandBuffers(owner->value, &allocation, &task->command), "vkAllocateCommandBuffers fill");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(owner->vkBeginCommandBuffer(task->command, &begin), "vkBeginCommandBuffer fill");
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        owner->vkCmdPipelineBarrier(task->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        auto external = [&](bool acquire) {
            if (!target.imported) return;
            VkBufferMemoryBarrier transition{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            transition.buffer = target.value; transition.size = VK_WHOLE_SIZE;
            transition.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : owner->family;
            transition.dstQueueFamilyIndex = acquire ? owner->family : VK_QUEUE_FAMILY_EXTERNAL;
            transition.srcAccessMask = acquire ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
            transition.dstAccessMask = acquire ? VK_ACCESS_TRANSFER_WRITE_BIT : 0;
            owner->vkCmdPipelineBarrier(task->command, acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                acquire ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &transition, 0, nullptr);
        };
        external(true);
        owner->vkCmdFillBuffer(task->command, target.value, offset + target.base_offset, bytes, word);
        external(false);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        owner->vkCmdPipelineBarrier(task->command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
        check(owner->vkEndCommandBuffer(task->command), "vkEndCommandBuffer fill");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &task->command;
        check(owner->vkQueueSubmit(owner->queue, 1, &submit, task->fence), "vkQueueSubmit fill"); task->submitted = true;
        return task;
    }
    void fill_device(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes) override {
        wait_device(enqueue_fill(value, offset, pattern, bytes, nullptr));
    }
    device_work enqueue_read(buffer& value, size_t offset, std::span<std::byte> bytes, native_queue* queue) override {
        require(!queue || !queue->recording(), CUDA_ERROR_NOT_SUPPORTED, "Vulkan readback cannot be recorded");
        if (&value.storage() != &value) return enqueue_read(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (value.host_address() || ((offset | bytes.size()) & 3)) return session::enqueue_read(value, offset, bytes, queue);
        std::shared_ptr<buffer> staging = allocate(bytes.size());
        auto continuation = std::make_shared<device_continuation>(device_work{}, [staging, bytes] {
            std::memcpy(bytes.data(), static_cast<vulkan_buffer&>(*staging).mapped, bytes.size());
        });
        continuation->task = transfer(*staging, 0, value, offset, bytes.size()); return continuation;
    }
    void synchronize() override {
        { std::lock_guard lock(owner->execution); check(owner->vkDeviceWaitIdle(owner->value), "vkDeviceWaitIdle"); }
        if (synchronization) { std::lock_guard lock(synchronization->execution); check(synchronization->vkDeviceWaitIdle(synchronization->value), "vkDeviceWaitIdle semaphore queue"); }
    }
    cudavmBackend kind() const override { return cudavmBackendVulkan; }
    device_work enqueue_timestamp(std::shared_ptr<device_timestamp> timestamp,native_queue*) override {
        if(!owner->timestamp_bits) return {};
        std::lock_guard lock(owner->execution);
        auto task=std::make_shared<vulkan_task>(owner);task->timestamp=std::move(timestamp);task->marker_profile=profile;
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};info.queryType=VK_QUERY_TYPE_TIMESTAMP;info.queryCount=1;
        check(owner->vkCreateQueryPool(owner->value,&info,nullptr,&task->query_pool),"vkCreateQueryPool");
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};allocate.commandPool=owner->pool;allocate.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;allocate.commandBufferCount=1;
        check(owner->vkAllocateCommandBuffers(owner->value,&allocate,&task->command),"vkAllocateCommandBuffers timestamp");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(owner->vkBeginCommandBuffer(task->command,&begin),"vkBeginCommandBuffer timestamp");
        owner->vkCmdResetQueryPool(task->command,task->query_pool,0,1);
        owner->vkCmdWriteTimestamp(task->command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,task->query_pool,0);
        check(owner->vkEndCommandBuffer(task->command),"vkEndCommandBuffer timestamp");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&task->command;
        check(owner->vkQueueSubmit(owner->queue,1,&submit,task->fence),"vkQueueSubmit timestamp");task->submitted=true;return task;
    }
    uint64_t shader_arena_capacity() const override { return std::min(packet_arena_capacity, uint64_t(owner->limits.maxStorageBufferRange)); }
    unsigned direct_binding_limit() const override {
        auto slots=std::min(owner->limits.maxPerStageDescriptorStorageBuffers,owner->limits.maxDescriptorSetStorageBuffers);
        slots=std::min(slots,owner->limits.maxPerStageResources);
        return slots>2?std::min(6u,slots-2):0;
    }
    bool can_bind_direct(const buffer& value) const override {
        auto native=dynamic_cast<const vulkan_buffer*>(&value.storage());
        return native && native->owner==owner && !native->imported && value.size%4==0 && value.storage_offset()%4==0 && value.size<=packet_arena_capacity && value.storage_offset()<=packet_arena_capacity-value.size && value.storage().size<=owner->limits.maxStorageBufferRange;
    }
    bool can_copy_direct(const buffer& value) const override {
        auto native = dynamic_cast<const vulkan_buffer*>(&value.storage());
        return native && native->owner == owner && !native->imported;
    }
    bool supports_native_spirv(std::span<const uint32_t> code) const override {
        if (!owner->native_spirv || code.size() < 5 || code[1] > 0x00010500u) return false;
        for (size_t i = 5; i < code.size();) {
            auto count = code[i] >> 16, opcode = code[i] & 0xffffu;
            if (!count || count > code.size() - i) return false;
            if (opcode == 17) {
                if (count != 2) return false;
                switch (code[i + 1]) {
                case 1: break;
                case 10: if (!owner->native_features.shaderFloat64) return false; break;
                case 11: if (!owner->native_features.shaderInt64) return false; break;
                case 22: if (!owner->native_features.shaderInt16) return false; break;
                case 39: if (!owner->native_int8) return false; break;
                default: return false;
                }
            }
            i += count;
        }
        return true;
    }
    unsigned native_block_limit() const override { return owner->limits.maxComputeWorkGroupInvocations; }
    shader_device_resources execution_resources() const override { return owner->resources; }
    shader_resource_limits shader_limits() const override {
        shader_resource_limits result; result.threads = owner->limits.maxComputeWorkGroupInvocations;
        std::copy_n(owner->limits.maxComputeWorkGroupSize, 3, result.block.begin());
        std::copy_n(owner->limits.maxComputeWorkGroupCount, 3, result.grid.begin());
        result.shared_bytes = owner->limits.maxComputeSharedMemorySize; return result;
    }
    void validate_native_spirv_shape(const launch_shape& shape) const override {
        uint64_t threads = 1;
        for (unsigned axis = 0; axis < 3; ++axis) {
            require(shape.block[axis] <= owner->limits.maxComputeWorkGroupSize[axis] && shape.grid[axis] <= owner->limits.maxComputeWorkGroupCount[axis], CUDA_ERROR_INVALID_VALUE, "native SPIR-V launch dimensions exceed Vulkan device limits");
            threads *= shape.block[axis];
        }
        require(threads <= owner->limits.maxComputeWorkGroupInvocations, CUDA_ERROR_INVALID_VALUE, "native SPIR-V block exceeds Vulkan invocation limits");
    }
    std::unique_ptr<compute_pipeline> compile_native_spirv(std::span<const uint32_t> code, const launch_shape& shape) override {
        require(supports_native_spirv(code), CUDA_ERROR_NOT_SUPPORTED, "native SPIR-V device capabilities are unavailable");
        validate_native_spirv_shape(shape);
        return compile_binary(code, packet_direct_bindings);
    }
    std::unique_ptr<compute_pipeline> compile_binary(std::span<const uint32_t> code, unsigned direct_buffers) {
        auto device=owner->value;
        auto compiled=std::make_unique<vulkan_pipeline>(owner);
        compiled->direct_buffers=direct_buffers;
        VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};shader_info.codeSize=code.size()*4;shader_info.pCode=code.data();
        check(owner->vkCreateShaderModule(device,&shader_info,nullptr,&compiled->shader),"vkCreateShaderModule");
        std::vector<VkDescriptorSetLayoutBinding> bindings(2+direct_buffers);
        for(unsigned i=0;i<bindings.size();++i) {bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
        VkDescriptorSetLayoutCreateInfo slots_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};slots_info.bindingCount=uint32_t(bindings.size());slots_info.pBindings=bindings.data();
        check(owner->vkCreateDescriptorSetLayout(device,&slots_info,nullptr,&compiled->slots),"vkCreateDescriptorSetLayout");
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};layout_info.setLayoutCount=1;layout_info.pSetLayouts=&compiled->slots;
        check(owner->vkCreatePipelineLayout(device,&layout_info,nullptr,&compiled->layout),"vkCreatePipelineLayout");
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pipeline_info.layout=compiled->layout;
        pipeline_info.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};pipeline_info.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pipeline_info.stage.module=compiled->shader;pipeline_info.stage.pName="main";
        check(owner->vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pipeline_info,nullptr,&compiled->pipeline),"vkCreateComputePipelines");
        return compiled;
    }
    device_work enqueue_dispatch(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters, native_queue* queue) override {
        return enqueue_dispatch_bound(pipeline,shape,memory,parameters,{},queue);
    }
    device_work enqueue_dispatch_bound(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters,std::span<buffer* const> bindings,native_queue* queue) override {
        std::lock_guard lock(owner->execution);
        auto recording = queue && queue->recording() ? dynamic_cast<vulkan_graph*>(queue) : nullptr;
        require(!recording || (recording->commands->owner == owner && !profile->enabled), CUDA_ERROR_NOT_SUPPORTED, "Vulkan dispatch cannot be recorded");
        std::array<buffer*,packet_direct_bindings> unused; unused.fill(&memory);
        if (bindings.empty()) bindings = unused;
        require(bindings.size()==pipeline.direct_buffers,CUDA_ERROR_INVALID_VALUE,"Vulkan direct binding count mismatch");
        for(auto resource:bindings)require(can_bind_direct(*resource),CUDA_ERROR_INVALID_VALUE,"incompatible Vulkan direct buffer");
        auto& compiled=static_cast<vulkan_pipeline&>(pipeline);auto device=owner->value;
        auto task = std::make_shared<vulkan_task>(owner);
        auto& descriptors = task->descriptors; auto& command = task->command;
        if (recording) recording->resources.push_back(task);
        if(profile->enabled && owner->timestamp_bits) {
            task->dispatch_profile=profile;
            VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};query.queryType=VK_QUERY_TYPE_TIMESTAMP;query.queryCount=2;
            check(owner->vkCreateQueryPool(device,&query,nullptr,&task->query_pool),"vkCreateQueryPool dispatch");
        }
        try {
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,uint32_t(2+bindings.size())};VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pool_info.maxSets=1;pool_info.poolSizeCount=1;pool_info.pPoolSizes=&pool_size;
            check(owner->vkCreateDescriptorPool(device,&pool_info,nullptr,&descriptors),"vkCreateDescriptorPool");
            VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};set_info.descriptorPool=descriptors;set_info.descriptorSetCount=1;set_info.pSetLayouts=&compiled.slots;VkDescriptorSet set;
            check(owner->vkAllocateDescriptorSets(device,&set_info,&set),"vkAllocateDescriptorSets");
            std::vector<VkDescriptorBufferInfo> buffers={{static_cast<vulkan_buffer&>(memory).value,0,memory.size},{static_cast<vulkan_buffer&>(parameters).value,0,parameters.size}};
            for(auto resource:bindings)buffers.push_back({static_cast<vulkan_buffer&>(resource->storage()).value,0,resource->storage().size});
            std::vector<VkWriteDescriptorSet> writes(buffers.size());
            for(unsigned i=0;i<writes.size();++i) {writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&buffers[i];}
            owner->vkUpdateDescriptorSets(device,uint32_t(writes.size()),writes.data(),0,nullptr);
            if (recording) { command = recording->commands->command; task->owns_command = false; }
            else {
                VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};allocate.commandPool=owner->pool;allocate.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;allocate.commandBufferCount=1;
                check(owner->vkAllocateCommandBuffers(device,&allocate,&command),"vkAllocateCommandBuffers");
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                check(owner->vkBeginCommandBuffer(command,&begin),"vkBeginCommandBuffer");
            }
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
            owner->vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            owner->vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,compiled.pipeline);owner->vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,compiled.layout,0,1,&set,0,nullptr);
            if(task->query_pool) {owner->vkCmdResetQueryPool(command,task->query_pool,0,2);owner->vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,task->query_pool,0);}
            owner->vkCmdDispatch(command,shape.grid[0],shape.grid[1],shape.grid[2]);
            if(task->query_pool) owner->vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,task->query_pool,1);
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT;
            owner->vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            if (!recording) {
                check(owner->vkEndCommandBuffer(command),"vkEndCommandBuffer");VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;
                check(owner->vkQueueSubmit(owner->queue,1,&submit,task->fence),"vkQueueSubmit"); task->submitted = true;
            }
        } catch(...) { throw; }
        return recording ? device_work{} : task;
    }
    void dispatch(compute_pipeline& pipeline,const launch_shape& shape,buffer& memory,buffer& parameters) override {
        wait_device(enqueue_dispatch(pipeline, shape, memory, parameters, nullptr));
    }
};
struct vulkan_backend : shader_backend {
    shader_resource_limits shader_limits(int ordinal) override {
        VkPhysicalDeviceProperties properties{};
        instance->load<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(devices.at(ordinal), &properties);
        shader_resource_limits result; result.threads = properties.limits.maxComputeWorkGroupInvocations;
        std::copy_n(properties.limits.maxComputeWorkGroupSize, 3, result.block.begin());
        std::copy_n(properties.limits.maxComputeWorkGroupCount, 3, result.grid.begin());
        result.shared_bytes = properties.limits.maxComputeSharedMemorySize; return result;
    }
    std::shared_ptr<vulkan_instance> instance = std::make_shared<vulkan_instance>();
    std::vector<VkPhysicalDevice> devices;
    device_resources<vulkan_device> resources;
    void physical_properties(int ordinal, void* chain) {
        require(instance->api_version >= VK_API_VERSION_1_1 || instance->external_memory, CUDA_ERROR_NOT_SUPPORTED, "Vulkan device identity properties are unavailable");
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; properties.pNext = chain;
        auto name = instance->api_version >= VK_API_VERSION_1_1 ? "vkGetPhysicalDeviceProperties2" : "vkGetPhysicalDeviceProperties2KHR";
        instance->load<PFN_vkGetPhysicalDeviceProperties2>(name)(devices.at(ordinal), &properties);
    }
    std::array<char, 16> device_uuid(int ordinal, bool) override {
        VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES}; physical_properties(ordinal, &ids);
        std::array<char, 16> result; std::memcpy(result.data(), ids.deviceUUID, result.size()); return result;
    }
    device_luid_info device_luid(int ordinal) override {
        VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES}; physical_properties(ordinal, &ids);
        require(ids.deviceLUIDValid, CUDA_ERROR_NOT_SUPPORTED, "Vulkan device LUID is unavailable");
        device_luid_info result; std::memcpy(result.bytes.data(), ids.deviceLUID, result.bytes.size()); result.node_mask = ids.deviceNodeMask; return result;
    }
    pci_address device_pci(int ordinal) override {
        auto enumerate = instance->load<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        uint32_t count = 0; check(enumerate(devices.at(ordinal), nullptr, &count, nullptr), "enumerate Vulkan device extensions");
        std::vector<VkExtensionProperties> extensions(count); check(enumerate(devices.at(ordinal), nullptr, &count, extensions.data()), "enumerate Vulkan device extensions");
        require(std::any_of(extensions.begin(), extensions.end(), [](auto& item) { return std::strcmp(item.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0; }), CUDA_ERROR_NOT_SUPPORTED, "Vulkan PCI bus information is unavailable");
        VkPhysicalDevicePCIBusInfoPropertiesEXT info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT}; physical_properties(ordinal, &info);
        return {info.pciDomain, info.pciBus, info.pciDevice, info.pciFunction};
    }
    std::vector<device_info> discover() override {
        auto enumerate = instance->load<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
        uint32_t count = 0;
        check(enumerate(instance->value, &count, nullptr), "vkEnumeratePhysicalDevices");
        std::vector<VkPhysicalDevice> native(count);
        if (count) check(enumerate(instance->value, &count, native.data()), "vkEnumeratePhysicalDevices");
        devices.clear();
        std::vector<device_info> result;
        for (auto device : native) {
            VkPhysicalDeviceProperties properties{};
            instance->load<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(device, &properties);
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
            VkPhysicalDeviceMemoryProperties memory{};
            instance->load<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(device, &memory);
            size_t bytes = 0;
            for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) bytes += static_cast<size_t>(memory.memoryHeaps[i].size);
            devices.push_back(device);
            result.push_back({properties.deviceName, bytes});
        }
        return result;
    }
    std::unique_ptr<session> open(int ordinal) override {
        auto device = devices.at(ordinal);
        return std::make_unique<vulkan_session>(resources.get(ordinal, [&] { return std::make_shared<vulkan_device>(instance, device); }));
    }
};
}
std::unique_ptr<backend> make_vulkan_backend() { return std::make_unique<vulkan_backend>(); }
}
