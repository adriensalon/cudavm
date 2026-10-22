#include "dynamic_library.hpp"
#include "cuda_graph_capture.hpp"
#include "ptx_parameters.hpp"
#include "nvidia_image.hpp"
#include "nvidia_managed_image.hpp"
#include <cctype>
#include <fstream>
#include <iterator>
#include <optional>
#include <cstdio>
namespace cudavm {
namespace {
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) throw failure(result, std::string(operation) + " failed: " + std::to_string(result));
}
struct cuda_api {
#if defined(_WIN32)
    dynamic_library driver{{"nvcuda.dll"}};
#else
    dynamic_library driver{{"libcuda.so.1", "libcuda.so"}};
#endif
    decltype(&cuInit) init = driver.get<decltype(init)>("cuInit");
    decltype(&cuDeviceGetCount) count = driver.get<decltype(count)>("cuDeviceGetCount");
    decltype(&cuDeviceGet) device = driver.get<decltype(device)>("cuDeviceGet");
    decltype(&cuDeviceGetName) name = driver.get<decltype(name)>("cuDeviceGetName");
    decltype(&cuDeviceTotalMem) memory = driver.get<decltype(memory)>("cuDeviceTotalMem_v2");
    decltype(&cuDeviceGetAttribute) attribute = driver.get<decltype(attribute)>("cuDeviceGetAttribute");
    using create_type = CUresult (CUDAAPI*)(CUcontext*, unsigned, CUdevice);
    create_type create = driver.get<create_type>("cuCtxCreate_v2");
    decltype(&cuCtxDestroy) destroy = driver.get<decltype(destroy)>("cuCtxDestroy_v2");
    decltype(&cuCtxPushCurrent) push = driver.get<decltype(push)>("cuCtxPushCurrent_v2");
    decltype(&cuCtxPopCurrent) pop = driver.get<decltype(pop)>("cuCtxPopCurrent_v2");
    decltype(&cuCtxSynchronize) sync = driver.get<decltype(sync)>("cuCtxSynchronize");
    decltype(&cuMemAlloc) allocate = driver.get<decltype(allocate)>("cuMemAlloc_v2");
    decltype(&cuMemAllocPitch) allocate_pitch = driver.get<decltype(allocate_pitch)>("cuMemAllocPitch_v2");
    decltype(&cuMemFree) free = driver.get<decltype(free)>("cuMemFree_v2");
    decltype(&cuMemcpyHtoD) write = driver.get<decltype(write)>("cuMemcpyHtoD_v2");
    decltype(&cuMemcpyDtoH) read = driver.get<decltype(read)>("cuMemcpyDtoH_v2");
    decltype(&cuMemcpyDtoD) copy = driver.get<decltype(copy)>("cuMemcpyDtoD_v2");
    decltype(&cuModuleLoad) load_file = driver.get<decltype(load_file)>("cuModuleLoad");
    decltype(&cuModuleLoadDataEx) load_image = driver.get<decltype(load_image)>("cuModuleLoadDataEx");
    decltype(&cuModuleUnload) unload = driver.get<decltype(unload)>("cuModuleUnload");
    decltype(&cuModuleGetFunction) function = driver.get<decltype(function)>("cuModuleGetFunction");
    decltype(&cuModuleGetGlobal) global = driver.get<decltype(global)>("cuModuleGetGlobal_v2");
    decltype(&cuLaunchKernel) launch = driver.get<decltype(launch)>("cuLaunchKernel");
    decltype(&cuFuncGetParamInfo) parameter_info = nullptr;
    cuda_api() {
        parameter_info = reinterpret_cast<decltype(parameter_info)>(driver.find("cuFuncGetParamInfo"));
        if (!parameter_info) {
            // A newer API can be available through procedure lookup without a
            // direct DLL export. Request its introduction-version contract.
            using resolver_type = CUresult (CUDAAPI*)(const char*, void**, int, cuuint64_t);
            auto resolve = reinterpret_cast<resolver_type>(driver.find("cuGetProcAddress"));
            void* address = nullptr;
            if (resolve && resolve("cuFuncGetParamInfo", &address, 12040, 0) == CUDA_SUCCESS)
                parameter_info = reinterpret_cast<decltype(parameter_info)>(address);
        }
    }
};
struct cuda_context {
    std::shared_ptr<cuda_api> api;
    CUcontext value = nullptr;
    CUdevice device;
    uint64_t graph_generation = 0;
    explicit cuda_context(std::shared_ptr<cuda_api> input, CUdevice device, unsigned flags) : api(std::move(input)), device(device) {
        check(api->create(&value, flags, device), "cuCtxCreate_v2");
        CUcontext popped = nullptr;
        check(api->pop(&popped), "cuCtxPopCurrent_v2");
    }
    ~cuda_context() { if (value) api->destroy(value); }
};
struct cuda_context_guard {
    cuda_context& context;
    explicit cuda_context_guard(cuda_context& input) : context(input) { check(context.api->push(context.value), "cuCtxPushCurrent_v2"); }
    ~cuda_context_guard() { CUcontext previous = nullptr; context.api->pop(&previous); }
};
struct cuda_buffer : buffer {
    std::shared_ptr<cuda_context> context;
    CUdeviceptr address = 0;
    std::shared_ptr<external_memory> imported;
    std::shared_ptr<executable> module;
    std::shared_ptr<host_allocation> mapped_host;
    std::shared_ptr<device_library> library;
    bool managed = false;
    cuda_buffer(std::shared_ptr<cuda_context> input, size_t bytes, CUdeviceptr pointer, std::shared_ptr<external_memory> memory)
        : buffer(bytes), context(std::move(input)), address(pointer), imported(std::move(memory)) {}
    cuda_buffer(std::shared_ptr<cuda_context> input, size_t size) : buffer(size), context(std::move(input)) {
        cuda_context_guard guard(*context);
        check(context->api->allocate(&address, size), "cuMemAlloc_v2");
    }
    ~cuda_buffer() {
        if (address && !module && !mapped_host && !library && context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->free(address);
            CUcontext previous = nullptr;
            context->api->pop(&previous);
        }
    }
    uint64_t native_address() const override { return address; }
    bool is_managed() const override { return managed; }
    size_t physical_size() const override { return imported || mapped_host || module || library ? 0 : size; }
};
struct cuda_mipmapped_storage {
    std::shared_ptr<cuda_context> context;
    CUmipmappedArray value = nullptr;
    explicit cuda_mipmapped_storage(std::shared_ptr<cuda_context> owner) : context(std::move(owner)) {}
    ~cuda_mipmapped_storage() {
        if (value && context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->driver.get<decltype(&cuMipmappedArrayDestroy)>("cuMipmappedArrayDestroy")(value);
            CUcontext previous{}; context->api->pop(&previous);
        }
    }
};
struct cuda_array_buffer : cuda_buffer {
    CUarray value = nullptr;
    CUarray cuda_array() const override { return value; }
    CUDA_ARRAY3D_DESCRIPTOR descriptor{};
    std::shared_ptr<cuda_mipmapped_storage> mipmapped;
    cuda_array_buffer(std::shared_ptr<cuda_context> owner, const CUDA_ARRAY3D_DESCRIPTOR& d)
        : cuda_buffer(std::move(owner), array_storage_bytes(d), 0, {}), descriptor(d) {}
    ~cuda_array_buffer() {
        if (value && !mipmapped && context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->driver.get<decltype(&cuArrayDestroy)>("cuArrayDestroy")(value);
            CUcontext previous{}; context->api->pop(&previous);
        }
    }
};
struct cuda_image_object : image_object {
    std::shared_ptr<cuda_context> context;
    bool surface;
    cuda_image_object(std::shared_ptr<cuda_context> owner, bool is_surface) : context(std::move(owner)), surface(is_surface) {}
    ~cuda_image_object() {
        if (handle && context->api->push(context->value) == CUDA_SUCCESS) {
            auto& driver = context->api->driver;
            if (surface) driver.get<decltype(&cuSurfObjectDestroy)>("cuSurfObjectDestroy")(handle);
            else driver.get<decltype(&cuTexObjectDestroy)>("cuTexObjectDestroy")(handle);
            CUcontext previous{}; context->api->pop(&previous);
        }
    }
};
CUDA_MEMCPY3D cuda_array_row(buffer* to, size_t destination, void* host_to,
                           buffer* from, size_t source, const void* host_from, size_t bytes) {
    CUDA_MEMCPY3D p{};
    p.srcMemoryType = p.dstMemoryType = CU_MEMORYTYPE_HOST;
    p.srcHost = host_from; p.dstHost = host_to;
    p.srcPitch = p.dstPitch = p.WidthInBytes = bytes; p.Height = p.Depth = 1;
    auto endpoint = [&](buffer* storage, size_t offset, bool target) {
        if (!storage) return;
        auto* image = dynamic_cast<cuda_array_buffer*>(storage);
        auto type = image ? CU_MEMORYTYPE_ARRAY : CU_MEMORYTYPE_DEVICE;
        size_t x = 0, y = 0, z = 0;
        if (image) {
            auto pitch = image->descriptor.Width * array_element_bytes(image->descriptor);
            auto height = std::max(size_t(1), image->descriptor.Height);
            x = offset % pitch; y = offset / pitch % height; z = offset / pitch / height;
        }
        auto address = image ? 0 : static_cast<cuda_buffer*>(storage)->address + offset;
        if (target) { p.dstMemoryType = type; p.dstArray = image ? image->value : nullptr; p.dstDevice = address; p.dstXInBytes = x; p.dstY = y; p.dstZ = z; }
        else { p.srcMemoryType = type; p.srcArray = image ? image->value : nullptr; p.srcDevice = address; p.srcXInBytes = x; p.srcY = y; p.srcZ = z; }
    };
    endpoint(to, destination, true); endpoint(from, source, false); return p;
}
// Reserve the graph's native address independently of physical memory. This
// lets a trim unmap idle storage without changing pointers embedded in kernels.
struct cuda_graph_buffer : cuda_buffer {
    CUmemAllocationProp properties{};
    size_t allocation_size = 0;
    CUmemGenericAllocationHandle physical = 0;
    bool mapped = false;
    size_t physical_size() const override { return physical ? allocation_size : 0; }
    cuda_graph_buffer(std::shared_ptr<cuda_context> owner, size_t bytes)
        : cuda_buffer(std::move(owner), bytes, 0, {}) {
        cuda_context_guard guard(*context);
        auto& driver = context->api->driver;
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE; properties.location.id = context->device;
        size_t granularity{};
        check(driver.get<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity")(
            &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity");
        require(granularity && bytes <= std::numeric_limits<size_t>::max() - (granularity - 1), CUDA_ERROR_OUT_OF_MEMORY, "graph allocation size overflow");
        allocation_size = ((bytes + granularity - 1) / granularity) * granularity;
        check(driver.get<decltype(&cuMemAddressReserve)>("cuMemAddressReserve")(&address, allocation_size, 0, 0, 0), "cuMemAddressReserve");
    }
    size_t restore_graph() override {
        if (mapped) return allocation_size;
        cuda_context_guard guard(*context);
        auto& driver = context->api->driver;
        if (!physical) check(driver.get<decltype(&cuMemCreate)>("cuMemCreate")(&physical, allocation_size, &properties, 0), "cuMemCreate");
        try {
            check(driver.get<decltype(&cuMemMap)>("cuMemMap")(address, allocation_size, 0, physical, 0), "cuMemMap");
            mapped = true;
            CUmemAccessDesc access{properties.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
            check(driver.get<decltype(&cuMemSetAccess)>("cuMemSetAccess")(address, allocation_size, &access, 1), "cuMemSetAccess");
        } catch (...) {
            if (mapped) driver.get<decltype(&cuMemUnmap)>("cuMemUnmap")(address, allocation_size);
            mapped = false; driver.get<decltype(&cuMemRelease)>("cuMemRelease")(physical); physical = 0; throw;
        }
        return allocation_size;
    }
    void trim_graph() override {
        cuda_context_guard guard(*context);
        auto& driver = context->api->driver;
        if (mapped) { check(driver.get<decltype(&cuMemUnmap)>("cuMemUnmap")(address, allocation_size), "cuMemUnmap"); mapped = false; }
        if (physical) { check(driver.get<decltype(&cuMemRelease)>("cuMemRelease")(physical), "cuMemRelease"); physical = 0; }
    }
    ~cuda_graph_buffer() {
        if (address && context->api->push(context->value) == CUDA_SUCCESS) {
            auto& driver = context->api->driver;
            if (mapped) driver.get<decltype(&cuMemUnmap)>("cuMemUnmap")(address, allocation_size);
            if (physical) driver.get<decltype(&cuMemRelease)>("cuMemRelease")(physical);
            driver.get<decltype(&cuMemAddressFree)>("cuMemAddressFree")(address, allocation_size);
            CUcontext previous{}; context->api->pop(&previous);
        }
        address = 0;
    }
};
struct cuda_event;
struct cuda_queue : native_queue {
    std::shared_ptr<cuda_context> context;
    CUstream value = nullptr;
    bool capturing = false;
    std::mutex completion_mutex;
    std::vector<std::shared_ptr<cuda_event>> completion_events;
    std::shared_ptr<cuda_event> acquire_completion();
    cuda_queue(std::shared_ptr<cuda_context> owner, int priority) : context(std::move(owner)) {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuStreamCreateWithPriority)>("cuStreamCreateWithPriority")(&value, CU_STREAM_NON_BLOCKING, priority), "cuStreamCreateWithPriority");
    }
    ~cuda_queue() {
        if (context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->driver.get<decltype(&cuStreamDestroy)>("cuStreamDestroy_v2")(value);
            CUcontext previous = nullptr; context->api->pop(&previous);
        }
    }
};
struct cuda_event : native_signal, device_timer {
    std::shared_ptr<cuda_context> context;
    CUevent value = nullptr;
    explicit cuda_event(std::shared_ptr<cuda_context> owner, unsigned flags) : context(std::move(owner)) {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuEventCreate)>("cuEventCreate")(&value, flags), "cuEventCreate");
    }
    bool can_wait(native_queue* queue) const override {
        // CUDA permits event waits from another CUDA context. Other adapters
        // retain scheduler completion, without attempting native handle casts.
        return dynamic_cast<cuda_queue*>(queue) != nullptr;
    }
    void enqueue_wait(native_queue* queue) const override {
        auto& lane = *static_cast<cuda_queue*>(queue);
        cuda_context_guard guard(*lane.context);
        check(lane.context->api->driver.get<decltype(&cuStreamWaitEvent)>("cuStreamWaitEvent")(lane.value, value, 0), "cuStreamWaitEvent");
    }
    float elapsed(const device_timer& end) const override {
        auto* last = dynamic_cast<const cuda_event*>(&end);
        require(last && last->context == context, CUDA_ERROR_INVALID_HANDLE, "event device clocks differ");
        cuda_context_guard guard(*context); float milliseconds = 0;
        check(context->api->driver.get<decltype(&cuEventElapsedTime)>("cuEventElapsedTime")(&milliseconds, value, last->value), "cuEventElapsedTime");
        return milliseconds;
    }
    ~cuda_event() {
        if (value && context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->driver.get<decltype(&cuEventDestroy)>("cuEventDestroy_v2")(value);
            CUcontext previous = nullptr; context->api->pop(&previous);
        }
    }
};
struct cuda_task : device_task {
    std::shared_ptr<cuda_event> event;
    bool recorded = false;
    explicit cuda_task(std::shared_ptr<cuda_context> owner, unsigned flags = CU_EVENT_DISABLE_TIMING)
        : event(std::make_shared<cuda_event>(std::move(owner), flags)) {}
    explicit cuda_task(std::shared_ptr<cuda_event> input) : event(std::move(input)) {}
    std::shared_ptr<native_signal> signal() const override { return recorded ? event : nullptr; }
    void record(CUstream stream) {
        check(event->context->api->driver.get<decltype(&cuEventRecord)>("cuEventRecord")(event->value, stream), "cuEventRecord"); recorded = true;
    }
    bool query() override {
        cuda_context_guard guard(*event->context);
        auto result = event->context->api->driver.get<decltype(&cuEventQuery)>("cuEventQuery")(event->value);
        if (result == CUDA_ERROR_NOT_READY) return false;
        check(result, "cuEventQuery"); return true;
    }
    ~cuda_task() {
        auto& context = *event->context;
        if (recorded && context.api->push(context.value) == CUDA_SUCCESS) {
            context.api->driver.get<decltype(&cuEventSynchronize)>("cuEventSynchronize")(event->value);
            CUcontext previous = nullptr; context.api->pop(&previous);
        }
    }
};
std::shared_ptr<cuda_event> cuda_queue::acquire_completion() {
    std::lock_guard lock(completion_mutex);
    // A completed task may still have native waiters or a published signal.
    // Only pool ownership permits re-recording; task completion alone does not.
    for (auto& event : completion_events) if (event.use_count() == 1) return event;
    auto event = std::make_shared<cuda_event>(context, CU_EVENT_DISABLE_TIMING);
    // Bound idle command resources even when many replays are queued at once.
    if (completion_events.size() < 8) completion_events.push_back(event);
    return event;
}
struct cuda_launch_event : device_launch_event {
    std::shared_ptr<cuda_context> context;
    explicit cuda_launch_event(std::shared_ptr<cuda_context> owner) : context(std::move(owner)) {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuEventCreate)>("cuEventCreate")(&handle, CU_EVENT_DISABLE_TIMING), "cuEventCreate for launch dependency");
    }
    ~cuda_launch_event() {
        if (handle && context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->driver.get<decltype(&cuEventDestroy)>("cuEventDestroy_v2")(handle);
            CUcontext previous = nullptr; context->api->pop(&previous);
        }
    }
};
template<class F> device_work cuda_enqueue(std::shared_ptr<cuda_context> context, native_queue* queue, F action) {
    cuda_context_guard guard(*context);
    auto& lane = *static_cast<cuda_queue*>(queue);
    auto stream = lane.value;
    if (lane.capturing) { action(stream); return {}; }
    auto done = std::make_shared<cuda_task>(lane.acquire_completion());
    try { action(stream); done->record(stream); }
    catch (...) {
        // Submission may have partially succeeded. Retain arguments and buffers
        // until this stream drains before reporting the submission error.
        context->api->driver.get<decltype(&cuStreamSynchronize)>("cuStreamSynchronize")(stream); throw;
    }
    return done;
}
struct cuda_graph : native_graph {
    std::shared_ptr<cuda_context> context;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    explicit cuda_graph(std::shared_ptr<cuda_context> owner) : context(std::move(owner)) {}
    ~cuda_graph() {
        if (context->api->push(context->value) == CUDA_SUCCESS) {
            auto& driver = context->api->driver;
            if (executable) driver.get<decltype(&cuGraphExecDestroy)>("cuGraphExecDestroy")(executable);
            if (graph) driver.get<decltype(&cuGraphDestroy)>("cuGraphDestroy")(graph);
            CUcontext previous = nullptr; context->api->pop(&previous);
        }
    }
    device_work enqueue(native_queue* queue) override {
        return cuda_enqueue(context, queue, [&](CUstream stream) {
            check(context->api->driver.get<decltype(&cuGraphLaunch)>("cuGraphLaunch")(executable, stream), "cuGraphLaunch native region");
        });
    }
};
#include "cuda_graph_definition.inl"
struct cuda_memory : external_memory, std::enable_shared_from_this<cuda_memory> {
    std::shared_ptr<cuda_context> context;
    CUexternalMemory value = nullptr;
    cuda_memory(std::shared_ptr<cuda_context> input, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& descriptor) : context(std::move(input)) {
        cuda_context_guard guard(*context);
        auto import = context->api->driver.get<decltype(&cuImportExternalMemory)>("cuImportExternalMemory");
        check(import(&value, &descriptor), "cuImportExternalMemory");
    }
    ~cuda_memory() {
        if (value && context->api->push(context->value) == CUDA_SUCCESS) {
            auto destroy = reinterpret_cast<decltype(&cuDestroyExternalMemory)>(context->api->driver.find("cuDestroyExternalMemory"));
            if (destroy) destroy(value);
            CUcontext previous = nullptr; context->api->pop(&previous);
        }
    }
    std::unique_ptr<buffer> map(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC& descriptor) override {
        cuda_context_guard guard(*context); CUdeviceptr pointer = 0;
        auto map = context->api->driver.get<decltype(&cuExternalMemoryGetMappedBuffer)>("cuExternalMemoryGetMappedBuffer");
        check(map(&pointer, value, &descriptor), "cuExternalMemoryGetMappedBuffer");
        try { return std::make_unique<cuda_buffer>(context, size_t(descriptor.size), pointer, shared_from_this()); }
        catch (...) { context->api->free(pointer); throw; }
    }
};
struct cuda_host : host_allocation {
    std::shared_ptr<cuda_context> context;
    bool registered;
    cuda_host(std::shared_ptr<cuda_context> owner, void* pointer, size_t bytes, unsigned flags, bool registration)
        : host_allocation(bytes), context(std::move(owner)), registered(registration) {
        cuda_context_guard guard(*context);
        if (registered) {
            check(context->api->driver.get<decltype(&cuMemHostRegister)>("cuMemHostRegister_v2")(pointer, bytes, flags), "cuMemHostRegister_v2");
            address = pointer;
        } else check(context->api->driver.get<decltype(&cuMemHostAlloc)>("cuMemHostAlloc")(&address, bytes, flags), "cuMemHostAlloc");
    }
    ~cuda_host() {
        if (address && context->api->push(context->value) == CUDA_SUCCESS) {
            if (registered) context->api->driver.get<decltype(&cuMemHostUnregister)>("cuMemHostUnregister")(address);
            else context->api->driver.get<decltype(&cuMemFreeHost)>("cuMemFreeHost")(address);
            CUcontext previous = nullptr; context->api->pop(&previous);
        }
    }
    unsigned flags(unsigned requested) override {
        if (registered) return requested;
        cuda_context_guard guard(*context); unsigned value = 0;
        check(context->api->driver.get<decltype(&cuMemHostGetFlags)>("cuMemHostGetFlags")(&value, address), "cuMemHostGetFlags");
        return value;
    }
};
struct cuda_semaphore : external_semaphore {
    std::shared_ptr<cuda_context> context;
    CUexternalSemaphore value = nullptr;
    cuda_semaphore(std::shared_ptr<cuda_context> input, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& descriptor) : context(std::move(input)) {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuImportExternalSemaphore)>("cuImportExternalSemaphore")(&value, &descriptor), "cuImportExternalSemaphore");
    }
    ~cuda_semaphore() {
        if (value && context->api->push(context->value) == CUDA_SUCCESS) {
            auto destroy = reinterpret_cast<decltype(&cuDestroyExternalSemaphore)>(context->api->driver.find("cuDestroyExternalSemaphore"));
            if (destroy) destroy(value);
            CUcontext previous = nullptr; context->api->pop(&previous);
        }
    }
    void signal(uint64_t epoch) override {
        cuda_context_guard guard(*context); CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{}; params.params.fence.value = epoch;
        check(context->api->driver.get<decltype(&cuSignalExternalSemaphoresAsync)>("cuSignalExternalSemaphoresAsync")(&value, &params, 1, nullptr), "cuSignalExternalSemaphoresAsync");
    }
    void wait(uint64_t epoch) override {
        cuda_context_guard guard(*context); CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{}; params.params.fence.value = epoch;
        check(context->api->driver.get<decltype(&cuWaitExternalSemaphoresAsync)>("cuWaitExternalSemaphoresAsync")(&value, &params, 1, nullptr), "cuWaitExternalSemaphoresAsync");
    }
};
struct cuda_module;
template<class T> void cuda_image_metadata(T& target, std::span<const std::byte> image) {
    auto metadata = nvidia_image_metadata(image);
    for (auto& [name, parameters] : metadata.kernels) {
        target.names.push_back(name);
        if (!metadata.unavailable.contains(name)) target.layouts.emplace(name, std::move(parameters));
    }
    target.layout_metadata = true;
}
std::vector<CUlaunchAttribute> cuda_launch_attributes(const launch_shape& shape) {
    std::vector<CUlaunchAttribute> result;
    auto add = [&](CUlaunchAttributeID id) -> CUlaunchAttributeValue& { result.push_back({}); result.back().id = id; return result.back().value; };
    const auto& options = shape.options;
    if (options.cooperative) add(CU_LAUNCH_ATTRIBUTE_COOPERATIVE).cooperative = 1;
    if (options.cluster) {
        const auto& dim = *options.cluster; add(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION).clusterDim = {dim[0], dim[1], dim[2]};
    }
    if (options.cluster_policy) add(CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE).clusterSchedulingPolicyPreference = *options.cluster_policy;
    if (options.priority) add(CU_LAUNCH_ATTRIBUTE_PRIORITY).priority = *options.priority;
    if (options.domain_map) {
        const auto& map = *options.domain_map; add(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP).memSyncDomainMap = {map[0], map[1]};
    }
    if (options.domain) add(CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN).memSyncDomain = *options.domain;
    if (options.access_policy) add(CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW).accessPolicyWindow = access_policy_to_native(*options.access_policy);
    if (options.shared_memory_mode) add(CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE).sharedMemoryMode = *options.shared_memory_mode;
    if (options.preferred_cluster) {
        const auto& dim = *options.preferred_cluster; add(CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION).preferredClusterDim = {dim[0], dim[1], dim[2]};
    }
    if (options.carveout) add(CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT).sharedMemCarveout = *options.carveout;
    if (options.nvlink) add(CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING).nvlinkUtilCentricScheduling = *options.nvlink;
    if (options.portable_cluster) add(CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE).portableClusterSizeMode = *options.portable_cluster;
    if (options.programmatic_serialization) add(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION).programmaticStreamSerializationAllowed = *options.programmatic_serialization;
    if (options.programmatic_event) {
        const auto& event = *options.programmatic_event;
        require(bool(event.native), CUDA_ERROR_INVALID_HANDLE, "programmatic event has no native binding");
        add(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT).programmaticEvent = {event.native->handle, event.flags, int(event.block_start)};
    }
    if (options.completion_event) {
        const auto& event = *options.completion_event;
        require(bool(event.native), CUDA_ERROR_INVALID_HANDLE, "launch completion event has no native binding");
        add(CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT).launchCompletionEvent = {event.native->handle, event.flags};
    }
    return result;
}
CUlaunchConfig cuda_launch_config(const launch_shape& shape, CUstream stream, std::vector<CUlaunchAttribute>& attributes) {
    return {shape.grid[0], shape.grid[1], shape.grid[2], shape.block[0], shape.block[1], shape.block[2], shape.shared_bytes,
        stream, attributes.data(), unsigned(attributes.size())};
}
struct cuda_kernel : kernel {
    std::shared_ptr<cuda_module> module;
    CUfunction value = nullptr;
    CUfunction cuda_function() const override { return value; }
    uint64_t generation = 0;
    uint64_t graph_generation() const override { return generation; }
    explicit cuda_kernel(std::shared_ptr<cuda_module> input) : module(std::move(input)) {}
    void launch(const launch_shape&, void**, void**) override;
    device_work enqueue_launch(const launch_shape&, void**, void**, std::span<const allocation_binding>, native_queue*) override;
    void parameter_info(size_t, size_t&, size_t&) override;
    std::vector<kernel_parameter> parameter_layout() override;
    std::optional<std::vector<kernel_parameter>> parameters;
    unsigned max_grid[3]{}, max_block[3]{}, max_threads = 0;
    int attribute(CUfunction_attribute) override;
    void set_attribute(CUfunction_attribute, int) override;
    void set_cache_config(CUfunc_cache) override;
    void set_shared_config(CUsharedconfig) override;
    int active_blocks(int, size_t, unsigned) override;
    void potential_block_size(int&, int&, CUoccupancyB2DSize, size_t, int, unsigned) override;
    size_t available_dynamic_shared(int, int) override;
    int cluster_occupancy(const launch_shape&, native_queue*, bool) override;
    std::array<unsigned char, 2> default_domain_map() override;
    void validate_configuration(const launch_shape&) override;
    void submit(const launch_shape&, CUstream, void**, void**);
};
struct cuda_module : executable, std::enable_shared_from_this<cuda_module> {
    std::shared_ptr<cuda_context> context;
    CUmodule value = nullptr;
    std::shared_ptr<device_library> library;
    std::unordered_map<std::string, std::vector<kernel_parameter>> layouts;
    bool layout_metadata = false;
    std::vector<std::string> names;
    explicit cuda_module(std::shared_ptr<cuda_context> input) : context(std::move(input)) {}
    ~cuda_module() {
        if (value && !library && context->api->push(context->value) == CUDA_SUCCESS) {
            context->api->unload(value);
            CUcontext previous = nullptr;
            context->api->pop(&previous);
        }
    }
    std::unique_ptr<kernel> find(const char* name) override {
        auto result = std::make_unique<cuda_kernel>(shared_from_this());
        cuda_context_guard guard(*context);
        check(context->api->function(&result->value, value, name), "cuModuleGetFunction");
        if (auto found = layouts.find(name); found != layouts.end()) result->parameters = found->second;
        CUdevice device = 0;
        check(context->api->driver.get<decltype(&cuCtxGetDevice)>("cuCtxGetDevice")(&device), "cuCtxGetDevice");
        auto attribute = [&](CUdevice_attribute which) { int value = 0; check(context->api->attribute(&value, which, device), "cuDeviceGetAttribute"); return unsigned(value); };
        for (unsigned i = 0; i < 3; ++i) {
            result->max_grid[i] = attribute(static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i));
            result->max_block[i] = attribute(static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X + i));
        }
        auto function_attribute = context->api->driver.get<decltype(&cuFuncGetAttribute)>("cuFuncGetAttribute");
        int threads = 0;
        check(function_attribute(&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, result->value), "cuFuncGetAttribute");
        result->max_threads = unsigned(threads);
        return result;
    }
    std::unique_ptr<buffer> find_global(const char* name) override {
        cuda_context_guard guard(*context); CUdeviceptr pointer=0; size_t bytes=0;
        check(context->api->global(&pointer,&bytes,value,name), "cuModuleGetGlobal_v2");
        auto result=std::make_unique<cuda_buffer>(context,bytes,pointer,std::shared_ptr<external_memory>{});
        result->module=shared_from_this();
        unsigned managed = 0;
        check(context->api->driver.get<decltype(&cuPointerGetAttribute)>("cuPointerGetAttribute")(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, pointer), "cuPointerGetAttribute");
        result->managed = managed != 0; return result;
    }
    void unload() override {
        require(!library, CUDA_ERROR_NOT_PERMITTED, "library modules are unloaded with their library");
        cuda_context_guard guard(*context);
        check(context->api->unload(value), "cuModuleUnload");
        value = nullptr;
    }
    std::vector<std::string> kernel_names() const override {
        std::vector<std::string> result;
        auto& driver = context->api->driver;
        auto count = reinterpret_cast<decltype(&cuModuleGetFunctionCount)>(driver.find("cuModuleGetFunctionCount"));
        auto enumerate = reinterpret_cast<decltype(&cuModuleEnumerateFunctions)>(driver.find("cuModuleEnumerateFunctions"));
        auto name = reinterpret_cast<decltype(&cuFuncGetName)>(driver.find("cuFuncGetName"));
        if (count && enumerate && name) {
            cuda_context_guard guard(*context); unsigned size = 0; check(count(&size, value), "cuModuleGetFunctionCount");
            std::vector<CUfunction> functions(size); if (size) check(enumerate(functions.data(), size, value), "cuModuleEnumerateFunctions");
            for (auto function : functions) { const char* text = nullptr; check(name(&text, function), "cuFuncGetName"); result.emplace_back(text); }
        } else { require(layout_metadata, CUDA_ERROR_NOT_SUPPORTED, "native binary function enumeration is unavailable"); result = names; }
        std::sort(result.begin(), result.end()); return result;
    }
};
void cuda_kernel::launch(const launch_shape& shape, void** parameters, void** extra) {
    cuda_context_guard guard(*module->context);
    auto& api = *module->context->api;
    submit(shape, nullptr, parameters, extra);
    // Scheduler completion must include device execution before releasing the
    // captured arguments/resources or publishing a following event.
    check(api.sync(), "cuCtxSynchronize after kernel launch");
}
device_work cuda_kernel::enqueue_launch(const launch_shape& shape, void** parameters, void** extra, std::span<const allocation_binding>, native_queue* queue) {
    return cuda_enqueue(module->context, queue, [&](CUstream stream) {
        submit(shape, stream, parameters, extra);
    });
}
void cuda_kernel::submit(const launch_shape& shape, CUstream stream, void** parameters, void** extra) {
    auto& api = *module->context->api;
    auto attributes = cuda_launch_attributes(shape);
    if (attributes.empty()) {
        check(api.launch(value, shape.grid[0], shape.grid[1], shape.grid[2], shape.block[0], shape.block[1], shape.block[2],
            shape.shared_bytes, stream, parameters, extra), "cuLaunchKernel");
    } else {
        auto config = cuda_launch_config(shape, stream, attributes);
        check(api.driver.get<decltype(&cuLaunchKernelEx)>("cuLaunchKernelEx")(&config, value, parameters, extra), "cuLaunchKernelEx");
    }
}
void cuda_kernel::parameter_info(size_t index, size_t& offset, size_t& size) {
    if (parameters) {
        require(index < parameters->size(), CUDA_ERROR_INVALID_VALUE, "kernel parameter index is out of range");
        offset = (*parameters)[index].offset; size = (*parameters)[index].size; return;
    }
    auto& api = *module->context->api;
    require(api.parameter_info != nullptr, CUDA_ERROR_NOT_SUPPORTED, "native cuFuncGetParamInfo is unavailable");
    cuda_context_guard guard(*module->context);
    check(api.parameter_info(value, index, &offset, &size), "cuFuncGetParamInfo");
}
std::vector<kernel_parameter> cuda_kernel::parameter_layout() {
    return parameters ? *parameters : kernel::parameter_layout();
}
int cuda_kernel::attribute(CUfunction_attribute which) {
    cuda_context_guard guard(*module->context); int result = 0;
    check(module->context->api->driver.get<decltype(&cuFuncGetAttribute)>("cuFuncGetAttribute")(&result, which, value), "cuFuncGetAttribute");
    return result;
}
void cuda_kernel::set_attribute(CUfunction_attribute which, int input) {
    cuda_context_guard guard(*module->context);
    check(module->context->api->driver.get<decltype(&cuFuncSetAttribute)>("cuFuncSetAttribute")(value, which, input), "cuFuncSetAttribute");
    ++generation;
}
void cuda_kernel::set_cache_config(CUfunc_cache config) {
    cuda_context_guard guard(*module->context);
    check(module->context->api->driver.get<decltype(&cuFuncSetCacheConfig)>("cuFuncSetCacheConfig")(value, config), "cuFuncSetCacheConfig");
    ++generation;
}
void cuda_kernel::set_shared_config(CUsharedconfig config) {
    cuda_context_guard guard(*module->context);
    check(module->context->api->driver.get<decltype(&cuFuncSetSharedMemConfig)>("cuFuncSetSharedMemConfig")(value, config), "cuFuncSetSharedMemConfig");
    ++generation;
}
int cuda_kernel::active_blocks(int block, size_t shared, unsigned flags) {
    cuda_context_guard guard(*module->context); int result = 0;
    check(module->context->api->driver.get<decltype(&cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)>("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")(&result, value, block, shared, flags), "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    return result;
}
void cuda_kernel::potential_block_size(int& grid, int& block, CUoccupancyB2DSize callback, size_t shared, int limit, unsigned flags) {
    cuda_context_guard guard(*module->context);
    check(module->context->api->driver.get<decltype(&cuOccupancyMaxPotentialBlockSizeWithFlags)>("cuOccupancyMaxPotentialBlockSizeWithFlags")(&grid, &block, value, callback, shared, limit, flags), "cuOccupancyMaxPotentialBlockSizeWithFlags");
}
size_t cuda_kernel::available_dynamic_shared(int blocks, int block) {
    cuda_context_guard guard(*module->context); size_t result = 0;
    check(module->context->api->driver.get<decltype(&cuOccupancyAvailableDynamicSMemPerBlock)>("cuOccupancyAvailableDynamicSMemPerBlock")(&result, value, blocks, block), "cuOccupancyAvailableDynamicSMemPerBlock");
    return result;
}
std::array<unsigned char, 2> cuda_kernel::default_domain_map() {
    cuda_context_guard guard(*module->context); CUstreamAttrValue result{};
    check(module->context->api->driver.get<decltype(&cuStreamGetAttribute)>("cuStreamGetAttribute")(
        nullptr, CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &result), "cuStreamGetAttribute for memory synchronization domains");
    return {result.memSyncDomainMap.default_, result.memSyncDomainMap.remote};
}
int cuda_kernel::cluster_occupancy(const launch_shape& shape, native_queue* queue, bool potential) {
    kernel::validate_configuration(shape);
    cuda_context_guard guard(*module->context); int result = 0;
    auto attributes = cuda_launch_attributes(shape);
    auto config = cuda_launch_config(shape, queue ? static_cast<cuda_queue*>(queue)->value : nullptr, attributes);
    auto& driver = module->context->api->driver;
    if (potential) check(driver.get<decltype(&cuOccupancyMaxPotentialClusterSize)>("cuOccupancyMaxPotentialClusterSize")(&result, value, &config), "cuOccupancyMaxPotentialClusterSize");
    else check(driver.get<decltype(&cuOccupancyMaxActiveClusters)>("cuOccupancyMaxActiveClusters")(&result, value, &config), "cuOccupancyMaxActiveClusters");
    return result;
}
void cuda_kernel::validate_configuration(const launch_shape& shape) {
    kernel::validate_configuration(shape);
    uint64_t threads = 1;
    for (unsigned i = 0; i < 3; ++i) {
        require(shape.grid[i] <= max_grid[i] && shape.block[i] <= max_block[i] && threads <= max_threads / shape.block[i],
                CUDA_ERROR_INVALID_VALUE, "launch dimensions exceed device/function limits");
        threads *= shape.block[i];
    }
    // Explicit shared-memory modes replace the function's dynamic opt-in
    // ceiling. Older native drivers have no mode attribute and use that ceiling.
    int mode = CU_SHARED_MEMORY_MODE_DEFAULT;
    {
        cuda_context_guard guard(*module->context);
        auto status = module->context->api->driver.get<decltype(&cuFuncGetAttribute)>("cuFuncGetAttribute")(
            &mode, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, value);
        if (shape.options.shared_memory_mode) check(status, "cuFuncGetAttribute for launch shared-memory modes");
        else if (status == CUDA_ERROR_INVALID_VALUE || status == CUDA_ERROR_NOT_SUPPORTED) mode = CU_SHARED_MEMORY_MODE_DEFAULT;
        else check(status, "cuFuncGetAttribute for shared-memory mode");
    }
    if (shape.options.shared_memory_mode) mode = *shape.options.shared_memory_mode;
    if (mode == CU_SHARED_MEMORY_MODE_DEFAULT) {
        auto maximum = attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
        require(shape.shared_bytes <= unsigned(maximum), CUDA_ERROR_INVALID_VALUE, "dynamic shared memory exceeds function limits");
    } else {
        auto which = mode == CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE ? CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK :
            mode == CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE ? CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN :
            CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK;
        int maximum = 0; auto& context = *module->context; cuda_context_guard guard(context);
        check(context.api->attribute(&maximum, which, context.device), "cuDeviceGetAttribute for shared-memory mode");
        auto fixed = attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
        require(maximum >= fixed && shape.shared_bytes <= unsigned(maximum - fixed), CUDA_ERROR_INVALID_VALUE, "shared memory exceeds the selected native mode limit");
    }
    const auto& options = shape.options;
    if (options == launch_options{}) return;
    auto& context = *module->context; cuda_context_guard guard(context); auto& api = *context.api;
    require(api.driver.find("cuLaunchKernelEx"), CUDA_ERROR_NOT_SUPPORTED, "native extended kernel launch is unavailable");
    auto device_attribute = [&](CUdevice_attribute which) {
        int result = 0; check(api.attribute(&result, which, context.device), "cuDeviceGetAttribute for launch admission"); return result;
    };
    if (options.cooperative) {
        require(device_attribute(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH), CUDA_ERROR_NOT_SUPPORTED, "device does not support cooperative launch");
        uint64_t capacity = uint64_t(active_blocks(int(threads), shape.shared_bytes, 0)) * device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        uint64_t groups = 1;
        for (auto dimension : shape.grid) {
            require(groups <= capacity / dimension, CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE, "cooperative grid exceeds native residency capacity");
            groups *= dimension;
        }
    }
    if (options.cluster || options.cluster_policy || options.preferred_cluster || options.portable_cluster) {
        auto clusters = cluster_occupancy(shape, nullptr, false);
        if (options.cluster) require(clusters > 0, CUDA_ERROR_INVALID_CLUSTER_SIZE, "cluster launch has no resident configuration");
    }
    if (options.access_policy) {
        validate_access_policy(*options.access_policy);
        require(options.access_policy->bytes <= size_t(device_attribute(CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE)),
            CUDA_ERROR_INVALID_VALUE, "access policy window exceeds the native device limit");
    }
    if (options.domain_map) {
        auto count = device_attribute(CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT);
        require((*options.domain_map)[0] < count && (*options.domain_map)[1] < count, CUDA_ERROR_INVALID_VALUE, "memory synchronization domain map exceeds device count");
    }
}
struct cuda_session : session {
    uint64_t graph_generation() const override { return context->graph_generation; }
    std::shared_ptr<cuda_context> context;
    cuda_session(std::shared_ptr<cuda_api> api, CUdevice device, unsigned flags = CU_CTX_MAP_HOST)
        : context(std::make_shared<cuda_context>(std::move(api), device, flags)) {}
    std::shared_ptr<device_launch_event> create_launch_event() override { return std::make_shared<cuda_launch_event>(context); }
    std::pair<int, int> stream_priority_range() override {
        cuda_context_guard guard(*context); int least = 0, greatest = 0;
        check(context->api->driver.get<decltype(&cuCtxGetStreamPriorityRange)>("cuCtxGetStreamPriorityRange")(&least, &greatest), "cuCtxGetStreamPriorityRange");
        return {least, greatest};
    }
    CUstreamAttrValue canonical_stream_attribute(CUstreamAttrID attribute, const CUstreamAttrValue& input, int priority) override {
        cuda_context_guard guard(*context);
        // Validate and canonicalize with a private stream. The scheduler may
        // still have earlier launches to submit on a public stream's queue.
        // Mutating that queue here would change their inherited attributes.
        auto queue = std::static_pointer_cast<cuda_queue>(create_queue(priority));
        auto& driver = context->api->driver; CUstreamAttrValue result{};
        check(driver.get<decltype(&cuStreamSetAttribute)>("cuStreamSetAttribute")(queue->value, attribute, &input), "cuStreamSetAttribute");
        check(driver.get<decltype(&cuStreamGetAttribute)>("cuStreamGetAttribute")(queue->value, attribute, &result), "cuStreamGetAttribute");
        return result;
    }
    std::array<unsigned char, 2> default_stream_domain_map() override {
        cuda_context_guard guard(*context); CUstreamAttrValue value{};
        check(context->api->driver.get<decltype(&cuStreamGetAttribute)>("cuStreamGetAttribute")(nullptr, CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &value), "cuStreamGetAttribute");
        return {value.memSyncDomainMap.default_, value.memSyncDomainMap.remote};
    }
    void reset_persisting_cache() override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuCtxResetPersistingL2Cache)>("cuCtxResetPersistingL2Cache")(), "cuCtxResetPersistingL2Cache");
    }
    unsigned get_context_flags(unsigned) override {
        cuda_context_guard guard(*context); unsigned flags = 0;
        check(context->api->driver.get<decltype(&cuCtxGetFlags)>("cuCtxGetFlags")(&flags), "cuCtxGetFlags"); return flags;
    }
    unsigned set_context_flags(unsigned flags) override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuCtxSetFlags)>("cuCtxSetFlags")(flags), "cuCtxSetFlags");
        ++context->graph_generation;
        return get_context_flags(flags);
    }
    unsigned context_api_version() override {
        cuda_context_guard guard(*context); unsigned version = 0;
        check(context->api->driver.get<decltype(&cuCtxGetApiVersion)>("cuCtxGetApiVersion")(context->value, &version), "cuCtxGetApiVersion"); return version;
    }
    size_t context_limit(CUlimit limit) override {
        cuda_context_guard guard(*context); size_t value = 0;
        check(context->api->driver.get<decltype(&cuCtxGetLimit)>("cuCtxGetLimit")(&value, limit), "cuCtxGetLimit"); return value;
    }
    void set_context_limit(CUlimit limit, size_t value) override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuCtxSetLimit)>("cuCtxSetLimit")(limit, value), "cuCtxSetLimit");
        ++context->graph_generation;
    }
    CUfunc_cache context_cache_config() override {
        cuda_context_guard guard(*context); CUfunc_cache value{};
        check(context->api->driver.get<decltype(&cuCtxGetCacheConfig)>("cuCtxGetCacheConfig")(&value), "cuCtxGetCacheConfig"); return value;
    }
    void set_context_cache_config(CUfunc_cache config) override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuCtxSetCacheConfig)>("cuCtxSetCacheConfig")(config), "cuCtxSetCacheConfig");
        ++context->graph_generation;
    }
    CUsharedconfig context_shared_config() override {
        cuda_context_guard guard(*context); CUsharedconfig value{};
        check(context->api->driver.get<decltype(&cuCtxGetSharedMemConfig)>("cuCtxGetSharedMemConfig")(&value), "cuCtxGetSharedMemConfig"); return value;
    }
    void set_context_shared_config(CUsharedconfig config) override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuCtxSetSharedMemConfig)>("cuCtxSetSharedMemConfig")(config), "cuCtxSetSharedMemConfig");
        ++context->graph_generation;
    }
    interop_capabilities interop() override {
        interop_capabilities result;
        auto& driver = context->api->driver;
        if (driver.find("cuImportExternalMemory") && driver.find("cuExternalMemoryGetMappedBuffer") && driver.find("cuDestroyExternalMemory")) {
#if defined(_WIN32)
            for (unsigned type = 2; type <= 7; ++type) result.memory |= handle_bit(type);
#else
            result.memory = handle_bit(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD);
#endif
        }
        if (driver.find("cuImportExternalSemaphore") && driver.find("cuDestroyExternalSemaphore") && driver.find("cuSignalExternalSemaphoresAsync") && driver.find("cuWaitExternalSemaphoresAsync")) {
#if defined(_WIN32)
            for (unsigned type : {2u, 3u, 4u, 5u, 10u}) result.semaphore |= handle_bit(type);
#else
            result.semaphore = handle_bit(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD) | handle_bit(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD);
#endif
        }
        return result;
    }
    std::shared_ptr<external_memory> import_memory(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC& d, const cudavmExternalMemoryMetadata* = nullptr) override {
        require(interop().memory & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "native CUDA memory handle type is unavailable");
        return std::make_shared<cuda_memory>(context, d);
    }
    std::shared_ptr<external_semaphore> import_semaphore(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC& d) override {
        require(interop().semaphore & handle_bit(d.type), CUDA_ERROR_NOT_SUPPORTED, "native CUDA semaphore handle type is unavailable");
        return std::make_shared<cuda_semaphore>(context, d);
    }
    std::unique_ptr<buffer> allocate(size_t size) override { return std::make_unique<cuda_buffer>(context, size); }
    std::shared_ptr<buffer> allocate_array(const CUDA_ARRAY3D_DESCRIPTOR& d) override {
        auto result = std::make_shared<cuda_array_buffer>(context, d);
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuArray3DCreate)>("cuArray3DCreate_v2")(&result->value, &d), "cuArray3DCreate_v2");
        return result;
    }
    std::shared_ptr<image_object> create_texture(buffer& storage, const CUDA_RESOURCE_DESC& input, size_t offset, const CUDA_TEXTURE_DESC& texture, const CUDA_RESOURCE_VIEW_DESC* view) override {
        auto result = std::make_shared<cuda_image_object>(context, false);
        auto resource = input;
        if (resource.resType == CU_RESOURCE_TYPE_ARRAY) resource.res.array.hArray = static_cast<cuda_array_buffer&>(storage).value;
        else if (resource.resType == CU_RESOURCE_TYPE_LINEAR) resource.res.linear.devPtr = storage.native_address() + offset;
        else resource.res.pitch2D.devPtr = storage.native_address() + offset;
        cuda_context_guard guard(*context);
        CUtexObject handle{};
        check(context->api->driver.get<decltype(&cuTexObjectCreate)>("cuTexObjectCreate")(&handle, &resource, &texture, view), "cuTexObjectCreate");
        result->handle = handle;
        return result;
    }
    std::shared_ptr<image_object> create_surface(buffer& storage) override {
        auto result = std::make_shared<cuda_image_object>(context, true);
        CUDA_RESOURCE_DESC resource{}; resource.resType = CU_RESOURCE_TYPE_ARRAY;
        resource.res.array.hArray = static_cast<cuda_array_buffer&>(storage).value;
        cuda_context_guard guard(*context);
        CUsurfObject handle{};
        check(context->api->driver.get<decltype(&cuSurfObjectCreate)>("cuSurfObjectCreate")(&handle, &resource), "cuSurfObjectCreate");
        result->handle = handle;
        return result;
    }
    std::vector<std::shared_ptr<buffer>> allocate_mipmapped_array(const CUDA_ARRAY3D_DESCRIPTOR& d, unsigned levels) override {
        auto parent = std::make_shared<cuda_mipmapped_storage>(context);
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuMipmappedArrayCreate)>("cuMipmappedArrayCreate")(&parent->value, &d, levels), "cuMipmappedArrayCreate");
        std::vector<std::shared_ptr<buffer>> result;
        for (unsigned i = 0; i < levels; ++i) {
            auto level = std::make_shared<cuda_array_buffer>(context, array_level_descriptor(d, i)); level->mipmapped = parent;
            check(context->api->driver.get<decltype(&cuMipmappedArrayGetLevel)>("cuMipmappedArrayGetLevel")(&level->value, parent->value, i), "cuMipmappedArrayGetLevel");
            // Native descriptors are authoritative for layer/face dimensions.
            check(context->api->driver.get<decltype(&cuArray3DGetDescriptor)>("cuArray3DGetDescriptor_v2")(&level->descriptor, level->value), "cuArray3DGetDescriptor_v2");
            result.push_back(level);
        }
        return result;
    }
    device_work transfer_array_row(buffer* to, size_t destination, void* host_to, buffer* from, size_t source,
                                   const void* host_from, size_t bytes, native_queue* queue, bool asynchronous) {
        auto p = cuda_array_row(to, destination, host_to, from, source, host_from, bytes);
        auto submit = [&](CUstream stream) {
            auto& driver = context->api->driver;
            auto to_context = to ? static_cast<cuda_buffer*>(to)->context : context;
            auto from_context = from ? static_cast<cuda_buffer*>(from)->context : context;
            if (to_context != from_context) {
                CUDA_MEMCPY3D_PEER peer{};
                peer.srcMemoryType=p.srcMemoryType; peer.srcHost=p.srcHost; peer.srcDevice=p.srcDevice; peer.srcArray=p.srcArray;
                peer.srcXInBytes=p.srcXInBytes; peer.srcY=p.srcY; peer.srcZ=p.srcZ; peer.srcPitch=p.srcPitch; peer.srcContext=from_context->value;
                peer.dstMemoryType=p.dstMemoryType; peer.dstHost=p.dstHost; peer.dstDevice=p.dstDevice; peer.dstArray=p.dstArray;
                peer.dstXInBytes=p.dstXInBytes; peer.dstY=p.dstY; peer.dstZ=p.dstZ; peer.dstPitch=p.dstPitch; peer.dstContext=to_context->value;
                peer.WidthInBytes=bytes; peer.Height=peer.Depth=1;
                if (asynchronous) check(driver.get<decltype(&cuMemcpy3DPeerAsync)>("cuMemcpy3DPeerAsync")(&peer,stream), "cuMemcpy3DPeerAsync");
                else check(driver.get<decltype(&cuMemcpy3DPeer)>("cuMemcpy3DPeer")(&peer), "cuMemcpy3DPeer");
            } else if (asynchronous) check(driver.get<decltype(&cuMemcpy3DAsync)>("cuMemcpy3DAsync_v2")(&p,stream), "cuMemcpy3DAsync_v2");
            else check(driver.get<decltype(&cuMemcpy3D)>("cuMemcpy3D_v2")(&p), "cuMemcpy3D_v2");
        };
        if (asynchronous) return cuda_enqueue(context, queue, submit);
        cuda_context_guard guard(*context); submit(nullptr); check(context->api->sync(), "cuCtxSynchronize after array transfer"); return {};
    }
    std::unique_ptr<buffer> allocate_managed(size_t bytes, unsigned flags) override {
        auto result = std::make_unique<cuda_buffer>(context, bytes, 0, std::shared_ptr<external_memory>{});
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuMemAllocManaged)>("cuMemAllocManaged")(&result->address, bytes, flags), "cuMemAllocManaged");
        result->managed = true; return result;
    }
    device_work enqueue_attach(CUdeviceptr pointer, size_t bytes, unsigned flags, native_queue* queue) override {
        return cuda_enqueue(context, queue, [&](CUstream stream) {
            check(context->api->driver.get<decltype(&cuStreamAttachMemAsync)>("cuStreamAttachMemAsync")(stream, pointer, bytes, flags), "cuStreamAttachMemAsync");
        });
    }
    std::unique_ptr<buffer> allocate_pool(size_t size) override {
        auto result = std::make_unique<cuda_graph_buffer>(context, size);
        result->restore_graph();
        return result;
    }
    void memory_info(size_t& free, size_t& total) override {
        cuda_context_guard guard(*context);
        check(context->api->driver.get<decltype(&cuMemGetInfo)>("cuMemGetInfo_v2")(&free, &total), "cuMemGetInfo_v2");
    }
    bool can_copy_direct(const buffer& value) const override {
        auto native = dynamic_cast<const cuda_buffer*>(&value.storage());
        return native && native->context->device == context->device && !native->imported && !native->mapped_host;
    }
    std::unique_ptr<buffer> reserve_graph(size_t size) override { return std::make_unique<cuda_graph_buffer>(context, size); }
    std::unique_ptr<host_allocation> allocate_host(size_t bytes, unsigned flags) override { return std::make_unique<cuda_host>(context, nullptr, bytes, flags, false); }
    std::unique_ptr<host_allocation> register_host(void* pointer, size_t bytes, unsigned flags) override { return std::make_unique<cuda_host>(context, pointer, bytes, flags, true); }
    std::unique_ptr<buffer> map_host(std::shared_ptr<host_allocation> host) override {
        cuda_context_guard guard(*context); CUdeviceptr pointer = 0;
        check(context->api->driver.get<decltype(&cuMemHostGetDevicePointer)>("cuMemHostGetDevicePointer_v2")(&pointer, host->address, 0), "cuMemHostGetDevicePointer_v2");
        auto result = std::make_unique<cuda_buffer>(context, host->size, pointer, std::shared_ptr<external_memory>{});
        result->mapped_host = std::move(host); return result;
    }
    std::shared_ptr<native_queue> create_queue(int priority) override { return std::make_shared<cuda_queue>(context, priority); }
    device_work enqueue_marker(native_queue* queue) override {
        return cuda_enqueue(context, queue, [](CUstream) {});
    }
    device_work enqueue_timestamp(std::shared_ptr<device_timestamp> timestamp, native_queue* queue) override {
        cuda_context_guard guard(*context);
        auto task = std::make_shared<cuda_task>(context, 0);
        task->record(static_cast<cuda_queue*>(queue)->value);
        timestamp->timer = task->event;
        return task;
    }
    std::shared_ptr<native_graph> compile_graph(std::span<const native_graph_node> nodes) override {
        auto& driver = context->api->driver;
        auto begin = reinterpret_cast<decltype(&cuStreamBeginCapture)>(driver.find("cuStreamBeginCapture_v2"));
        auto end = reinterpret_cast<decltype(&cuStreamEndCapture)>(driver.find("cuStreamEndCapture"));
        auto instantiate = reinterpret_cast<decltype(&cuGraphInstantiateWithFlags)>(driver.find("cuGraphInstantiateWithFlags"));
        cuda_graph_capture capture{
            reinterpret_cast<cuda_graph_capture::capture_info>(driver.find("cuStreamGetCaptureInfo_v2")),
            reinterpret_cast<cuda_graph_capture::capture_dependencies>(driver.find("cuStreamUpdateCaptureDependencies")),
            reinterpret_cast<decltype(&cuStreamUpdateCaptureDependencies)>(driver.find("cuStreamUpdateCaptureDependencies_v2")),
            reinterpret_cast<decltype(&cuGraphKernelNodeSetAttribute)>(driver.find("cuGraphKernelNodeSetAttribute")),
            reinterpret_cast<decltype(&cuGraphNodeGetType)>(driver.find("cuGraphNodeGetType"))};
        if (!begin || !end || !instantiate || !capture.available(nodes)) return {};
        cuda_context_guard guard(*context);
        cuda_queue lane(context, 0);
        auto result = std::make_shared<cuda_graph>(context);
        auto status = begin(lane.value, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        if (status == CUDA_ERROR_NOT_SUPPORTED) return {};
        check(status, "cuStreamBeginCapture native region");
        lane.capturing = true;
        try {
            capture.record(nodes, &lane, lane.value);
        } catch (...) {
            // End invalidated capture before destroying the private lane. No
            // recorded command has reached the GPU, so fallback is atomic.
            end(lane.value, &result->graph); lane.capturing = false;
            try { throw; } catch (const failure& error) {
                if (error.code == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED || error.code == CUDA_ERROR_NOT_SUPPORTED) return {};
                throw;
            }
        }
        lane.capturing = false;
        check(end(lane.value, &result->graph), "cuStreamEndCapture native region");
        status = instantiate(&result->executable, result->graph, capture.set_attribute ? CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY : 0);
        if (status == CUDA_ERROR_NOT_SUPPORTED) return {};
        check(status, "cuGraphInstantiateWithFlags native region");
        return result;
    }
    std::shared_ptr<native_graph_definition> create_graph() override {
        return std::make_shared<cuda_graph_definition>(context);
    }
    device_work enqueue_write(buffer& value, size_t offset, std::span<const std::byte> bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_write(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (dynamic_cast<cuda_array_buffer*>(&value)) return transfer_array_row(&value,offset,nullptr,nullptr,0,bytes.data(),bytes.size(),queue,true);
        return cuda_enqueue(context, queue, [&](CUstream stream) { check(context->api->driver.get<decltype(&cuMemcpyHtoDAsync)>("cuMemcpyHtoDAsync_v2")(static_cast<cuda_buffer&>(value).address + offset, bytes.data(), bytes.size(), stream), "cuMemcpyHtoDAsync_v2"); });
    }
    device_work enqueue_read(buffer& value, size_t offset, std::span<std::byte> bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_read(value.storage(), offset + value.storage_offset(), bytes, queue);
        if (dynamic_cast<cuda_array_buffer*>(&value)) return transfer_array_row(nullptr,0,bytes.data(),&value,offset,nullptr,bytes.size(),queue,true);
        return cuda_enqueue(context, queue, [&](CUstream stream) { check(context->api->driver.get<decltype(&cuMemcpyDtoHAsync)>("cuMemcpyDtoHAsync_v2")(bytes.data(), static_cast<cuda_buffer&>(value).address + offset, bytes.size(), stream), "cuMemcpyDtoHAsync_v2"); });
    }
    device_work enqueue_copy(buffer& to, size_t destination, buffer& from, size_t source, size_t bytes, native_queue* queue) override {
        if (&to.storage() != &to || &from.storage() != &from) return enqueue_copy(to.storage(), destination + to.storage_offset(), from.storage(), source + from.storage_offset(), bytes, queue);
        if (dynamic_cast<cuda_array_buffer*>(&to) || dynamic_cast<cuda_array_buffer*>(&from)) return transfer_array_row(&to,destination,nullptr,&from,source,nullptr,bytes,queue,true);
        auto& target = static_cast<cuda_buffer&>(to); auto& input = static_cast<cuda_buffer&>(from);
        return cuda_enqueue(context, queue, [&](CUstream stream) {
            if (target.context != input.context)
                check(context->api->driver.get<decltype(&cuMemcpyPeerAsync)>("cuMemcpyPeerAsync")(target.address + destination, target.context->value, input.address + source, input.context->value, bytes, stream), "cuMemcpyPeerAsync");
            else check(context->api->driver.get<decltype(&cuMemcpyDtoDAsync)>("cuMemcpyDtoDAsync_v2")(target.address + destination, input.address + source, bytes, stream), "cuMemcpyDtoDAsync_v2");
        });
    }
    device_work enqueue_fill(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes, native_queue* queue) override {
        if (&value.storage() != &value) return enqueue_fill(value.storage(), offset + value.storage_offset(), pattern, bytes, queue);
        return cuda_enqueue(context, queue, [&](CUstream stream) {
            auto address = static_cast<cuda_buffer&>(value).address + offset;
            auto& driver = context->api->driver;
            if (pattern.size() == 1) { unsigned char v; std::memcpy(&v, pattern.data(), 1); check(driver.get<decltype(&cuMemsetD8Async)>("cuMemsetD8Async")(address, v, bytes, stream), "cuMemsetD8Async"); }
            else if (pattern.size() == 2) { unsigned short v; std::memcpy(&v, pattern.data(), 2); check(driver.get<decltype(&cuMemsetD16Async)>("cuMemsetD16Async")(address, v, bytes / 2, stream), "cuMemsetD16Async"); }
            else { unsigned v; std::memcpy(&v, pattern.data(), 4); check(driver.get<decltype(&cuMemsetD32Async)>("cuMemsetD32Async")(address, v, bytes / 4, stream), "cuMemsetD32Async"); }
        });
    }
    void fill_device(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes) override {
        if (!bytes) return;
        cuda_queue queue(context, 0);
        wait_device(enqueue_fill(value, offset, pattern, bytes, &queue));
    }
    std::unique_ptr<buffer> allocate_pitched(size_t width, size_t height, unsigned element, size_t& pitch) override {
        cuda_context_guard guard(*context); CUdeviceptr pointer = 0; size_t value = 0;
        check(context->api->allocate_pitch(&pointer, &value, width, height, element), "cuMemAllocPitch_v2");
        try {
            auto result = std::make_unique<cuda_buffer>(context, value * height, pointer, std::shared_ptr<external_memory>{});
            pitch = value; return result;
        } catch (...) { context->api->free(pointer); throw; }
    }
    void write_device(buffer& value, size_t offset, std::span<const std::byte> data) override {
        if (&value.storage() != &value) { write_device(value.storage(), offset + value.storage_offset(), data); return; }
        if (dynamic_cast<cuda_array_buffer*>(&value)) { transfer_array_row(&value,offset,nullptr,nullptr,0,data.data(),data.size(),nullptr,false); return; }
        cuda_context_guard guard(*context);
        check(context->api->write(static_cast<cuda_buffer&>(value).address + offset, data.data(), data.size()), "cuMemcpyHtoD_v2");
        // Pageable HtoD can return after staging while DMA is still pending.
        // The synchronous session path must finish before an independent native
        // stream can use the buffer or a scheduler action publishes completion.
        check(context->api->driver.get<decltype(&cuStreamSynchronize)>("cuStreamSynchronize")(nullptr), "cuStreamSynchronize after host write");
    }
    void read_device(buffer& value, size_t offset, std::span<std::byte> data) override {
        if (&value.storage() != &value) { read_device(value.storage(), offset + value.storage_offset(), data); return; }
        if (dynamic_cast<cuda_array_buffer*>(&value)) { transfer_array_row(nullptr,0,data.data(),&value,offset,nullptr,data.size(),nullptr,false); return; }
        cuda_context_guard guard(*context);
        check(context->api->read(data.data(), static_cast<cuda_buffer&>(value).address + offset, data.size()), "cuMemcpyDtoH_v2");
    }
    void copy_device(buffer& destination, size_t to, buffer& source, size_t from, size_t size) override {
        if (&destination.storage() != &destination || &source.storage() != &source) { copy_device(destination.storage(), to + destination.storage_offset(), source.storage(), from + source.storage_offset(), size); return; }
        if (dynamic_cast<cuda_array_buffer*>(&destination) || dynamic_cast<cuda_array_buffer*>(&source)) { transfer_array_row(&destination,to,nullptr,&source,from,nullptr,size,nullptr,false); return; }
        cuda_context_guard guard(*context);
        auto& target = static_cast<cuda_buffer&>(destination); auto& input = static_cast<cuda_buffer&>(source);
        if (target.context != input.context)
            check(context->api->driver.get<decltype(&cuMemcpyPeer)>("cuMemcpyPeer")(target.address + to, target.context->value, input.address + from, input.context->value, size), "cuMemcpyPeer");
        else check(context->api->copy(target.address + to, input.address + from, size), "cuMemcpyDtoD_v2");
        check(context->api->sync(), "cuCtxSynchronize");
    }
    void synchronize() override { cuda_context_guard guard(*context); check(context->api->sync(), "cuCtxSynchronize"); }
    std::shared_ptr<executable> load_image(const void* image, unsigned count, CUjit_option* options, void** values) override {
        auto result = std::make_shared<cuda_module>(context);
        cuda_context_guard guard(*context);
        check(context->api->load_image(&result->value, image, count, options, values), "cuModuleLoadDataEx");
        auto first = *static_cast<const unsigned char*>(image);
        if (first == '.' || first == '/' || first == ' ' || first == '\n' || first == '\r' || first == '\t') {
            result->layouts = ptx_parameter_layouts(static_cast<const char*>(image));
            result->layout_metadata = true;
            result->names = ptx_entry_names(static_cast<const char*>(image));
            try { result->data_views = decode_data_views(static_cast<const char*>(image)); }
            catch (const std::exception& e) { throw failure(CUDA_ERROR_INVALID_IMAGE, e.what()); }
        }
        else if (auto bytes = nvidia_image_length(image)) cuda_image_metadata(*result, std::span(static_cast<const std::byte*>(image), bytes));
        return result;
    }
    std::shared_ptr<executable> load_file(const char* path) override {
        auto result = std::make_shared<cuda_module>(context);
        cuda_context_guard guard(*context);
        check(context->api->load_file(&result->value, path), "cuModuleLoad");
        std::ifstream file(path, std::ios::binary);
        std::string image((std::istreambuf_iterator<char>(file)), {});
        if (!image.empty() && (image.front() == '.' || image.front() == '/' || std::isspace(static_cast<unsigned char>(image.front())))) {
            result->layouts = ptx_parameter_layouts(image);
            result->layout_metadata = true;
            result->names = ptx_entry_names(image);
            try { result->data_views = decode_data_views(image); }
            catch (const std::exception& e) { throw failure(CUDA_ERROR_INVALID_IMAGE, e.what()); }
        }
        else if (!image.empty()) cuda_image_metadata(*result, std::as_bytes(std::span(image.data(), image.size())));
        return result;
    }
};
struct cuda_device_library : device_library, std::enable_shared_from_this<cuda_device_library> {
    std::shared_ptr<cuda_api> api;
    CUlibrary value = nullptr;
    std::unordered_map<std::string, std::vector<kernel_parameter>> layouts;
    bool layout_metadata = false;
    std::vector<std::string> names;
    std::map<std::string, data_view_descriptor> views;
    explicit cuda_device_library(std::shared_ptr<cuda_api> input) : api(std::move(input)) {}
    ~cuda_device_library() { if (value) { auto fn = reinterpret_cast<decltype(&cuLibraryUnload)>(api->driver.find("cuLibraryUnload")); if (fn) fn(value); } }
    CUkernel kernel_handle(const char* name) {
        CUkernel result = nullptr; check(api->driver.get<decltype(&cuLibraryGetKernel)>("cuLibraryGetKernel")(&result, value, name), "cuLibraryGetKernel"); return result;
    }
    void find(const char* name) override { (void)kernel_handle(name); }
    std::vector<std::string> kernel_names() override {
        std::vector<std::string> result;
        auto count = reinterpret_cast<decltype(&cuLibraryGetKernelCount)>(api->driver.find("cuLibraryGetKernelCount"));
        auto enumerate = reinterpret_cast<decltype(&cuLibraryEnumerateKernels)>(api->driver.find("cuLibraryEnumerateKernels"));
        if (count && enumerate) {
            unsigned size = 0; check(count(&size, value), "cuLibraryGetKernelCount"); std::vector<CUkernel> kernels(size);
            if (size) check(enumerate(kernels.data(), size, value), "cuLibraryEnumerateKernels");
            for (auto item : kernels) { const char* name = nullptr; check(api->driver.get<decltype(&cuKernelGetName)>("cuKernelGetName")(&name, item), "cuKernelGetName"); result.emplace_back(name); }
        } else { require(layout_metadata, CUDA_ERROR_NOT_SUPPORTED, "native binary kernel enumeration is unavailable"); result = names; }
        std::sort(result.begin(), result.end()); return result;
    }
    std::vector<kernel_parameter> parameters(const char* name) override {
        if (auto it = layouts.find(name); it != layouts.end()) return it->second;
        auto item = kernel_handle(name); auto fn = api->driver.get<decltype(&cuKernelGetParamInfo)>("cuKernelGetParamInfo");
        std::vector<kernel_parameter> result;
        for (size_t i = 0; i <= 4096; ++i) { size_t offset = 0, size = 0; auto code = fn(item, i, &offset, &size);
            if (code == CUDA_ERROR_INVALID_VALUE) return result; check(code, "cuKernelGetParamInfo");
            require(i < 4096, CUDA_ERROR_NOT_SUPPORTED, "kernel parameter inventory exceeds transport limit"); result.push_back({offset, size}); }
        return result;
    }
    int attribute(const char* name, CUfunction_attribute which, int device) override {
        int result = 0; check(api->driver.get<decltype(&cuKernelGetAttribute)>("cuKernelGetAttribute")(&result, which, kernel_handle(name), device), "cuKernelGetAttribute"); return result;
    }
    void set_attribute(const char* name, CUfunction_attribute which, int number, int device) override {
        check(api->driver.get<decltype(&cuKernelSetAttribute)>("cuKernelSetAttribute")(which, number, kernel_handle(name), device), "cuKernelSetAttribute");
    }
    void set_cache_config(const char* name, CUfunc_cache config, int device) override {
        check(api->driver.get<decltype(&cuKernelSetCacheConfig)>("cuKernelSetCacheConfig")(kernel_handle(name), config, device), "cuKernelSetCacheConfig");
    }
    std::unique_ptr<buffer> managed(const char* name, session& owner) override {
        auto& target = static_cast<cuda_session&>(owner); cuda_context_guard guard(*target.context);
        CUdeviceptr pointer = 0; size_t bytes = 0;
        check(api->driver.get<decltype(&cuLibraryGetManaged)>("cuLibraryGetManaged")(&pointer, &bytes, value, name), "cuLibraryGetManaged");
        auto result = std::make_unique<cuda_buffer>(target.context, bytes, pointer, std::shared_ptr<external_memory>{});
        result->library = shared_from_this(); result->managed = true; return result;
    }
    void* unified_function(const char* name) override {
        void* result = nullptr;
        check(api->driver.get<decltype(&cuLibraryGetUnifiedFunction)>("cuLibraryGetUnifiedFunction")(&result, value, name), "cuLibraryGetUnifiedFunction");
        return result;
    }
    std::shared_ptr<executable> bind(session& owner) override {
        auto& target = static_cast<cuda_session&>(owner); auto result = std::make_shared<cuda_module>(target.context);
        cuda_context_guard guard(*target.context); check(api->driver.get<decltype(&cuLibraryGetModule)>("cuLibraryGetModule")(&result->value, value), "cuLibraryGetModule");
        result->library = shared_from_this(); result->layouts = layouts; result->layout_metadata = layout_metadata; result->names = names; result->data_views = views; return result;
    }
    std::shared_ptr<executable> bind_registered(session& owner, std::span<const std::byte> image, const std::map<std::string, std::shared_ptr<buffer>>& storage) override {
        auto& target = static_cast<cuda_session&>(owner); cuda_context_guard guard(*target.context);
        CUdevice device = 0; int major = 0, minor = 0;
        check(api->driver.get<decltype(&cuCtxGetDevice)>("cuCtxGetDevice")(&device), "cuCtxGetDevice");
        check(api->attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute");
        check(api->attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute");
        auto selected = nvidia_select_binding_image(image, unsigned(major * 10 + minor));
        if (selected.ptx) {
            CUlinkState link = nullptr;
            check(api->driver.get<decltype(&cuLinkCreate)>("cuLinkCreate_v2")(0, nullptr, nullptr, &link), "cuLinkCreate");
            try {
                selected.bytes.push_back(std::byte{});
                check(api->driver.get<decltype(&cuLinkAddData)>("cuLinkAddData_v2")(link, CU_JIT_INPUT_PTX, selected.bytes.data(), selected.bytes.size(), "managed registration", 0, nullptr, nullptr), "cuLinkAddData");
                void* binary = nullptr; size_t bytes = 0;
                check(api->driver.get<decltype(&cuLinkComplete)>("cuLinkComplete")(link, &binary, &bytes), "cuLinkComplete");
                selected.bytes.assign(static_cast<std::byte*>(binary), static_cast<std::byte*>(binary) + bytes);
            } catch (...) { api->driver.get<decltype(&cuLinkDestroy)>("cuLinkDestroy")(link); throw; }
            check(api->driver.get<decltype(&cuLinkDestroy)>("cuLinkDestroy")(link), "cuLinkDestroy");
        }
        std::map<std::string, std::pair<CUdeviceptr, size_t>> bindings;
        for (auto& [name, native] : storage) bindings.emplace(name, std::pair(native->native_address(), native->size));
        auto binary = nvidia_bind_managed_symbols(selected.bytes, bindings);
        return target.load_image(binary.data(), 0, nullptr, nullptr);
    }
    void unload() override { check(api->driver.get<decltype(&cuLibraryUnload)>("cuLibraryUnload")(value), "cuLibraryUnload"); value = nullptr; }
};
struct cuda_backend : backend {
    std::shared_ptr<cuda_api> api = std::make_shared<cuda_api>();
    CUmoduleLoadingMode loading_mode() override {
        CUmoduleLoadingMode mode{}; check(api->driver.get<decltype(&cuModuleGetLoadingMode)>("cuModuleGetLoadingMode")(&mode), "cuModuleGetLoadingMode"); return mode;
    }
    std::shared_ptr<device_library> load_library(const void* image, const char* path, CUjit_option* options, void** values, unsigned count,
                                               CUlibraryOption* library_options, void** library_values, unsigned library_count) override {
        auto result = std::make_shared<cuda_device_library>(api);
        if (path) check(api->driver.get<decltype(&cuLibraryLoadFromFile)>("cuLibraryLoadFromFile")(&result->value, path, options, values, count, library_options, library_values, library_count), "cuLibraryLoadFromFile");
        else check(api->driver.get<decltype(&cuLibraryLoadData)>("cuLibraryLoadData")(&result->value, image, options, values, count, library_options, library_values, library_count), "cuLibraryLoadData");
        std::string text;
        if (path) { std::ifstream file(path, std::ios::binary); text.assign(std::istreambuf_iterator<char>(file), {}); }
        else if (image) { auto first = *static_cast<const unsigned char*>(image); if (first == '.' || first == '/' || std::isspace(first)) text = static_cast<const char*>(image);
            else if (auto bytes = nvidia_image_length(image)) text.assign(static_cast<const char*>(image), bytes); }
        if (!text.empty() && (text.front() == '.' || text.front() == '/' || std::isspace(static_cast<unsigned char>(text.front())))) {
            result->layouts = ptx_parameter_layouts(text); result->layout_metadata = true; result->names = ptx_entry_names(text);
            try { result->views = decode_data_views(text); } catch (const std::exception& e) { throw failure(CUDA_ERROR_INVALID_IMAGE, e.what()); }
        }
        else if (!text.empty()) cuda_image_metadata(*result, std::as_bytes(std::span(text.data(), text.size())));
        return result;
    }
    std::vector<device_info> discover() override {
        check(api->init(0), "cuInit");
        int count = 0;
        check(api->count(&count), "cuDeviceGetCount");
        std::vector<device_info> devices;
        for (int i = 0; i < count; ++i) {
            CUdevice device{};
            char name[256]{};
            size_t bytes = 0;
            check(api->device(&device, i), "cuDeviceGet");
            check(api->name(name, sizeof(name), device), "cuDeviceGetName");
            check(api->memory(&bytes, device), "cuDeviceTotalMem_v2");
            devices.push_back({name, bytes});
        }
        return devices;
    }
    std::unique_ptr<session> open(int ordinal) override {
        CUdevice device{};
        check(api->device(&device, ordinal), "cuDeviceGet");
        return std::make_unique<cuda_session>(api, device);
    }
    std::unique_ptr<session> open(int ordinal, unsigned flags) override {
        CUdevice device{}; check(api->device(&device, ordinal), "cuDeviceGet");
        return std::make_unique<cuda_session>(api, device, flags);
    }
    int attribute(int ordinal, CUdevice_attribute attribute) override {
        CUdevice device{};
        int result = 0;
        check(api->device(&device, ordinal), "cuDeviceGet");
        check(api->attribute(&result, attribute, device), "cuDeviceGetAttribute");
        return result;
    }
    std::array<char, 16> device_uuid(int ordinal, bool instance) override {
        CUdevice device{}; check(api->device(&device, ordinal), "cuDeviceGet"); CUuuid uuid{};
        auto name = instance ? "cuDeviceGetUuid_v2" : "cuDeviceGetUuid";
        check(api->driver.get<decltype(&cuDeviceGetUuid)>(name)(&uuid, device), name);
        std::array<char, 16> result; std::memcpy(result.data(), uuid.bytes, result.size()); return result;
    }
    device_luid_info device_luid(int ordinal) override {
        CUdevice device{}; check(api->device(&device, ordinal), "cuDeviceGet"); device_luid_info result;
        check(api->driver.get<decltype(&cuDeviceGetLuid)>("cuDeviceGetLuid")(result.bytes.data(), &result.node_mask, device), "cuDeviceGetLuid"); return result;
    }
    pci_address device_pci(int ordinal) override {
        CUdevice device{}; check(api->device(&device, ordinal), "cuDeviceGet"); char text[32]{}; pci_address result;
        check(api->driver.get<decltype(&cuDeviceGetPCIBusId)>("cuDeviceGetPCIBusId")(text, sizeof(text), device), "cuDeviceGetPCIBusId");
        require(std::sscanf(text, "%x:%x:%x.%x", &result.domain, &result.bus, &result.device, &result.function) == 4, CUDA_ERROR_UNKNOWN, "invalid native PCI address"); return result;
    }
    std::optional<int> device_from_pci(const char* text) override {
        CUdevice device{}; check(api->driver.get<decltype(&cuDeviceGetByPCIBusId)>("cuDeviceGetByPCIBusId")(&device, text), "cuDeviceGetByPCIBusId");
        int count = 0; check(api->count(&count), "cuDeviceGetCount");
        for (int i = 0; i < count; ++i) { CUdevice candidate{}; check(api->device(&candidate, i), "cuDeviceGet"); if (candidate == device) return i; }
        throw failure(CUDA_ERROR_INVALID_DEVICE, "native PCI device is absent from the selected adapter");
    }
};
}
std::unique_ptr<backend> make_cuda_backend() { return std::make_unique<cuda_backend>(); }
}
