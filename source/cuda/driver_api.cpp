#include "backend.hpp"
#include "driver_api.hpp"
#include "opengl_graphics.hpp"
#include "command_queue.hpp"
#include "command_domain.hpp"
#include "../cudavm/module_descriptor.hpp"
#include "shader_ptx.hpp"
#include "ptx_parameters.hpp"
#include "nvidia_image.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <charconv>
#include <cstdio>
#include <string_view>

#if defined(__clang__) && defined(_WIN32)
// The SDK declares these APIs without dll attributes. Definitions must export
// their exact SDK symbols from this compatibility DLL.
#pragma clang diagnostic ignored "-Wdll-attribute-on-redeclaration"
#endif

// These APIs are also used by the callback configurator below. Export decoration
// must precede their first use when compiling the Windows DLL.
extern "C" CUDAVM_API CUresult CUDAAPI cuFuncGetAttribute(int*, CUfunction_attribute, CUfunction);
extern "C" CUDAVM_API CUresult CUDAAPI cuDeviceGetAttribute(int*, CUdevice_attribute, CUdevice);
extern "C" CUDAVM_API CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int*, CUfunction, int, size_t, unsigned);

namespace cudavm {
namespace {
struct stream_state;
struct graph_memory_region;
struct capture_state;
struct host_entry;
struct context {
    CUcontext handle = nullptr;
    int device = 0;
    bool primary = false;
    bool retiring = false;
    uint64_t id = 0;
    unsigned flags = 0;
    std::pair<int, int> priority_range{0, -1};
    std::shared_ptr<session> native;
    std::shared_ptr<opengl_graphics> gl;
    std::mutex execution;
    std::unique_ptr<command_queue> queue;
    std::shared_ptr<cudavm::command_domain> command_domain;
    std::shared_ptr<stream_state> legacy;
    std::unordered_map<uint64_t, std::shared_ptr<stream_state>> per_thread;
    std::vector<std::weak_ptr<stream_state>> streams;
    // Commands executed on another context can still own this context's buffers.
    std::vector<std::weak_ptr<command_completion>> foreign_work;
    completion transport_tail; // Scalar shader transport snapshots all bindings.
};
struct stream_state {
    std::weak_ptr<context> owner;
    unsigned flags = 0;
    int priority = 0;
    unsigned long long id = 0;
    uint64_t sequence = 0;
    bool legacy = false;
    completion tail;
    std::shared_ptr<capture_state> capture;
    std::vector<CUgraphNode> capture_dependencies;
    std::vector<CUgraphEdgeData> capture_edges;
    std::shared_ptr<native_queue> native;
    launch_options attributes;
    int synchronization_policy = -1; // Native default inherits the context policy.
};
struct event_state {
    std::weak_ptr<context> owner;
    unsigned flags = 0;
    completion record;
    std::weak_ptr<capture_state> capture;
    std::vector<CUgraphNode> capture_dependencies;
    std::vector<CUgraphEdgeData> capture_edges;
    std::shared_ptr<device_timestamp> timestamp;
};
#include "memory_pool_state.inl"
struct allocation {
    std::shared_ptr<context> owner;
    std::shared_ptr<buffer> native;
    CUgraphicsResource graphics = nullptr;
    CUexternalMemory external = nullptr;
    CUmodule module = nullptr;
    unsigned long long id = 0;
    unsigned sync_memops = 0;
    std::weak_ptr<host_entry> host;
    bool graph_owned = false;
    std::weak_ptr<graph_memory_region> graph_memory;
    std::shared_ptr<memory_pool> pool;
    completion async_alloc;
    completion async_free;
    unsigned long long free_stream_id = 0;
    CUlibrary library = nullptr;
};
struct memory_entry { std::shared_ptr<context> owner; std::shared_ptr<external_memory> native; uint64_t size; };
struct array_entry {
    std::shared_ptr<context> owner;
    std::shared_ptr<buffer> native;
    CUDA_ARRAY3D_DESCRIPTOR descriptor{};
    CUmipmappedArray parent = nullptr;
    bool retiring = false;
    CUdeviceptr image_address = 0;
};
struct image_object_entry {
    std::shared_ptr<context> owner;
    std::shared_ptr<array_entry> array;
    std::shared_ptr<image_object> native;
    std::shared_ptr<buffer> descriptor;
    CUDA_RESOURCE_DESC resource{};
    CUDA_TEXTURE_DESC texture{};
    std::optional<CUDA_RESOURCE_VIEW_DESC> view;
    bool retiring = false;
};
struct mipmapped_array_entry { std::shared_ptr<context> owner; std::vector<CUarray> levels; bool retiring = false; };
struct semaphore_entry { std::shared_ptr<context> owner; std::shared_ptr<external_semaphore> native; CUexternalSemaphoreHandleType type; };
struct graphics_entry { std::shared_ptr<context> owner; unsigned object, flags; CUdeviceptr mapped = 0; };
struct global_symbol { CUdeviceptr address; size_t bytes; };
struct module_entry {
    std::shared_ptr<context> owner;
    std::shared_ptr<executable> native;
    std::unordered_map<std::string, CUfunction> functions;
    std::unordered_map<std::string, global_symbol> globals;
    CUlibrary library = nullptr;
    bool retiring = false;
};
struct function_entry {
    CUmodule module;
    std::shared_ptr<context> owner;
    std::shared_ptr<kernel> native;
    std::string name;
    std::unordered_set<CUfunction_attribute> overrides;
    bool cache_override = false;
};
struct host_entry {
    std::shared_ptr<host_allocation> native;
    std::weak_ptr<context> owner;
    int device = 0;
    unsigned flags = 0;
    unsigned long long id = 0;
    bool registered = false, retiring = false;
    CUdeviceptr alias = 0;
    std::vector<std::weak_ptr<command_completion>> work;
};
#include "user_object_state.inl"
#include "graph_memory_state.inl"
#include "graph_state.inl"
#include "library_types_state.inl"
struct state {
    std::mutex mutex;
    bool initialized = false, selected = false;
    bool shutting_down = false;
    cudavmBackend kind = cudavmBackendAuto, requested = cudavmBackendAuto;
    unsigned int allowed_backends = 31;
    std::unique_ptr<backend> provider;
    // Keep the destructor service alive until all context work has retired.
    std::shared_ptr<user_destructor_queue> user_destructors;
    std::unordered_map<CUuserObject, user_object_entry> user_objects;
    std::vector<device_info> devices;
    std::unordered_map<CUcontext, std::shared_ptr<context>> contexts;
    std::vector<std::weak_ptr<registered_managed_image>> registered_images;
    std::unordered_map<int, std::shared_ptr<context>> primaries;
    // Primary identity and retain ownership survive reset; the active session
    // and all resource generations do not.
    struct primary_record { CUcontext handle = nullptr; unsigned retains = 0; };
    std::unordered_map<int, primary_record> primary_records;
    std::unordered_map<int, unsigned> primary_flags;
    std::map<CUdeviceptr, allocation> allocations;
    std::unordered_map<CUarray, std::shared_ptr<array_entry>> arrays;
    std::unordered_map<CUmipmappedArray, mipmapped_array_entry> mipmapped_arrays;
    std::unordered_map<uint64_t, std::shared_ptr<image_object_entry>> textures, surfaces;
    std::uintptr_t next_array = 0xd000000, next_mipmapped_array = 0xe000000;
    // Extracted registry nodes keep earlier allocation generations until their
    // frees retire. A failed dependency can restore a node without allocation.
    std::list<std::map<CUdeviceptr, allocation>::node_type> pool_retiring;
    struct device_buffer_entry { int device; std::weak_ptr<buffer> native; };
    std::vector<device_buffer_entry> device_buffers;
    std::unordered_map<CUdeviceptr, std::shared_ptr<graph_memory_region>> graph_live;
    std::unordered_map<int, std::shared_ptr<graph_pool_stats>> graph_pools;
    struct graph_cache_entry { std::shared_ptr<context> owner; std::shared_ptr<buffer> native; };
    std::vector<graph_cache_entry> graph_cache;
    std::unordered_map<CUmemoryPool, std::shared_ptr<memory_pool>> memory_pools;
    std::unordered_map<int, CUmemoryPool> default_pools, current_pools;
    std::unordered_map<int, std::shared_ptr<context>> pool_storage;
    std::unordered_map<int, std::shared_ptr<command_domain>> command_domains;
    std::uintptr_t next_pool = 0xa000000;
    std::map<uintptr_t, std::shared_ptr<host_entry>> hosts;
    std::unordered_map<CUmodule, module_entry> modules;
    std::unordered_map<CUfunction, function_entry> functions;
    std::unordered_map<CUlibrary, std::shared_ptr<library_entry>> libraries;
    std::unordered_map<CUkernel, std::shared_ptr<library_kernel_entry>> library_kernels;
    std::uintptr_t next_library = 0xb000000, next_library_kernel = 0xc000000;
    std::unordered_map<CUexternalMemory, memory_entry> memories;
    std::unordered_map<CUexternalSemaphore, semaphore_entry> semaphores;
    std::unordered_map<CUgraphicsResource, graphics_entry> graphics;
    std::unordered_map<CUstream, std::shared_ptr<stream_state>> streams;
    std::unordered_map<CUevent, event_state> events;
    std::unordered_set<CUevent> reset_events;
    std::unordered_map<CUgraph, graph_state> graphs;
    std::unordered_map<CUgraphNode, CUgraph> graph_nodes;
    std::unordered_map<CUgraphExec, graph_exec> graph_execs;
    uint64_t epoch = 0;
    std::uintptr_t next_module = 0x1000000, next_function = 0x2000000;
    std::uintptr_t next_context = 0x10000;
    std::uintptr_t next_interop = 0x3000000;
    CUdeviceptr next_address = 0x10000000000ull;
    unsigned long long next_buffer_id = 1;
    std::uintptr_t next_stream = 0x4000000, next_event = 0x5000000;
    unsigned long long next_stream_id = 1;
    uint64_t next_capture_id = 1;
    std::uintptr_t next_graph = 0x6000000, next_graph_node = 0x7000000, next_graph_exec = 0x8000000;
    uint64_t next_graph_id = 1;
    std::uintptr_t next_user_object = 0x9000000;
    // Buffers must be destroyed before native sessions and the backend loader.
    ~state() { graph_execs.clear(); graph_nodes.clear(); graphs.clear(); graph_live.clear(); graph_cache.clear(); user_objects.clear(); functions.clear(); modules.clear(); textures.clear(); surfaces.clear(); arrays.clear(); mipmapped_arrays.clear(); allocations.clear(); hosts.clear(); graphics.clear(); memories.clear(); semaphores.clear(); primaries.clear(); contexts.clear(); }
};
void synchronize_context(const std::shared_ptr<context>&);
void capture_unsafe(const std::shared_ptr<context>&);
void capture_retire_context(const std::shared_ptr<context>&);
void retire_context(const std::shared_ptr<context>&);
void bind_context_libraries(const std::shared_ptr<context>&);
void configure_library_function(CUfunction);
CUfunction resolve_library_function(CUfunction, const std::shared_ptr<context>&);
void destroy(const std::shared_ptr<context>&, bool);
void retire_primary_hosts(const std::shared_ptr<context>&);
void collect_pool_memory();
void pool_release(allocation&, CUdeviceptr, bool);
void pool_retire_context(const std::shared_ptr<context>&);
void pool_synchronize();
void wait_stream(const std::shared_ptr<context>&, CUstream);
thread_local std::unique_lock<std::mutex>* driver_mutex_lock = nullptr;
unsigned context_schedule(const std::shared_ptr<context>&);
template<class F> void driver_wait(F function) {
    auto* lock = driver_mutex_lock;
    require(lock && lock->owns_lock(), CUDA_ERROR_UNKNOWN, "queue wait requires a Driver API boundary");
    lock->unlock();
    try { function(); } catch (...) { lock->lock(); throw; }
    lock->lock();
}
void drain_context_work(const std::shared_ptr<context>& owner) {
    auto* queue = owner->queue.get();
    std::vector<completion> foreign;
    std::erase_if(owner->foreign_work, [](const auto& item) { return item.expired(); });
    for (auto& item : owner->foreign_work) if (auto done = item.lock()) foreign.push_back(std::move(done));
    auto schedule = context_schedule(owner);
    if (queue || !foreign.empty()) driver_wait([&] {
        if (queue) queue->drain(true, schedule);
        for (auto& done : foreign) done->wait(schedule);
    });
    require(!owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "context is being destroyed");
}
template<class F> auto native_call(const std::shared_ptr<context>& owner, F function, bool capture_safe = false) {
    if (!capture_safe) capture_unsafe(owner);
    drain_context_work(owner);
    auto domain = command_domain_read(owner->command_domain);
    std::lock_guard lock(owner->execution);
    return function();
}
state& global() {
    // GPU drivers and the OpenGL owner thread cannot be called from Windows
    // DLL process detach (loader lock). The registry has process lifetime;
    // explicit context release/reset and cudavmShutdown perform native cleanup
    // while ordinary API calls are still safe.
    static state* value = new state;
    return *value;
}
std::vector<CUcontext>& context_stack() {
    static thread_local std::vector<CUcontext> value;
    return value;
}
std::string& diagnostic_text() {
    // Function-local initialization also works when a statically linked client
    // calls CUDA from an ordinary global initializer before main.
    static thread_local std::string value;
    return value;
}

void restore_registered_images();
void initialize() {
    auto& value = global();
    require(!value.shutting_down, CUDA_ERROR_NOT_PERMITTED, "shutdown is in progress");
    if (value.initialized) return;
    auto requested = value.requested;
    if (!value.selected) {
        if (auto name = backend_environment("CUDAVM_BACKEND")) {
            bool found = *name == "auto";
            for (int i = 0; i < 5; ++i) {
                auto kind = static_cast<cudavmBackend>(i);
                if (*name == backend_name(kind)) { requested = kind; found = true; break; }
            }
            require(found, CUDA_ERROR_INVALID_VALUE, "CUDAVM_BACKEND must be auto, cuda, opencl, opengl, vulkan or directx");
        }
    }
    // Failed candidates release their loaders before trying another backend.
    // Only Auto falls back, and state is committed after successful discovery.
    auto chosen = discover_backend(requested, value.allowed_backends);
    value.kind = chosen.kind;
    value.provider = std::move(chosen.provider);
    value.devices = std::move(chosen.devices);
    value.initialized = true;
    try { restore_registered_images(); }
    catch (...) {
        value.libraries.clear(); value.allocations.clear(); value.device_buffers.clear();
        value.provider.reset(); value.devices.clear(); value.initialized = false; throw;
    }
    diagnostic_text().clear();
}
void driver_ready() { require(global().initialized && !global().shutting_down, CUDA_ERROR_NOT_INITIALIZED, "Driver is uninitialized or shutting down"); }
void valid_device(int device) {
    require(device >= 0 && static_cast<size_t>(device) < global().devices.size(), CUDA_ERROR_INVALID_DEVICE, "device ordinal out of range");
}
int primary_device(CUcontext handle) {
    if (handle) for (auto& [device, record] : global().primary_records) if (record.handle == handle) return device;
    return -1;
}
std::shared_ptr<context> lookup(CUcontext handle) {
    auto it = global().contexts.find(handle);
    require(it != global().contexts.end() || primary_device(handle) < 0,
        CUDA_ERROR_CONTEXT_IS_DESTROYED, "primary context is inactive after release or reset");
    require(it != global().contexts.end() && !it->second->retiring, CUDA_ERROR_INVALID_CONTEXT, "invalid, retiring or destroyed context");
    return it->second;
}
std::shared_ptr<context> current() {
    auto& stack = context_stack();
    require(!stack.empty(), CUDA_ERROR_INVALID_CONTEXT, "no context is current on this host thread");
    return lookup(stack.back());
}
unsigned context_schedule(const std::shared_ptr<context>& owner) {
    auto schedule = owner->flags & CU_CTX_SCHED_MASK;
    if (schedule != CU_CTX_SCHED_AUTO) return schedule;
    auto processors = std::max(1u, std::thread::hardware_concurrency());
    return global().contexts.size() > processors ? CU_CTX_SCHED_YIELD : CU_CTX_SCHED_SPIN;
}
void validate_context_flags(unsigned flags, bool allow_map_host = true) {
    auto schedule = flags & CU_CTX_SCHED_MASK;
    require(!(flags & ~unsigned(CU_CTX_FLAGS_MASK)) && (allow_map_host || !(flags & CU_CTX_MAP_HOST)) &&
        (schedule == CU_CTX_SCHED_AUTO || schedule == CU_CTX_SCHED_SPIN ||
         schedule == CU_CTX_SCHED_YIELD || schedule == CU_CTX_SCHED_BLOCKING_SYNC),
        CUDA_ERROR_INVALID_VALUE, "invalid context flags or scheduling combination");
    require(global().kind == cudavmBackendCuda || !(flags & (CU_CTX_COREDUMP_ENABLE | CU_CTX_USER_COREDUMP_ENABLE)),
        CUDA_ERROR_NOT_SUPPORTED, "GPU coredump flags require native CUDA support");
}
template<class F> auto context_configuration(const std::shared_ptr<context>& owner, F function) {
    std::lock_guard lock(owner->execution); return function();
}
void set_context_flags(const std::shared_ptr<context>& owner, unsigned flags) {
    validate_context_flags(flags);
    // MAP_HOST is a creation-time property. cuCtxSetFlags accepts its deprecated
    // spelling but neither enables nor clears it on an existing context.
    flags = (flags & ~unsigned(CU_CTX_MAP_HOST)) | (owner->flags & CU_CTX_MAP_HOST);
    auto actual = context_configuration(owner, [&] { return owner->native->set_context_flags(flags); });
    owner->flags = actual;
    if (owner->primary) global().primary_flags[owner->device] = actual & ~unsigned(CU_CTX_MAP_HOST);
}
void set_primary_flags(int device, unsigned flags, bool legacy = false) {
    valid_device(device); validate_context_flags(flags, false);
    auto found = global().primaries.find(device);
    if (found != global().primaries.end()) {
        require(!legacy, CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE, "primary context is already active");
        set_context_flags(lookup(found->second->handle), flags);
    } else global().primary_flags[device] = flags | ((flags & CU_CTX_USER_COREDUMP_ENABLE) ? CU_CTX_COREDUMP_ENABLE : 0);
}
std::shared_ptr<context> create(int device, bool primary, unsigned flags = 0, CUcontext identity = nullptr) {
    valid_device(device);
    validate_context_flags(flags, true);
    if (primary) flags |= CU_CTX_MAP_HOST; // Primary contexts always support mapped pinned allocations.
    auto result = std::make_shared<context>();
    result->native = global().provider->open(device, flags);
    result->flags = result->native->get_context_flags(flags);
    result->priority_range = result->native->stream_priority_range();
    result->device = device;
    if (global().kind != cudavmBackendCuda) {
        auto& domain = global().command_domains[device];
        if (!domain) domain = std::make_shared<command_domain>();
        result->command_domain = domain;
    }
    result->primary = primary;
    result->id = global().next_context;
    result->handle = identity ? identity : reinterpret_cast<CUcontext>(result->id);
    global().next_context += 0x100;
    global().contexts.emplace(result->handle, result);
    try { bind_context_libraries(result); } catch (...) { destroy(result, false); throw; }
    return result;
}
std::shared_ptr<context> primary(int device) {
    valid_device(device);
    auto it = global().primaries.find(device);
    if (it != global().primaries.end()) { require(!it->second->retiring, CUDA_ERROR_INVALID_CONTEXT, "primary context is being destroyed"); return it->second; }
    auto& record = global().primary_records[device];
    auto result = create(device, true, global().primary_flags[device], record.handle);
    record.handle = result->handle;
    global().primary_flags[device] = result->flags & ~unsigned(CU_CTX_MAP_HOST);
    global().primaries.emplace(device, result);
    return result;
}
void set_current(CUcontext handle) {
    auto& stack = context_stack();
    if (handle) {
        if (primary_device(handle) < 0 || global().contexts.contains(handle)) lookup(handle);
        if (stack.empty()) stack.push_back(handle); else stack.back() = handle;
    } else if (!stack.empty()) stack.pop_back();
}
void destroy(const std::shared_ptr<context>& owner, bool pop_current = true) {
    capture_retire_context(owner);
    owner->retiring = true;
    retire_context(owner);
    if (owner->primary) retire_primary_hosts(owner);
    pool_retire_context(owner);
    std::erase_if(global().streams, [&](auto& item) { return item.second->owner.lock() == owner; });
    std::erase_if(global().events, [&](auto& item) {
        if (item.second.owner.lock() != owner) return false;
        if (owner->primary) global().reset_events.insert(item.first);
        return true;
    });
    std::erase_if(global().graph_execs, [&](auto& item) { return item.second.owner.lock() == owner; });
    std::erase_if(global().functions, [&](auto& item) { return item.second.owner == owner; });
    std::erase_if(global().modules, [&](auto& item) { return item.second.owner == owner; });
    std::erase_if(global().textures, [&](auto& item) { return item.second->owner == owner; });
    std::erase_if(global().surfaces, [&](auto& item) { return item.second->owner == owner; });
    std::erase_if(global().arrays, [&](auto& item) { return item.second->owner == owner; });
    std::erase_if(global().mipmapped_arrays, [&](auto& item) { return item.second.owner == owner; });
    std::erase_if(global().graph_live, [&](auto& item) { return item.second->owner.lock() == owner; });
    std::erase_if(global().graph_cache, [&](auto& item) { return item.owner == owner; });
    auto& allocations = global().allocations;
    for (auto it = allocations.begin(); it != allocations.end();)
        if (it->second.owner == owner) it = allocations.erase(it); else ++it;
    std::erase_if(global().graphics, [&](auto& item) { return item.second.owner == owner; });
    std::erase_if(global().memories, [&](auto& item) { return item.second.owner == owner; });
    std::erase_if(global().semaphores, [&](auto& item) { return item.second.owner == owner; });
    global().contexts.erase(owner->handle);
    if (owner->primary) { global().primaries.erase(owner->device); global().primary_flags.erase(owner->device); }
    // Other host threads retain opaque handles; lookup rejects these after destruction.
    if (pop_current) std::erase(context_stack(), owner->handle);
}
void reset_primary(int device) {
    valid_device(device);
    auto it = global().primaries.find(device);
    if (it != global().primaries.end()) {
        auto owner = it->second;
        require(!owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "primary context is being destroyed");
        destroy(owner, false);
    }
    global().primary_flags.erase(device);
}
void release_primary(int device) {
    valid_device(device);
    auto it = global().primary_records.find(device);
    require(it != global().primary_records.end() && it->second.retains,
        CUDA_ERROR_INVALID_CONTEXT, "primary context has no retain to release");
    if (--it->second.retains == 0) reset_primary(device);
}
module_entry& module_lookup(CUmodule handle, const std::shared_ptr<context>& owner) {
    auto it = global().modules.find(handle);
    require(it != global().modules.end(), CUDA_ERROR_INVALID_HANDLE, "invalid or unloaded module");
    require(!it->second.retiring, CUDA_ERROR_NOT_PERMITTED, "registered module is resetting");
    require(it->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "module belongs to a different context");
    return it->second;
}
CUfunction find_function(CUmodule handle, const char* name, const std::shared_ptr<context>& owner) {
    require(name && *name, CUDA_ERROR_INVALID_VALUE, "null or empty kernel name");
    auto& module = module_lookup(handle, owner);
    auto existing = module.functions.find(name);
    if (existing != module.functions.end()) return existing->second;
    auto native = module.native->find(name);
    auto function = reinterpret_cast<CUfunction>(global().next_function);
    global().next_function += 0x100;
    global().functions.emplace(function, function_entry{handle, owner, std::move(native), name});
    try { module.functions.emplace(name, function); }
    catch (...) { global().functions.erase(function); throw; }
    try { configure_library_function(function); }
    catch (...) { module.functions.erase(name); global().functions.erase(function); throw; }
    return function;
}
function_entry& function_lookup(CUfunction handle, const std::shared_ptr<context>& owner) {
    handle = resolve_library_function(handle, owner);
    auto it = global().functions.find(handle);
    require(it != global().functions.end(), CUDA_ERROR_INVALID_HANDLE, "invalid or unloaded function");
    require(it->second.owner == owner, CUDA_ERROR_INVALID_CONTEXT, "function belongs to a different context");
    module_lookup(it->second.module, owner);
    return it->second;
}
function_entry& function_resources(CUfunction handle) {
    if (global().library_kernels.contains(reinterpret_cast<CUkernel>(handle))) handle = resolve_library_function(handle, current());
    auto it = global().functions.find(handle);
    require(it != global().functions.end(), CUDA_ERROR_INVALID_HANDLE, "invalid or unloaded function");
    require(!it->second.owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "function context is being destroyed");
    module_lookup(it->second.module, it->second.owner);
    return it->second;
}
void launch(CUfunction function, const launch_shape& shape, CUstream stream, void** parameters, void** extra,
            const std::shared_ptr<context>& owner);
CUdeviceptr reserve_device_address(size_t bytes) {
    constexpr CUdeviceptr alignment = 256;
    require(bytes <= std::numeric_limits<CUdeviceptr>::max() - global().next_address - alignment,
            CUDA_ERROR_OUT_OF_MEMORY, "device address space exhausted");
    auto address = global().next_address;
    global().next_address += (static_cast<CUdeviceptr>(bytes) + alignment - 1) & ~(alignment - 1);
    return address;
}
CUdeviceptr add_shared_allocation(const std::shared_ptr<context>& owner, std::shared_ptr<buffer> storage,
                           CUgraphicsResource graphics = nullptr, CUexternalMemory external = nullptr, CUmodule module = nullptr) {
    auto bytes = storage->size;
    CUdeviceptr address = storage->native_address();
    if (!address) address = reserve_device_address(bytes);
    require(!global().allocations.contains(address), CUDA_ERROR_UNKNOWN, "native device address collision");
    std::erase_if(global().device_buffers, [](auto& entry) { return entry.native.expired(); });
    global().device_buffers.push_back({owner->device, storage});
    global().allocations.emplace(address, allocation{owner, std::move(storage), graphics, external, module, global().next_buffer_id++, 0, {}, false, {}, {}, {}, {}, 0});
    return address;
}
CUdeviceptr add_allocation(const std::shared_ptr<context>& owner, std::unique_ptr<buffer> native,
                           CUgraphicsResource graphics = nullptr, CUexternalMemory external = nullptr, CUmodule module = nullptr) {
    return add_shared_allocation(owner, std::move(native), graphics, external, module);
}
std::shared_ptr<library_entry> library_lookup(CUlibrary);
global_symbol library_managed(CUlibrary, const char*);
global_symbol find_global(CUmodule handle, const char* name, const std::shared_ptr<context>& owner) {
    require(name && *name,CUDA_ERROR_INVALID_VALUE,"null or empty global name");
    auto& module=module_lookup(handle,owner);
    if(auto it=module.globals.find(name);it!=module.globals.end())return it->second;
    if (module.library) {
        auto library = library_lookup(module.library);
        if (auto found = library->managed.find(name); found != library->managed.end()) { module.globals.emplace(name, found->second); return found->second; }
    }
    auto native = module.native;
    if (auto it = native->data_views.find(name); it != native->data_views.end()) {
        auto view = it->second;
        require(!native->data_views.contains(view.target), CUDA_ERROR_INVALID_IMAGE, "data view target is another view");
        global_symbol target{};
        try { target = find_global(handle, view.target.c_str(), owner); }
        catch (const failure& error) {
            if (error.code == CUDA_ERROR_NOT_FOUND) throw failure(CUDA_ERROR_INVALID_IMAGE, "data view target is absent from module");
            throw;
        }
        require(view.offset <= target.bytes && view.bytes <= target.bytes - view.offset,
                CUDA_ERROR_INVALID_IMAGE, "data view exceeds module storage");
        global_symbol result{target.address + view.offset, size_t(view.bytes)};
        module_lookup(handle, owner).globals.emplace(name, result);
        return result;
    }
    auto storage = native_call(owner, [&] { module_lookup(handle, owner); return native->find_global(name); }, true);
    auto library = module_lookup(handle, owner).library;
    if (storage->is_managed() && library && library_lookup(library)->native) {
        auto result = library_managed(library, name); module_lookup(handle, owner).globals.emplace(name, result); return result;
    }
    auto address=add_allocation(owner,std::move(storage),nullptr,nullptr,handle);
    global_symbol result{address, global().allocations.at(address).native->size};
    try { module_lookup(handle, owner).globals.emplace(name,result); }
    catch(...) { global().allocations.erase(address); throw; }
    return result;
}
CUmodule add_module(const std::shared_ptr<context>& owner, std::shared_ptr<executable> native) {
    auto handle=reinterpret_cast<CUmodule>(global().next_module); global().next_module+=0x100;
    global().modules.emplace(handle,module_entry{owner,std::move(native),{}, {}});
    try {
        auto& module=global().modules.at(handle); std::vector<allocation_binding> bindings;
        for(const auto& name:module.native->global_names()) {
            auto address=find_global(handle,name.c_str(),owner).address;
            bindings.push_back({address,global().allocations.at(address).native.get()});
        }
        native_call(owner, [&] { module.native->bind_globals(bindings); }, true);
        std::vector<std::string> views;
        for (const auto& [name, view] : module.native->data_views) views.push_back(name);
        for (const auto& name : views) (void)find_global(handle, name.c_str(), owner);
    } catch(...) {
        std::erase_if(global().allocations,[&](auto& item){return item.second.module==handle;});
        global().modules.erase(handle); throw;
    }
    return handle;
}
#include "library_state.inl"
CUdeviceptr allocate(const std::shared_ptr<context>& owner, size_t bytes) {
    require(bytes != 0, CUDA_ERROR_INVALID_VALUE, "zero-sized Driver allocation");
    return add_allocation(owner, native_call(owner, [&] { return owner->native->allocate(bytes); }));
}
using allocation_it = std::map<CUdeviceptr, allocation>::iterator;
bool allocation_visible(const allocation& entry, const std::shared_ptr<context>& owner) {
    return entry.owner == owner || entry.native->is_managed() || (entry.pool && entry.pool->props.location.id == owner->device);
}
allocation_it containing(CUdeviceptr pointer) {
    auto& allocations = global().allocations;
    auto it = allocations.upper_bound(pointer);
    if (it == allocations.begin()) return allocations.end();
    --it;
    if (pointer - it->first >= it->second.native->size) return allocations.end();
    return it;
}
void collect_graph_memory() {
    for (auto& [address, entry] : global().allocations) if (auto memory = entry.graph_memory.lock()) {
        if (memory->transition && memory->transition->ready.load(std::memory_order_acquire)) {
            memory->planned_live = memory->live.load(); memory->transition.reset();
        }
    }
    std::erase_if(global().graph_live, [](auto& entry) { return !entry.second->planned_live && !entry.second->transition; });
    for (auto it = global().allocations.begin(); it != global().allocations.end();) {
        auto& entry = it->second;
        if (!entry.graph_owned || !entry.graph_memory.expired()) { ++it; continue; }
        if (entry.native->graph_charge) {
            global().graph_cache.push_back({entry.owner, entry.native});
            entry.native->graph_charge->associate(false);
        }
        it = global().allocations.erase(it);
    }
}
#include "host_memory.inl"
struct region { allocation* value; size_t offset; };
region resolve(CUdeviceptr pointer, size_t bytes, const std::shared_ptr<context>& owner) {
    auto it = containing(pointer);
    if (it == global().allocations.end()) {
        auto host = host_containing(static_cast<uintptr_t>(pointer));
        if (host != global().hosts.end() && (host->second->native->flags(host->second->flags) & CU_MEMHOSTALLOC_DEVICEMAP)) {
            auto alias = host_alias(host->second, owner);
            if (alias == host->first) it = containing(pointer);
        }
    }
    require(it != global().allocations.end(), CUDA_ERROR_INVALID_VALUE, "unknown device allocation");
    require(!it->second.async_free, CUDA_ERROR_INVALID_VALUE, "allocation has an enqueued free");
    size_t offset = static_cast<size_t>(pointer - it->first);
    require(bytes <= it->second.native->size - offset, CUDA_ERROR_INVALID_VALUE, "device allocation range exceeded");
    if (it->second.graph_owned) {
        auto memory = it->second.graph_memory.lock();
        require(memory && (resolving_graph_memory || memory->planned_live), CUDA_ERROR_INVALID_VALUE, "graph allocation is not live");
    }
    auto host = it->second.host.lock();
    if (host) require(!host->retiring && host_visible(host, owner),
                      CUDA_ERROR_INVALID_VALUE, "mapped host memory is retiring or unavailable in this context");
    else require(allocation_visible(it->second, owner), CUDA_ERROR_INVALID_CONTEXT, "allocation belongs to a different context or device");
    return {&it->second, offset};
}
void release(CUdeviceptr pointer, const std::shared_ptr<context>& owner) {
    auto it = global().allocations.find(pointer);
    require(it != global().allocations.end(), CUDA_ERROR_INVALID_VALUE, "free requires an allocation base address");
    require(allocation_visible(it->second, owner), CUDA_ERROR_INVALID_CONTEXT, "allocation belongs to a different context or device");
    require(!it->second.graphics, CUDA_ERROR_INVALID_VALUE, "a mapped graphics pointer must be released by graphics unmap");
    require(!it->second.module, CUDA_ERROR_INVALID_VALUE, "module globals are released by module unload");
    require(!it->second.library, CUDA_ERROR_INVALID_VALUE, "library managed globals are released by library unload");
    require(it->second.host.expired(), CUDA_ERROR_INVALID_VALUE, "mapped host aliases are released by host free/unregister");
    auto memory = it->second.graph_memory.lock();
    if (it->second.graph_owned) require(memory && memory->planned_live && !memory->internal_free(), CUDA_ERROR_INVALID_VALUE, "graph allocation cannot be freed outside its owning graph");
    auto expected = it->second.native;
    auto storage_owner = it->second.owner;
    synchronize_context(expected->is_managed() ? storage_owner : owner);
    it = global().allocations.find(pointer);
    require(it != global().allocations.end() && it->second.native == expected, CUDA_ERROR_INVALID_VALUE, "allocation changed while waiting for free");
    if (memory) {
        require(memory->live.load(), CUDA_ERROR_INVALID_VALUE, "graph allocation is not live");
        memory->live = false; memory->planned_live = false; memory->transition.reset();
        global().graph_live.erase(pointer);
    } else { if (it->second.pool) pool_release(it->second, pointer, false); global().allocations.erase(it); }
}
void write(CUdeviceptr destination, const void* source, size_t bytes, const std::shared_ptr<context>& owner) {
    if (!bytes) return;
    require(source != nullptr, CUDA_ERROR_INVALID_VALUE, "null host source");
    host_use host{host_range(reinterpret_cast<uintptr_t>(source), bytes)};
    auto region = resolve(destination, bytes, owner);
    host_use mapped{region.value->host.lock()};
    auto native = region.value->native;
    native_call(owner, [&] { owner->native->write(*native, region.offset, {static_cast<const std::byte*>(source), bytes}); });
}
void read(void* destination, CUdeviceptr source, size_t bytes, const std::shared_ptr<context>& owner) {
    if (!bytes) return;
    require(destination != nullptr, CUDA_ERROR_INVALID_VALUE, "null host destination");
    host_use host{host_range(reinterpret_cast<uintptr_t>(destination), bytes)};
    auto region = resolve(source, bytes, owner);
    host_use mapped{region.value->host.lock()};
    auto native = region.value->native;
    native_call(owner, [&] { owner->native->read(*native, region.offset, {static_cast<std::byte*>(destination), bytes}); });
}
void copy(CUdeviceptr destination, CUdeviceptr source, size_t bytes, const std::shared_ptr<context>& owner) {
    if (!bytes) return;
    auto to = resolve(destination, bytes, owner), from = resolve(source, bytes, owner);
    host_use mapped{to.value->host.lock(), from.value->host.lock()};
    require(to.value != from.value || to.offset + bytes <= from.offset || from.offset + bytes <= to.offset,
            CUDA_ERROR_INVALID_VALUE, "overlapping copy regions");
    auto to_native = to.value->native, from_native = from.value->native;
    native_call(owner, [&] { owner->native->copy(*to_native, to.offset, *from_native, from.offset, bytes); });
}
template<class T> void fill(CUdeviceptr destination, T pattern, size_t count, const std::shared_ptr<context>& owner) {
    require(destination % alignof(T) == 0 && count <= std::numeric_limits<size_t>::max() / sizeof(T),
            CUDA_ERROR_INVALID_VALUE, "fill alignment or element count is invalid");
    if (!count) return;
    auto region = resolve(destination, count * sizeof(T), owner);
    host_use mapped{region.value->host.lock()};
    auto native = region.value->native;
    native_call(owner, [&] { owner->native->fill(*native, region.offset, {reinterpret_cast<const std::byte*>(&pattern), sizeof(T)}, count * sizeof(T)); });
}
template<class F> CUresult driver_call(F function) noexcept {
    try {
        require(!executing_host_function, CUDA_ERROR_NOT_PERMITTED, "CUDA calls are forbidden inside host functions");
        std::unique_lock lock(global().mutex);
        struct boundary {
            std::unique_lock<std::mutex>* previous = driver_mutex_lock;
            explicit boundary(std::unique_lock<std::mutex>& value) { driver_mutex_lock = &value; }
            ~boundary() { driver_mutex_lock = previous; }
        } scope(lock);
        collect_graph_memory(); collect_pool_memory(); function(); collect_graph_memory(); collect_pool_memory(); return CUDA_SUCCESS;
    }
    catch (const failure& e) { diagnostic_text() = e.what(); return e.code; }
    catch (const std::bad_alloc&) { diagnostic_text() = "host allocation failed"; return CUDA_ERROR_OUT_OF_MEMORY; }
    catch (const std::exception& e) { diagnostic_text() = e.what(); return CUDA_ERROR_UNKNOWN; }
    catch (...) { diagnostic_text() = "unknown backend failure"; return CUDA_ERROR_UNKNOWN; }
}
bool capture_node(const std::shared_ptr<context>&, CUstream, graph_node);
void capture_check_stream(const std::shared_ptr<stream_state>&);
bool capture_record(CUevent, CUstream, unsigned);
bool capture_wait(CUstream, CUevent, unsigned);
std::function<void()> graph_prepare_semaphores(const graph_node&);
bool capture_graph_launch(const std::shared_ptr<context>&, CUstream, const graph_exec&);
#include "streams.inl"
#include "arrays.inl"
#include "image_objects.inl"
#include "memory.inl"
#include "async_memory.inl"
#include "launch_state.inl"
#include "stream_attribute_state.inl"
#include "kernel_launch.inl"
#include "memory_operations.inl"
#include "graph_edges.inl"
#include "graph_memory.inl"
#include "graphs.inl"
#include "graph_priorities.inl"
#include "user_objects.inl"
#include "graph_instantiation.inl"
#include "capture.inl"
#include "memory_pools.inl"
const char* error_name(CUresult error) {
    switch (error) {
#define DRIVER_ERROR(value, description) case value: return #value;
#include "error_catalog.inl"
#undef DRIVER_ERROR
    default: return nullptr;
    }
}
const char* error_string(CUresult error) {
    switch (error) {
#define DRIVER_ERROR(value, description) case value: return description;
#include "error_catalog.inl"
#undef DRIVER_ERROR
    default: return nullptr;
    }
}
#include "interop.inl"
#include "graph_interop.inl"
#include "graph_generic.inl"
#include "graph_parameters.inl"
#include "graph_diagnostics.inl"
#include "function_state.inl"
#include "device_state.inl"
}
}
using namespace cudavm;
#define DRIVER_API(name, parameters, ...) extern "C" CUDAVM_API CUresult CUDAAPI name parameters { return driver_call([&] { __VA_ARGS__; }); }

#include "interop_exports.inl"
#include "device_exports.inl"
#include "memory_exports.inl"
#include "array_exports.inl"
#include "image_object_exports.inl"
#include "host_memory_exports.inl"
#include "stream_exports.inl"
#include "memory_operation_exports.inl"
#include "async_memory_exports.inl"
#include "memory_pool_exports.inl"
#include "graph_exports.inl"
#include "graph_memory_exports.inl"
#include "graph_memory_operation_exports.inl"
#include "graph_advanced_exports.inl"
#include "graph_interop_exports.inl"
#include "graph_generic_exports.inl"
#include "graph_lifetime_exports.inl"
#include "graph_priority_exports.inl"
#include "graph_parameter_exports.inl"
#include "graph_diagnostic_exports.inl"
#include "capture_exports.inl"
#include "context_exports.inl"
#include "function_exports.inl"
#include "launch_exports.inl"
#include "library_exports.inl"

extern "C" CUDAVM_CUDA_API CUresult cudavmDriverLoadRegisteredLibrary(CUlibrary* output, const void* image, size_t bytes) {
    return driver_call([&] { driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null registered library output"); *output = registered_library_load(image, bytes); });
}
extern "C" CUDAVM_CUDA_API CUresult cudavmDriverCreateManagedRegistration(void** output, const void* image, size_t bytes, const char* const* names, const void* const* slots, const size_t* sizes, size_t count) {
    return driver_call([&] {
        require(output && image && bytes && static_cast<const char*>(image)[bytes - 1] == 0 && names && slots && sizes && count,
            CUDA_ERROR_INVALID_VALUE, "invalid managed registration image");
        auto value = std::make_shared<registered_managed_image>(); value->image.assign(static_cast<const char*>(image), bytes - 1);
        for (size_t i = 0; i < count; ++i) {
            require(names[i] && *names[i] && slots[i] && sizes[i], CUDA_ERROR_INVALID_VALUE, "missing managed symbol name, host slot or size");
            value->names.emplace_back(names[i]); value->slots.push_back(slots[i]); value->sizes.push_back(sizes[i]);
        }
        auto handle = std::make_unique<std::shared_ptr<registered_managed_image>>(value);
        std::erase_if(global().registered_images, [](const auto& item) { return item.expired(); });
        global().registered_images.push_back(value); *output = handle.release();
    });
}
extern "C" CUDAVM_CUDA_API CUresult cudavmDriverLoadManagedRegistration(CUlibrary* output, void* image) {
    return driver_call([&] {
        driver_ready(); require(output && image, CUDA_ERROR_INVALID_VALUE, "null managed registration image or output");
        *output = registered_managed_load(*static_cast<std::shared_ptr<registered_managed_image>*>(image));
    });
}
extern "C" CUDAVM_CUDA_API void cudavmDriverReleaseManagedRegistration(void* image) {
    delete static_cast<std::shared_ptr<registered_managed_image>*>(image);
}
extern "C" CUDAVM_CUDA_API CUresult cudavmDriverRetireManagedRegistration(void* input) {
    return driver_call([&] {
        require(input, CUDA_ERROR_INVALID_VALUE, "null managed registration");
        auto image = *static_cast<std::shared_ptr<registered_managed_image>*>(input); auto epoch = global().epoch;
        image->retired = true;
        try { if (image->epoch == epoch && global().libraries.contains(image->facade)) library_unload(image->facade); }
        catch (...) { if (epoch == global().epoch) { image->retired = false; throw; } }
    });
}
extern "C" CUDAVM_CUDA_API CUresult cudavmDriverNvidiaImageSize(const void* image, size_t* output) {
    return driver_call([&] { require(output, CUDA_ERROR_INVALID_VALUE, "null NVIDIA image extent output"); auto size = nvidia_image_length(image); require(size, CUDA_ERROR_INVALID_IMAGE, "NVIDIA image is not a fatbinary or cubin"); *output = size; });
}
extern "C" CUDAVM_CUDA_API CUresult cudavmDriverValidateNvidiaImage(const void* image, size_t size) {
    return driver_call([&] { require(image && size, CUDA_ERROR_INVALID_IMAGE, "empty NVIDIA image"); nvidia_image_metadata(std::span(static_cast<const std::byte*>(image), size)); });
}
extern "C" CUDAVM_CUDA_API CUresult cudavmDriverReleaseRegisteredModules(CUcontext context, const CUlibrary* libraries, size_t count) {
    return driver_call([&] {
        driver_ready(); require(!count || libraries, CUDA_ERROR_INVALID_VALUE, "missing registered libraries"); auto owner = lookup(context);
        std::vector<CUmodule> modules;
        for (size_t i = 0; i < count; ++i) {
            auto entry = library_lookup(libraries[i]); require(entry->registered, CUDA_ERROR_NOT_PERMITTED, "library is not owned by Runtime registration");
            if (auto found = entry->modules.find(owner->id); found != entry->modules.end() && global().modules.contains(found->second)) {
                modules.push_back(found->second);
            }
        }
        for (auto module : modules) registered_module_release(module, owner);
    });
}

extern "C" CUDAVM_API const char* cudavmBackendName(cudavmBackend kind) { return backend_name(kind); }
extern "C" CUDAVM_API const char* cudavmGetDiagnostic() { return diagnostic_text().c_str(); }
extern "C" CUDAVM_API int cudavmSelectBackend(cudavmBackend kind) {
    return static_cast<int>(driver_call([&] {
        require(!global().initialized, CUDA_ERROR_NOT_PERMITTED, "backend cannot change after initialization");
        require(kind >= cudavmBackendAuto && kind <= cudavmBackendDirectX, CUDA_ERROR_INVALID_VALUE, "invalid backend");
        require(kind == cudavmBackendAuto || (backend_enabled(kind) && (global().allowed_backends & (1u << kind))),
                CUDA_ERROR_NOT_SUPPORTED, "requested backend is disabled by this build or program policy");
        global().requested = kind; global().selected = true;
    }));
}
extern "C" CUDAVM_API unsigned int cudavmGetEnabledBackends() {
    unsigned int result = 0;
    for (auto kind : enabled_backends()) result |= 1u << static_cast<unsigned>(kind);
    return result;
}
extern "C" CUDAVM_API int cudavmConfigureBackends(unsigned int mask) {
    return static_cast<int>(driver_call([&] {
        auto& value = global();
        require(!value.initialized, CUDA_ERROR_NOT_PERMITTED, "backend policy cannot change after initialization");
        require(mask && !(mask & ~31u), CUDA_ERROR_INVALID_VALUE, "invalid backend policy mask");
        require(!(mask & ~cudavmGetEnabledBackends()), CUDA_ERROR_NOT_SUPPORTED,
                "backend policy requests an adapter absent from this build");
        auto allowed = value.allowed_backends & mask;
        require(allowed != 0, CUDA_ERROR_INVALID_VALUE, "backend policies have no common enabled adapter");
        value.allowed_backends = allowed;
    }));
}
extern "C" CUDAVM_API unsigned int cudavmGetAllowedBackends() {
    std::lock_guard lock(global().mutex);
    return global().allowed_backends & cudavmGetEnabledBackends();
}
extern "C" CUDAVM_API int cudavmGetBackend(cudavmBackend* output) {
    return static_cast<int>(driver_call([&] {
        driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null backend output");
        *output = global().kind;
    }));
}
extern "C" CUDAVM_API int cudavmEnableComputeProfiling(int enabled) {
    return static_cast<int>(driver_call([&] {
        driver_ready();require(enabled==0 || enabled==1,CUDA_ERROR_INVALID_VALUE,"invalid compute profiling flag");
        auto owner=current();require(owner->native->kind()!=cudavmBackendCuda,CUDA_ERROR_NOT_SUPPORTED,"compute profiling requires a shader adapter");
        owner->native->profile->enabled=bool(enabled);
    }));
}
extern "C" CUDAVM_API int cudavmEventComputeElapsedTime(float* milliseconds,void* start,void* end) {
    return static_cast<int>(driver_call([&] {
        driver_ready();require(milliseconds,CUDA_ERROR_INVALID_VALUE,"null compute time output");
        auto first=static_cast<CUevent>(start),last=static_cast<CUevent>(end);float ignored{};event_elapsed(&ignored,first,last);
        auto a=event_lookup(first).timestamp,b=event_lookup(last).timestamp;
        require(a && b && a->compute_measured && b->compute_measured,CUDA_ERROR_NOT_SUPPORTED,"compute profiling was not enabled for both recorded events");
        require(b->compute_nanoseconds>=a->compute_nanoseconds,CUDA_ERROR_INVALID_VALUE,"compute timing events are reversed");
        *milliseconds=float(double(b->compute_nanoseconds-a->compute_nanoseconds)/1000000.0);
    }));
}
extern "C" CUDAVM_API int cudavmShutdown() {
    return static_cast<int>(driver_call([&] {
        auto& value = global();
        require(!value.shutting_down, CUDA_ERROR_NOT_PERMITTED, "shutdown is already in progress");
        value.shutting_down = true;
        for (auto& [handle, owner] : value.contexts) owner->retiring = true;
        for (auto& [handle, owner] : value.contexts) retire_context(owner);
        value.streams.clear(); value.events.clear(); value.reset_events.clear();
        value.graph_execs.clear(); value.graph_nodes.clear(); value.graphs.clear();
        value.functions.clear(); value.modules.clear(); value.library_kernels.clear(); value.libraries.clear();
        value.graph_live.clear(); value.textures.clear(); value.surfaces.clear(); value.arrays.clear(); value.mipmapped_arrays.clear(); value.allocations.clear(); value.pool_retiring.clear(); value.hosts.clear(); value.graphics.clear(); value.memories.clear(); value.semaphores.clear();
        value.primaries.clear(); value.primary_records.clear(); value.primary_flags.clear(); value.contexts.clear();
        value.graph_cache.clear(); value.graph_pools.clear();
        value.memory_pools.clear(); value.default_pools.clear(); value.current_pools.clear();
        value.pool_storage.clear(); value.command_domains.clear(); value.device_buffers.clear();
        value.provider.reset(); value.devices.clear(); value.initialized = false;
        value.shutting_down = false;
        context_stack().clear(); ++value.epoch;
    }));
}
DRIVER_API(cuInit, (unsigned flags), require(flags == 0, CUDA_ERROR_INVALID_VALUE, "cuInit flags must be zero"); initialize())
DRIVER_API(cuDriverGetVersion, (int* version), require(version, CUDA_ERROR_INVALID_VALUE, "null version output"); *version = CUDA_VERSION)
DRIVER_API(cuDeviceGetCount, (int* count), driver_ready(); require(count, CUDA_ERROR_INVALID_VALUE, "null device count"); *count = static_cast<int>(global().devices.size()))
DRIVER_API(cuDeviceGet, (CUdevice* device, int ordinal), driver_ready(); require(device, CUDA_ERROR_INVALID_VALUE, "null device output"); valid_device(ordinal); *device = ordinal)
DRIVER_API(cuDeviceGetAttribute, (int* output, CUdevice_attribute attribute, CUdevice device), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null device attribute output"); valid_device(device);
    *output = attribute == CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS || attribute == CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR ? 1 :
    attribute == CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES || attribute == CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES ||
    attribute == CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED || attribute == CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED ? 0 :
    global().kind != cudavmBackendCuda && (attribute == CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH || attribute == CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH) ? 0 :
    global().kind != cudavmBackendCuda && (attribute == CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY || attribute == CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS ||
    attribute == CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST || attribute == CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS ||
    attribute == CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES || attribute == CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS) ? 0 :
    attribute == CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED || attribute == CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED || (attribute == CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY && global().kind != cudavmBackendCuda) ? 1 : attribute == CU_DEVICE_ATTRIBUTE_MAX_PITCH ? static_cast<int>(maximum_pitch(device)) : provider_device_attribute(device, attribute))
DRIVER_API(cuDeviceGetName, (char* name, int length, CUdevice device), driver_ready(); require(name && length > 0, CUDA_ERROR_INVALID_VALUE, "invalid name output"); valid_device(device); auto& text = global().devices[device].name; size_t count = std::min(text.size(), static_cast<size_t>(length - 1)); std::memcpy(name, text.data(), count); name[count] = 0)
DRIVER_API(cuDeviceTotalMem, (size_t* bytes, CUdevice device), driver_ready(); require(bytes, CUDA_ERROR_INVALID_VALUE, "null memory output"); valid_device(device); auto memory = global().devices[device].memory_bytes; require(memory != 0, CUDA_ERROR_NOT_SUPPORTED, "backend memory capacity query is not implemented"); *bytes = memory)
DRIVER_API(cuCtxCreate, (CUcontext* output, CUctxCreateParams* parameters, unsigned flags, CUdevice device), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null context output"); require(parameters == nullptr, CUDA_ERROR_NOT_SUPPORTED, "advanced context parameters are not implemented"); auto owner = create(device, false, flags); context_stack().push_back(owner->handle); *output = owner->handle)
DRIVER_API(cuCtxDestroy, (CUcontext handle), driver_ready(); auto owner = lookup(handle); require(!owner->primary, CUDA_ERROR_INVALID_CONTEXT, "primary contexts must be released"); destroy(owner))
DRIVER_API(cuCtxGetCurrent, (CUcontext* output), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null context output"); *output = context_stack().empty() ? nullptr : context_stack().back())
DRIVER_API(cuCtxSetCurrent, (CUcontext handle), driver_ready(); set_current(handle))
DRIVER_API(cuCtxPushCurrent, (CUcontext handle), driver_ready(); lookup(handle); context_stack().push_back(handle))
DRIVER_API(cuCtxPopCurrent, (CUcontext* output), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null context output"); auto owner = current(); *output = owner->handle; context_stack().pop_back())
DRIVER_API(cuCtxGetDevice, (CUdevice* device), driver_ready(); require(device, CUDA_ERROR_INVALID_VALUE, "null device output"); *device = current()->device)
DRIVER_API(cuCtxGetFlags, (unsigned* flags), driver_ready(); require(flags, CUDA_ERROR_INVALID_VALUE, "null flags output"); *flags = current()->flags)
DRIVER_API(cuCtxSynchronize, (), driver_ready(); synchronize_context(current()))
DRIVER_API(cuDevicePrimaryCtxRetain, (CUcontext* output, CUdevice device), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null context output"); auto owner = primary(device); ++global().primary_records.at(device).retains; *output = owner->handle)
DRIVER_API(cuDevicePrimaryCtxRelease, (CUdevice device), driver_ready(); release_primary(device))
DRIVER_API(cuDevicePrimaryCtxGetState, (CUdevice device, unsigned* flags, int* active), driver_ready(); valid_device(device); require(flags && active, CUDA_ERROR_INVALID_VALUE, "null primary state output"); auto it = global().primary_flags.find(device); *flags = it == global().primary_flags.end() ? 0 : it->second; *active = global().primaries.contains(device) ? 1 : 0)
DRIVER_API(cuMemAlloc, (CUdeviceptr* output, size_t bytes), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null allocation output"); *output = allocate(current(), bytes))
DRIVER_API(cuMemFree, (CUdeviceptr pointer), driver_ready(); release(pointer, current()))
DRIVER_API(cuMemGetAddressRange, (CUdeviceptr* base, size_t* bytes, CUdeviceptr pointer), driver_ready(); auto owner = current(); auto region = resolve(pointer, 0, owner); if (base) *base = pointer - region.offset; if (bytes) *bytes = region.value->native->size)
DRIVER_API(cuMemcpyHtoD, (CUdeviceptr to, const void* from, size_t bytes), driver_ready(); write(to, from, bytes, current()))
DRIVER_API(cuMemcpyDtoH, (void* to, CUdeviceptr from, size_t bytes), driver_ready(); read(to, from, bytes, current()))
DRIVER_API(cuMemcpyDtoD, (CUdeviceptr to, CUdeviceptr from, size_t bytes), driver_ready(); copy(to, from, bytes, current()))
DRIVER_API(cuMemsetD8, (CUdeviceptr to, unsigned char pattern, size_t count), driver_ready(); fill(to, pattern, count, current()))
DRIVER_API(cuMemsetD16, (CUdeviceptr to, unsigned short pattern, size_t count), driver_ready(); fill(to, pattern, count, current()))
DRIVER_API(cuMemsetD32, (CUdeviceptr to, unsigned pattern, size_t count), driver_ready(); fill(to, pattern, count, current()))
DRIVER_API(cuModuleLoad, (CUmodule* output, const char* path), driver_ready(); require(output && path && *path, CUDA_ERROR_INVALID_VALUE, "invalid module file arguments"); auto owner = current(); auto native = native_call(owner, [&] { return owner->native->load_file(path); }, true); *output = add_module(owner, std::move(native)))
DRIVER_API(cuModuleLoadData, (CUmodule* output, const void* image), driver_ready(); require(output && image, CUDA_ERROR_INVALID_VALUE, "invalid module image arguments"); auto owner = current(); auto native = native_call(owner, [&] { return owner->native->load_image(image, 0, nullptr, nullptr); }, true); *output = add_module(owner, std::move(native)))
DRIVER_API(cuModuleLoadDataEx, (CUmodule* output, const void* image, unsigned count, CUjit_option* options, void** values), driver_ready(); require(output && image && (!count || (options && values)), CUDA_ERROR_INVALID_VALUE, "invalid module JIT arguments"); auto owner = current(); auto native = native_call(owner, [&] { return owner->native->load_image(image, count, options, values); }, true); *output = add_module(owner, std::move(native)))
DRIVER_API(cuModuleGetFunction, (CUfunction* output, CUmodule module, const char* name), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null function output"); *output = find_function(module, name, current()))
DRIVER_API(cuModuleGetGlobal, (CUdeviceptr* output, size_t* bytes, CUmodule module, const char* name), driver_ready(); require(output || bytes, CUDA_ERROR_INVALID_VALUE, "missing global query outputs"); auto symbol=find_global(module,name,current()); if(output)*output=symbol.address; if(bytes)*bytes=symbol.bytes)
DRIVER_API(cuModuleUnload, (CUmodule module), driver_ready(); auto owner = current(); require(!module_lookup(module, owner).library, CUDA_ERROR_NOT_PERMITTED, "library modules are unloaded with their library"); auto native = module_lookup(module, owner).native; synchronize_context(owner); require(module_lookup(module, owner).native == native, CUDA_ERROR_INVALID_HANDLE, "module changed while waiting"); native->unload(); std::erase_if(global().functions, [&](auto& item) { return item.second.module == module; }); std::erase_if(global().allocations,[&](auto& item){return item.second.module==module;}); global().modules.erase(module))
DRIVER_API(cuFuncGetParamInfo, (CUfunction function, size_t index, size_t* offset, size_t* size), driver_ready(); require(offset, CUDA_ERROR_INVALID_VALUE, "null parameter offset output"); auto& value = function_lookup(function, current()); size_t local_offset = 0, local_size = 0; value.native->parameter_info(index, local_offset, local_size); *offset = local_offset; if (size) *size = local_size)
DRIVER_API(cuFuncGetParamCount, (CUfunction function, size_t* count), driver_ready(); require(count, CUDA_ERROR_INVALID_VALUE, "null parameter count output"); *count = function_lookup(function, current()).native->parameter_layout().size())
DRIVER_API(cuFuncGetModule, (CUmodule* output, CUfunction function), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null function module output"); *output = function_lookup(function, current()).module)
DRIVER_API(cuFuncGetName, (const char** output, CUfunction function), driver_ready(); require(output && function, CUDA_ERROR_INVALID_VALUE, "null function name output or handle"); *output = function_lookup(function, current()).name.c_str())
DRIVER_API(cuLaunchKernel, (CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream, void** parameters, void** extra), driver_ready(); launch(function, {{gx, gy, gz}, {bx, by, bz}, shared}, stream, parameters, extra, launch_context(function, stream)))

DRIVER_API(cuPointerGetAttribute, (void* output, CUpointer_attribute attribute, CUdeviceptr pointer),
    driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null pointer attribute output"); pointer_attribute(output, attribute, pointer, false))
DRIVER_API(cuGetErrorName, (CUresult error, const char** output), require(output, CUDA_ERROR_INVALID_VALUE, "null error name output"); *output = error_name(error); require(*output, CUDA_ERROR_INVALID_VALUE, "unrecognized error code"))
DRIVER_API(cuGetErrorString, (CUresult error, const char** output), require(output, CUDA_ERROR_INVALID_VALUE, "null error string output"); *output = error_string(error); require(*output, CUDA_ERROR_INVALID_VALUE, "unrecognized error code"))

// Driver-owned metadata probes: no initialization and no Runtime behavior.
extern "C" CUDAVM_CUDA_API uint64_t cudavmDriverEpoch() {
    std::lock_guard lock(global().mutex); return global().epoch;
}
extern "C" CUDAVM_CUDA_API int cudavmDriverIsLibraryKernel(const void* handle) {
    std::lock_guard lock(global().mutex); return global().library_kernels.contains(reinterpret_cast<CUkernel>(const_cast<void*>(handle)));
}
extern "C" CUDAVM_CUDA_API int cudavmDriverContextIsPrimary(CUcontext handle) {
    std::lock_guard lock(global().mutex);
    return primary_device(handle) >= 0;
}
extern "C" CUDAVM_CUDA_API int cudavmDriverPrimaryDevice(CUcontext handle) {
    std::lock_guard lock(global().mutex); return primary_device(handle);
}
extern "C" CUDAVM_CUDA_API uint64_t cudavmDriverContextGeneration(CUcontext handle) {
    std::lock_guard lock(global().mutex); auto it = global().contexts.find(handle);
    return it != global().contexts.end() && !it->second->retiring ? it->second->id : 0;
}
extern "C" CUDAVM_CUDA_API int cudavmDriverContextExists(CUcontext handle) {
    std::lock_guard lock(global().mutex); auto it = global().contexts.find(handle); return it != global().contexts.end() && !it->second->retiring;
}
extern "C" CUDAVM_CUDA_API int cudavmDriverIsDevicePointer(CUdeviceptr pointer) {
    std::lock_guard lock(global().mutex); return containing(pointer) != global().allocations.end();
}
extern "C" CUDAVM_CUDA_API int cudavmDriverIsHostPointer(CUdeviceptr pointer) {
    if (executing_host_function) return 0;
    std::lock_guard lock(global().mutex);
    auto entry = host_containing(static_cast<uintptr_t>(pointer));
    if (entry != global().hosts.end()) return !entry->second->retiring;
    auto alias = containing(pointer);
    auto host = alias != global().allocations.end() ? alias->second.host.lock() : std::shared_ptr<host_entry>{};
    return host && !host->retiring;
}
extern "C" CUDAVM_CUDA_API int cudavmDriverIsManagedPointer(CUdeviceptr pointer) {
    std::lock_guard lock(global().mutex); auto it = containing(pointer);
    return it != global().allocations.end() && it->second.native->is_managed();
}
extern "C" CUDAVM_CUDA_API cudavmBackend cudavmDriverBackend() {
    std::lock_guard lock(global().mutex); return global().kind;
}
extern "C" CUDAVM_CUDA_API void cudavmDriverSetDiagnostic(const char* message) {
    diagnostic_text() = message;
}
#include "per_thread_exports.inl"
