#pragma once
#include <cuda/cudavm_cuda.h>
#include <cuda.h>
#include <cuda/cudavm_interop.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <thread>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include "../cudavm/data_view_descriptor.hpp"

namespace cudavm {
struct session;
struct failure : std::runtime_error {
    CUresult code;
    failure(CUresult value, std::string message) : std::runtime_error(std::move(message)), code(value) {}
};
inline void require(bool condition, CUresult error, const char* message) {
    if (!condition) throw failure(error, message);
}
struct device_info {
    std::string name;
    size_t memory_bytes = 0;
};
// Limits of the CUDA-shaped shader transport, intersected with the adapter's
// native dispatch limits. These do not describe NVIDIA SM residency.
struct shader_resource_limits {
    unsigned threads = 1024;
    std::array<unsigned, 3> block{1024, 1024, 1024};
    std::array<unsigned, 3> grid{65535, 65535, 65535};
    unsigned shared_bytes = 32768;
};
int shader_device_attribute(const shader_resource_limits&, CUdevice_attribute);
struct device_luid_info { std::array<char, 8> bytes{}; unsigned node_mask = 0; };
struct pci_address {
    unsigned domain = 0, bus = 0, device = 0, function = 0;
    bool operator==(const pci_address&) const = default;
};
struct graph_pool_stats {
    std::mutex mutex;
    uint64_t used = 0, used_high = 0, reserved = 0, reserved_high = 0;
};
struct pool_backing_stats {
    std::mutex mutex;
    uint64_t reserved = 0, high = 0;
};
// Reservation follows the buffer, including commands retained after pool
// destruction. Only logical used bytes belong to the allocation registry.
struct pool_backing_charge {
    std::shared_ptr<pool_backing_stats> stats;
    uint64_t bytes;
    pool_backing_charge(std::shared_ptr<pool_backing_stats> value, uint64_t size)
        : stats(std::move(value)), bytes(size) {
        std::lock_guard lock(stats->mutex);
        require(bytes <= UINT64_MAX - stats->reserved, CUDA_ERROR_OUT_OF_MEMORY, "pool reservation overflow");
        stats->reserved += bytes; stats->high = std::max(stats->high, stats->reserved);
    }
    ~pool_backing_charge() { std::lock_guard lock(stats->mutex); stats->reserved -= bytes; }
};
// A charge follows physical storage, including captures retained by workers.
// Detached cache blocks remain reserved but are no longer associated with a graph.
struct graph_pool_charge {
    std::shared_ptr<graph_pool_stats> pool;
    uint64_t bytes;
    bool mapped = true;
    graph_pool_charge(std::shared_ptr<graph_pool_stats> value, uint64_t size) : pool(std::move(value)), bytes(size) {
        std::lock_guard lock(pool->mutex);
        pool->reserved += bytes; pool->used += bytes;
        pool->reserved_high = std::max(pool->reserved_high, pool->reserved);
        pool->used_high = std::max(pool->used_high, pool->used);
    }
    void associate(bool value) {
        std::lock_guard lock(pool->mutex);
        if (mapped == value) return;
        mapped = value;
        if (value) { pool->used += bytes; pool->used_high = std::max(pool->used_high, pool->used); }
        else pool->used -= bytes;
    }
    ~graph_pool_charge() {
        std::lock_guard lock(pool->mutex);
        pool->reserved -= bytes; if (mapped) pool->used -= bytes;
    }
};
struct buffer {
    size_t size;
    // Pool backing retains its private storage session beyond public contexts.
    std::shared_ptr<session> pool_session;
    std::shared_ptr<graph_pool_charge> graph_charge;
    std::shared_ptr<pool_backing_charge> pool_charge;
    explicit buffer(size_t bytes) : size(bytes) {}
    virtual ~buffer() = default;
    // Native CUDA allocations preserve their native addresses. Other adapters
    // receive stable CUDA virtual addresses from the common allocation registry.
    virtual uint64_t native_address() const { return 0; }
    virtual bool is_managed() const { return false; }
    virtual buffer& storage() { return *this; }
    virtual const buffer& storage() const { return *this; }
    virtual size_t storage_offset() const { return 0; }
    virtual std::byte* host_address() const { return nullptr; }
    virtual size_t graph_backing_size() const { return size; }
    virtual size_t physical_size() const { return graph_backing_size(); }
    virtual size_t restore_graph() { return size; }
    virtual void trim_graph() { throw failure(CUDA_ERROR_NOT_SUPPORTED, "graph backing cannot be unmapped"); }
};
// Transfer layout for ordinary packed CUDA numeric arrays. Sampling and
// sparse/compressed image layouts have separate backend contracts.
inline size_t array_element_bytes(const CUDA_ARRAY3D_DESCRIPTOR& d) {
    require(d.NumChannels == 1 || d.NumChannels == 2 || d.NumChannels == 4, CUDA_ERROR_INVALID_VALUE, "invalid array channel count");
    size_t component = 0;
    switch (d.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8: case CU_AD_FORMAT_SIGNED_INT8: component = 1; break;
    case CU_AD_FORMAT_UNSIGNED_INT16: case CU_AD_FORMAT_SIGNED_INT16: case CU_AD_FORMAT_HALF: component = 2; break;
    case CU_AD_FORMAT_UNSIGNED_INT32: case CU_AD_FORMAT_SIGNED_INT32: case CU_AD_FORMAT_FLOAT: component = 4; break;
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "array format requires a separate image layout");
    }
    return component * d.NumChannels;
}
inline CUDA_ARRAY3D_DESCRIPTOR array_level_descriptor(CUDA_ARRAY3D_DESCRIPTOR d, unsigned level) {
    d.Width = std::max(size_t(1), d.Width >> level);
    if (d.Height) d.Height = std::max(size_t(1), d.Height >> level);
    if (d.Depth && !(d.Flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP))) d.Depth = std::max(size_t(1), d.Depth >> level);
    return d;
}
inline size_t array_storage_bytes(const CUDA_ARRAY3D_DESCRIPTOR& d) {
    size_t bytes = array_element_bytes(d);
    for (auto extent : {d.Width, std::max(size_t(1), d.Height), std::max(size_t(1), d.Depth)}) {
        require(extent && extent <= std::numeric_limits<size_t>::max() / bytes, CUDA_ERROR_INVALID_VALUE, "array storage size overflow");
        bytes *= extent;
    }
    return bytes;
}
// A logical allocation retains its backing and reservation independently of
// the pool handle. Reuse waits for the frees following queued snapshots;
// completed cache views with other retained owners remain unavailable.
struct pool_buffer : buffer {
    std::shared_ptr<buffer> backing;
    size_t offset;
    pool_buffer(std::shared_ptr<buffer> parent, size_t begin, size_t bytes)
        : buffer(bytes), backing(std::move(parent)), offset(begin) {
        require(offset <= backing->size && bytes <= backing->size - offset,
                CUDA_ERROR_INVALID_VALUE, "pool slice exceeds backing");
        pool_session = backing->pool_session;
        pool_charge = backing->pool_charge;
    }
    buffer& storage() override { return backing->storage(); }
    const buffer& storage() const override { return backing->storage(); }
    size_t storage_offset() const override { return backing->storage_offset() + offset; }
    uint64_t native_address() const override {
        auto address = backing->native_address(); return address ? address + offset : 0;
    }
    size_t physical_size() const override { return 0; }
};
// Shader graph addresses are registry identities; no native storage is needed
// until upload or replay. Parameter validation can still inspect the byte range.
struct graph_address_buffer : buffer {
    using buffer::buffer;
    size_t physical_size() const override { return 0; }
};
struct host_allocation {
    void* address = nullptr;
    size_t size;
    explicit host_allocation(size_t bytes) : size(bytes) {}
    virtual ~host_allocation() = default;
    virtual unsigned flags(unsigned requested) { return requested; }
};
struct native_queue {
    virtual ~native_queue() = default;
    virtual bool recording() const { return false; }
};
// An immutable, already submitted GPU dependency. It never promises completion
// of host continuations, staging readback or scheduler actions.
struct native_signal {
    virtual ~native_signal() = default;
    virtual bool can_wait(native_queue*) const = 0;
    virtual void enqueue_wait(native_queue*) const = 0;
};
// Polling never waits for device execution. A task owns its native fence and
// any staging storage until the fence completes, including on error paths.
struct device_task {
    virtual ~device_task() = default;
    virtual bool query() = 0;
    virtual std::shared_ptr<native_signal> signal() const { return {}; }
    // Cooperative memory waits have no native work between polling transfers.
    virtual bool yields_to_exclusive() const { return false; }
};
using device_work = std::shared_ptr<device_task>;
struct native_signal_group : native_signal {
    std::vector<std::shared_ptr<native_signal>> signals;
    bool can_wait(native_queue* queue) const override {
        return std::all_of(signals.begin(), signals.end(), [&](const auto& signal) { return signal->can_wait(queue); });
    }
    void enqueue_wait(native_queue* queue) const override {
        for (const auto& signal : signals) signal->enqueue_wait(queue);
    }
};
struct device_group : device_task {
    std::vector<device_work> tasks;
    std::shared_ptr<native_signal> signal() const override {
        auto result = std::make_shared<native_signal_group>();
        for (const auto& task : tasks) if (task) {
            auto signal = task->signal();
            if (!signal) return {};
            result->signals.push_back(std::move(signal));
        }
        return result->signals.empty() ? nullptr : result;
    }
    bool query() override {
        bool ready = true;
        for (auto& task : tasks) if (task) { if (task->query()) task.reset(); else ready = false; }
        return ready;
    }
};
inline void wait_device(const device_work& task) {
    if (task) while (!task->query()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
struct device_continuation : device_task {
    device_work task;
    std::function<void()> finish;
    device_continuation(device_work work, std::function<void()> action) : task(std::move(work)), finish(std::move(action)) {}
    ~device_continuation() { task.reset(); } // Retire the fence before staging captures.
    bool query() override {
        if (task && !task->query()) return false;
        if (finish) { auto action = std::move(finish); action(); }
        task.reset(); return true;
    }
};
// Recording must not execute work or retain temporary host spans. The graph
// owns its native descriptors; callers retain the recorded resource snapshots.
using native_graph_command = std::function<device_work(native_queue*)>;
struct native_graph_dependency {
    size_t node = 0;
    CUgraphEdgeData edge{};
    bool operator==(const native_graph_dependency& other) const {
        return node == other.node && std::memcmp(&edge, &other.edge, sizeof(edge)) == 0;
    }
};
struct native_graph_node {
    native_graph_command command;
    std::vector<native_graph_dependency> dependencies;
    int priority = 0;
};
struct native_graph {
    virtual ~native_graph() = default;
    virtual device_work enqueue(native_queue*) = 0;
};
inline uint32_t repeated_fill_word(std::span<const std::byte> pattern) {
    require(pattern.size() == 1 || pattern.size() == 2 || pattern.size() == 4, CUDA_ERROR_INVALID_VALUE, "invalid native fill pattern");
    uint32_t word = 0;
    for (size_t i = 0; i < sizeof(word); ++i)
        reinterpret_cast<std::byte*>(&word)[i] = pattern[i % pattern.size()];
    return word;
}
struct device_launch_event {
    CUevent handle = nullptr;
    virtual ~device_launch_event() = default;
};
struct device_timer {
    virtual ~device_timer() = default;
    virtual float elapsed(const device_timer& end) const = 0;
};
struct device_timestamp {
    std::shared_ptr<device_timer> timer;
    uint64_t ticks = 0;
    double nanoseconds_per_tick = 0;
    unsigned valid_bits = 64;
    uint64_t compute_nanoseconds = 0;
    bool compute_measured = false;
};
struct compute_profile {
    std::atomic<bool> enabled{false};
    std::atomic<uint64_t> nanoseconds{0};
    void snapshot(device_timestamp& stamp) const {
        stamp.compute_nanoseconds=nanoseconds.load();stamp.compute_measured=enabled.load();
    }
};
struct host_buffer : buffer {
    std::shared_ptr<host_allocation> storage;
    explicit host_buffer(std::shared_ptr<host_allocation> value) : buffer(value->size), storage(std::move(value)) {}
    uint64_t native_address() const override { return reinterpret_cast<uintptr_t>(storage->address); }
    std::byte* host_address() const override { return static_cast<std::byte*>(storage->address); }
    size_t physical_size() const override { return 0; }
};
std::unique_ptr<host_allocation> allocate_locked_host(size_t);
std::unique_ptr<host_allocation> register_locked_host(void*, size_t);
struct external_memory {
    virtual ~external_memory() = default;
    virtual std::unique_ptr<buffer> map(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC&) = 0;
};
struct external_semaphore {
    virtual ~external_semaphore() = default;
    virtual void signal(uint64_t value) = 0;
    virtual void wait(uint64_t value) = 0;
};
struct interop_capabilities { uint64_t memory = 0, semaphore = 0, metadata = 0; };
constexpr uint64_t handle_bit(unsigned type) { return uint64_t(1) << type; }
// Launch snapshots carry values and registry-checked memory addresses. Public
// stream/event handles require translation before reaching a native descriptor.
struct access_policy_window {
    void* base = nullptr; size_t bytes = 0; float hit_ratio = 0;
    CUaccessProperty hit = CU_ACCESS_PROPERTY_NORMAL, miss = CU_ACCESS_PROPERTY_NORMAL;
    bool operator==(const access_policy_window&) const = default;
};
inline access_policy_window access_policy_from_native(const CUaccessPolicyWindow& value) {
    return {value.base_ptr, value.num_bytes, value.hitRatio, value.hitProp, value.missProp};
}
inline CUaccessPolicyWindow access_policy_to_native(const access_policy_window& value) {
    return {value.base, value.bytes, value.hit_ratio, value.hit, value.miss};
}
inline void validate_access_policy(const access_policy_window& value) {
    require(value.hit_ratio >= 0 && value.hit_ratio <= 1 && value.hit >= CU_ACCESS_PROPERTY_NORMAL &&
        value.hit <= CU_ACCESS_PROPERTY_PERSISTING && value.miss >= CU_ACCESS_PROPERTY_NORMAL &&
        value.miss <= CU_ACCESS_PROPERTY_STREAMING, CUDA_ERROR_INVALID_VALUE, "invalid access policy window");
}
struct launch_options {
    bool cooperative = false;
    std::optional<std::array<unsigned, 3>> cluster;
    std::optional<CUclusterSchedulingPolicy> cluster_policy;
    std::optional<int> priority;
    std::optional<std::array<unsigned char, 2>> domain_map;
    std::optional<CUlaunchMemSyncDomain> domain;
    std::optional<access_policy_window> access_policy;
    std::optional<CUsharedMemoryMode> shared_memory_mode;
    std::optional<std::array<unsigned, 3>> preferred_cluster;
    std::optional<unsigned> carveout, nvlink;
    std::optional<CUlaunchAttributePortableClusterMode> portable_cluster;
    std::optional<bool> programmatic_serialization;
    struct event_record {
        CUevent handle = nullptr;
        int flags = 0;
        bool block_start = false;
        std::shared_ptr<device_launch_event> native;
        bool operator==(const event_record&) const = default;
    };
    std::optional<event_record> programmatic_event, completion_event;
    bool operator==(const launch_options&) const = default;
};
struct launch_shape {
    unsigned grid[3], block[3], shared_bytes;
    launch_options options{};
};
struct allocation_binding { uint64_t address; buffer* native; bool surface_descriptor = false; };
struct image_object {
    uint64_t handle = 0;
    virtual ~image_object() = default;
};
struct kernel_parameter { size_t offset, size; };
struct compute_pipeline {
    unsigned direct_buffers = 0;
    virtual ~compute_pipeline() = default;
};
struct kernel {
    virtual ~kernel() = default;
    virtual uint64_t graph_generation() const { return 0; }
    virtual bool can_record_launch(std::span<const allocation_binding>) const { return false; }
    virtual int attribute(CUfunction_attribute) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "function resource attribute is unavailable on this adapter");
    }
    virtual void set_attribute(CUfunction_attribute, int) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "function resource configuration is unavailable on this adapter");
    }
    virtual void set_cache_config(CUfunc_cache) {}
    virtual void set_shared_config(CUsharedconfig) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "function shared-memory bank configuration is unavailable");
    }
    virtual int active_blocks(int, size_t, unsigned) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native function residency information is unavailable on this adapter");
    }
    virtual void potential_block_size(int&, int&, CUoccupancyB2DSize, size_t, int, unsigned) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native function residency information is unavailable on this adapter");
    }
    virtual size_t available_dynamic_shared(int, int) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native function residency information is unavailable on this adapter");
    }
    virtual int cluster_occupancy(const launch_shape&, native_queue*, bool) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native cluster residency information is unavailable on this adapter");
    }
    virtual std::array<unsigned char, 2> default_domain_map() {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native memory synchronization domains are unavailable on this adapter");
    }
    virtual void launch(const launch_shape&, void** parameters, void** extra) = 0;
    virtual void launch_bound(const launch_shape& shape, void** parameters, void** extra, std::span<const allocation_binding>) { launch(shape, parameters, extra); }
    virtual device_work enqueue_launch(const launch_shape& shape, void** parameters, void** extra,
                                       std::span<const allocation_binding> bindings, native_queue*) {
        launch_bound(shape, parameters, extra, bindings); return {};
    }
    virtual void parameter_info(size_t, size_t&, size_t&) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "kernel parameter introspection is unavailable");
    }
    virtual std::vector<kernel_parameter> parameter_layout() {
        std::vector<kernel_parameter> result;
        for (size_t i = 0; i <= 4096; ++i) {
            size_t offset = 0, size = 0;
            try { parameter_info(i, offset, size); }
            catch (const failure& error) { if (error.code == CUDA_ERROR_INVALID_VALUE) return result; throw; }
            require(i < 4096 && size && offset <= 65536 && size <= 65536 - offset,
                    CUDA_ERROR_NOT_SUPPORTED, "kernel parameter layout exceeds capture limits");
            result.push_back({offset, size});
        }
        return result;
    }
    virtual void validate_configuration(const launch_shape& shape) {
        for (unsigned i = 0; i < 3; ++i)
            require(shape.grid[i] && shape.block[i], CUDA_ERROR_INVALID_VALUE, "launch dimensions must be nonzero");
        if (shape.options.preferred_cluster) {
            const auto required = shape.options.cluster.value_or(std::array<unsigned, 3>{1, 1, 1});
            for (unsigned axis = 0; axis < 3; ++axis) {
                auto preferred = (*shape.options.preferred_cluster)[axis];
                require(preferred && preferred % required[axis] == 0 && shape.grid[axis] % preferred == 0 && (axis != 2 || preferred == required[axis]),
                    CUDA_ERROR_INVALID_CLUSTER_SIZE, "preferred cluster dimensions do not divide the grid or required cluster");
            }
        }
    }
    virtual void validate_launch(const launch_shape& shape, std::span<const std::byte>, std::span<const allocation_binding>) {
        validate_configuration(shape);
    }
};
struct executable {
    std::map<std::string, data_view_descriptor> data_views;
    virtual ~executable() = default;
    virtual std::unique_ptr<kernel> find(const char* name) = 0;
    virtual std::vector<std::string> kernel_names() const { throw failure(CUDA_ERROR_NOT_SUPPORTED, "module kernel enumeration is unavailable"); }
    virtual std::vector<std::string> global_names() const { return {}; }
    virtual std::unique_ptr<buffer> find_global(const char*) { throw failure(CUDA_ERROR_NOT_FOUND, "global symbol is absent from module"); }
    virtual void bind_globals(std::span<const allocation_binding>) {}
    virtual void unload() = 0;
};
struct session {
    virtual uint64_t graph_generation() const { return 0; }
    std::mutex pool_execution;
    std::shared_ptr<compute_profile> profile=std::make_shared<compute_profile>();
    virtual ~session() = default;
    virtual unsigned native_block_limit() const {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native block resource limit is unavailable");
    }
    virtual shader_resource_limits shader_limits() const {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "shader dispatch resource limits are unavailable");
    }
    virtual std::shared_ptr<device_launch_event> create_launch_event() {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native launch dependency events are unavailable");
    }
    virtual unsigned get_context_flags(unsigned requested) { return requested; }
    virtual unsigned set_context_flags(unsigned flags) { return flags; }
    virtual unsigned context_api_version() { return 3020; }
    virtual std::pair<int, int> stream_priority_range() { return {0, -1}; }
    virtual CUstreamAttrValue canonical_stream_attribute(CUstreamAttrID, const CUstreamAttrValue& value, int) { return value; }
    virtual std::array<unsigned char, 2> default_stream_domain_map() {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native memory synchronization domains are unavailable");
    }
    virtual void reset_persisting_cache() {} // No persisting L2 partition on shader transports.
    virtual size_t context_limit(CUlimit limit) {
        if (limit == CU_LIMIT_MAX_L2_FETCH_GRANULARITY || limit == CU_LIMIT_PERSISTING_L2_CACHE_SIZE) return 0;
        throw failure(CUDA_ERROR_UNSUPPORTED_LIMIT, "this context resource limit is not implemented on the adapter");
    }
    virtual void set_context_limit(CUlimit limit, size_t) { (void)context_limit(limit); }
    // Shader adapters expose no cache partition control. These CUDA preferences
    // may be ignored; they do not change shader resource admission.
    virtual CUfunc_cache context_cache_config() { return CU_FUNC_CACHE_PREFER_NONE; }
    virtual void set_context_cache_config(CUfunc_cache) {}
    virtual CUsharedconfig context_shared_config() {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native shared-memory bank configuration is unavailable");
    }
    virtual void set_context_shared_config(CUsharedconfig) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native shared-memory bank configuration is unavailable");
    }
    virtual interop_capabilities interop() { return {}; }
    virtual std::shared_ptr<external_memory> import_memory(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC&, const cudavmExternalMemoryMetadata* = nullptr) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "external memory handle type is unavailable on this device");
    }
    virtual std::shared_ptr<external_semaphore> import_semaphore(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC&) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "external semaphore handle type is unavailable on this device");
    }
    virtual std::unique_ptr<buffer> allocate(size_t bytes) = 0;
    // Buffer storage implements allocation and byte transfers on shader
    // adapters. It does not imply native image sampling support.
    virtual std::shared_ptr<buffer> allocate_array(const CUDA_ARRAY3D_DESCRIPTOR& d) { return allocate(array_storage_bytes(d)); }
    virtual std::shared_ptr<image_object> create_texture(buffer&, const CUDA_RESOURCE_DESC&, size_t, const CUDA_TEXTURE_DESC&, const CUDA_RESOURCE_VIEW_DESC*) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native texture objects are unavailable");
    }
    virtual std::shared_ptr<image_object> create_surface(buffer&) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native surface objects are unavailable");
    }
    virtual std::vector<std::shared_ptr<buffer>> allocate_mipmapped_array(const CUDA_ARRAY3D_DESCRIPTOR& d, unsigned levels) {
        std::vector<std::shared_ptr<buffer>> result;
        for (unsigned i = 0; i < levels; ++i) result.push_back(allocate_array(array_level_descriptor(d, i)));
        return result;
    }
    virtual std::unique_ptr<buffer> allocate_managed(size_t, unsigned) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native managed memory is unavailable on this adapter");
    }
    virtual device_work enqueue_attach(CUdeviceptr, size_t, unsigned, native_queue*) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native managed/pageable stream association is unavailable");
    }
    virtual std::unique_ptr<buffer> allocate_pool(size_t bytes) { return allocate(bytes); }
    virtual void memory_info(size_t&, size_t&) { throw failure(CUDA_ERROR_NOT_SUPPORTED, "native memory query is unavailable"); }
    // Copy eligibility is independent of shader word alignment and slot limits.
    virtual bool can_copy_direct(const buffer&) const { return false; }
    virtual std::unique_ptr<buffer> reserve_graph(size_t bytes) { return std::make_unique<graph_address_buffer>(bytes); }
    virtual std::unique_ptr<host_allocation> allocate_host(size_t bytes, unsigned) { return allocate_locked_host(bytes); }
    virtual std::unique_ptr<host_allocation> register_host(void* pointer, size_t bytes, unsigned) { return register_locked_host(pointer, bytes); }
    virtual std::unique_ptr<buffer> map_host(std::shared_ptr<host_allocation> value) { return std::make_unique<host_buffer>(std::move(value)); }
    virtual std::shared_ptr<native_queue> create_queue(int) { return {}; }
    virtual bool command_regions() const { return false; }
    virtual bool can_record_copy(const buffer&, size_t, const buffer&, size_t, size_t) const { return false; }
    virtual bool can_record_fill(const buffer&, size_t, size_t) const { return false; }
    virtual std::shared_ptr<native_graph> compile_graph(std::span<const native_graph_node>) { return {}; }
    virtual device_work enqueue_marker(native_queue*) { return {}; }
    // Values are published before the returned task completes. Timing uses
    // either a native timer pair or ticks with a nonzero clock period.
    virtual device_work enqueue_timestamp(std::shared_ptr<device_timestamp>, native_queue*) { return {}; }
    virtual std::unique_ptr<buffer> allocate_pitched(size_t width, size_t height, unsigned, size_t& pitch) {
        require(width <= std::numeric_limits<size_t>::max() - 255, CUDA_ERROR_INVALID_VALUE, "pitched width overflow");
        auto value = (width + 255) & ~size_t(255);
        require(value && height <= std::numeric_limits<size_t>::max() / value, CUDA_ERROR_INVALID_VALUE, "pitched allocation size overflow");
        auto result = allocate(value * height); pitch = value; return result;
    }
    virtual void write_device(buffer&, size_t offset, std::span<const std::byte>) = 0;
    virtual void read_device(buffer&, size_t offset, std::span<std::byte>) = 0;
    virtual void copy_device(buffer& destination, size_t destination_offset, buffer& source, size_t source_offset, size_t bytes) = 0;
    void write(buffer& value, size_t offset, std::span<const std::byte> bytes) {
        if (value.pool_session && value.pool_session.get() != this) {
            auto& storage = *value.pool_session; std::lock_guard lock(storage.pool_execution);
            storage.write(value, offset, bytes); return;
        }
        if (auto host = value.host_address()) std::memcpy(host + offset, bytes.data(), bytes.size());
        else write_device(value, offset, bytes);
    }
    void read(buffer& value, size_t offset, std::span<std::byte> bytes) {
        if (value.pool_session && value.pool_session.get() != this) {
            auto& storage = *value.pool_session; std::lock_guard lock(storage.pool_execution);
            storage.read(value, offset, bytes); return;
        }
        if (auto host = value.host_address()) std::memcpy(bytes.data(), host + offset, bytes.size());
        else read_device(value, offset, bytes);
    }
    void copy(buffer& to, size_t destination, buffer& from, size_t source, size_t bytes) {
        if (can_copy_direct(to) && can_copy_direct(from)) {
            copy_device(to, destination, from, source, bytes); return;
        }
        auto to_session = to.pool_session ? to.pool_session.get() : this;
        auto from_session = from.pool_session ? from.pool_session.get() : this;
        if (to_session != this || from_session != this) {
            if (to_session == from_session) {
                std::lock_guard lock(to_session->pool_execution);
                to_session->copy(to, destination, from, source, bytes);
            } else {
                std::vector<std::byte> staging(std::min(bytes, size_t(1024 * 1024)));
                for (size_t position = 0; position < bytes;) {
                    auto count = std::min(bytes - position, staging.size());
                    read(from, source + position, {staging.data(), count});
                    write(to, destination + position, {staging.data(), count}); position += count;
                }
            }
            return;
        }
        auto to_host = to.host_address(), from_host = from.host_address();
        if (to_host) read(from, source, {to_host + destination, bytes});
        else if (from_host) write(to, destination, {from_host + source, bytes});
        else copy_device(to, destination, from, source, bytes);
    }
    virtual device_work enqueue_write(buffer& value, size_t offset, std::span<const std::byte> bytes, native_queue*) { write(value, offset, bytes); return {}; }
    virtual device_work enqueue_read(buffer& value, size_t offset, std::span<std::byte> bytes, native_queue*) { read(value, offset, bytes); return {}; }
    virtual device_work enqueue_copy(buffer& to, size_t destination, buffer& from, size_t source, size_t bytes, native_queue* queue) {
        if (auto host = to.host_address()) return enqueue_read(from, source, {host + destination, bytes}, queue);
        if (auto host = from.host_address()) return enqueue_write(to, destination, {host + source, bytes}, queue);
        copy(to, destination, from, source, bytes); return {};
    }
    virtual device_work enqueue_fill(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes, native_queue* queue) {
        if (value.host_address()) { fill(value, offset, pattern, bytes); return {}; }
        auto staging = std::make_shared<std::vector<std::byte>>(bytes);
        for (size_t i = 0; i < bytes; i += pattern.size()) std::memcpy(staging->data() + i, pattern.data(), pattern.size());
        auto continuation = std::make_shared<device_continuation>(device_work{}, [staging] {});
        continuation->task = enqueue_write(value, offset, *staging, queue); return continuation;
    }
    virtual void synchronize() = 0;
    virtual std::shared_ptr<executable> load_image(const void*, unsigned, CUjit_option*, void**);
    virtual std::shared_ptr<executable> load_file(const char*);

    virtual bool supports_native_spirv(std::span<const uint32_t>) const { return false; }
    virtual bool supports_native_dxil(std::string_view) const { return false; }
    virtual void validate_native_dxil_shape(const launch_shape&) const {}
    virtual std::unique_ptr<compute_pipeline> compile_native_dxil(std::string_view, const launch_shape&) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native DXIL ingestion is unavailable");
    }
    virtual void validate_native_spirv_shape(const launch_shape&) const {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native Vulkan SPIR-V ABI is unavailable on this adapter");
    }
    virtual std::unique_ptr<compute_pipeline> compile_native_spirv(std::span<const uint32_t>, const launch_shape&) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native Vulkan SPIR-V ingestion is unavailable on this adapter");
    }
    // Direct resources must share this session's native resource owner and
    // support word accesses. Other allocations use the staged transport.
    virtual unsigned direct_binding_limit() const { return 0; }
    virtual bool can_bind_direct(const buffer&) const { return false; }
    virtual device_work enqueue_dispatch_bound(compute_pipeline& pipeline, const launch_shape& shape,
            buffer& memory, buffer& arguments, std::span<buffer* const> bindings, native_queue* queue) {
        require(bindings.empty(), CUDA_ERROR_NOT_SUPPORTED, "direct shader bindings are unavailable");
        return enqueue_dispatch(pipeline, shape, memory, arguments, queue);
    }
    virtual void dispatch(compute_pipeline&, const launch_shape&, buffer&, buffer&) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native shader dispatch is unavailable");
    }
    virtual device_work enqueue_dispatch(compute_pipeline& pipeline, const launch_shape& shape, buffer& memory, buffer& arguments, native_queue*) {
        dispatch(pipeline, shape, memory, arguments); return {};
    }
    virtual cudavmBackend kind() const { return cudavmBackendCuda; }
    virtual void fill_device(buffer& destination, size_t offset, std::span<const std::byte> pattern, size_t bytes) {
        require(!pattern.empty() && bytes % pattern.size() == 0, CUDA_ERROR_INVALID_VALUE, "invalid fill pattern");
        std::vector<std::byte> staging(std::min(bytes, size_t(1024 * 1024)));
        for (size_t i = 0; i < staging.size(); ++i) staging[i] = pattern[i % pattern.size()];
        for (size_t position = 0; position < bytes;) {
            size_t count = std::min(bytes - position, staging.size());
            write(destination, offset + position, std::span<const std::byte>(staging.data(), count));
            position += count;
        }
    }
    void fill(buffer& value, size_t offset, std::span<const std::byte> pattern, size_t bytes) {
        if (value.pool_session && value.pool_session.get() != this) {
            auto& storage = *value.pool_session; std::lock_guard lock(storage.pool_execution);
            storage.fill(value, offset, pattern, bytes); return;
        }
        if (auto host = value.host_address()) {
            require(!pattern.empty() && bytes % pattern.size() == 0, CUDA_ERROR_INVALID_VALUE, "invalid fill pattern");
            for (size_t i = 0; i < bytes; i += pattern.size()) std::memcpy(host + offset + i, pattern.data(), pattern.size());
        } else fill_device(value, offset, pattern, bytes);
    }
};
struct device_library {
    virtual ~device_library() = default;
    virtual std::shared_ptr<executable> bind(session&) = 0;
    virtual std::shared_ptr<executable> bind_registered(session&, std::span<const std::byte>, const std::map<std::string, std::shared_ptr<buffer>>&) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "independent managed registration bindings are unavailable");
    }
    virtual void find(const char*) = 0;
    virtual std::vector<std::string> kernel_names() = 0;
    virtual std::vector<kernel_parameter> parameters(const char*) = 0;
    virtual int attribute(const char*, CUfunction_attribute, int) = 0;
    virtual void set_attribute(const char*, CUfunction_attribute, int, int) = 0;
    virtual void set_cache_config(const char*, CUfunc_cache, int) = 0;
    virtual std::unique_ptr<buffer> managed(const char*, session&) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native library managed symbols are unavailable");
    }
    virtual void* unified_function(const char*) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native unified functions are unavailable");
    }
    virtual void unload() = 0;
};
struct backend {
    virtual ~backend() = default;
    virtual std::vector<device_info> discover() = 0;
    virtual std::unique_ptr<session> open(int device) = 0;
    virtual std::array<char, 16> device_uuid(int, bool) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native device UUID is unavailable on this adapter");
    }
    virtual device_luid_info device_luid(int) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native device LUID is unavailable on this adapter");
    }
    virtual pci_address device_pci(int) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "native device PCI address is unavailable on this adapter");
    }
    virtual std::optional<int> device_from_pci(const char*) { return {}; }
    virtual CUmoduleLoadingMode loading_mode() {
        auto* mode = std::getenv("CUDA_MODULE_LOADING"); return mode && std::string_view(mode) == "EAGER" ? CU_MODULE_EAGER_LOADING : CU_MODULE_LAZY_LOADING;
    }
    virtual std::shared_ptr<device_library> load_library(const void*, const char*, CUjit_option*, void**, unsigned, CUlibraryOption*, void**, unsigned) { return {}; }
    virtual std::unique_ptr<session> open(int device, unsigned flags) {
        auto result = open(device); result->set_context_flags(flags); return result;
    }
    virtual shader_resource_limits shader_limits(int) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "shader dispatch resource limits are unavailable");
    }
    virtual int attribute(int, CUdevice_attribute) {
        throw failure(CUDA_ERROR_NOT_SUPPORTED, "device resource attribute is unavailable");
    }
};
struct shader_backend : backend {
    int attribute(int device, CUdevice_attribute which) override { return shader_device_attribute(shader_limits(device), which); }
};
std::unique_ptr<backend> make_cuda_backend();
std::unique_ptr<backend> make_opencl_backend();
std::unique_ptr<backend> make_opengl_backend();
std::unique_ptr<backend> make_vulkan_backend();
std::unique_ptr<backend> make_directx_backend();
std::unique_ptr<backend> make_backend(cudavmBackend);
const char* backend_name(cudavmBackend);
std::optional<std::string> backend_environment(const char*);
std::span<const cudavmBackend> enabled_backends();
bool backend_enabled(cudavmBackend);
struct backend_selection {
    cudavmBackend kind;
    std::unique_ptr<backend> provider;
    std::vector<device_info> devices;
};
backend_selection discover_backend(cudavmBackend, unsigned int allowed_mask);
}
