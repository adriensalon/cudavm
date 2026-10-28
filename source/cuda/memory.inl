// Shared, synchronous linear-memory geometry. All endpoints are preflighted
// before the first native transfer; padding is never included in row writes.
size_t memory_add(size_t a, size_t b) {
    require(b <= std::numeric_limits<size_t>::max() - a, CUDA_ERROR_INVALID_VALUE, "memory geometry addition overflow");
    return a + b;
}
size_t memory_mul(size_t a, size_t b) {
    require(!a || b <= std::numeric_limits<size_t>::max() / a, CUDA_ERROR_INVALID_VALUE, "memory geometry multiplication overflow");
    return a * b;
}
size_t maximum_pitch(int device) {
    if (device_kind(device) != cudavmBackendCuda) return std::numeric_limits<int>::max();
    auto pitch = global().provider->attribute(device, CU_DEVICE_ATTRIBUTE_MAX_PITCH);
    require(pitch > 0, CUDA_ERROR_UNKNOWN, "native maximum pitch is invalid");
    return static_cast<size_t>(pitch);
}
CUdeviceptr allocate_pitch(const std::shared_ptr<context>& owner, size_t width, size_t height, unsigned element, size_t& pitch) {
    require(width && height && (element == 4 || element == 8 || element == 16), CUDA_ERROR_INVALID_VALUE, "invalid pitched allocation dimensions or element size");
    // The CUDA adapter delegates alignment to NVIDIA; shader sessions use a
    // 256-byte pitch with their ordinary persistent native buffer allocation.
    memory_mul(memory_add(width, 255) & ~size_t(255), height);
    require(width <= maximum_pitch(owner->device), CUDA_ERROR_INVALID_VALUE, "allocation width exceeds maximum pitch");
    size_t native_pitch = 0;
    auto native = native_call(owner, [&] { return owner->native->allocate_pitched(width, height, element, native_pitch); });
    require(native_pitch >= width && native_pitch <= maximum_pitch(owner->device), CUDA_ERROR_INVALID_VALUE, "allocation pitch exceeds maximum pitch");
    auto address = add_allocation(owner, std::move(native));
    pitch = native_pitch;
    return address;
}
struct memory_box {
    CUdeviceptr base = 0;
    size_t first = 0, pitch = 0, slice = 0, rows = 0, depth = 0, span = 0;
    allocation* device = nullptr;
    CUarray array = nullptr;
    std::shared_ptr<buffer> storage;
    size_t allocation_offset = 0;
    std::shared_ptr<context> owner;
    std::shared_ptr<host_entry> host;
    size_t offset(size_t row) const { return first + (row / rows) * slice + (row % rows) * pitch; }
    CUdeviceptr address(size_t row) const { return base + offset(row); }
    bool device_storage() const { return device || array; }
};
memory_box memory_endpoint(CUmemorytype type, const void* host, CUdeviceptr device,
                           size_t pitch, size_t slice_height, size_t x, size_t y, size_t z,
                           size_t width, size_t height, size_t depth, const std::shared_ptr<context>& owner, bool check_pitch = true, CUarray array = nullptr) {
    if (type == CU_MEMORYTYPE_ARRAY) {
        require(array, CUDA_ERROR_INVALID_VALUE, "null array copy endpoint");
        auto entry = array_lookup(array, owner); auto& d = entry->descriptor;
        require(!(d.Flags & CUDA_ARRAY3D_DEFERRED_MAPPING), CUDA_ERROR_INVALID_VALUE, "array backing has not been mapped");
        auto element = array_element_bytes(d), row_bytes = array_storage_width(d) * element;
        auto array_height = array_storage_height(d), array_depth = std::max(size_t(1), d.Depth);
        const bool compressed = array_format_properties(d.Format).encoding == array_storage_encoding::block_compressed;
        require(x % element == 0 && (!compressed || width % element == 0) && x <= row_bytes && width <= row_bytes - x &&
                y <= array_height && height <= array_height - y && z <= array_depth && depth <= array_depth - z,
                CUDA_ERROR_INVALID_VALUE, "array copy exceeds extent or element alignment");
        memory_box result; result.owner = owner;
        result.storage = std::shared_ptr<buffer>(entry->native, &entry->native->storage()); result.array = array;
        result.allocation_offset = entry->native->storage_offset();
        result.pitch = row_bytes;
        result.slice = entry->native->array_slice_stride() ? entry->native->array_slice_stride() : row_bytes * array_height;
        result.rows = height; result.depth = depth;
        if (!width || !height || !depth) return result;
        result.first = z * result.slice + y * row_bytes + x;
        result.span = (depth - 1) * result.slice + (height - 1) * row_bytes + width;
        return result;
    }
    require(type == CU_MEMORYTYPE_HOST || type == CU_MEMORYTYPE_DEVICE || type == CU_MEMORYTYPE_UNIFIED,
            CUDA_ERROR_INVALID_VALUE, "invalid copy memory type");
    memory_box result;
    result.owner = owner; result.pitch = pitch; result.rows = height; result.depth = depth;
    if (!width || !height || !depth) return result;
    require((!check_pitch || pitch <= maximum_pitch(owner->device)) && memory_add(x, width) <= pitch,
            CUDA_ERROR_INVALID_VALUE, "copy width or X offset exceeds pitch");
    if (slice_height) require(memory_add(y, height) <= slice_height, CUDA_ERROR_INVALID_VALUE, "copy rows exceed slice height");
    require(slice_height || (depth == 1 && z == 0), CUDA_ERROR_INVALID_VALUE, "3D copies require a slice height");
    result.slice = memory_mul(pitch, slice_height);
    result.first = memory_add(memory_add(memory_mul(z, result.slice), memory_mul(y, pitch)), x);
    result.span = memory_add(memory_add(memory_mul(depth - 1, result.slice), memory_mul(height - 1, pitch)), width);
    auto end = memory_add(result.first, result.span);
    memory_mul(height, depth);
    result.base = type == CU_MEMORYTYPE_HOST ? reinterpret_cast<std::uintptr_t>(host) : device;
    require(result.base && end <= std::numeric_limits<CUdeviceptr>::max() - result.base,
            CUDA_ERROR_INVALID_VALUE, "null or overflowing copy pointer");
    bool is_device = type == CU_MEMORYTYPE_DEVICE || (type == CU_MEMORYTYPE_UNIFIED && containing(device) != global().allocations.end());
    if (is_device) {
        auto region = resolve(result.base, end, owner);
        result.device = region.value;
        result.storage = region.value->native; result.allocation_offset = region.offset;
        result.host = region.value->host.lock();
    } else {
        auto device = containing(result.base);
        require(device == global().allocations.end() || device->second.native->is_managed() || !device->second.host.expired(), CUDA_ERROR_INVALID_VALUE, "host endpoint is a device pointer");
        if (device != global().allocations.end() && device->second.native->is_managed()) {
            require(end <= device->second.native->size - size_t(result.base - device->first), CUDA_ERROR_INVALID_VALUE, "managed host endpoint exceeds allocation");
            result.storage = device->second.native;
        }
        result.host = host_range(result.base, end);
    }
    return result;
}
void memory_disjoint(const memory_box& to, const memory_box& from, size_t width) {
    if ((to.array || from.array) && to.array != from.array) return;
    bool shared_host = to.host && to.host == from.host;
    if (!shared_host && to.device != from.device && (to.device || from.device)) return;
    auto address = [&](const memory_box& box, size_t row) {
        if (shared_host && box.device) return reinterpret_cast<uintptr_t>(box.host->native->address) + box.allocation_offset + box.offset(row);
        return box.address(row);
    };
    if (address(to, 0) + to.span <= address(from, 0) || address(from, 0) + from.span <= address(to, 0)) return;
    size_t i = 0, j = 0, to_rows = to.rows * to.depth, from_rows = from.rows * from.depth;
    // Rows are sorted and disjoint, so a merge detects cross-row overlap too.
    while (i < to_rows && j < from_rows) {
        auto a = address(to, i), b = address(from, j);
        require(a + width <= b || b + width <= a, CUDA_ERROR_INVALID_VALUE, "overlapping multidimensional copy regions");
        if (a < b) ++i; else ++j;
    }
}
void execute_copy_boxes(const memory_box& to, const memory_box& from, size_t width) {
    auto domain = command_domain_read(to.owner->command_domain);
    auto source_domain = from.owner->command_domain != to.owner->command_domain ? command_domain_read(from.owner->command_domain) : std::shared_lock<command_domain>{};
    if (!width || !to.rows || !to.depth) return;
    std::unique_lock to_lock(to.owner->execution, std::defer_lock);
    std::unique_lock from_lock(from.owner->execution, std::defer_lock);
    if (to.owner == from.owner) to_lock.lock(); else std::lock(to_lock, from_lock);
    std::vector<std::byte> staging;
    bool direct = to.device_storage() && from.device_storage() && to.owner->native->can_copy_direct(*to.storage) && to.owner->native->can_copy_direct(*from.storage);
    if (to.device_storage() && from.device_storage() && to.owner != from.owner && !direct) staging.resize(std::min(width, size_t(1024 * 1024)));
    for (size_t i = 0; i < to.rows * to.depth; ++i) {
        auto destination = to.address(i), source = from.address(i);
        if (to.device_storage() && from.device_storage()) {
            if (to.owner == from.owner || direct) to.owner->native->copy(*to.storage, to.allocation_offset + to.offset(i), *from.storage, from.allocation_offset + from.offset(i), width);
            else {
                for (size_t position = 0; position < width;) {
                    auto count = std::min(width - position, staging.size());
                    from.owner->native->read(*from.storage, from.allocation_offset + from.offset(i) + position, {staging.data(), count});
                    to.owner->native->write(*to.storage, to.allocation_offset + to.offset(i) + position, {staging.data(), count});
                    position += count;
                }
            }
        } else if (to.device_storage()) to.owner->native->write(*to.storage, to.allocation_offset + to.offset(i), {reinterpret_cast<const std::byte*>(static_cast<std::uintptr_t>(source)), width});
        else if (from.device_storage()) from.owner->native->read(*from.storage, from.allocation_offset + from.offset(i), {reinterpret_cast<std::byte*>(static_cast<std::uintptr_t>(destination)), width});
        else std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(destination)), reinterpret_cast<const void*>(static_cast<std::uintptr_t>(source)), width);
    }
}
device_work enqueue_copy_boxes(const memory_box& to, const memory_box& from, size_t width, native_queue* queue,
                               const std::shared_ptr<context>& executing = {}) {
    if (!width || !to.rows || !to.depth) return {};
    auto owner = executing ? executing : to.owner;
    auto accessible = [&](const memory_box& box) {
        return !box.device_storage() || owner->native->can_copy_direct(*box.storage);
    };
    auto foreign_pool = [&](const memory_box& box) {
        return box.storage && box.storage->pool_session && box.storage->pool_session.get() != owner->native.get() && !accessible(box);
    };
    if (((to.owner != owner || from.owner != owner) && (!accessible(to) || !accessible(from))) ||
        (!to.device_storage() && !to.host) || (!from.device_storage() && !from.host) || foreign_pool(to) || foreign_pool(from)) {
        execute_copy_boxes(to, from, width); return {};
    }
    auto group = std::make_shared<device_group>(); group->tasks.reserve(to.rows * to.depth);
    std::lock_guard lock(owner->execution);
    auto& native = *owner->native;
    for (size_t i = 0; i < to.rows * to.depth; ++i) {
        device_work task;
        if (to.device_storage() && from.device_storage()) task = native.enqueue_copy(*to.storage, to.allocation_offset + to.offset(i), *from.storage, from.allocation_offset + from.offset(i), width, queue);
        else if (to.device_storage()) task = native.enqueue_write(*to.storage, to.allocation_offset + to.offset(i), {reinterpret_cast<const std::byte*>(static_cast<uintptr_t>(from.address(i))), width}, queue);
        else if (from.device_storage()) task = native.enqueue_read(*from.storage, from.allocation_offset + from.offset(i), {reinterpret_cast<std::byte*>(static_cast<uintptr_t>(to.address(i))), width}, queue);
        else std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(to.address(i))), reinterpret_cast<const void*>(static_cast<uintptr_t>(from.address(i))), width);
        group->tasks.push_back(std::move(task));
    }
    return group;
}
void copy_boxes(const memory_box& to, const memory_box& from, size_t width) {
    capture_unsafe(to.owner);
    if (!width || !to.rows || !to.depth) return;
    memory_disjoint(to, from, width);
    host_use use{to.host, from.host};
    drain_context_work(to.owner);
    if (from.owner != to.owner) drain_context_work(from.owner);
    require(!to.owner->retiring, CUDA_ERROR_INVALID_CONTEXT, "copy context is being destroyed");
    execute_copy_boxes(to, from, width);
}
void copy_2d(const CUDA_MEMCPY2D& p, const std::shared_ptr<context>& owner) {
    auto from = memory_endpoint(p.srcMemoryType, p.srcHost, p.srcDevice, p.srcPitch, 0, p.srcXInBytes, p.srcY, 0, p.WidthInBytes, p.Height, 1, owner, true, p.srcArray);
    auto to = memory_endpoint(p.dstMemoryType, p.dstHost, p.dstDevice, p.dstPitch, 0, p.dstXInBytes, p.dstY, 0, p.WidthInBytes, p.Height, 1, owner, true, p.dstArray);
    copy_boxes(to, from, p.WidthInBytes);
}
template<class Params> void copy_3d(const Params& p, const std::shared_ptr<context>& to_owner, const std::shared_ptr<context>& from_owner) {
    require(!p.srcLOD && !p.dstLOD, CUDA_ERROR_INVALID_VALUE, "copy LOD must be zero");
    if constexpr (requires { p.reserved0; }) require(!p.reserved0 && !p.reserved1, CUDA_ERROR_INVALID_VALUE, "copy reserved pointers must be null");
    auto from = memory_endpoint(p.srcMemoryType, p.srcHost, p.srcDevice, p.srcPitch, p.srcHeight, p.srcXInBytes, p.srcY, p.srcZ, p.WidthInBytes, p.Height, p.Depth, from_owner, true, p.srcArray);
    auto to = memory_endpoint(p.dstMemoryType, p.dstHost, p.dstDevice, p.dstPitch, p.dstHeight, p.dstXInBytes, p.dstY, p.dstZ, p.WidthInBytes, p.Height, p.Depth, to_owner, true, p.dstArray);
    copy_boxes(to, from, p.WidthInBytes);
}
template<class T> void execute_fill_box(const memory_box& to, T value, size_t bytes) {
    if (!bytes || !to.rows || !to.depth) return;
    auto domain = command_domain_read(to.owner->command_domain);
    std::lock_guard lock(to.owner->execution);
    for (size_t i = 0; i < to.rows * to.depth; ++i)
        to.owner->native->fill(*to.storage, to.allocation_offset + to.offset(i), {reinterpret_cast<const std::byte*>(&value), sizeof(T)}, bytes);
}
template<class T> device_work enqueue_fill_box(const memory_box& to, T value, size_t bytes, native_queue* queue) {
    if (!bytes || !to.rows || !to.depth) return {};
    if (to.owner->command_domain && to.storage->pool_session && to.storage->pool_session.get() != to.owner->native.get()) {
        execute_fill_box(to, value, bytes); return {};
    }
    auto group = std::make_shared<device_group>(); group->tasks.reserve(to.rows * to.depth);
    std::lock_guard lock(to.owner->execution);
    for (size_t i = 0; i < to.rows * to.depth; ++i)
        group->tasks.push_back(to.owner->native->enqueue_fill(*to.storage, to.allocation_offset + to.offset(i), {reinterpret_cast<const std::byte*>(&value), sizeof(T)}, bytes, queue));
    return group;
}
template<class T> void fill_box(CUdeviceptr pointer, size_t pitch, size_t slice_height, T value,
                               size_t width, size_t height, size_t depth, const std::shared_ptr<context>& owner) {
    require(pointer % alignof(T) == 0 && pitch % alignof(T) == 0, CUDA_ERROR_INVALID_VALUE, "invalid multidimensional fill alignment");
    auto bytes = memory_mul(width, sizeof(T));
    auto to = memory_endpoint(CU_MEMORYTYPE_DEVICE, nullptr, pointer, pitch, slice_height, 0, 0, 0, bytes, height, depth, owner);
    if (!bytes || !height || !depth) return;
    capture_unsafe(owner);
    host_use use{to.host};
    drain_context_work(owner);
    execute_fill_box(to, value, bytes);
}
void copy_unified(CUdeviceptr to, CUdeviceptr from, size_t bytes, const std::shared_ptr<context>& owner) {
    auto source = memory_endpoint(CU_MEMORYTYPE_UNIFIED, nullptr, from, bytes, 0, 0, 0, 0, bytes, 1, 1, owner, false);
    auto destination = memory_endpoint(CU_MEMORYTYPE_UNIFIED, nullptr, to, bytes, 0, 0, 0, 0, bytes, 1, 1, owner, false);
    copy_boxes(destination, source, bytes);
}
size_t pointer_attribute_size(CUpointer_attribute attribute) {
    require(attribute >= CU_POINTER_ATTRIBUTE_CONTEXT && attribute <= CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID,
            CUDA_ERROR_INVALID_VALUE, "invalid pointer attribute kind");
    switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT: return sizeof(CUcontext);
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
    case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR: return sizeof(CUdeviceptr);
    case CU_POINTER_ATTRIBUTE_HOST_POINTER: return sizeof(void*);
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
    case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID: return sizeof(unsigned long long);
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
    case CU_POINTER_ATTRIBUTE_MAPPING_SIZE: return sizeof(size_t);
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: return sizeof(int);
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
    case CU_POINTER_ATTRIBUTE_MAPPED:
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: return sizeof(unsigned);
    case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: return sizeof(CUmemoryPool);
    default: throw failure(CUDA_ERROR_NOT_SUPPORTED, "pointer attribute is not implemented");
    }
}
void pointer_attribute(void* output, CUpointer_attribute attribute, CUdeviceptr pointer, bool bulk) {
    auto size = pointer_attribute_size(attribute);
    auto it = containing(pointer);
    require(it == global().allocations.end() || !it->second.async_free, CUDA_ERROR_INVALID_VALUE, "allocation has an enqueued free");
    auto host = host_containing(static_cast<uintptr_t>(pointer));
    if (it != global().allocations.end()) if (auto mapped = it->second.host.lock())
        host = global().hosts.find(reinterpret_cast<uintptr_t>(mapped->native->address));
    if (it == global().allocations.end() || host != global().hosts.end()) {
        if (host != global().hosts.end() && !host->second->retiring) {
            auto& entry = *host->second;
            switch (attribute) {
            case CU_POINTER_ATTRIBUTE_CONTEXT: { auto owner = entry.owner.lock(); *static_cast<CUcontext*>(output) = owner && !owner->retiring ? owner->handle : nullptr; break; }
            case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: *static_cast<unsigned*>(output) = CU_MEMORYTYPE_HOST; break;
            case CU_POINTER_ATTRIBUTE_HOST_POINTER: *static_cast<void**>(output) = reinterpret_cast<void*>(host->first + (it != global().allocations.end() ? pointer - it->first : pointer - host->first)); break;
            case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: *static_cast<int*>(output) = entry.device; break;
            case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR: *static_cast<CUdeviceptr*>(output) = host->first; break;
            case CU_POINTER_ATTRIBUTE_RANGE_SIZE: *static_cast<size_t*>(output) = entry.native->size; break;
            case CU_POINTER_ATTRIBUTE_BUFFER_ID:
            case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID: *static_cast<unsigned long long*>(output) = entry.id; break;
            case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
                if (bulk && !host_visible(host->second, current())) *static_cast<CUdeviceptr*>(output) = 0;
                else if (entry.native->flags(entry.flags) & CU_MEMHOSTALLOC_DEVICEMAP) *static_cast<CUdeviceptr*>(output) = host_alias(host->second, current()) + (it != global().allocations.end() ? pointer - it->first : pointer - host->first);
                else { require(bulk, CUDA_ERROR_NOT_SUPPORTED, "host memory was not mapped"); *static_cast<CUdeviceptr*>(output) = 0; }
                break;
            case CU_POINTER_ATTRIBUTE_MAPPED: *static_cast<unsigned*>(output) = (entry.native->flags(entry.flags) & CU_MEMHOSTALLOC_DEVICEMAP) != 0; break;
            default: std::memset(output, 0, size); break;
            }
            return;
        }
        require(bulk, CUDA_ERROR_INVALID_VALUE, "unknown device pointer");
        std::memset(output, 0, size); return;
    }
    auto& entry = it->second;
    switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT: *static_cast<CUcontext*>(output) = entry.pool || entry.library ? nullptr : entry.owner->handle; break;
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: *static_cast<unsigned*>(output) = CU_MEMORYTYPE_DEVICE; break;
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: *static_cast<CUdeviceptr*>(output) = pointer; break;
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
        if (entry.native->is_managed()) { *static_cast<void**>(output) = reinterpret_cast<void*>(uintptr_t(pointer)); break; }
        require(bulk, CUDA_ERROR_INVALID_VALUE, "device allocation has no host pointer alias"); *static_cast<void**>(output) = nullptr; break;
    case CU_POINTER_ATTRIBUTE_IS_MANAGED: *static_cast<unsigned*>(output) = entry.native->is_managed(); break;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: *static_cast<int*>(output) = entry.owner->device; break;
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
    case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR: *static_cast<CUdeviceptr*>(output) = it->first; break;
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
    case CU_POINTER_ATTRIBUTE_MAPPING_SIZE: *static_cast<size_t*>(output) = entry.native->size; break;
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
    case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID: *static_cast<unsigned long long*>(output) = entry.id; break;
    case CU_POINTER_ATTRIBUTE_MAPPED: *static_cast<unsigned*>(output) = 1; break;
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: *static_cast<unsigned*>(output) = entry.sync_memops; break;
    case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: *static_cast<CUmemoryPool*>(output) = entry.pool ? entry.pool->handle : nullptr; break;
    default: break; // The size preflight rejects unsupported attributes.
    }
}
