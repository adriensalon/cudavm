std::shared_ptr<memory_pool> pool_lookup(CUmemoryPool handle) {
    auto found = global().memory_pools.find(handle);
    require(found != global().memory_pools.end(), CUDA_ERROR_INVALID_VALUE, "invalid or destroyed memory pool");
    return found->second;
}
void pool_location(const CUmemLocation& location, CUmemAllocationType type) {
    require(type == CU_MEM_ALLOCATION_TYPE_PINNED && location.type == CU_MEM_LOCATION_TYPE_DEVICE,
            CUDA_ERROR_NOT_SUPPORTED, "memory pools require pinned device storage");
    valid_device(location.id);
}
CUmemoryPool pool_create(const CUmemPoolProps& props, bool builtin = false) {
    pool_location(props.location, props.allocType);
    require(props.handleTypes == CU_MEM_HANDLE_TYPE_NONE && !props.usage && !props.gpuDirectRDMACapable,
            CUDA_ERROR_NOT_SUPPORTED, "exported pools and allocation hints are unavailable");
    require(!props.win32SecurityAttributes && std::all_of(std::begin(props.reserved), std::end(props.reserved), [](auto byte) { return byte == 0; }),
            CUDA_ERROR_INVALID_VALUE, "nonzero pool security or reserved fields");
    auto pool = std::make_shared<memory_pool>(); pool->props = props; pool->builtin = builtin;
    pool->handle = reinterpret_cast<CUmemoryPool>(global().next_pool); global().next_pool += 0x100;
    global().memory_pools.emplace(pool->handle, pool); return pool->handle;
}
CUmemoryPool pool_default(int device) {
    valid_device(device);
    if (auto found = global().default_pools.find(device); found != global().default_pools.end()) return found->second;
    CUmemPoolProps props{}; props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED; props.location.type = CU_MEM_LOCATION_TYPE_DEVICE; props.location.id = device;
    auto handle = pool_create(props, true);
    try { global().default_pools.emplace(device, handle); } catch (...) { global().memory_pools.erase(handle); throw; }
    return handle;
}
CUmemoryPool pool_current(int device) {
    valid_device(device);
    auto found = global().current_pools.find(device);
    return found == global().current_pools.end() ? pool_default(device) : found->second;
}
void pool_set_current(int device, CUmemoryPool handle) {
    valid_device(device); auto pool = pool_lookup(handle);
    require(pool->props.location.id == device, CUDA_ERROR_INVALID_VALUE, "pool belongs to another device");
    global().current_pools.insert_or_assign(device, handle);
}
void pool_trim(const std::shared_ptr<memory_pool>& pool, uint64_t keep) {
    for (auto it = pool->blocks.begin(); it != pool->blocks.end();) {
        bool idle = true; size_t views = 0;
        for (auto& weak : it->slices) if (auto slice = weak.lock()) {
            ++views;
            auto cached = std::find_if(pool->cache.begin(), pool->cache.end(),
                [&](auto& block) { return block.native == slice; });
            if (cached == pool->cache.end() || slice.use_count() != 2) { idle = false; break; }
        }
        // A retained command/view, even after logical free, protects its range
        // and the entire backing block from trimming.
        if (!idle || it->native.use_count() != static_cast<long>(1 + views)) { ++it; continue; }
        auto bytes = it->native->pool_charge->bytes;
        uint64_t reserved;
        { std::lock_guard lock(pool->backing->mutex); reserved = pool->backing->reserved; }
        if (reserved <= keep) break;
        // Trimming whole blocks may leave more than the requested minimum.
        if (bytes > reserved - keep) { ++it; continue; }
        auto native = it->native.get();
        std::erase_if(pool->cache, [&](auto& block) { return &block.native->storage() == native; });
        it = pool->blocks.erase(it);
    }
}
void pool_destroy(CUmemoryPool handle) {
    auto pool = pool_lookup(handle);
    require(!pool->builtin, CUDA_ERROR_INVALID_VALUE, "default memory pools cannot be destroyed");
    pool->destroyed = true;
    pool->cache.clear();
    pool->blocks.clear();
    std::erase_if(global().current_pools, [&](auto& entry) { return entry.second == handle; });
    global().memory_pools.erase(handle);
}
void pool_release(allocation& entry, CUdeviceptr address, bool cache) {
    auto pool = entry.pool; auto bytes = entry.native->size;
    if (cache && !pool->destroyed) pool->cache.push_back({entry.owner, entry.native, address, entry.free_stream_id, entry.async_free});
    pool->used -= bytes;
}
bool pool_overlap(const buffer& a, const buffer& b) {
    return &a.storage() == &b.storage() && a.storage_offset() < b.storage_offset() + b.size &&
           b.storage_offset() < a.storage_offset() + a.size;
}
using pool_intervals = std::vector<std::pair<size_t, size_t>>;
void pool_cut(pool_intervals& ranges, size_t begin, size_t end) {
    pool_intervals remaining;
    for (auto [first, last] : ranges) {
        if (end <= first || begin >= last) remaining.emplace_back(first, last);
        else {
            if (first < begin) remaining.emplace_back(first, begin);
            if (end < last) remaining.emplace_back(end, last);
        }
    }
    ranges = std::move(remaining);
}
pool_intervals pool_unclaimed(const allocation& entry) {
    auto& native = *entry.native;
    pool_intervals ranges{{native.storage_offset(), native.storage_offset() + native.size}};
    auto cut = [&](const allocation& newer) {
        if (newer.id > entry.id && pool_overlap(native, *newer.native))
            pool_cut(ranges, newer.native->storage_offset(), newer.native->storage_offset() + newer.native->size);
    };
    for (auto& [address, newer] : global().allocations) cut(newer);
    for (auto& node : global().pool_retiring) cut(node.mapped());
    return ranges;
}
// Keep unused pieces attached to the stream/free that made them available.
// Otherwise splitting a view would turn its remainder into unrestricted gaps.
std::vector<memory_pool::cached_block> pool_cache_pieces(const std::shared_ptr<context>& owner,
    const std::shared_ptr<buffer>& native, CUdeviceptr address, unsigned long long stream_id,
    const completion& free, const pool_intervals& ranges) {
    std::vector<memory_pool::cached_block> pieces;
    for (auto [begin, end] : ranges) {
        if (begin > SIZE_MAX - 255) continue;
        begin = (begin + 255) & ~size_t(255);
        if (begin >= end) continue;
        auto& view = static_cast<pool_buffer&>(*native);
        auto offset = begin - native->storage_offset();
        auto piece = std::make_shared<pool_buffer>(view.backing, view.offset + offset, end - begin);
        pieces.push_back({owner, std::move(piece), address + offset, stream_id, free});
    }
    return pieces;
}
void pool_publish_cache(const std::shared_ptr<memory_pool>& pool, std::vector<memory_pool::cached_block> pieces) {
    auto& block = *std::find_if(pool->blocks.begin(), pool->blocks.end(), [&](auto& backing) {
        return !pieces.empty() && backing.native.get() == &pieces.front().native->storage();
    });
    pool->cache.reserve(pool->cache.size() + pieces.size());
    block.slices.reserve(block.slices.size() + pieces.size());
    for (auto& piece : pieces) { block.slices.push_back(piece.native); pool->cache.push_back(std::move(piece)); }
}
void collect_pool_memory() {
    // Reused ranges can have several queued generations. Discard allocations
    // whose activation failed before restoring the original failed free.
    auto failed_allocation = [](allocation& entry) {
        if (!entry.async_alloc || !entry.async_alloc->ready.load(std::memory_order_acquire)) return false;
        if (!entry.async_alloc->error) { entry.async_alloc.reset(); return false; }
        entry.pool->used -= entry.native->size; return true;
    };
    std::erase_if(global().allocations, [&](auto& item) { return failed_allocation(item.second); });
    std::erase_if(global().pool_retiring, [&](auto& node) { return failed_allocation(node.mapped()); });
    for (auto it = global().pool_retiring.begin(); it != global().pool_retiring.end();) {
        auto& entry = it->mapped();
        if (!entry.async_free->ready.load(std::memory_order_acquire)) { ++it; continue; }
        if (entry.async_free->error) {
            // A descendant may still be propagating the same error. Its native
            // range must disappear before the original allocation is restored.
            bool occupied = std::any_of(global().allocations.begin(), global().allocations.end(),
                [&](auto& item) { return pool_overlap(*entry.native, *item.second.native); });
            if (occupied) { ++it; continue; }
            entry.async_free.reset();
            global().allocations.insert(std::move(*it));
        } else {
            if (!entry.pool->destroyed) {
                auto pieces = pool_cache_pieces(entry.owner, entry.native, it->key(), entry.free_stream_id,
                                                entry.async_free, pool_unclaimed(entry));
                if (!pieces.empty()) pool_publish_cache(entry.pool, std::move(pieces));
            }
            pool_release(entry, it->key(), false);
        }
        it = global().pool_retiring.erase(it);
    }
    for (auto it = global().allocations.begin(); it != global().allocations.end();) {
        auto& entry = it->second;
        if (!entry.async_free || !entry.async_free->ready.load(std::memory_order_acquire)) { ++it; continue; }
        if (entry.async_free->error) { entry.async_free.reset(); ++it; continue; }
        if (entry.pool) pool_release(entry, it->first, true);
        it = global().allocations.erase(it);
    }
}
void pool_retire_context(const std::shared_ptr<context>& owner) {
    // Async allocations and cached storage belong to a private device session,
    // rather than the public context being retired.
    (void)owner;
    collect_pool_memory();
}
std::shared_ptr<context> pool_storage_context(int device) {
    auto& storage = global().pool_storage[device];
    if (!storage) {
        auto value = std::make_shared<context>(); value->device = device;
        value->native = global().provider->open(device);
        storage = std::move(value);
    }
    return storage;
}
void pool_synchronize() {
    collect_pool_memory();
    for (auto& [handle, pool] : global().memory_pools) pool_trim(pool, pool->threshold);
}
void pool_attribute(const std::shared_ptr<memory_pool>& pool, CUmemPool_attribute attr, void* value, bool set) {
    require(value, CUDA_ERROR_INVALID_VALUE, "null memory pool attribute value");
    int* policy = nullptr;
    switch (attr) {
    case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: policy = &pool->reuse_events; break;
    case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: policy = &pool->reuse_opportunistic; break;
    case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: policy = &pool->reuse_internal; break;
    default: break;
    }
    if (policy) {
        if (set) { int v; std::memcpy(&v, value, sizeof(v)); *policy = v != 0; }
        else std::memcpy(value, policy, sizeof(*policy));
        return;
    }
    if (set) {
        require(attr == CU_MEMPOOL_ATTR_RELEASE_THRESHOLD || attr == CU_MEMPOOL_ATTR_USED_MEM_HIGH || attr == CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                CUDA_ERROR_INVALID_VALUE, "memory pool attribute is read-only or unknown");
        uint64_t v; std::memcpy(&v, value, sizeof(v));
        if (attr == CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) pool->threshold = v;
        else { require(v == 0, CUDA_ERROR_INVALID_VALUE, "high watermark reset requires zero");
            if (attr == CU_MEMPOOL_ATTR_USED_MEM_HIGH) pool->used_high = pool->used;
            else { std::lock_guard lock(pool->backing->mutex); pool->backing->high = pool->backing->reserved; } }
        return;
    }
    uint64_t v = 0; int integer = 0; bool small = false;
    switch (attr) {
    case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: v = pool->threshold; break;
    case CU_MEMPOOL_ATTR_USED_MEM_CURRENT: v = pool->used; break;
    case CU_MEMPOOL_ATTR_USED_MEM_HIGH: v = pool->used_high; break;
    case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: { std::lock_guard lock(pool->backing->mutex); v = pool->backing->reserved; break; }
    case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: { std::lock_guard lock(pool->backing->mutex); v = pool->backing->high; break; }
    case CU_MEMPOOL_ATTR_MAX_POOL_SIZE: v = pool->props.maxSize; break;
    case CU_MEMPOOL_ATTR_ALLOCATION_TYPE: small = true; integer = pool->props.allocType; break;
    case CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES: small = true; integer = pool->props.handleTypes; break;
    case CU_MEMPOOL_ATTR_LOCATION_ID: small = true; integer = pool->props.location.id; break;
    case CU_MEMPOOL_ATTR_LOCATION_TYPE: small = true; integer = pool->props.location.type; break;
    case CU_MEMPOOL_ATTR_LOCALITY_DOMAIN_ID: small = true; integer = -1; break;
    case CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED: small = true; break;
    default: throw failure(CUDA_ERROR_INVALID_VALUE, "unknown memory pool attribute");
    }
    if (small) std::memcpy(value, &integer, sizeof(integer)); else std::memcpy(value, &v, sizeof(v));
}
void pool_set_access(const std::shared_ptr<memory_pool>& pool, const CUmemAccessDesc* descriptions, size_t count) {
    require(!count || descriptions, CUDA_ERROR_INVALID_VALUE, "null pool access array");
    for (size_t i = 0; i < count; ++i) {
        auto& p = descriptions[i]; pool_location(p.location, CU_MEM_ALLOCATION_TYPE_PINNED);
        require(p.flags == CU_MEM_ACCESS_FLAGS_PROT_NONE || p.flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                CUDA_ERROR_INVALID_VALUE, "invalid memory pool access flags");
        require(p.location.id == pool->props.location.id && p.flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                CUDA_ERROR_NOT_SUPPORTED, "peer pool access and owner access removal are unavailable");
    }
}
bool pool_reusable(const std::shared_ptr<memory_pool>& pool, unsigned long long free_stream,
                   const completion& free, const std::shared_ptr<stream_state>& stream,
                   const command_order& order, bool internal) {
    if (free && free->ready.load(std::memory_order_acquire) && free->error) return false;
    if (free_stream == stream->id) return true;
    if (pool->reuse_events && free) {
        auto before = free->order.find(free_stream);
        auto after = order.find(free_stream);
        if (before != free->order.end() && after != order.end() && after->second >= before->second) return true;
    }
    if (pool->reuse_opportunistic && free && free->ready.load(std::memory_order_acquire)) return true;
    return internal && pool->reuse_internal;
}
struct pool_range_plan {
    memory_pool::backing_block* block = nullptr;
    size_t offset = 0;
    std::vector<CUdeviceptr> retiring;
    std::vector<completion> waits;
    std::shared_ptr<buffer> exact;
    CUdeviceptr address = 0;
};
pool_range_plan pool_find_range(const std::shared_ptr<memory_pool>& pool, size_t bytes,
                               const std::shared_ptr<stream_state>& stream, const command_order& order, bool internal) {
    for (auto& block : pool->blocks) {
        std::erase_if(block.slices, [](auto& slice) { return slice.expired(); });
        std::vector<std::pair<size_t, size_t>> occupied;
        auto belongs = [&](const buffer& native) { return &native.storage() == block.native.get(); };
        for (auto& [address, entry] : global().allocations) if (entry.pool == pool && belongs(*entry.native))
            if (!entry.async_free || !pool_reusable(pool, entry.free_stream_id, entry.async_free, stream, order, internal))
                occupied.emplace_back(entry.native->storage_offset(), entry.native->size);
        for (auto& node : global().pool_retiring) {
            auto& entry = node.mapped();
            if (entry.pool != pool || !belongs(*entry.native) || pool_reusable(pool, entry.free_stream_id, entry.async_free, stream, order, internal)) continue;
            // A newer generation governs the portions it claimed, even when
            // the older free's originating policy is no longer enabled.
            for (auto [begin, end] : pool_unclaimed(entry)) occupied.emplace_back(begin, end - begin);
        }
        for (auto& cached : pool->cache) if (belongs(*cached.native))
            if (cached.native.use_count() != 1 || !pool_reusable(pool, cached.stream_id, cached.free, stream, order, internal))
                occupied.emplace_back(cached.native->storage_offset(), cached.native->size);
        for (auto& weak : block.slices) if (auto slice = weak.lock()) {
            bool known = std::any_of(global().allocations.begin(), global().allocations.end(), [&](auto& item) { return item.second.native == slice; }) ||
                std::any_of(global().pool_retiring.begin(), global().pool_retiring.end(), [&](auto& node) { return node.mapped().native == slice; }) ||
                std::any_of(pool->cache.begin(), pool->cache.end(), [&](auto& cached) { return cached.native == slice; });
            if (!known) occupied.emplace_back(slice->storage_offset(), slice->size);
        }
        std::sort(occupied.begin(), occupied.end());
        size_t start = 0;
        for (auto [offset, size] : occupied) {
            if (offset >= start && bytes <= offset - start) break;
            auto end = offset + size;
            if (end > SIZE_MAX - 255) { start = SIZE_MAX; break; }
            start = std::max(start, (end + 255) & ~size_t(255));
        }
        if (start > block.native->size || bytes > block.native->size - start) continue;
        pool_range_plan plan; plan.block = &block; plan.offset = start;
        auto overlaps = [&](const buffer& native) {
            return &native.storage() == block.native.get() && native.storage_offset() < start + bytes &&
                   start < native.storage_offset() + native.size;
        };
        for (auto& [address, entry] : global().allocations) if (entry.pool == pool && overlaps(*entry.native)) {
            plan.retiring.push_back(address); plan.waits.push_back(entry.async_free);
            if (entry.native->storage_offset() == start && entry.native->size == bytes) { plan.exact = entry.native; plan.address = address; }
        }
        for (auto& node : global().pool_retiring) if (node.mapped().pool == pool && overlaps(*node.mapped().native))
            plan.waits.push_back(node.mapped().async_free);
        for (auto& cached : pool->cache) if (overlaps(*cached.native)) {
            // Eligible cached views were included in the gap; their sole strong
            // owner is the cache until the plan captures an exact view below.
            if (cached.native->storage_offset() == start && cached.native->size == bytes) { plan.exact = cached.native; plan.address = cached.address; }
        }
        return plan;
    }
    return {};
}
CUdeviceptr pool_allocate(size_t bytes, const std::shared_ptr<memory_pool>& pool, CUstream handle) {
    auto stream = stream_lookup(handle); auto owner = stream->owner.lock(); capture_check_stream(stream);
    require(pool->props.location.id == owner->device, CUDA_ERROR_NOT_SUPPORTED, "foreign-device pool allocation is unavailable");
    if (!bytes) return 0;
    if (stream->capture) {
        auto& frontier = stream->capture_dependencies; auto& edges = stream->capture_edges;
        frontier.reserve(1); edges.reserve(1);
        CUDA_MEM_ALLOC_NODE_PARAMS p{}; p.poolProps = pool->props; p.poolProps.maxSize = 0; p.bytesize = bytes;
        auto dependencies = capture_node_frontier(stream, CU_GRAPH_NODE_TYPE_MEM_ALLOC);
        auto node = graph_memory_add_alloc(stream->capture->graph, dependencies.nodes.data(), dependencies.nodes.size(), p, true, owner, dependencies.edges.data());
        frontier.assign(1, node); edges.assign(1, CUgraphEdgeData{}); return p.dptr;
    }
    auto storage = pool_storage_context(pool->props.location.id);
    command_order order;
    for (auto& dependency : stream_dependencies(owner, stream)) inherit_order(order, dependency);
    auto cached = std::find_if(pool->cache.begin(), pool->cache.end(), [&](auto& block) {
        return block.owner == storage && block.native->size == bytes && block.native.use_count() == 1 &&
               pool_reusable(pool, block.stream_id, block.free, stream, order, false);
    });
    CUdeviceptr address; bool reused = cached != pool->cache.end();
    memory_pool::backing_block fresh;
    memory_pool::backing_block* selected = nullptr;
    pool_range_plan plan;
    std::list<std::map<CUdeviceptr, allocation>::node_type> retiring;
    std::vector<memory_pool::cached_block> remaining_cache;
    if (reused) {
        address = cached->address;
        allocation entry{}; entry.owner = storage; entry.native = cached->native; entry.id = global().next_buffer_id++;
        global().allocations.emplace(address, std::move(entry));
    } else {
        plan = pool_find_range(pool, bytes, stream, order, false);
        selected = plan.block;
        size_t offset = plan.offset;
        if (!selected) {
            try {
                uint64_t reserved;
                { std::lock_guard lock(pool->backing->mutex); reserved = pool->backing->reserved; }
                require(!pool->props.maxSize || (bytes <= pool->props.maxSize && reserved <= pool->props.maxSize - bytes),
                        CUDA_ERROR_OUT_OF_MEMORY, "memory pool maximum size exceeded");
                // Small allocations share a 64 KiB native buffer. A finite maximum
                // caps growth; native physical granularity is still checked below.
                size_t capacity = std::max(bytes, size_t(64 * 1024));
                if (pool->props.maxSize) capacity = static_cast<size_t>(std::min<uint64_t>(capacity, pool->props.maxSize - reserved));
                std::lock_guard lock(storage->native->pool_execution);
                std::unique_ptr<buffer> native;
                try { native = storage->native->allocate_pool(capacity); }
                catch (const failure& error) {
                    if (capacity == bytes || error.code != CUDA_ERROR_OUT_OF_MEMORY) throw;
                    native = storage->native->allocate_pool(bytes);
                }
                // Preferred growth can round past a finite limit even when a
                // smaller native allocation fits. Retry before charging/publishing.
                if (capacity != bytes && pool->props.maxSize && native->physical_size() > pool->props.maxSize - reserved) {
                    native.reset(); native = storage->native->allocate_pool(bytes);
                }
                native->pool_session = storage->native;
                auto physical = native->physical_size();
                require(!pool->props.maxSize || (physical <= pool->props.maxSize && reserved <= pool->props.maxSize - physical),
                        CUDA_ERROR_OUT_OF_MEMORY, "memory pool backing exceeds maximum size");
                native->pool_charge = std::make_shared<pool_backing_charge>(pool->backing, physical);
                fresh.native = std::move(native);
                global().device_buffers.push_back({storage->device, fresh.native});
                pool->blocks.reserve(pool->blocks.size() + 1);
                selected = &fresh; offset = 0;
            } catch (const failure& error) {
                if (error.code != CUDA_ERROR_OUT_OF_MEMORY || !pool->reuse_internal) throw;
                plan = pool_find_range(pool, bytes, stream, order, true);
                if (!plan.block) throw;
                selected = plan.block; offset = plan.offset;
            }
        }
        selected->slices.reserve(selected->slices.size() + 1);
        for (auto& block : pool->cache) if (&block.native->storage() == selected->native.get()) {
            auto begin = block.native->storage_offset(), end = begin + block.native->size;
            if (begin >= offset + bytes || end <= offset) continue;
            pool_intervals ranges{{begin, end}}; pool_cut(ranges, offset, offset + bytes);
            auto pieces = pool_cache_pieces(block.owner, block.native, block.address, block.stream_id, block.free, ranges);
            for (auto& piece : pieces) remaining_cache.push_back(std::move(piece));
        }
        pool->cache.reserve(pool->cache.size() + remaining_cache.size());
        selected->slices.reserve(selected->slices.size() + 1 + remaining_cache.size());
        // Reserve restoration slots before extracting any registry entries.
        for (auto pointer : plan.retiring) { (void)pointer; retiring.emplace_back(); }
        auto slot = retiring.begin();
        for (auto pointer : plan.retiring) *slot++ = global().allocations.extract(pointer);
        try {
            if (plan.exact) {
                address = plan.address;
                allocation entry{}; entry.owner = storage; entry.native = plan.exact; entry.id = global().next_buffer_id++;
                global().allocations.emplace(address, std::move(entry));
            } else address = add_allocation(storage, std::make_unique<pool_buffer>(selected->native, offset, bytes));
        } catch (...) { for (auto& node : retiring) global().allocations.insert(std::move(node)); throw; }
    }
    completion activation;
    try { activation = submit_stream(owner, handle, [] {}, plan.waits); }
    catch (...) {
        global().allocations.erase(address);
        for (auto& node : retiring) global().allocations.insert(std::move(node));
        throw;
    }
    global().allocations.at(address).pool = pool;
    if (!plan.waits.empty()) global().allocations.at(address).async_alloc = activation;
    global().pool_retiring.splice(global().pool_retiring.end(), retiring);
    if (reused) pool->cache.erase(cached);
    else {
        std::erase_if(pool->cache, [&](auto& block) {
            return &block.native->storage() == selected->native.get() && block.native->storage_offset() < plan.offset + bytes &&
                   plan.offset < block.native->storage_offset() + block.native->size;
        });
        for (auto& piece : remaining_cache) { selected->slices.push_back(piece.native); pool->cache.push_back(std::move(piece)); }
        if (!plan.exact) selected->slices.push_back(global().allocations.at(address).native);
        if (selected == &fresh) pool->blocks.push_back(std::move(fresh));
    }
    pool->used += bytes; pool->used_high = std::max(pool->used_high, pool->used);
    return address;
}
void pool_free(CUdeviceptr pointer, CUstream handle) {
    auto stream = stream_lookup(handle); auto owner = stream->owner.lock(); capture_check_stream(stream);
    if (!pointer) return;
    if (stream->capture) {
        auto& frontier = stream->capture_dependencies; auto& edges = stream->capture_edges;
        frontier.reserve(1); edges.reserve(1);
        auto dependencies = capture_node_frontier(stream, CU_GRAPH_NODE_TYPE_MEM_FREE);
        auto node = graph_memory_add_free(stream->capture->graph, dependencies.nodes.data(), dependencies.nodes.size(), pointer, dependencies.edges.data());
        frontier.assign(1, node); edges.assign(1, CUgraphEdgeData{}); return;
    }
    auto it = global().allocations.find(pointer);
    require(it != global().allocations.end(), CUDA_ERROR_INVALID_VALUE, "async free requires an allocation base");
    auto& entry = it->second;
    require(allocation_visible(entry, owner), CUDA_ERROR_INVALID_CONTEXT, "allocation belongs to another context or device");
    if (entry.graph_owned) {
        auto memory = entry.graph_memory.lock();
        require(memory && memory->planned_live && !memory->internal_free(), CUDA_ERROR_INVALID_VALUE, "graph allocation cannot be freed outside its owning graph");
        auto native = entry.native;
        auto done = submit_stream(owner, handle, [memory, native] { memory->live = false; });
        memory->planned_live = false; memory->transition = done; return;
    }
    require(!entry.async_free && !entry.graph_owned && !entry.graphics && !entry.external && !entry.module && !entry.library && entry.host.expired(),
            CUDA_ERROR_INVALID_VALUE, "allocation cannot be freed asynchronously");
    auto native = entry.native;
    entry.async_free = submit_stream(owner, handle, [native] {}); entry.free_stream_id = stream->id;
}
