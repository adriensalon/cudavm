auto host_containing(uintptr_t pointer) {
    auto& entries = global().hosts;
    auto entry = entries.upper_bound(pointer);
    if (entry == entries.begin()) return entries.end();
    --entry;
    return pointer - entry->first < entry->second->native->size ? entry : entries.end();
}
std::shared_ptr<host_entry> host_range(uintptr_t pointer, size_t bytes) {
    if (!bytes) return {};
    require(pointer && bytes <= std::numeric_limits<uintptr_t>::max() - pointer,
            CUDA_ERROR_INVALID_VALUE, "null or overflowing host range");
    auto entry = host_containing(pointer);
    if (entry != global().hosts.end()) {
        require(!entry->second->retiring && bytes <= entry->second->native->size - (pointer - entry->first),
                CUDA_ERROR_INVALID_VALUE, "host range exceeds allocation or is being released");
        return entry->second;
    }
    auto next = global().hosts.lower_bound(pointer);
    require(next == global().hosts.end() || next->first >= pointer + bytes,
            CUDA_ERROR_INVALID_VALUE, "host range crosses a registered allocation boundary");
    return {};
}
void host_track(const std::shared_ptr<host_entry>& entry, const completion& done) {
    if (!entry) return;
    std::erase_if(entry->work, [](auto& item) { return item.expired(); });
    entry->work.push_back(done);
}
// Synchronous APIs may release the registry mutex while waiting for previous
// device work. Publish their own use so concurrent host free/unregister waits.
struct host_use {
    std::vector<std::shared_ptr<host_entry>> pins;
    completion done;
    host_use(std::initializer_list<std::shared_ptr<host_entry>> entries) {
        for (auto& entry : entries) if (entry && std::find(pins.begin(), pins.end(), entry) == pins.end()) pins.push_back(entry);
        if (pins.empty()) return;
        done = std::make_shared<command_completion>();
        for (auto& entry : pins) host_track(entry, done);
    }
    ~host_use() {
        if (!done) return;
        pins.clear();
        { std::lock_guard lock(done->mutex); done->time = std::chrono::steady_clock::now(); done->ready.store(true, std::memory_order_release); }
        done->changed.notify_all();
    }
};
void host_flags(unsigned flags, bool registration) {
    const unsigned allowed = registration ? 15 : 7;
    require(!(flags & ~allowed), CUDA_ERROR_INVALID_VALUE, "invalid host allocation/registration flags");
    if (registration) require(!(flags & (CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_READ_ONLY)),
                              CUDA_ERROR_NOT_SUPPORTED, "I/O and read-only host registrations are not implemented");
    else require(!(flags & CU_MEMHOSTALLOC_WRITECOMBINED) || global().kind == cudavmBackendCuda,
                 CUDA_ERROR_NOT_SUPPORTED, "write-combined host allocation requires native CUDA");
}
bool host_visible(const std::shared_ptr<host_entry>& entry, const std::shared_ptr<context>& owner) {
    return entry->owner.lock() == owner || (entry->flags & CU_MEMHOSTALLOC_PORTABLE) ||
        (global().kind == cudavmBackendCuda && !entry->registered && global().provider->attribute(owner->device, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING));
}
CUdeviceptr host_alias(const std::shared_ptr<host_entry>& entry, const std::shared_ptr<context>& owner) {
    require(!entry->retiring && host_visible(entry, owner),
            CUDA_ERROR_INVALID_VALUE, "host mapping is unavailable in this context");
    require(entry->native->flags(entry->flags) & CU_MEMHOSTALLOC_DEVICEMAP, CUDA_ERROR_NOT_SUPPORTED, "host memory was not mapped");
    if (entry->alias && global().allocations.contains(entry->alias)) return entry->alias;
    auto native = owner->native->map_host(entry->native);
    auto address = add_allocation(owner, std::move(native));
    global().allocations.at(address).host = entry;
    entry->alias = address; return address;
}
CUdeviceptr host_device_pointer(void* pointer, unsigned flags, const std::shared_ptr<context>& owner) {
    require(!flags, CUDA_ERROR_INVALID_VALUE, "host device pointer flags must be zero");
    auto address = reinterpret_cast<uintptr_t>(pointer);
    auto found = host_containing(address);
    require(found != global().hosts.end(), CUDA_ERROR_INVALID_VALUE, "unknown host allocation");
    return host_alias(found->second, owner) + address - found->first;
}
void* allocate_host(size_t bytes, unsigned flags, const std::shared_ptr<context>& owner) {
    capture_unsafe(owner);
    host_flags(flags, false);
    require(bytes && bytes <= std::numeric_limits<uintptr_t>::max(), CUDA_ERROR_INVALID_VALUE, "zero-sized host allocation");
    auto entry = std::make_shared<host_entry>();
    entry->native = owner->native->allocate_host(bytes, flags);
    entry->owner = owner; entry->device = owner->device; entry->flags = flags; entry->id = global().next_buffer_id++;
    auto pointer = reinterpret_cast<uintptr_t>(entry->native->address);
    require(global().hosts.emplace(pointer, entry).second, CUDA_ERROR_UNKNOWN, "duplicate host allocation address");
    return entry->native->address;
}
void register_host(void* pointer, size_t bytes, unsigned flags, const std::shared_ptr<context>& owner) {
    capture_unsafe(owner);
    host_flags(flags, true);
    auto address = reinterpret_cast<uintptr_t>(pointer);
    require(address && bytes && bytes <= std::numeric_limits<uintptr_t>::max() - address,
            CUDA_ERROR_INVALID_VALUE, "invalid host registration range");
    auto next = global().hosts.lower_bound(address);
    require(host_containing(address) == global().hosts.end() && (next == global().hosts.end() || next->first >= address + bytes),
            CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, "host registration overlaps allocated or registered storage");
    auto device = global().allocations.lower_bound(address);
    require(containing(address) == global().allocations.end() && (device == global().allocations.end() || device->first >= address + bytes),
            CUDA_ERROR_INVALID_VALUE, "device storage cannot be registered as host memory");
    auto entry = std::make_shared<host_entry>();
    entry->native = owner->native->register_host(pointer, bytes, flags);
    entry->owner = owner; entry->device = owner->device; entry->flags = flags;
    entry->registered = true; entry->id = global().next_buffer_id++;
    global().hosts.emplace(address, entry);
}
void release_host(void* pointer, bool registration) {
    capture_unsafe({});
    if (!pointer && !registration) return;
    auto address = reinterpret_cast<uintptr_t>(pointer);
    auto found = global().hosts.find(address);
    auto invalid = registration ? CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED : CUDA_ERROR_INVALID_VALUE;
    require(found != global().hosts.end() && !found->second->retiring && found->second->registered == registration,
            invalid, "host release requires the matching allocation/registration base");
    auto entry = found->second;
    std::vector<completion> work;
    for (auto& item : entry->work) if (auto done = item.lock()) work.push_back(std::move(done));
    entry->retiring = true;
    std::exception_ptr error;
    if (!work.empty()) driver_wait([&] { for (auto& done : work) { try { done->wait(); } catch (...) { if (!error) error = std::current_exception(); } } });
    found = global().hosts.find(address);
    require(found != global().hosts.end() && found->second == entry, invalid, "host registration changed while waiting for release");
    if (entry->alias) global().allocations.erase(entry->alias);
    global().hosts.erase(found);
    entry.reset();
    if (error) std::rethrow_exception(error);
}
unsigned get_host_flags(void* pointer) {
    auto entry = host_containing(reinterpret_cast<uintptr_t>(pointer));
    require(entry != global().hosts.end() && !entry->second->retiring, CUDA_ERROR_INVALID_VALUE, "unknown or retiring host memory");
    return entry->second->native->flags(entry->second->flags);
}
void retire_primary_hosts(const std::shared_ptr<context>& owner) {
    std::vector<std::shared_ptr<host_entry>> entries;
    std::vector<completion> work;
    for (auto& [address, entry] : global().hosts) if (entry->owner.lock() == owner) {
        entries.push_back(entry);
        for (auto& item : entry->work) if (auto done = item.lock()) work.push_back(std::move(done));
    }
    for (auto& entry : entries) entry->retiring = true;
    if (!work.empty()) driver_wait([&] { for (auto& done : work) { try { done->wait(); } catch (...) {} } });
    for (auto& entry : entries) {
        if (entry->alias) global().allocations.erase(entry->alias);
        global().hosts.erase(reinterpret_cast<uintptr_t>(entry->native->address));
    }
}
