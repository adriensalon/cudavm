// Registry access is protected by Driver's mutex. Workers use only captured
// native resources and immutable completion records, never the registry.
// Priorities are scheduler hints; lower values are selected first among ready
// kernel commands. Running native work is never preempted.
unsigned stream_schedule(const std::shared_ptr<context>& owner, const std::shared_ptr<stream_state>& stream) {
    switch (stream->synchronization_policy) {
    case CU_SYNC_POLICY_SPIN: return CU_CTX_SCHED_SPIN;
    case CU_SYNC_POLICY_YIELD: return CU_CTX_SCHED_YIELD;
    case CU_SYNC_POLICY_BLOCKING_SYNC: return CU_CTX_SCHED_BLOCKING_SYNC;
    default: return context_schedule(owner);
    }
}
int stream_priority(int requested, const std::shared_ptr<context>& owner) {
    return std::clamp(requested, owner->priority_range.second, owner->priority_range.first);
}
uint64_t stream_thread_id() {
    static std::atomic<uint64_t> next{1};
    static thread_local uint64_t id = next.fetch_add(1);
    return id;
}
std::shared_ptr<stream_state> new_stream(const std::shared_ptr<context>& owner, unsigned flags, bool legacy = false) {
    auto value = std::make_shared<stream_state>();
    value->owner = owner; value->flags = flags; value->legacy = legacy; value->id = global().next_stream_id++;
    owner->streams.push_back(value); return value;
}
std::shared_ptr<stream_state> stream_lookup(CUstream handle, std::shared_ptr<context> owner = {}) {
    if (!handle || handle == CU_STREAM_LEGACY || handle == CU_STREAM_PER_THREAD) {
        if (!owner) owner = current();
        if (handle == CU_STREAM_PER_THREAD) {
            auto id = stream_thread_id();
            auto& value = owner->per_thread[id];
            if (!value) value = new_stream(owner, 0);
            return value;
        }
        if (!owner->legacy) owner->legacy = new_stream(owner, 0, true);
        return owner->legacy;
    }
    auto found = global().streams.find(handle);
    require(found != global().streams.end(), CUDA_ERROR_INVALID_HANDLE, "invalid or destroyed stream");
    auto stream_owner = found->second->owner.lock();
    require(stream_owner && !stream_owner->retiring && (!owner || owner == stream_owner), CUDA_ERROR_INVALID_HANDLE, "stream belongs to a different or retiring context");
    return found->second;
}
std::vector<completion> stream_dependencies(const std::shared_ptr<context>& owner, const std::shared_ptr<stream_state>& stream) {
    std::vector<completion> result;
    if (stream->tail) result.push_back(stream->tail);
    std::erase_if(owner->streams, [](const auto& item) { return item.expired(); });
    if (stream->legacy) {
        for (auto& item : owner->streams) if (auto other = item.lock())
            if (other != stream && !(other->flags & CU_STREAM_NON_BLOCKING) && other->tail) result.push_back(other->tail);
    } else if (!(stream->flags & CU_STREAM_NON_BLOCKING) && owner->legacy && owner->legacy->tail) {
        result.push_back(owner->legacy->tail);
    }
    return result;
}
std::shared_ptr<native_queue> stream_native(const std::shared_ptr<context>& owner, const std::shared_ptr<stream_state>& stream) {
    if (!stream->native) stream->native = owner->native->create_queue(stream->priority);
    return stream->native;
}
completion submit_stream(const std::shared_ptr<context>& owner, CUstream handle, std::function<void()> action,
                         std::vector<completion> additional = {}, int priority = 0, std::function<device_work()> start = {}, bool exclusive = false,
                         std::function<void(std::exception_ptr)> notify = {}, std::shared_ptr<native_queue> lane = {}) {
    owner->native->check_device_fault();
    auto stream = stream_lookup(handle, owner);
    capture_check_stream(stream);
    if (stream->capture) {
        stream->capture->invalidated = true;
        throw failure(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "this stream operation cannot be captured");
    }
    auto dependencies = stream_dependencies(owner, stream);
    command_order order;
    for (auto& dependency : dependencies) inherit_order(order, dependency);
    for (auto& dependency : additional) inherit_order(order, dependency);
    order[stream->id] = stream->sequence + 1;
    if (owner->transport_tail) dependencies.push_back(owner->transport_tail);
    dependencies.insert(dependencies.end(), additional.begin(), additional.end());
    // Managed pointers can be reached through device globals and nested records.
    // Conservatively track every visible managed owner before submission, so
    // free, context retirement and library unload also wait for foreign queues.
    std::vector<std::shared_ptr<context>> managed_owners;
    for (auto& [address, entry] : global().allocations) if (entry.native->is_managed() && entry.owner != owner) {
        require(!entry.owner->retiring && (!entry.library || !global().libraries.at(entry.library)->retiring), CUDA_ERROR_INVALID_VALUE, "managed storage is retiring");
        if (std::find(managed_owners.begin(), managed_owners.end(), entry.owner) == managed_owners.end()) managed_owners.push_back(entry.owner);
    }
    for (auto& target : managed_owners) {
        std::erase_if(target->foreign_work, [](auto& item) { return item.expired(); });
        target->foreign_work.reserve(target->foreign_work.size() + 1);
    }
    if (!owner->queue) owner->queue = std::make_unique<command_queue>(owner->command_domain);
    // Retain a destroyed stream until its last queued command completes, so
    // the legacy stream still observes its outstanding blocking work.
    auto native_start = start ? std::function<device_work()>([stream, start = std::move(start)] { return start(); }) : std::function<device_work()>{};
    auto callback = notify ? std::function<void(std::exception_ptr)>([stream, notify = std::move(notify)](auto error) { notify(error); }) : std::function<void(std::exception_ptr)>{};
    auto done = owner->queue->submit(std::move(dependencies), [stream, action = std::move(action)] { action(); }, priority, std::move(native_start), exclusive, std::move(order), std::move(callback), std::move(lane));
    for (auto& target : managed_owners) target->foreign_work.push_back(done);
    ++stream->sequence; stream->tail = done; return done;
}
void wait_stream(const std::shared_ptr<context>& owner, CUstream handle) {
    owner->native->check_device_fault();
    auto done = submit_stream(owner, handle, [] {});
    auto schedule = stream_schedule(owner, stream_lookup(handle, owner));
    observe_context_errors(owner, [&] { driver_wait([&] { done->wait(schedule); }); });
}
void synchronize_context(const std::shared_ptr<context>& owner) {
    owner->native->check_device_fault();
    for (auto& item : owner->streams) if (auto stream = item.lock()) if (stream->capture) {
        stream->capture->invalidated = true;
        throw failure(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "cannot synchronize a context containing capturing streams");
    }
    native_call(owner, [&] { owner->native->synchronize(); });
    pool_synchronize();
}
void retire_context(const std::shared_ptr<context>& owner) {
    auto* queue = owner->queue.get();
    std::vector<completion> foreign;
    for (auto& item : owner->foreign_work) if (auto done = item.lock()) foreign.push_back(std::move(done));
    if (queue || !foreign.empty()) driver_wait([&] {
        if (queue) queue->drain(false);
        for (auto& done : foreign) { try { done->wait(); } catch (...) {} }
    });
    std::lock_guard lock(owner->execution);
    // Teardown must still release workers/resources after a deferred error.
    try { owner->native->synchronize(); } catch (...) {}
}
void stream_status(CUstream handle, bool wait) {
    auto stream = stream_lookup(handle);
    capture_check_stream(stream);
    if (stream->capture) {
        stream->capture->invalidated = true;
        throw failure(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "cannot query or synchronize a capturing stream");
    }
    auto owner = stream->owner.lock();
    // NULL/legacy queries and synchronization include previous blocking work.
    owner->native->check_device_fault();
    auto dependencies = stream->legacy ? stream_dependencies(owner, stream) : std::vector<completion>{stream->tail};
    auto schedule = stream_schedule(owner, stream);
    observe_context_errors(owner, [&] {
        if (wait) driver_wait([&] { for (auto& done : dependencies) if (done) done->wait(schedule); });
        else for (auto& done : dependencies) if (done) done->query();
    });
    if (wait) pool_synchronize();
}
CUstream create_stream(unsigned flags, int priority) {
    require(!(flags & ~CU_STREAM_NON_BLOCKING), CUDA_ERROR_INVALID_VALUE, "invalid stream flags");
    auto value = new_stream(current(), flags);
    value->priority = stream_priority(priority, value->owner.lock());
    auto handle = reinterpret_cast<CUstream>(global().next_stream); global().next_stream += 0x100;
    global().streams.emplace(handle, std::move(value)); return handle;
}
event_state& event_lookup(CUevent handle) {
    if (global().reset_events.contains(handle)) {
        auto& stack = context_stack();
        auto found = stack.empty() ? global().contexts.end() : global().contexts.find(stack.back());
        require(found != global().contexts.end() && !found->second->retiring,
            CUDA_ERROR_INVALID_CONTEXT, "event query has no active current context");
        throw failure(CUDA_ERROR_CONTEXT_IS_DESTROYED, "event belongs to a reset primary context generation");
    }
    auto found = global().events.find(handle);
    require(found != global().events.end() && !found->second.owner.expired(), CUDA_ERROR_INVALID_HANDLE, "invalid or destroyed event");
    return found->second;
}
CUevent create_event(unsigned flags) {
    require(!(flags & ~(CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS)), CUDA_ERROR_INVALID_VALUE, "invalid event flags");
    require(!(flags & CU_EVENT_INTERPROCESS) || (flags & CU_EVENT_DISABLE_TIMING), CUDA_ERROR_INVALID_VALUE, "interprocess events require disabled timing");
    require(!(flags & CU_EVENT_INTERPROCESS), CUDA_ERROR_NOT_SUPPORTED, "interprocess events are not implemented");
    auto handle = reinterpret_cast<CUevent>(global().next_event); global().next_event += 0x100;
    global().events.emplace(handle, event_state{current(), flags, {}, {}, {}}); return handle;
}
void record_event(CUevent handle, CUstream stream, unsigned flags) {
    if (capture_record(handle, stream, flags)) return;
    require(flags == 0, CUDA_ERROR_INVALID_VALUE, "external event records require stream capture");
    auto& event = event_lookup(handle); auto owner = current();
    require(event.owner.lock() == owner, CUDA_ERROR_INVALID_HANDLE, "event and stream contexts differ");
    auto timestamp = (event.flags & CU_EVENT_DISABLE_TIMING) ? std::shared_ptr<device_timestamp>{} : std::make_shared<device_timestamp>();
    auto lane = stream_native(owner,stream_lookup(stream,owner));
    auto record = submit_stream(owner, stream, [] {}, {}, 0, [owner,lane,timestamp] {
        std::lock_guard lock(owner->execution);
        return timestamp ? owner->native->enqueue_timestamp(timestamp,lane.get()) : owner->native->enqueue_marker(lane.get());
    }, false, {}, lane);
    event.record = std::move(record); event.timestamp = std::move(timestamp);
    event.capture.reset(); event.capture_dependencies.clear(); event.capture_edges.clear();
}
void wait_event(CUstream stream, CUevent handle, unsigned flags) {
    if (capture_wait(stream, handle, flags)) return;
    require(flags == 0, CUDA_ERROR_INVALID_VALUE, "external event waits require stream capture");
    auto record = event_lookup(handle).record;
    auto owner = current(); auto lane = stream_native(owner, stream_lookup(stream, owner));
    submit_stream(owner, stream, [] {}, record ? std::vector<completion>{record} : std::vector<completion>{}, 0, [owner, lane] {
        std::lock_guard lock(owner->execution);
        return owner->native->enqueue_marker(lane.get());
    }, false, {}, lane);
}
void event_status(CUevent handle, bool wait) {
    if (auto capture = event_lookup(handle).capture.lock()) {
        capture->invalidated = true;
        throw failure(CUDA_ERROR_CAPTURED_EVENT, "cannot query or synchronize a captured event");
    }
    auto& event = event_lookup(handle);
    auto owner = event.owner.lock(); owner->native->check_device_fault();
    auto done = event.record;
    auto schedule = event.flags & CU_EVENT_BLOCKING_SYNC ? unsigned(CU_CTX_SCHED_BLOCKING_SYNC) : context_schedule(event.owner.lock());
    observe_context_errors(owner, [&] { if (done) { if (wait) driver_wait([&] { done->wait(schedule); }); else done->query(); } });
    if (wait) pool_synchronize();
}
void event_elapsed(float* output, CUevent start, CUevent end) {
    require(output, CUDA_ERROR_INVALID_VALUE, "null elapsed time output");
    auto& a = event_lookup(start); auto& b = event_lookup(end);
    require(a.capture.expired() && b.capture.expired(), CUDA_ERROR_CAPTURED_EVENT, "elapsed time cannot query a captured event");
    require(a.record && b.record && !(a.flags & CU_EVENT_DISABLE_TIMING) && !(b.flags & CU_EVENT_DISABLE_TIMING),
            CUDA_ERROR_INVALID_HANDLE, "elapsed time requires recorded timing events");
    require(a.owner.lock() == b.owner.lock(), CUDA_ERROR_INVALID_HANDLE, "timing events belong to different contexts");
    a.record->query(); b.record->query();
    if (a.timestamp && b.timestamp && a.timestamp->timer && b.timestamp->timer) {
        *output = a.timestamp->timer->elapsed(*b.timestamp->timer); return;
    }
    if(a.timestamp && b.timestamp && a.timestamp->nanoseconds_per_tick && b.timestamp->nanoseconds_per_tick) {
        require(a.timestamp->nanoseconds_per_tick==b.timestamp->nanoseconds_per_tick && a.timestamp->valid_bits==b.timestamp->valid_bits,
                CUDA_ERROR_INVALID_HANDLE,"event device clocks differ");
        auto bits=a.timestamp->valid_bits;
        auto delta=b.timestamp->ticks-a.timestamp->ticks;
        if(bits<64) delta&=(uint64_t(1)<<bits)-1;
        *output=float(double(delta)*a.timestamp->nanoseconds_per_tick/1000000.0); return;
    }
    throw failure(CUDA_ERROR_NOT_SUPPORTED,"GPU event timestamps are unavailable for this event or adapter");
}
void host_function(CUstream stream, CUhostFn function, void* data) {
    require(function, CUDA_ERROR_INVALID_VALUE, "null host function");
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_HOST; node.host = {function, data};
    if (capture_node(current(), stream, std::move(node))) return;
    submit_stream(current(), stream, [function, data] {
        struct guard {
            guard() { executing_host_function = true; }
            ~guard() { executing_host_function = false; }
        } scope;
        function(data);
    });
}
void stream_callback(CUstream stream, CUstream reported, CUstreamCallback callback, void* data, unsigned flags) {
    require(callback && flags == 0, CUDA_ERROR_INVALID_VALUE, "invalid stream callback or flags");
    submit_stream(current(), stream, {}, {}, 0, {}, false, [reported, callback, data](std::exception_ptr error) {
        CUresult status = CUDA_SUCCESS;
        if (error) try { std::rethrow_exception(error); }
            catch (const failure& failure) { status = failure.code; }
            catch (const std::bad_alloc&) { status = CUDA_ERROR_OUT_OF_MEMORY; }
            catch (...) { status = CUDA_ERROR_UNKNOWN; }
        struct guard {
            guard() { executing_host_function = true; }
            ~guard() { executing_host_function = false; }
        } scope;
        callback(reported, status, data);
    });
}
