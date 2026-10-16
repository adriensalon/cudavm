void user_count(unsigned count) {
    require(count && count <= static_cast<unsigned>(std::numeric_limits<int>::max()), CUDA_ERROR_INVALID_VALUE, "invalid user object reference count");
}
user_object_entry& user_lookup(CUuserObject handle) {
    auto it = global().user_objects.find(handle);
    require(it != global().user_objects.end() && !it->second.object.expired(), CUDA_ERROR_INVALID_VALUE, "invalid or released user object");
    return it->second;
}
CUuserObject user_create(void* data, CUhostFn destroy, unsigned count, unsigned flags) {
    user_count(count);
    require(destroy && flags == CU_USER_OBJECT_NO_DESTRUCTOR_SYNC, CUDA_ERROR_INVALID_VALUE, "invalid user object destructor or flags");
    auto& state = global();
    if (!state.user_destructors) state.user_destructors = std::make_shared<user_destructor_queue>();
    std::erase_if(state.user_objects, [](auto& item) { return item.second.object.expired(); });
    auto object = std::make_shared<user_object>(); object->queue = state.user_destructors;
    object->task = std::make_unique<user_destructor>(); object->task->function = destroy; object->task->data = data;
    auto handle = graph_handle<CUuserObject>(state.next_user_object);
    state.user_objects.emplace(handle, user_object_entry{object, {object, count}});
    object->armed = true; return handle;
}
void user_retain(CUuserObject handle, unsigned count) {
    user_count(count); auto& entry = user_lookup(handle);
    auto object = entry.object.lock();
    require(object != nullptr, CUDA_ERROR_INVALID_VALUE, "user object completed its final queued use");
    require(count <= std::numeric_limits<uint64_t>::max() - entry.caller.count, CUDA_ERROR_INVALID_VALUE, "user reference count overflow");
    entry.caller.object = std::move(object); entry.caller.count += count;
}
void user_release(CUuserObject handle, unsigned count) {
    user_count(count); auto& entry = user_lookup(handle);
    require(count <= entry.caller.count, CUDA_ERROR_INVALID_VALUE, "caller does not own these references");
    entry.caller.count -= count;
    if (!entry.caller.count) entry.caller.object.reset();
}
void graph_user_retain(CUgraph graph, CUuserObject handle, unsigned count, unsigned flags) {
    user_count(count); require(!(flags & ~CU_GRAPH_USER_OBJECT_MOVE), CUDA_ERROR_INVALID_VALUE, "invalid graph user object flags");
    auto& objects = graph_lookup(graph).objects; auto& entry = user_lookup(handle);
    auto object = entry.object.lock();
    require(object != nullptr, CUDA_ERROR_INVALID_VALUE, "user object completed its final queued use");
    require(!flags || count <= entry.caller.count, CUDA_ERROR_INVALID_VALUE, "caller does not own the moved references");
    auto it = objects.find(handle);
    require(it == objects.end() || count <= std::numeric_limits<uint64_t>::max() - it->second.count, CUDA_ERROR_INVALID_VALUE, "graph reference count overflow");
    auto& reference = objects.try_emplace(handle, user_reference{std::move(object), 0}).first->second;
    reference.count += count;
    if (flags) { entry.caller.count -= count; if (!entry.caller.count) entry.caller.object.reset(); }
}
void graph_user_release(CUgraph graph, CUuserObject handle, unsigned count) {
    user_count(count); user_lookup(handle); auto& objects = graph_lookup(graph).objects;
    auto it = objects.find(handle);
    require(it != objects.end() && count <= it->second.count, CUDA_ERROR_INVALID_VALUE, "graph does not own these references");
    it->second.count -= count; if (!it->second.count) objects.erase(it);
}
