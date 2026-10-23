struct captured_launch {
    std::shared_ptr<context> owner;
    std::shared_ptr<kernel> function;
    launch_shape shape;
    std::vector<std::byte> arguments;
    std::vector<kernel_parameter> parameters;
    std::vector<std::shared_ptr<buffer>> resources;
    std::vector<allocation_binding> bindings;
    std::vector<std::shared_ptr<host_entry>> hosts;
    std::vector<std::shared_ptr<image_object_entry>> images;
    void execute() {
        std::lock_guard lock(owner->execution);
        size_t bytes = arguments.size();
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, arguments.data(), CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
        function->launch_bound(shape, nullptr, bytes ? extra : nullptr, bindings);
    }
    device_work enqueue(native_queue* queue) {
        std::lock_guard lock(owner->execution);
        size_t bytes = arguments.size();
        void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, arguments.data(), CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
        return function->enqueue_launch(shape, nullptr, bytes ? extra : nullptr, bindings, queue);
    }
};
std::shared_ptr<captured_launch> prepare_launch(CUfunction handle, const launch_shape& shape, void** parameters, void** extra,
                                               const std::shared_ptr<context>& owner) {
    auto function = function_lookup(handle, owner).native;
    access_policy_resources(owner, shape.options);
    require(!parameters || !extra, CUDA_ERROR_INVALID_VALUE, "kernelParams and extra cannot both be supplied");
    for (unsigned i = 0; i < 3; ++i)
        require(shape.grid[i] && shape.block[i], CUDA_ERROR_INVALID_VALUE, "launch dimensions must be nonzero");
    std::vector<kernel_parameter> layout;
    try { layout = function->parameter_layout(); }
    catch (const failure& error) { if (!extra || error.code != CUDA_ERROR_NOT_SUPPORTED) throw; }
    size_t required_bytes = 0;
    for (auto parameter : layout) {
        require(parameter.size && parameter.offset <= 65536 && parameter.size <= 65536 - parameter.offset,
                CUDA_ERROR_NOT_SUPPORTED, "kernel argument capture exceeds 64 KiB");
        required_bytes = std::max(required_bytes, parameter.offset + parameter.size);
    }
    auto command = std::make_shared<captured_launch>();
    command->owner = owner; command->function = std::move(function); command->shape = shape;
    if (extra) {
        const void* buffer = nullptr; size_t bytes = 0; bool got_buffer = false, got_size = false, ended = false;
        for (unsigned i = 0; i < 5;) {
            auto token = extra[i++];
            if (token == CU_LAUNCH_PARAM_END) { ended = true; break; }
            require(i < 5, CUDA_ERROR_INVALID_VALUE, "invalid launch extra list");
            auto value = extra[i++];
            if (token == CU_LAUNCH_PARAM_BUFFER_POINTER && !got_buffer) { buffer = value; got_buffer = true; }
            else if (token == CU_LAUNCH_PARAM_BUFFER_SIZE && !got_size && value) { std::memcpy(&bytes, value, sizeof(bytes)); got_size = true; }
            else throw failure(CUDA_ERROR_INVALID_VALUE, "invalid launch extra token");
        }
        require(ended && got_buffer && got_size && bytes >= required_bytes && bytes <= 65536 && (!bytes || buffer),
                CUDA_ERROR_INVALID_VALUE, "incomplete or oversized packed launch arguments");
        command->arguments.resize(bytes);
        if (bytes) std::memcpy(command->arguments.data(), buffer, bytes);
    } else {
        require(layout.empty() || parameters, CUDA_ERROR_INVALID_VALUE, "missing kernel parameters");
        command->arguments.resize(required_bytes);
        for (size_t i = 0; i < layout.size(); ++i) {
            require(parameters[i], CUDA_ERROR_INVALID_VALUE, "null kernel parameter storage");
            std::memcpy(command->arguments.data() + layout[i].offset, parameters[i], layout[i].size);
        }
    }
    for (auto parameter : layout) if (parameter.size == sizeof(CUdeviceptr)) {
        CUdeviceptr pointer{}; std::memcpy(&pointer, command->arguments.data() + parameter.offset, sizeof(pointer));
        auto found = containing(pointer);
        if (found != global().allocations.end()) require(!found->second.async_free, CUDA_ERROR_INVALID_VALUE, "kernel allocation has an enqueued free");
        if (found != global().allocations.end() && found->second.graph_owned) {
            auto memory = found->second.graph_memory.lock();
            require(memory && (resolving_graph_memory || memory->planned_live), CUDA_ERROR_INVALID_VALUE, "kernel graph allocation is not live");
        }
    }
    // Shader transport can follow arbitrary device pointers, including globals
    // and records. Retain the context's allocation snapshot, not only top-level
    // pointer arguments. Workers never consult the mutable registry.
    for (auto& [address, host] : global().hosts) if (!host->retiring && host_visible(host, owner)) {
        if (host->native->flags(host->flags) & CU_MEMHOSTALLOC_DEVICEMAP) host_alias(host, owner);
    }
    for (const auto& [address, allocation] : global().allocations) {
        if (allocation.async_free) continue;
        auto host = allocation.host.lock();
        if (host ? (host->retiring || !host_visible(host, owner)) : !allocation_visible(allocation, owner)) continue;
        if (allocation.graph_owned) {
            auto memory = allocation.graph_memory.lock();
            if (!memory || (!resolving_graph_memory && !memory->planned_live)) continue;
        }
        command->resources.push_back(allocation.native);
        command->bindings.push_back({address, allocation.native.get()});
        if (host) command->hosts.push_back(std::move(host));
    }
    std::unordered_set<uint64_t> bound_addresses;
    for (auto binding : command->bindings) bound_addresses.insert(binding.address);
    for (auto* objects : {&global().textures, &global().surfaces}) for (auto& [handle, image] : *objects) {
        if (image->owner != owner || image->retiring || (image->array && image->array->retiring)) continue;
        command->images.push_back(image);
        if (!image->descriptor) continue;
        command->bindings.push_back({handle, image->descriptor.get(), objects == &global().surfaces, true});
        auto address = image->backing_address;
        if (bound_addresses.insert(address).second)
            command->bindings.push_back({address, image->backing.get()});
    }
    context_configuration(owner, [&] { command->function->validate_launch(shape, command->arguments, command->bindings); });
    command->parameters = std::move(layout);
    return command;
}
void launch(CUfunction handle, const launch_shape& input, CUstream stream, void** parameters, void** extra,
            const std::shared_ptr<context>& owner) {
    auto library_kernel = global().library_kernels.contains(reinterpret_cast<CUkernel>(handle)) ? reinterpret_cast<CUkernel>(handle) : nullptr;
    handle = resolve_library_function(handle, owner);
    auto target = stream_lookup(stream, owner);
    auto shape = stream_launch_shape(input, target);
    auto bind_event = [&](auto& attribute) {
        if (!attribute) return;
        auto& event = event_lookup(attribute->handle);
        require(event.owner.lock() == owner, CUDA_ERROR_INVALID_HANDLE, "launch dependency event belongs to a different context");
        require((event.flags & CU_EVENT_DISABLE_TIMING) && !(event.flags & CU_EVENT_INTERPROCESS), CUDA_ERROR_INVALID_VALUE, "launch dependency events must disable timing and be local");
        auto capture = event.capture.lock();
        require(!capture || capture == target->capture, CUDA_ERROR_CAPTURED_EVENT, "launch dependency event belongs to another capture");
        // Capture can end while preparation releases the registry lock. Keep
        // a native binding until capture_node decides whether to capture.
        attribute->native = context_configuration(owner, [&] { return owner->native->create_launch_event(); });
    };
    bind_event(shape.options.programmatic_event); bind_event(shape.options.completion_event);
    graph_memory_scope scope(bool(target->capture));
    auto command = prepare_launch(handle, shape, parameters, extra, owner);
    graph_node node; node.type = CU_GRAPH_NODE_TYPE_KERNEL; node.owner = owner;
    node.function = handle; node.shape = shape; node.arguments = command->arguments;
    node.library_kernel = library_kernel;
    node.kernel_layout = command->parameters; node.kernel_extra = extra != nullptr;
    node.priority = target->priority;
    if (shape.options.priority) node.priority = *shape.options.priority;
    node.shape.options.priority.reset();
    auto priority = stream_priority(node.priority, owner);
    for (auto& host : command->hosts) node.host_regions.push_back(host);
    if (capture_node(owner, stream, std::move(node))) return;
    for (auto& host : command->hosts) { std::erase_if(host->work, [](auto& item) { return item.expired(); }); host->work.reserve(host->work.size() + 1); }
    auto queue = stream_native(owner, target);
    auto done = submit_stream(owner, stream, {}, {}, priority, [command, queue] { return command->enqueue(queue.get()); }, global().kind != cudavmBackendCuda, {}, global().kind == cudavmBackendCuda ? queue : nullptr);
    for (const auto& attribute : {shape.options.programmatic_event, shape.options.completion_event}) if (attribute) {
        auto& event = event_lookup(attribute->handle); event.record = done; event.timestamp.reset();
        event.capture.reset(); event.capture_dependencies.clear(); event.capture_edges.clear();
    }
    for (auto& host : command->hosts) host_track(host, done);
}
