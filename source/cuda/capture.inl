thread_local CUstreamCaptureMode thread_capture_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;

void capture_retire_context(const std::shared_ptr<context>& owner) {
    std::vector<std::shared_ptr<capture_state>> captures;
    for (auto& item : owner->streams) if (auto stream = item.lock()) if (stream->capture) {
        if (std::find(captures.begin(), captures.end(), stream->capture) == captures.end()) captures.push_back(stream->capture);
        stream->capture.reset(); stream->capture_dependencies.clear(); stream->capture_edges.clear();
    }
    // A capture can include streams from other contexts. Detach every member
    // before destroying its graph, including members of a retiring context.
    for (auto& capture : captures) {
        capture->invalidated = true;
        for (auto& item : capture->streams) if (auto stream = item.lock()) {
            stream->capture.reset(); stream->capture_dependencies.clear(); stream->capture_edges.clear();
        }
        if (capture->owns_graph) graph_destroy(capture->graph);
    }
}

void capture_fail(const std::shared_ptr<capture_state>& capture, CUresult code, const char* message) {
    capture->invalidated = true;
    throw failure(code, message);
}
void capture_check_stream(const std::shared_ptr<stream_state>& stream) {
    if (stream->capture)
        require(!stream->capture->invalidated, CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, "stream capture was invalidated");
    if (stream->legacy) {
        auto owner = stream->owner.lock();
        for (auto& item : owner->streams) if (auto other = item.lock())
            if (other->capture && !(other->flags & CU_STREAM_NON_BLOCKING))
                capture_fail(other->capture, CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, "legacy stream would depend on a capturing stream");
    }
}
void capture_unsafe(const std::shared_ptr<context>&) {
    if (thread_capture_mode == CU_STREAM_CAPTURE_MODE_RELAXED) return;
    bool forbidden = false;
    for (auto& [handle, owner] : global().contexts) for (auto& item : owner->streams) if (auto stream = item.lock()) {
        auto capture = stream->capture;
        if (capture && capture->mode != CU_STREAM_CAPTURE_MODE_RELAXED &&
            (capture->thread == stream_thread_id() ||
             (thread_capture_mode == CU_STREAM_CAPTURE_MODE_GLOBAL && capture->mode == CU_STREAM_CAPTURE_MODE_GLOBAL))) {
            capture->invalidated = true; forbidden = true;
        }
    }
    require(!forbidden, CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "synchronous operation is unsafe during stream capture");
}
void capture_dependencies_validate(CUgraph graph, const CUgraphNode* nodes, const CUgraphEdgeData* data, size_t count) {
    graph_dependencies_validate(graph, nodes, data, count, {}, false);
}
void capture_set_dependencies(const std::shared_ptr<stream_state>& stream, std::vector<CUgraphNode> nodes, std::vector<CUgraphEdgeData> edges) {
    stream->capture_dependencies = std::move(nodes); stream->capture_edges = std::move(edges);
}
void capture_begin(CUstream handle, CUstreamCaptureMode mode, CUgraph graph = nullptr,
                   const CUgraphNode* dependencies = nullptr, const CUgraphEdgeData* data = nullptr, size_t count = 0) {
    require(mode >= CU_STREAM_CAPTURE_MODE_GLOBAL && mode <= CU_STREAM_CAPTURE_MODE_RELAXED, CUDA_ERROR_INVALID_VALUE, "invalid stream capture mode");
    auto stream = stream_lookup(handle); capture_check_stream(stream);
    require(!stream->legacy, CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "cannot capture the legacy stream");
    require(!stream->capture, CUDA_ERROR_ILLEGAL_STATE, "stream is already capturing");
    if (graph) { graph_lookup(graph); capture_dependencies_validate(graph, dependencies, data, count); }
    std::vector<CUgraphNode> initial; std::vector<CUgraphEdgeData> edges;
    for (size_t i = 0; i < count; ++i) graph_dependency_append(initial, edges, dependencies[i], data ? data[i] : CUgraphEdgeData{});
    auto capture = std::make_shared<capture_state>();
    capture->mode = mode; capture->thread = stream_thread_id(); capture->id = global().next_capture_id++;
    capture->origin = stream; capture->streams.push_back(stream); capture->owns_graph = !graph;
    capture_set_dependencies(stream, std::move(initial), std::move(edges));
    capture->graph = graph ? graph : graph_create(0); stream->capture = std::move(capture);
}
struct capture_frontier { std::vector<CUgraphNode> nodes; std::vector<CUgraphEdgeData> edges; };
capture_frontier capture_node_frontier(const std::shared_ptr<stream_state>& stream, CUgraphNodeType type, bool serialize = false) {
    capture_frontier result;
    for (size_t i = 0; i < stream->capture_dependencies.size(); ++i) {
        auto edge = stream->capture_edges[i]; auto node = stream->capture_dependencies[i];
        if (type != CU_GRAPH_NODE_TYPE_KERNEL && edge.type == CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC) edge = {};
        if (serialize && graph_edge_default(edge) && graph_node_lookup(node).type == CU_GRAPH_NODE_TYPE_KERNEL)
            edge = {CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC, 0, CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC, {}};
        graph_dependency_append(result.nodes, result.edges, node, edge);
    }
    return result;
}
bool capture_node(const std::shared_ptr<context>& owner, CUstream handle, graph_node node) {
    auto stream = stream_lookup(handle, owner); capture_check_stream(stream);
    if (!stream->capture) return false;
    auto programmatic = node.shape.options.programmatic_event, completion = node.shape.options.completion_event;
    auto serialize = node.shape.options.programmatic_serialization.value_or(false);
    node.shape.options.programmatic_event.reset(); node.shape.options.completion_event.reset(); node.shape.options.programmatic_serialization.reset();
    graph_validate(node);
    auto frontier = capture_node_frontier(stream, node.type, serialize);
    for (const auto& attribute : {programmatic, completion}) if (attribute) {
        auto& event = event_lookup(attribute->handle);
        require(event.owner.lock() == owner, CUDA_ERROR_INVALID_HANDLE, "launch dependency event belongs to another context");
        auto captured = event.capture.lock();
        require(!captured || captured == stream->capture, CUDA_ERROR_CAPTURED_EVENT, "launch dependency event belongs to another capture");
        event.capture_dependencies.reserve(1); event.capture_edges.reserve(1);
    }
    // Reserve frontier storage before committing the graph node.
    stream->capture_dependencies.reserve(1); stream->capture_edges.reserve(1);
    auto added = graph_add(stream->capture->graph, frontier.nodes.data(), frontier.nodes.size(), std::move(node), frontier.edges.data());
    stream->capture_dependencies.assign(1, added); stream->capture_edges.assign(1, CUgraphEdgeData{});
    auto record = [&](const auto& attribute, CUgraphEdgeData edge) {
        if (!attribute) return;
        auto& event = event_lookup(attribute->handle);
        event.capture_dependencies.assign(1, added); event.capture_edges.assign(1, edge);
        event.capture = stream->capture; event.record.reset(); event.timestamp.reset();
    };
    record(programmatic, {static_cast<unsigned char>(programmatic && programmatic->block_start ? CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER : CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC), 0, CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC, {}});
    record(completion, {CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER, 0, CU_GRAPH_DEPENDENCY_TYPE_DEFAULT, {}});
    return true;
}
CUgraph capture_end(CUstream handle) {
    auto stream = stream_lookup(handle); auto capture = stream->capture;
    require(capture != nullptr, CUDA_ERROR_ILLEGAL_STATE, "stream is not capturing");
    require(capture->origin.lock() == stream, CUDA_ERROR_STREAM_CAPTURE_UNMATCHED, "capture must end on its origin stream");
    bool wrong_thread = capture->mode != CU_STREAM_CAPTURE_MODE_RELAXED && capture->thread != stream_thread_id();
    bool unjoined = false;
    if (!capture->invalidated && !wrong_thread) {
        std::vector<CUgraphNode> reachable = stream->capture_dependencies;
        for (size_t i = 0; i < reachable.size(); ++i) {
            for (auto dependency : graph_node_lookup(reachable[i]).dependencies)
                if (std::find(reachable.begin(), reachable.end(), dependency) == reachable.end()) reachable.push_back(dependency);
        }
        // Frontier nodes explicitly removed by UpdateCaptureDependencies are
        // exempt from the fork/join check, including their predecessors.
        auto excluded = capture->excluded;
        for (size_t i = 0; i < excluded.size(); ++i) {
            auto it = global().graph_nodes.find(excluded[i]);
            if (it == global().graph_nodes.end()) continue;
            for (auto dependency : graph_node_lookup(excluded[i]).dependencies)
                if (std::find(excluded.begin(), excluded.end(), dependency) == excluded.end()) excluded.push_back(dependency);
        }
        for (auto& item : capture->streams) if (auto other = item.lock()) if (other != stream)
            for (auto dependency : other->capture_dependencies)
                if (std::find(reachable.begin(), reachable.end(), dependency) == reachable.end() &&
                    std::find(excluded.begin(), excluded.end(), dependency) == excluded.end()) unjoined = true;
    }
    for (auto& item : capture->streams) if (auto other = item.lock()) {
        other->capture.reset(); other->capture_dependencies.clear(); other->capture_edges.clear();
    }
    for (auto& [event, value] : global().events) if (value.capture.lock() == capture) {
        value.capture.reset(); value.capture_dependencies.clear(); value.capture_edges.clear();
    }
    if (capture->invalidated || unjoined || wrong_thread) {
        if (capture->owns_graph) graph_destroy(capture->graph);
        throw failure(wrong_thread ? CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD : capture->invalidated ? CUDA_ERROR_STREAM_CAPTURE_INVALIDATED : CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
                      wrong_thread ? "discarded capture ended on the wrong thread" : capture->invalidated ? "discarded invalidated capture" : "capture branches were not rejoined to the origin");
    }
    return capture->graph;
}
void capture_info(CUstream handle, CUstreamCaptureStatus* status, cuuint64_t* id, CUgraph* graph,
                  const CUgraphNode** dependencies, const CUgraphEdgeData** edges, size_t* count) {
    require(status && (!edges || dependencies), CUDA_ERROR_INVALID_VALUE, "invalid capture info outputs");
    auto stream = stream_lookup(handle);
    // An invalidated capture is still queryable until EndCapture releases it.
    if (!stream->capture) capture_check_stream(stream);
    auto capture = stream->capture;
    *status = capture ? (capture->invalidated ? CU_STREAM_CAPTURE_STATUS_INVALIDATED : CU_STREAM_CAPTURE_STATUS_ACTIVE) : CU_STREAM_CAPTURE_STATUS_NONE;
    if (capture && !capture->invalidated) {
        require(!dependencies || edges || std::all_of(stream->capture_edges.begin(), stream->capture_edges.end(), graph_edge_default),
                CUDA_ERROR_LOSSY_QUERY, "capture query would discard dependency data");
        if (id) *id = capture->id;
        if (graph) *graph = capture->graph;
        if (dependencies) *dependencies = stream->capture_dependencies.data();
        if (edges) *edges = stream->capture_edges.data();
        if (count) *count = stream->capture_dependencies.size();
    }
}
void capture_update_dependencies(CUstream handle, const CUgraphNode* dependencies, const CUgraphEdgeData* data, size_t count, unsigned flags) {
    require(flags == CU_STREAM_ADD_CAPTURE_DEPENDENCIES || flags == CU_STREAM_SET_CAPTURE_DEPENDENCIES, CUDA_ERROR_INVALID_VALUE, "invalid capture dependency flags");
    auto stream = stream_lookup(handle); capture_check_stream(stream);
    require(stream->capture != nullptr, CUDA_ERROR_ILLEGAL_STATE, "stream is not capturing");
    capture_dependencies_validate(stream->capture->graph, dependencies, data, count);
    auto result = flags == CU_STREAM_SET_CAPTURE_DEPENDENCIES ? std::vector<CUgraphNode>{} : stream->capture_dependencies;
    auto edges = flags == CU_STREAM_SET_CAPTURE_DEPENDENCIES ? std::vector<CUgraphEdgeData>{} : stream->capture_edges;
    for (size_t i = 0; i < count; ++i) graph_dependency_append(result, edges, dependencies[i], data ? data[i] : CUgraphEdgeData{});
    auto excluded = stream->capture->excluded;
    for (auto node : stream->capture_dependencies) if (std::find(result.begin(), result.end(), node) == result.end()) excluded.push_back(node);
    capture_set_dependencies(stream, std::move(result), std::move(edges));
    stream->capture->excluded = std::move(excluded);
}
bool capture_record(CUevent handle, CUstream stream_handle, unsigned flags) {
    auto stream = stream_lookup(stream_handle); capture_check_stream(stream);
    if (!stream->capture) return false;
    auto& event = event_lookup(handle);
    require(event.owner.lock() == stream->owner.lock(), CUDA_ERROR_INVALID_HANDLE, "event and capture contexts differ");
    require(flags == 0 || flags == CU_EVENT_RECORD_EXTERNAL, CUDA_ERROR_INVALID_VALUE, "invalid event record flags");
    if (flags) { graph_node node; node.type = CU_GRAPH_NODE_TYPE_EVENT_RECORD; node.event = handle; node.owner = stream->owner; capture_node(stream->owner.lock(), stream_handle, std::move(node)); }
    auto dependencies = stream->capture_dependencies; auto edges = stream->capture_edges;
    event.capture_dependencies = std::move(dependencies); event.capture_edges = std::move(edges);
    event.capture = stream->capture; event.record.reset(); event.timestamp.reset();
    return true;
}
bool capture_wait(CUstream handle, CUevent event_handle, unsigned flags) {
    require(flags == 0 || flags == CU_EVENT_WAIT_EXTERNAL, CUDA_ERROR_INVALID_VALUE, "invalid event wait flags");
    auto stream = stream_lookup(handle); capture_check_stream(stream);
    require(!flags || stream->capture, CUDA_ERROR_INVALID_VALUE, "external event waits require stream capture");
    auto& event = event_lookup(event_handle);
    auto captured = event.capture.lock();
    if (flags && stream->capture) {
        graph_node node; node.type = CU_GRAPH_NODE_TYPE_WAIT_EVENT; node.event = event_handle; node.owner = event.owner;
        return capture_node(stream->owner.lock(), handle, std::move(node));
    }
    if (captured) {
        if (stream->capture && stream->capture != captured) {
            captured->invalidated = true;
            capture_fail(stream->capture, CUDA_ERROR_STREAM_CAPTURE_MERGE, "cannot merge independent captures");
        }
        require(!stream->legacy, CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, "legacy stream cannot join capture");
        require(!captured->invalidated, CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, "event capture is invalidated");
        auto dependencies = stream->capture_dependencies;
        auto edges = stream->capture_edges;
        for (size_t i = 0; i < event.capture_dependencies.size(); ++i)
            graph_dependency_append(dependencies, edges, event.capture_dependencies[i], event.capture_edges[i]);
        if (!stream->capture) captured->streams.push_back(stream);
        capture_set_dependencies(stream, std::move(dependencies), std::move(edges)); stream->capture = captured; return true;
    }
    // Waiting on an event that has never been recorded is a no-op in CUDA.
    if (stream->capture && !event.record) return true;
    if (stream->capture) capture_fail(stream->capture, CUDA_ERROR_STREAM_CAPTURE_ISOLATION, "capturing stream cannot depend on an uncaptured event");
    return false;
}
