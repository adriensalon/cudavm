#pragma once
// Editable graphs retain descriptions and stable graph-memory reservations.
// Each launch resolves and retains resources before publishing its batch.
struct graph_parameter_storage {
    std::vector<std::byte> arguments;
    std::vector<void*> parameters;
    std::array<void*, 5> extra{};
    size_t bytes = 0;
    std::vector<CUexternalSemaphore> semaphores;
    std::vector<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> signals;
    std::vector<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> waits;
    std::vector<CUstreamBatchMemOpParams> memops;
};
struct graph_node {
    CUgraphNode handle = nullptr;
    unsigned local_id = 0;
    CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
    std::vector<CUgraphNode> dependencies;
    std::vector<CUgraphEdgeData> dependency_edges;
    std::weak_ptr<context> owner;
    CUcontext parameter_context = nullptr;
    CUDA_HOST_NODE_PARAMS host{};
    CUcontext host_context = nullptr;
    CUDA_MEMCPY3D copy{};
    // Peer capture keeps endpoint contexts separate from the ordering stream.
    bool peer_copy = false;
    std::weak_ptr<context> copy_source, copy_destination;
    CUDA_MEMSET_NODE_PARAMS fill{};
    CUfunction function = nullptr;
    CUkernel library_kernel = nullptr;
    std::string kernel_name;
    int priority = 0; // Requested value, preserved before scheduler clamping.
    launch_shape shape{};
    std::vector<std::byte> arguments;
    std::vector<kernel_parameter> kernel_layout;
    bool kernel_extra = false;
    // Capturing an executable preserves node type/ports and its enabled state.
    // A new top-level executable can subsequently enable this kernel again.
    bool snapshot_disabled = false;
    // Heap views survive graph vector relocation. Generic queries retain older
    // generations until node destruction, as required by GetParams.
    std::shared_ptr<graph_parameter_storage> parameter_view;
    std::vector<std::shared_ptr<graph_parameter_storage>> parameter_history;
    // Remember pinned storage identities without extending their idle lifetime.
    std::vector<std::weak_ptr<host_entry>> host_regions;
    std::shared_ptr<graph_memory_region> memory;
    std::shared_ptr<unsigned> memory_free_token;
    CUgraph child = nullptr;
    std::shared_ptr<const std::vector<graph_node>> child_nodes;
    user_references child_objects;
    CUevent event = nullptr;
    std::vector<CUexternalSemaphore> semaphores;
    std::vector<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS> signals;
    std::vector<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS> waits;
    std::vector<CUstreamBatchMemOpParams> memops;
};
struct graph_state {
    unsigned id = 0;
    uint64_t next_local_id = 0;
    std::vector<graph_node> nodes;
    user_references objects;
    // Original handles remain identities even if their source graph is gone.
    std::unordered_map<CUgraphNode, CUgraphNode> originals;
    CUgraph parent = nullptr;
    bool moved = false;
    std::vector<std::weak_ptr<graph_memory_instance>> instances;
};
struct capture_state {
    CUgraph graph = nullptr;
    bool owns_graph = true, invalidated = false;
    CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
    uint64_t id = 0, thread = 0;
    std::weak_ptr<stream_state> origin;
    std::vector<std::weak_ptr<stream_state>> streams;
    std::vector<CUgraphNode> excluded;
};
struct graph_exec {
    unsigned id = 0;
    unsigned source_id = 0;
    std::weak_ptr<context> owner;
    std::vector<graph_node> nodes;
    std::vector<bool> enabled;
    user_references objects;
    unsigned long long flags = 0;
    completion tail;
    std::shared_ptr<graph_memory_instance> memory_instance;
};
