#pragma once
// Graphs reserve stable addresses separately from trimmable physical storage.
// Public liveness follows queued allocation/free transitions, while pool used
// bytes describe mappings retained between launches.
struct graph_memory_region {
    CUdeviceptr address = 0;
    CUgraph origin = nullptr;
    std::weak_ptr<context> owner;
    CUDA_MEM_ALLOC_NODE_PARAMS params{};
    std::vector<CUmemAccessDesc> access;
    struct free_description { std::weak_ptr<unsigned> token; CUgraph graph; bool internal; };
    std::vector<free_description> frees;
    bool planned_live = false; // Protected by the Driver registry mutex.
    std::atomic<bool> live{false};
    completion transition;
    bool internal_free() const {
        return std::any_of(frees.begin(), frees.end(), [](auto& entry) { return entry.internal && !entry.token.expired(); });
    }
};
struct graph_memory_instance {};
inline thread_local bool resolving_graph_memory = false;
struct graph_memory_scope {
    bool previous = resolving_graph_memory;
    explicit graph_memory_scope(bool enabled = true) { if (enabled) resolving_graph_memory = true; }
    ~graph_memory_scope() { resolving_graph_memory = previous; }
};
