#pragma once
#include "backend.hpp"

namespace cudavm {
// Private recording lanes never submit work. Returned tasks retain immutable
// uploads, transport storage, pipelines and descriptors until the region retires.
struct command_region_queue : native_queue {
    std::vector<device_work> resources;
    bool recording() const override { return true; }
};
inline bool command_region_compatible(std::span<const native_graph_node> nodes) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i].command || nodes[i].priority != nodes.front().priority) return false;
        for (auto& dependency : nodes[i].dependencies) {
            CUgraphEdgeData full{};
            if (dependency.node >= i || std::memcmp(&dependency.edge, &full, sizeof(full))) return false;
        }
    }
    return !nodes.empty();
}
}
