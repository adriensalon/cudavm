#pragma once
#include "backend.hpp"

namespace cudavm {
// The capture frontier is replaced for every facade node, so recording on one
// private stream does not introduce edges between independent DAG branches.
struct cuda_graph_capture {
    using capture_info = CUresult(CUDAAPI*)(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*, const CUgraphNode**, size_t*);
    using capture_dependencies = CUresult(CUDAAPI*)(CUstream, CUgraphNode*, size_t, unsigned);
    capture_info info = nullptr;
    capture_dependencies update = nullptr;
    decltype(&cuStreamUpdateCaptureDependencies) update_edges = nullptr;
    decltype(&cuGraphKernelNodeSetAttribute) set_attribute = nullptr;
    decltype(&cuGraphNodeGetType) node_type = nullptr;

    bool available(std::span<const native_graph_node> nodes) const {
        return info && (update || update_edges) && (!set_attribute || node_type) &&
            (set_attribute || std::none_of(nodes.begin(), nodes.end(), [](auto& node) { return node.priority != 0; }));
    }
    void record(std::span<const native_graph_node> nodes, native_queue* queue, CUstream stream) const {
        std::vector<std::vector<CUgraphNode>> frontiers;
        for (auto& node : nodes) {
            std::vector<CUgraphNode> dependencies;
            std::vector<CUgraphEdgeData> edges;
            for (auto& dependency : node.dependencies) {
                require(dependency.node < frontiers.size(), CUDA_ERROR_INVALID_VALUE, "native graph dependency is not an ancestor");
                auto& frontier = frontiers[dependency.node];
                require(!dependency.edge.from_port || frontier.size() == 1, CUDA_ERROR_NOT_SUPPORTED, "native kernel port requires a single recorded node");
                for (auto predecessor : frontier) {
                    // Without edge metadata, parallel ports become one ordinary
                    // completion dependency rather than duplicate native edges.
                    if (!update_edges && std::find(dependencies.begin(), dependencies.end(), predecessor) != dependencies.end()) continue;
                    dependencies.push_back(predecessor); edges.push_back(dependency.edge);
                }
            }
            auto status = update_edges ? update_edges(stream, dependencies.data(), edges.data(), dependencies.size(), CU_STREAM_SET_CAPTURE_DEPENDENCIES) :
                update(stream, dependencies.data(), dependencies.size(), CU_STREAM_SET_CAPTURE_DEPENDENCIES);
            require(status == CUDA_SUCCESS, status, "updating native graph capture dependencies failed");
            node.command(queue);
            CUstreamCaptureStatus capture_status{}; const CUgraphNode* frontier = nullptr; size_t count = 0;
            status = info(stream, &capture_status, nullptr, nullptr, &frontier, &count);
            require(status == CUDA_SUCCESS, status, "querying native graph capture frontier failed");
            frontiers.emplace_back();
            if (count) frontiers.back().assign(frontier, frontier + count);
            // Launch capture preserves launch attributes. Apply the effective
            // priority even when it comes from the replay stream.
            if (set_attribute) for (auto captured : frontiers.back()) {
                CUgraphNodeType type{};
                status = node_type(captured, &type);
                require(status == CUDA_SUCCESS, status, "querying native graph node type failed");
                if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
                    CUkernelNodeAttrValue attribute{}; attribute.priority = node.priority;
                    status = set_attribute(captured, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &attribute);
                    require(status == CUDA_SUCCESS, status, "setting native graph kernel priority failed");
                }
            }
        }
    }
};
}
