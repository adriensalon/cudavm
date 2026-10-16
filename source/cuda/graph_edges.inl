graph_node& graph_node_lookup(CUgraphNode);
bool graph_edge_equal(const CUgraphEdgeData& a, const CUgraphEdgeData& b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}
bool graph_edge_default(const CUgraphEdgeData& edge) { return graph_edge_equal(edge, CUgraphEdgeData{}); }
void graph_edge_validate(const CUgraphEdgeData& edge, CUgraphNodeType source, std::optional<CUgraphNodeType> target = {}) {
    require(!edge.to_port && std::all_of(std::begin(edge.reserved), std::end(edge.reserved), [](auto byte) { return !byte; }),
            CUDA_ERROR_INVALID_VALUE, "invalid graph edge destination port or reserved fields");
    require(edge.type <= CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC && edge.from_port <= CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER,
            CUDA_ERROR_INVALID_VALUE, "invalid graph dependency type or source port");
    require(!edge.from_port || source == CU_GRAPH_NODE_TYPE_KERNEL, CUDA_ERROR_INVALID_VALUE, "only kernel nodes expose dependency ports");
    if (edge.type == CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
        require(source == CU_GRAPH_NODE_TYPE_KERNEL && (!target || *target == CU_GRAPH_NODE_TYPE_KERNEL) && edge.from_port,
                CUDA_ERROR_INVALID_VALUE, "programmatic dependencies require kernel endpoints and a nonzero source port");
    else require(edge.from_port != CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC, CUDA_ERROR_INVALID_VALUE, "programmatic ports require programmatic dependencies");
}
void graph_dependencies_validate(CUgraph graph, const CUgraphNode* nodes, const CUgraphEdgeData* data, size_t count,
                                 std::optional<CUgraphNodeType> target = {}, bool reject_duplicates = true) {
    require(!count || nodes, CUDA_ERROR_INVALID_VALUE, "null graph dependencies");
    for (size_t i = 0; i < count; ++i) {
        auto it = global().graph_nodes.find(nodes[i]);
        require(it != global().graph_nodes.end() && it->second == graph, CUDA_ERROR_INVALID_VALUE, "dependency belongs to another graph");
        auto edge = data ? data[i] : CUgraphEdgeData{};
        graph_edge_validate(edge, graph_node_lookup(nodes[i]).type, target);
        if (reject_duplicates) for (size_t j = 0; j < i; ++j)
            require(nodes[j] != nodes[i] || !graph_edge_equal(edge, data ? data[j] : CUgraphEdgeData{}),
                    CUDA_ERROR_INVALID_VALUE, "duplicate graph dependency");
    }
}
// Metadata is part of an edge's identity: one predecessor can supply more than
// one dependency port. Keep the arrays parallel through capture and graph edits.
void graph_dependency_append(std::vector<CUgraphNode>& nodes, std::vector<CUgraphEdgeData>& edges,
                             CUgraphNode node, const CUgraphEdgeData& edge) {
    for (size_t i = 0; i < nodes.size(); ++i) if (nodes[i] == node && graph_edge_equal(edges[i], edge)) return;
    nodes.push_back(node); edges.push_back(edge);
}
