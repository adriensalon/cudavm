#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#include <algorithm>
#include <cstring>
#include <set>
#include <utility>

namespace cudavm {
void check_topology_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define TOPOLOGY_RUNTIME(call) ::cudavm::check_topology_runtime((call), #call)
template<class Handle> std::set<uintptr_t> topology_set(std::span<const Handle> nodes) {
    std::set<uintptr_t> result;
    for (auto node : nodes) require(result.insert(reinterpret_cast<uintptr_t>(node)).second, "enumeration contains no duplicate node");
    return result;
}
template<class Handle, class Query> void topology_query_nodes(std::span<const CUgraphNode> expected, Query query) {
    struct guarded_count { size_t prefix = 0x12345678, value = 91, suffix = 0x76543210; } count;
    const auto check_count = [&] { require(count.prefix == 0x12345678 && count.suffix == 0x76543210, "graph count output guards"); };
    query(static_cast<Handle*>(nullptr), &count.value); check_count();
    require(count.value == expected.size(), "count-only graph query matches host topology");
    const auto wanted = topology_set<CUgraphNode>(expected);
    for (size_t capacity : {size_t(0), size_t(1), expected.empty() ? 0 : expected.size() - 1, expected.size(), expected.size() + 3}) {
        auto sentinel = reinterpret_cast<Handle>(uintptr_t(0x123456789ab));
        std::vector<Handle> output(capacity + 2, sentinel);
        count.value = capacity; query(output.data() + 1, &count.value); check_count();
        const auto written = std::min(capacity, expected.size());
        require(count.value == written && output.front() == sentinel && output.back() == sentinel, "graph node query extent and guards");
        auto actual = topology_set<Handle>(std::span<const Handle>(output.data() + 1, written));
        for (auto node : actual) require(wanted.contains(node), "graph node query returns only expected live nodes");
        if (capacity >= expected.size()) require(actual == wanted, "complete unordered graph node set");
        for (size_t i = written; i < capacity; ++i) require(output[i + 1] == nullptr, "oversized graph node query null padding");
    }
}
using topology_edge = std::pair<CUgraphNode, CUgraphNode>;
std::set<std::pair<uintptr_t, uintptr_t>> topology_edge_set(std::span<const topology_edge> edges) {
    std::set<std::pair<uintptr_t, uintptr_t>> result;
    for (const auto& [from, to] : edges)
        require(result.emplace(reinterpret_cast<uintptr_t>(from), reinterpret_cast<uintptr_t>(to)).second, "host graph has no duplicate default edges");
    return result;
}
template<class Handle, class Edge, class Query> void topology_query_edges(std::span<const topology_edge> expected, Query query, bool with_metadata = true) {
    size_t count = 91; query(static_cast<Handle*>(nullptr), static_cast<Handle*>(nullptr), static_cast<Edge*>(nullptr), &count);
    require(count == expected.size(), "count-only edge query matches host topology");
    const auto wanted = topology_edge_set(expected);
    for (bool metadata : {false, true})
    for (size_t capacity : {size_t(0), size_t(1), expected.size(), expected.size() + 3}) {
        if (metadata && !with_metadata) continue;
        auto sentinel = reinterpret_cast<Handle>(uintptr_t(0x123456789ab));
        std::vector<Handle> from(capacity + 2, sentinel), to = from;
        std::vector<Edge> data(capacity + 2); std::memset(data.data(), 0xa5, data.size() * sizeof(Edge));
        const auto before = data;
        count = capacity; query(from.data() + 1, to.data() + 1, metadata ? data.data() + 1 : nullptr, &count);
        const auto written = std::min(capacity, expected.size());
        require(count == written && from.front() == sentinel && from.back() == sentinel && to.front() == sentinel && to.back() == sentinel,
            "edge query extent and endpoint guards");
        require(std::memcmp(&data.front(), &before.front(), sizeof(Edge)) == 0 && std::memcmp(&data.back(), &before.back(), sizeof(Edge)) == 0,
            "edge metadata output guards");
        std::set<std::pair<uintptr_t, uintptr_t>> actual;
        for (size_t i = 0; i < written; ++i) {
            auto edge = std::pair{reinterpret_cast<uintptr_t>(from[i + 1]), reinterpret_cast<uintptr_t>(to[i + 1])};
            require(wanted.contains(edge) && actual.insert(edge).second, "edge query returns unique expected endpoint pairs");
            if (metadata) { Edge zero{}; require(std::memcmp(&data[i + 1], &zero, sizeof(Edge)) == 0, "ordinary edge metadata is zero"); }
        }
        if (capacity >= expected.size()) require(actual == wanted, "complete unordered graph edge set");
        for (size_t i = written; i < capacity; ++i) require(!from[i + 1] && !to[i + 1], "oversized edge query null endpoint padding");
    }
}
struct topology_graph {
    CUgraph value{};
    std::vector<CUgraphNode> nodes;
    std::vector<topology_edge> edges;
    explicit topology_graph(bool runtime) {
        if (runtime) { cudaGraph_t graph{}; TOPOLOGY_RUNTIME(cudaGraphCreate(&graph, 0)); value = reinterpret_cast<CUgraph>(graph); }
        else CONFORMANCE_CUDA(cuGraphCreate(&value, 0));
    }
    ~topology_graph() { if (value) cuGraphDestroy(value); }
    CUgraphNode empty(std::initializer_list<CUgraphNode> dependencies, bool runtime) {
        CUgraphNode node{};
        if (runtime) {
            std::vector<cudaGraphNode_t> deps; for (auto dependency : dependencies) deps.push_back(reinterpret_cast<cudaGraphNode_t>(dependency));
            cudaGraphNode_t result{}; TOPOLOGY_RUNTIME(cudaGraphAddEmptyNode(&result, reinterpret_cast<cudaGraph_t>(value), deps.data(), deps.size()));
            node = reinterpret_cast<CUgraphNode>(result);
        } else CONFORMANCE_CUDA(cuGraphAddEmptyNode(&node, value, dependencies.begin(), dependencies.size()));
        nodes.push_back(node); for (auto dependency : dependencies) edges.emplace_back(dependency, node);
        return node;
    }
    void change(std::initializer_list<topology_edge> changes, bool adding, bool runtime, bool metadata = false) {
        std::vector<CUgraphNode> from, to;
        for (const auto& [source, target] : changes) { from.push_back(source); to.push_back(target); }
        std::vector<CUgraphEdgeData> data(changes.size());
        if (runtime) {
            std::vector<cudaGraphNode_t> sources, targets;
            for (auto node : from) sources.push_back(reinterpret_cast<cudaGraphNode_t>(node));
            for (auto node : to) targets.push_back(reinterpret_cast<cudaGraphNode_t>(node));
            std::vector<cudaGraphEdgeData> runtime_data(changes.size());
            if (adding) TOPOLOGY_RUNTIME(cudaGraphAddDependencies(reinterpret_cast<cudaGraph_t>(value), sources.data(), targets.data(), metadata ? runtime_data.data() : nullptr, changes.size()));
            else TOPOLOGY_RUNTIME(cudaGraphRemoveDependencies(reinterpret_cast<cudaGraph_t>(value), sources.data(), targets.data(), metadata ? runtime_data.data() : nullptr, changes.size()));
        } else if (adding) CONFORMANCE_CUDA(cuGraphAddDependencies(value, from.data(), to.data(), metadata ? data.data() : nullptr, changes.size()));
        else CONFORMANCE_CUDA(cuGraphRemoveDependencies(value, from.data(), to.data(), metadata ? data.data() : nullptr, changes.size()));
        for (const auto& edge : changes) { if (adding) edges.push_back(edge); else std::erase(edges, edge); }
    }
    void remove(CUgraphNode node, bool runtime) {
        if (runtime) TOPOLOGY_RUNTIME(cudaGraphDestroyNode(reinterpret_cast<cudaGraphNode_t>(node)));
        else CONFORMANCE_CUDA(cuGraphDestroyNode(node));
        std::erase(nodes, node); std::erase_if(edges, [&](const auto& edge) { return edge.first == node || edge.second == node; });
    }
    void expect() const {
        std::vector<CUgraphNode> roots;
        for (auto node : nodes) if (std::none_of(edges.begin(), edges.end(), [&](const auto& edge) { return edge.second == node; })) roots.push_back(node);
        topology_query_nodes<CUgraphNode>(nodes, [&](auto* output, auto* count) { CONFORMANCE_CUDA(cuGraphGetNodes(value, output, count)); });
        topology_query_nodes<CUgraphNode>(roots, [&](auto* output, auto* count) { CONFORMANCE_CUDA(cuGraphGetRootNodes(value, output, count)); });
        topology_query_nodes<cudaGraphNode_t>(nodes, [&](auto* output, auto* count) { TOPOLOGY_RUNTIME(cudaGraphGetNodes(reinterpret_cast<cudaGraph_t>(value), output, count)); });
        topology_query_nodes<cudaGraphNode_t>(roots, [&](auto* output, auto* count) { TOPOLOGY_RUNTIME(cudaGraphGetRootNodes(reinterpret_cast<cudaGraph_t>(value), output, count)); });
        topology_query_edges<CUgraphNode, CUgraphEdgeData>(edges, [&](auto* from, auto* to, auto* data, auto* count) { CONFORMANCE_CUDA(cuGraphGetEdges(value, from, to, data, count)); });
        topology_query_edges<cudaGraphNode_t, cudaGraphEdgeData>(edges, [&](auto* from, auto* to, auto* data, auto* count) { TOPOLOGY_RUNTIME(cudaGraphGetEdges(reinterpret_cast<cudaGraph_t>(value), from, to, data, count)); });
        for (auto node : nodes) for (bool dependents : {false, true}) {
            std::vector<CUgraphNode> expected;
            for (const auto& edge : edges) if ((dependents ? edge.first : edge.second) == node) expected.push_back(dependents ? edge.second : edge.first);
            topology_query_nodes<CUgraphNode>(expected, [&](auto* output, auto* count) {
                std::vector<CUgraphEdgeData> data(output ? *count : 0);
                if (dependents) CONFORMANCE_CUDA(cuGraphNodeGetDependentNodes(node, output, data.empty() ? nullptr : data.data(), count));
                else CONFORMANCE_CUDA(cuGraphNodeGetDependencies(node, output, data.empty() ? nullptr : data.data(), count));
                for (size_t i = 0; output && i < *count; ++i) { CUgraphEdgeData zero{}; require(std::memcmp(&data[i], &zero, sizeof(zero)) == 0, "neighbor default edge data"); }
            });
            topology_query_nodes<cudaGraphNode_t>(expected, [&](auto* output, auto* count) {
                std::vector<cudaGraphEdgeData> data(output ? *count : 0);
                if (dependents) TOPOLOGY_RUNTIME(cudaGraphNodeGetDependentNodes(reinterpret_cast<cudaGraphNode_t>(node), output, data.empty() ? nullptr : data.data(), count));
                else TOPOLOGY_RUNTIME(cudaGraphNodeGetDependencies(reinterpret_cast<cudaGraphNode_t>(node), output, data.empty() ? nullptr : data.data(), count));
                for (size_t i = 0; output && i < *count; ++i) { cudaGraphEdgeData zero{}; require(std::memcmp(&data[i], &zero, sizeof(zero)) == 0, "Runtime neighbor default edge data"); }
            });
        }
    }
};
void graph_topology_enumeration() {
    for (bool runtime : {false, true}) {
        topology_graph graph(runtime); graph.expect();
        auto a = graph.empty({}, runtime), b = graph.empty({}, !runtime); graph.expect();
        auto c = graph.empty({a}, !runtime), d = graph.empty({a}, runtime); graph.expect();
        graph.empty({b, c, d}, !runtime); graph.expect();
    }
}
void graph_topology_edits() {
    for (bool runtime : {false, true}) {
        topology_graph graph(runtime);
        auto a = graph.empty({}, runtime), b = graph.empty({}, !runtime), c = graph.empty({}, runtime), d = graph.empty({}, !runtime);
        graph.change({{a, c}, {b, c}, {c, d}}, true, runtime); graph.expect();
        graph.change({}, true, !runtime); graph.change({}, false, runtime); graph.expect();
        graph.change({{b, c}}, false, !runtime, true); graph.expect();
        graph.change({{b, d}}, true, !runtime, true); graph.expect();
        graph.remove(c, runtime); graph.expect();
        graph.remove(b, !runtime); graph.expect();
        graph.remove(a, runtime); graph.expect(); graph.remove(d, !runtime); graph.expect();
    }
}
void topology_launch(CUgraph graph, bool runtime) {
    CUgraphExec executable{};
    if (runtime) { cudaGraphExec_t result{}; TOPOLOGY_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), 0)); executable = reinterpret_cast<CUgraphExec>(result); }
    else CONFORMANCE_CUDA(cuGraphInstantiate(&executable, graph, 0));
    struct cleanup { CUgraphExec value; ~cleanup() { cuGraphExecDestroy(value); } } cleanup{executable};
    if (runtime) TOPOLOGY_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(executable), nullptr));
    else CONFORMANCE_CUDA(cuGraphLaunch(executable, nullptr));
    CONFORMANCE_CUDA(cuCtxSynchronize());
}
void graph_topology_execution() {
    for (bool runtime : {false, true}) {
        device_buffer storage(557);
        void* output{}; CONFORMANCE_CUDA(cuMemAllocHost(&output, 601));
        struct cleanup { void* value; ~cleanup() { cuCtxSynchronize(); cuMemFreeHost(value); } } cleanup{output};
        topology_graph graph(runtime);
        CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        CUgraphNode read{}, left{}, right{}, root{};
        CUDA_MEMCPY3D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_DEVICE; copy.srcDevice = storage.value;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = static_cast<unsigned char*>(output) + 19;
        copy.srcPitch = copy.dstPitch = copy.WidthInBytes = 557; copy.srcHeight = copy.dstHeight = copy.Height = copy.Depth = 1;
        CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&read, graph.value, nullptr, 0, &copy, context));
        const auto fill = [&](CUgraphNode* node, size_t offset, size_t count, unsigned value) {
            CUDA_MEMSET_NODE_PARAMS params{}; params.dst = storage.value + offset; params.value = value; params.elementSize = 1; params.width = count; params.height = 1;
            CONFORMANCE_CUDA(cuGraphAddMemsetNode(node, graph.value, nullptr, 0, &params, context));
        };
        fill(&left, 23, 173, 0x35); fill(&right, 257, 211, 0x71); fill(&root, 0, 557, 0xa9);
        graph.nodes = {read, left, right, root}; const auto join = graph.empty({}, runtime);
        graph.change({{join, read}, {left, join}, {right, join}, {root, left}, {root, right}}, true, runtime, true); graph.expect();
        for (unsigned seed : {0u, 0x7du, 0xffu}) {
            CONFORMANCE_CUDA(cuMemsetD8(storage.value, static_cast<unsigned char>(seed), 557)); std::memset(output, 0xd3, 601);
            topology_launch(graph.value, !runtime);
            std::vector<unsigned char> expected(601, 0xd3); std::fill_n(expected.begin() + 19, 557, 0xa9);
            std::fill_n(expected.begin() + 42, 173, 0x35); std::fill_n(expected.begin() + 19 + 257, 211, 0x71);
            require(std::memcmp(output, expected.data(), expected.size()) == 0, "reverse-insertion fork/join graph publishes exact payload and guards");
        }
    }
}
void graph_topology_cycles() {
    for (bool runtime : {false, true}) {
        topology_graph graph(runtime); auto a = graph.empty({}, runtime), b = graph.empty({a}, !runtime), c = graph.empty({b}, runtime);
        graph.change({{c, a}}, true, runtime); graph.expect();
        CUgraphExec driver{}; cudaGraphExec_t executable{};
        require(cuGraphInstantiate(&driver, graph.value, 0) == CUDA_ERROR_INVALID_VALUE, "Driver rejects cyclic graph instantiation");
        require(cudaGraphInstantiateWithFlags(&executable, reinterpret_cast<cudaGraph_t>(graph.value), 0) == cudaErrorInvalidValue, "Runtime rejects cyclic graph instantiation");
        graph.change({{c, a}}, false, !runtime); graph.expect(); topology_launch(graph.value, runtime);
    }
}
template<class Function> Function topology_procedure(const char* name) {
    void* pointer{}; CONFORMANCE_CUDA(cuGetProcAddress(name, &pointer, 10000, CU_GET_PROC_ADDRESS_DEFAULT, nullptr));
    require(pointer, "legacy topology procedure address"); return reinterpret_cast<Function>(pointer);
}
void graph_topology_legacy() {
    topology_graph graph(false); auto a = graph.empty({}, false), b = graph.empty({}, true);
    CONFORMANCE_CUDA(topology_procedure<PFN_cuGraphAddDependencies_v10000>("cuGraphAddDependencies")(graph.value, &a, &b, 1));
    graph.edges.emplace_back(a, b); graph.expect();
    topology_query_edges<CUgraphNode, CUgraphEdgeData>(graph.edges, [&](auto* from, auto* to, auto* data, auto* count) {
        CONFORMANCE_CUDA(topology_procedure<PFN_cuGraphGetEdges_v10000>("cuGraphGetEdges")(graph.value, from, to, count));
    }, false);
    topology_query_nodes<CUgraphNode>(std::span<const CUgraphNode>(&a, 1), [&](auto* output, auto* count) { CONFORMANCE_CUDA(topology_procedure<PFN_cuGraphNodeGetDependencies_v10000>("cuGraphNodeGetDependencies")(b, output, count)); });
    topology_query_nodes<CUgraphNode>(std::span<const CUgraphNode>(&b, 1), [&](auto* output, auto* count) { CONFORMANCE_CUDA(topology_procedure<PFN_cuGraphNodeGetDependentNodes_v10000>("cuGraphNodeGetDependentNodes")(a, output, count)); });
    CONFORMANCE_CUDA(topology_procedure<PFN_cuGraphRemoveDependencies_v10000>("cuGraphRemoveDependencies")(graph.value, &a, &b, 1));
    graph.edges.clear(); graph.expect();
}
void graph_topology_validation() {
    topology_graph graph(false), foreign(true); auto a = graph.empty({}, false), b = graph.empty({a}, true), other = foreign.empty({}, true);
    size_t count = 1; CUgraphNode node{};
    require(cuGraphGetNodes(graph.value, &node, nullptr) == CUDA_ERROR_INVALID_VALUE, "Driver null node count");
    require(cudaGraphGetNodes(reinterpret_cast<cudaGraph_t>(graph.value), nullptr, nullptr) == cudaErrorInvalidValue, "Runtime null node count");
    require(cuGraphGetEdges(graph.value, &node, nullptr, nullptr, &count) == CUDA_ERROR_INVALID_VALUE, "incomplete Driver edge arrays");
    cudaGraphNode_t runtime_node{};
    require(cudaGraphGetEdges(reinterpret_cast<cudaGraph_t>(graph.value), &runtime_node, nullptr, nullptr, &count) == cudaErrorInvalidValue, "incomplete Runtime edge arrays");
    for (const auto& [from, to] : {topology_edge{a, a}, {a, b}, {other, a}}) {
        require(cuGraphAddDependencies(graph.value, &from, &to, nullptr, 1) == CUDA_ERROR_INVALID_VALUE, "invalid Driver edge endpoints or duplicate");
        auto source = reinterpret_cast<cudaGraphNode_t>(from), target = reinterpret_cast<cudaGraphNode_t>(to);
        require(cudaGraphAddDependencies(reinterpret_cast<cudaGraph_t>(graph.value), &source, &target, nullptr, 1) == cudaErrorInvalidValue, "invalid Runtime edge endpoints or duplicate");
        graph.expect();
    }
    require(cuGraphRemoveDependencies(graph.value, &b, &a, nullptr, 1) == CUDA_ERROR_INVALID_VALUE, "removal of absent edge");
    for (unsigned field : {0u, 1u, 2u, 3u}) {
        CUgraphEdgeData data{};
        if (field == 0) data.from_port = 1; else if (field == 1) data.to_port = 1; else if (field == 2) data.type = 0xff; else data.reserved[0] = 1;
        require(cuGraphAddDependencies(graph.value, &b, &a, &data, 1) == CUDA_ERROR_INVALID_VALUE, "invalid ordinary edge metadata");
    }
    std::array<CUgraphNode, 2> duplicate{a, a};
    require(cuGraphAddEmptyNode(&node, graph.value, duplicate.data(), duplicate.size()) == CUDA_ERROR_INVALID_VALUE, "duplicate creation dependencies");
    require(cuGraphAddDependencies(graph.value, nullptr, &a, nullptr, 1) == CUDA_ERROR_INVALID_VALUE, "null nonempty dependency array");
    graph.expect(); topology_launch(graph.value, true);
}
std::string topology_case = "all";
void execute_graph_topology() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (topology_case == "all" || topology_case == name) { execute(); selected = true; } };
    run("enumeration", graph_topology_enumeration); run("edits", graph_topology_edits); run("execution", graph_topology_execution);
    run("cycles", graph_topology_cycles); run("legacy", graph_topology_legacy); run("validation", graph_topology_validation);
    require(selected, "unknown graph_topology case: " + topology_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::topology_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_graph_topology);
}
