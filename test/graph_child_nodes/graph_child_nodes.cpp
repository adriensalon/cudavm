#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>

namespace cudavm {
void check_child_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CHILD_RUNTIME(call) ::cudavm::check_child_runtime((call), #call)
struct child_patch { size_t offset, count; unsigned value; };
struct child_storage {
    device_buffer device{557}; void* output{};
    child_storage() { TEST_CUDA(cuMemAllocHost(&output, 601)); }
    ~child_storage() { if (output) { cuCtxSynchronize(); cuMemFreeHost(output); } }
    void reset(unsigned seed) { TEST_CUDA(cuMemsetD8(device.value, static_cast<unsigned char>(seed), 557)); std::memset(output, 0xd3, 601); }
    void expect(unsigned seed, std::initializer_list<child_patch> patches) const {
        std::vector<unsigned char> expected(601, 0xd3);
        std::fill_n(expected.begin() + 19, 557, static_cast<unsigned char>(seed));
        std::fill_n(expected.begin() + 35, 512, 0x35);
        for (const auto& patch : patches) std::fill_n(expected.begin() + 35 + patch.offset, patch.count, static_cast<unsigned char>(patch.value));
        std::fill_n(expected.begin() + 35 + 87, 73, 0x71);
        require(std::memcmp(output, expected.data(), expected.size()) == 0, "nested graph ordered patches, untouched device bytes and host guards");
    }
};
CUDA_MEMSET_NODE_PARAMS child_fill_params(const child_storage& storage, child_patch patch) {
    CUDA_MEMSET_NODE_PARAMS p{}; p.dst = storage.device.value + 16 + patch.offset; p.value = patch.value;
    p.elementSize = 1; p.width = patch.count; p.height = 1; return p;
}
struct child_graph {
    CUgraph value{}; CUcontext context{};
    explicit child_graph(bool runtime) {
        TEST_CUDA(cuCtxGetCurrent(&context));
        if (runtime) { cudaGraph_t graph{}; CHILD_RUNTIME(cudaGraphCreate(&graph, 0)); value = reinterpret_cast<CUgraph>(graph); }
        else TEST_CUDA(cuGraphCreate(&value, 0));
    }
    child_graph(const child_graph&) = delete;
    child_graph& operator=(const child_graph&) = delete;
    ~child_graph() { if (value) cuGraphDestroy(value); }
    void destroy(bool runtime) {
        if (runtime) CHILD_RUNTIME(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(value)));
        else TEST_CUDA(cuGraphDestroy(value)); value = nullptr;
    }
    CUgraphNode fill(const child_storage& storage, child_patch patch, bool runtime, CUgraphNode dependency = nullptr) const {
        const auto p = child_fill_params(storage, patch); CUgraphNode node{};
        if (runtime) {
            cudaMemsetParams params{reinterpret_cast<void*>(static_cast<uintptr_t>(p.dst)), 0, p.value, 1, p.width, 1};
            auto dep = reinterpret_cast<cudaGraphNode_t>(dependency); cudaGraphNode_t result{};
            CHILD_RUNTIME(cudaGraphAddMemsetNode(&result, reinterpret_cast<cudaGraph_t>(value), dependency ? &dep : nullptr, dependency ? 1 : 0, &params)); node = reinterpret_cast<CUgraphNode>(result);
        } else TEST_CUDA(cuGraphAddMemsetNode(&node, value, dependency ? &dependency : nullptr, dependency ? 1 : 0, &p, context));
        return node;
    }
    CUgraphNode child(CUgraph source, bool runtime, CUgraphNode dependency = nullptr) const {
        CUgraphNode node{};
        if (runtime) {
            auto dep = reinterpret_cast<cudaGraphNode_t>(dependency); cudaGraphNode_t result{};
            CHILD_RUNTIME(cudaGraphAddChildGraphNode(&result, reinterpret_cast<cudaGraph_t>(value), dependency ? &dep : nullptr, dependency ? 1 : 0, reinterpret_cast<cudaGraph_t>(source)));
            node = reinterpret_cast<CUgraphNode>(result);
        } else TEST_CUDA(cuGraphAddChildGraphNode(&node, value, dependency ? &dependency : nullptr, dependency ? 1 : 0, source));
        CUgraphNodeType driver_type{}; cudaGraphNodeType runtime_type{};
        TEST_CUDA(cuGraphNodeGetType(node, &driver_type)); CHILD_RUNTIME(cudaGraphNodeGetType(reinterpret_cast<cudaGraphNode_t>(node), &runtime_type));
        require(driver_type == CU_GRAPH_NODE_TYPE_GRAPH && runtime_type == cudaGraphNodeTypeGraph, "both surfaces classify child graph nodes");
        return node;
    }
};
CUgraph child_owned_graph(CUgraphNode node, CUgraph original) {
    struct guarded_graph { uint64_t prefix = 0x123456789abcdef0; CUgraph value{}; uint64_t suffix = 0xfedcba9876543210; } driver;
    cudaGraph_t runtime{}; TEST_CUDA(cuGraphChildGraphNodeGetGraph(node, &driver.value));
    CHILD_RUNTIME(cudaGraphChildGraphNodeGetGraph(reinterpret_cast<cudaGraphNode_t>(node), &runtime));
    require(driver.value && driver.value != original && driver.value == reinterpret_cast<CUgraph>(runtime), "both surfaces return one distinct owned child clone");
    require(driver.prefix == 0x123456789abcdef0 && driver.suffix == 0xfedcba9876543210, "child graph output guards");
    return driver.value;
}
CUgraphNode child_cloned_fill(CUgraphNode original, CUgraph graph) {
    CUgraphNode driver{}; cudaGraphNode_t runtime{};
    TEST_CUDA(cuGraphNodeFindInClone(&driver, original, graph));
    CHILD_RUNTIME(cudaGraphNodeFindInClone(&runtime, reinterpret_cast<cudaGraphNode_t>(original), reinterpret_cast<cudaGraph_t>(graph)));
    require(driver && driver != original && driver == reinterpret_cast<CUgraphNode>(runtime), "owned child maps its original fill to a distinct node");
    return driver;
}
void child_set_fill(CUgraphNode node, const child_storage& storage, child_patch patch, bool runtime) {
    const auto p = child_fill_params(storage, patch);
    if (runtime) {
        cudaMemsetParams params{reinterpret_cast<void*>(static_cast<uintptr_t>(p.dst)), 0, p.value, 1, p.width, 1};
        CHILD_RUNTIME(cudaGraphMemsetNodeSetParams(reinterpret_cast<cudaGraphNode_t>(node), &params));
    } else TEST_CUDA(cuGraphMemsetNodeSetParams(node, &p));
}
void child_expect_fill(CUgraphNode node, const child_storage& storage, child_patch patch) {
    CUDA_MEMSET_NODE_PARAMS p{}; TEST_CUDA(cuGraphMemsetNodeGetParams(node, &p)); const auto expected = child_fill_params(storage, patch);
    require(p.dst == expected.dst && p.width == expected.width && p.height == 1 && p.elementSize == 1 && p.value == expected.value, "independent child fill parameter snapshot");
}
struct child_parent {
    child_graph graph;
    CUgraphNode root{}, child{}, tail{}, read{};
    child_parent(const child_storage& storage, CUgraph source, bool runtime) : graph(runtime) {
        root = graph.fill(storage, {0, 512, 0x35}, runtime);
        child = graph.child(source, !runtime, root); tail = graph.fill(storage, {87, 73, 0x71}, runtime, child);
        CUDA_MEMCPY3D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_DEVICE; copy.srcDevice = storage.device.value;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = static_cast<unsigned char*>(storage.output) + 19;
        copy.srcPitch = copy.dstPitch = copy.WidthInBytes = 557; copy.srcHeight = copy.dstHeight = copy.Height = copy.Depth = 1;
        TEST_CUDA(cuGraphAddMemcpyNode(&read, graph.value, &tail, 1, &copy, graph.context));
    }
};
struct child_executable {
    CUgraphExec value{};
    child_executable(CUgraph graph, bool runtime) {
        if (runtime) { cudaGraphExec_t result{}; CHILD_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), 0)); value = reinterpret_cast<CUgraphExec>(result); }
        else TEST_CUDA(cuGraphInstantiate(&value, graph, 0));
    }
    ~child_executable() { if (value) { cuCtxSynchronize(); cuGraphExecDestroy(value); } }
    void run(child_storage& storage, unsigned seed, std::initializer_list<child_patch> patches, bool runtime) const {
        storage.reset(seed);
        if (runtime) CHILD_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), nullptr));
        else TEST_CUDA(cuGraphLaunch(value, nullptr));
        TEST_CUDA(cuCtxSynchronize()); storage.expect(seed, patches);
    }
};
void graph_child_nodes_snapshot() {
    for (bool runtime : {false, true}) {
        child_storage storage; child_graph source(runtime);
        const auto fill = source.fill(storage, {43, 137, 0xab}, runtime);
        child_parent first(storage, source.value, runtime);
        const auto owned = child_owned_graph(first.child, source.value); const auto cloned_fill = child_cloned_fill(fill, owned);
        child_expect_fill(cloned_fill, storage, {43, 137, 0xab});
        child_set_fill(fill, storage, {43, 137, 0x99}, !runtime);
        child_expect_fill(cloned_fill, storage, {43, 137, 0xab});
        child_parent second(storage, source.value, !runtime);
        require(child_owned_graph(second.child, source.value) != owned, "each child embedding owns its own graph clone");
        child_executable original(first.graph.value, runtime), changed(second.graph.value, !runtime);
        for (unsigned seed : {0u, 0x7du, 0xffu}) {
            original.run(storage, seed, {{43, 137, 0xab}}, !runtime);
            changed.run(storage, seed, {{43, 137, 0x99}}, runtime);
        }
    }
}
void graph_child_nodes_edits() {
    for (bool runtime : {false, true}) {
        child_storage storage; child_graph source(runtime); const auto fill = source.fill(storage, {43, 137, 0xab}, !runtime);
        child_parent first(storage, source.value, runtime), second(storage, source.value, !runtime);
        const auto owned = child_owned_graph(first.child, source.value), owned_other = child_owned_graph(second.child, source.value);
        const auto copy = child_cloned_fill(fill, owned), other = child_cloned_fill(fill, owned_other);
        child_executable before(first.graph.value, runtime);
        child_set_fill(copy, storage, {17, 193, 0x99}, runtime);
        child_expect_fill(fill, storage, {43, 137, 0xab}); child_expect_fill(other, storage, {43, 137, 0xab});
        child_expect_fill(copy, storage, {17, 193, 0x99});
        child_executable after(first.graph.value, !runtime), independent(second.graph.value, runtime);
        before.run(storage, 0x91, {{43, 137, 0xab}}, runtime);
        after.run(storage, 0x7d, {{17, 193, 0x99}}, !runtime);
        independent.run(storage, 0xff, {{43, 137, 0xab}}, !runtime);
    }
}
void graph_child_nodes_nested() {
    for (bool runtime : {false, true}) {
        child_storage storage; child_graph leaf(runtime), middle(!runtime);
        leaf.fill(storage, {43, 137, 0xab}, runtime);
        const auto root = middle.fill(storage, {17, 53, 0x99}, !runtime);
        const auto child = middle.child(leaf.value, runtime, root);
        middle.fill(storage, {125, 87, 0x3d}, !runtime, child);
        child_parent parent(storage, middle.value, runtime);
        const auto owned = child_owned_graph(parent.child, middle.value);
        size_t count = 0; TEST_CUDA(cuGraphGetNodes(owned, nullptr, &count)); require(count == 3, "middle child retains all nested nodes");
        child_executable executable(parent.graph.value, !runtime);
        for (unsigned seed : {0u, 0x7du, 0xffu}) executable.run(storage, seed, {{17, 53, 0x99}, {43, 137, 0xab}, {125, 87, 0x3d}}, runtime);
    }
}
void graph_child_nodes_executable() {
    for (bool runtime : {false, true}) {
        child_storage storage; child_graph source(runtime), replacement(!runtime);
        const auto fill = source.fill(storage, {43, 137, 0xab}, runtime); replacement.fill(storage, {43, 137, 0x99}, !runtime);
        child_parent parent(storage, source.value, runtime);
        const auto copied = child_cloned_fill(fill, child_owned_graph(parent.child, source.value));
        child_executable changed(parent.graph.value, runtime), independent(parent.graph.value, !runtime);
        if (runtime) CHILD_RUNTIME(cudaGraphExecChildGraphNodeSetParams(reinterpret_cast<cudaGraphExec_t>(changed.value), reinterpret_cast<cudaGraphNode_t>(parent.child), reinterpret_cast<cudaGraph_t>(replacement.value)));
        else TEST_CUDA(cuGraphExecChildGraphNodeSetParams(changed.value, parent.child, replacement.value));
        child_expect_fill(copied, storage, {43, 137, 0xab}); child_expect_fill(fill, storage, {43, 137, 0xab});
        replacement.destroy(runtime);
        for (unsigned seed : {0u, 0x7du, 0xffu}) {
            changed.run(storage, seed, {{43, 137, 0x99}}, !runtime);
            independent.run(storage, seed, {{43, 137, 0xab}}, runtime);
        }
    }
}
void graph_child_nodes_lifetime() {
    for (bool runtime : {false, true}) {
        child_storage storage; child_graph source(runtime); source.fill(storage, {43, 137, 0xab}, runtime);
        child_parent parent(storage, source.value, !runtime);
        source.destroy(!runtime);
        child_executable executable(parent.graph.value, runtime);
        parent.graph.destroy(runtime);
        for (unsigned seed : {0u, 0x7du, 0xffu}) executable.run(storage, seed, {{43, 137, 0xab}}, !runtime);
    }
}
void graph_child_nodes_validation() {
    child_storage storage; child_graph empty(false), foreign(true);
    const auto foreign_node = foreign.fill(storage, {43, 137, 0xab}, true);
    child_parent parent(storage, empty.value, true);
    const auto owned = child_owned_graph(parent.child, empty.value);
    size_t count = 1; TEST_CUDA(cuGraphGetNodes(owned, nullptr, &count)); require(count == 0, "empty child clone has no nodes");
    CUgraph output{}; CUgraphNode node{}; cudaGraphNode_t runtime_node{}; cudaGraph_t runtime_graph{};
    require(cuGraphChildGraphNodeGetGraph(parent.child, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver child graph output");
    require(cudaGraphChildGraphNodeGetGraph(reinterpret_cast<cudaGraphNode_t>(parent.child), nullptr) == cudaErrorInvalidValue, "null Runtime child graph output");
    require(cuGraphChildGraphNodeGetGraph(parent.root, &output) == CUDA_ERROR_INVALID_VALUE, "fill node has no child graph");
    require(cudaGraphChildGraphNodeGetGraph(reinterpret_cast<cudaGraphNode_t>(parent.root), &runtime_graph) == cudaErrorInvalidValue, "Runtime rejects non-child query");
    require(cuGraphAddChildGraphNode(nullptr, parent.graph.value, nullptr, 0, empty.value) == CUDA_ERROR_INVALID_VALUE, "null Driver child-node output");
    require(cudaGraphAddChildGraphNode(nullptr, reinterpret_cast<cudaGraph_t>(parent.graph.value), nullptr, 0, reinterpret_cast<cudaGraph_t>(empty.value)) == cudaErrorInvalidValue, "null Runtime child-node output");
    require(cuGraphAddChildGraphNode(&node, parent.graph.value, &foreign_node, 1, empty.value) == CUDA_ERROR_INVALID_VALUE, "foreign Driver child dependency");
    const auto dependency = reinterpret_cast<cudaGraphNode_t>(foreign_node);
    require(cudaGraphAddChildGraphNode(&runtime_node, reinterpret_cast<cudaGraph_t>(parent.graph.value), &dependency, 1, reinterpret_cast<cudaGraph_t>(empty.value)) == cudaErrorInvalidValue, "foreign Runtime child dependency");
    child_executable executable(parent.graph.value, true);
    require(cuGraphExecChildGraphNodeSetParams(executable.value, parent.root, empty.value) == CUDA_ERROR_INVALID_VALUE, "Driver child setter rejects fill node");
    require(cudaGraphExecChildGraphNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable.value), reinterpret_cast<cudaGraphNode_t>(parent.root), reinterpret_cast<cudaGraph_t>(empty.value)) == cudaErrorInvalidValue, "Runtime child setter rejects fill node");
    require(child_owned_graph(parent.child, empty.value) == owned, "rejected child operations preserve ownership");
    executable.run(storage, 0x91, {}, false);
}
std::string child_case = "all";
void execute_graph_child_nodes() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (child_case == "all" || child_case == name) { execute(); selected = true; } };
    run("snapshot", graph_child_nodes_snapshot); run("edits", graph_child_nodes_edits); run("nested", graph_child_nodes_nested);
    run("executable", graph_child_nodes_executable); run("lifetime", graph_child_nodes_lifetime); run("validation", graph_child_nodes_validation);
    require(selected, "unknown graph_child_nodes case: " + child_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::child_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_graph_child_nodes);
}
