#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>

namespace cudavm {
void check_clone_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CLONE_RUNTIME(call) ::cudavm::check_clone_runtime((call), #call)
struct clone_storage {
    static constexpr size_t payload = 512, device_size = payload + 32, host_size = device_size + 32;
    device_buffer device{device_size};
    void* host{};
    clone_storage() { CONFORMANCE_CUDA(cuMemAllocHost(&host, host_size)); }
    ~clone_storage() { if (host) cuMemFreeHost(host); }
    void reset(unsigned seed) {
        CONFORMANCE_CUDA(cuMemsetD8(device.value, static_cast<unsigned char>(seed), device_size));
        std::memset(host, 0xd3, host_size);
    }
    void expect(unsigned seed, unsigned base, size_t offset, size_t count, unsigned replacement) const {
        std::vector<unsigned char> expected(host_size, 0xd3);
        std::fill_n(expected.begin() + 16, device_size, static_cast<unsigned char>(seed));
        std::fill_n(expected.begin() + 32, payload, static_cast<unsigned char>(base));
        std::fill_n(expected.begin() + 32 + offset, count, static_cast<unsigned char>(replacement));
        require(std::memcmp(host, expected.data(), expected.size()) == 0,
            "clone execution preserves independently selected parameters, dependencies and guards");
    }
};
CUDA_MEMSET_NODE_PARAMS clone_fill(CUdeviceptr device, size_t offset, size_t bytes, unsigned value) {
    CUDA_MEMSET_NODE_PARAMS params{};
    params.dst = device + 16 + offset; params.value = value; params.elementSize = 1; params.width = bytes; params.height = 1;
    return params;
}
struct clone_graph {
    CUgraph value{};
    std::array<CUgraphNode, 3> nodes{};
    explicit clone_graph(bool runtime) {
        if (runtime) {
            cudaGraph_t graph{}; CLONE_RUNTIME(cudaGraphCreate(&graph, 0)); value = reinterpret_cast<CUgraph>(graph);
        } else CONFORMANCE_CUDA(cuGraphCreate(&value, 0));
    }
    clone_graph(const clone_graph& original, bool runtime) {
        if (runtime) {
            cudaGraph_t graph{}; CLONE_RUNTIME(cudaGraphClone(&graph, reinterpret_cast<cudaGraph_t>(original.value)));
            value = reinterpret_cast<CUgraph>(graph);
        } else CONFORMANCE_CUDA(cuGraphClone(&value, original.value));
        try {
            for (size_t i = 0; i < nodes.size(); ++i) {
                CUgraphNode a{}; cudaGraphNode_t b{};
                CONFORMANCE_CUDA(cuGraphNodeFindInClone(&a, original.nodes[i], value));
                CLONE_RUNTIME(cudaGraphNodeFindInClone(&b, reinterpret_cast<cudaGraphNode_t>(original.nodes[i]),
                    reinterpret_cast<cudaGraph_t>(value)));
                require(a && a != original.nodes[i] && a == reinterpret_cast<CUgraphNode>(b), "both facades map to a distinct cloned node");
                nodes[i] = a;
                for (size_t j = 0; j < i; ++j) require(nodes[j] != nodes[i], "clone mapping is one-to-one");
                CUgraphNodeType type{}; CONFORMANCE_CUDA(cuGraphNodeGetType(a, &type));
                require(type == (i == 2 ? CU_GRAPH_NODE_TYPE_MEMCPY : CU_GRAPH_NODE_TYPE_MEMSET), "cloned node type");
            }
            for (size_t i = 0; i < nodes.size(); ++i) {
                CUgraphNode dependency{}; size_t count = 1;
                CONFORMANCE_CUDA(cuGraphNodeGetDependencies(nodes[i], &dependency, nullptr, &count));
                require(count == (i == 0 ? 0u : 1u) && (i == 0 || dependency == nodes[i - 1]), "cloned dependency correspondence");
            }
        } catch (...) { cuGraphDestroy(value); value = nullptr; throw; }
    }
    clone_graph(const clone_graph&) = delete;
    clone_graph& operator=(const clone_graph&) = delete;
    ~clone_graph() { if (value) cuGraphDestroy(value); }
    void destroy(bool runtime) {
        if (runtime) CLONE_RUNTIME(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(value)));
        else CONFORMANCE_CUDA(cuGraphDestroy(value));
        value = nullptr;
    }
    void build(clone_storage& storage) {
        CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        auto base = clone_fill(storage.device.value, 0, clone_storage::payload, 0x35);
        auto patch = clone_fill(storage.device.value, 43, 137, 0xab);
        CONFORMANCE_CUDA(cuGraphAddMemsetNode(&nodes[0], value, nullptr, 0, &base, context));
        CONFORMANCE_CUDA(cuGraphAddMemsetNode(&nodes[1], value, &nodes[0], 1, &patch, context));
        CUDA_MEMCPY3D read{};
        read.srcMemoryType = CU_MEMORYTYPE_DEVICE; read.srcDevice = storage.device.value;
        read.dstMemoryType = CU_MEMORYTYPE_HOST; read.dstHost = static_cast<unsigned char*>(storage.host) + 16;
        read.srcPitch = read.dstPitch = read.WidthInBytes = clone_storage::device_size;
        read.srcHeight = read.dstHeight = read.Height = read.Depth = 1;
        CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&nodes[2], value, &nodes[1], 1, &read, context));
    }
    void set(unsigned index, const CUDA_MEMSET_NODE_PARAMS& params, bool runtime) {
        if (runtime) {
            cudaMemsetParams p{};
            p.dst = reinterpret_cast<void*>(static_cast<uintptr_t>(params.dst)); p.pitch = params.pitch;
            p.value = params.value; p.elementSize = params.elementSize; p.width = params.width; p.height = params.height;
            CLONE_RUNTIME(cudaGraphMemsetNodeSetParams(reinterpret_cast<cudaGraphNode_t>(nodes[index]), &p));
        } else CONFORMANCE_CUDA(cuGraphMemsetNodeSetParams(nodes[index], &params));
    }
    void expect_fill(unsigned index, const CUDA_MEMSET_NODE_PARAMS& expected) const {
        CUDA_MEMSET_NODE_PARAMS a{}; cudaMemsetParams b{};
        CONFORMANCE_CUDA(cuGraphMemsetNodeGetParams(nodes[index], &a));
        CLONE_RUNTIME(cudaGraphMemsetNodeGetParams(reinterpret_cast<cudaGraphNode_t>(nodes[index]), &b));
        require(a.dst == expected.dst && a.value == expected.value && a.width == expected.width
            && a.height == 1 && a.elementSize == 1 && a.pitch == expected.pitch
            && reinterpret_cast<uintptr_t>(b.dst) == expected.dst && b.value == expected.value
            && b.width == expected.width && b.height == 1 && b.elementSize == 1 && b.pitch == expected.pitch,
            "clone/source retain exact independent memset parameters through both facades");
    }
    void run(clone_storage& storage, bool runtime, unsigned seed, unsigned base, size_t offset, size_t bytes, unsigned replacement) {
        storage.reset(seed);
        CUgraphExec executable{};
        if (runtime) {
            cudaGraphExec_t result{}; CLONE_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(value), 0));
            executable = reinterpret_cast<CUgraphExec>(result);
        } else CONFORMANCE_CUDA(cuGraphInstantiate(&executable, value, 0));
        struct executable_cleanup { CUgraphExec value; ~executable_cleanup() { if (value) cuGraphExecDestroy(value); } } cleanup{executable};
        CUstream stream{}; CONFORMANCE_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        struct stream_cleanup { CUstream value; ~stream_cleanup() { cuStreamSynchronize(value); cuStreamDestroy(value); } } drain{stream};
        if (runtime) CONFORMANCE_CUDA(cuGraphLaunch(executable, stream));
        else CLONE_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaStream_t>(stream)));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream));
        storage.expect(seed, base, offset, bytes, replacement);
        if (runtime) CONFORMANCE_CUDA(cuGraphExecDestroy(executable));
        else CLONE_RUNTIME(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(executable)));
        cleanup.value = nullptr;
    }
};
void graph_clone_correspondence() {
    for (bool runtime : {false, true}) {
        clone_storage storage; clone_graph original(runtime); original.build(storage);
        clone_graph copy(original, !runtime);
        for (auto* graph : {&original, &copy}) {
            graph->expect_fill(0, clone_fill(storage.device.value, 0, clone_storage::payload, 0x35));
            graph->expect_fill(1, clone_fill(storage.device.value, 43, 137, 0xab));
            for (unsigned seed : {0u, 0x7du, 0xffu}) graph->run(storage, runtime, seed, 0x35, 43, 137, 0xab);
        }
    }
}
void graph_clone_edits() {
    for (bool runtime : {false, true}) {
        clone_storage storage; clone_graph original(runtime); original.build(storage);
        clone_graph copy(original, !runtime);
        copy.set(1, clone_fill(storage.device.value, 87, 193, 0x71), runtime);
        original.expect_fill(1, clone_fill(storage.device.value, 43, 137, 0xab));
        copy.expect_fill(1, clone_fill(storage.device.value, 87, 193, 0x71));
        original.run(storage, runtime, 13, 0x35, 43, 137, 0xab);
        copy.run(storage, !runtime, 97, 0x35, 87, 193, 0x71);
        original.set(0, clone_fill(storage.device.value, 0, clone_storage::payload, 0x99), !runtime);
        original.set(1, clone_fill(storage.device.value, 12, 37, 0x3d), runtime);
        copy.expect_fill(0, clone_fill(storage.device.value, 0, clone_storage::payload, 0x35));
        copy.expect_fill(1, clone_fill(storage.device.value, 87, 193, 0x71));
        original.run(storage, !runtime, 181, 0x99, 12, 37, 0x3d);
        copy.run(storage, runtime, 233, 0x35, 87, 193, 0x71);
    }
}
void graph_clone_lifetime() {
    for (bool runtime : {false, true}) {
        clone_storage storage; clone_graph original(runtime); original.build(storage);
        clone_graph copy(original, !runtime);
        original.destroy(!runtime);
        copy.set(1, clone_fill(storage.device.value, 87, 193, 0x71), !runtime);
        copy.expect_fill(1, clone_fill(storage.device.value, 87, 193, 0x71));
        for (unsigned seed : {19u, 127u, 251u}) copy.run(storage, runtime, seed, 0x35, 87, 193, 0x71);
        copy.destroy(runtime);
    }
}
void graph_clone_validation() {
    for (bool runtime : {false, true}) {
        clone_storage storage; clone_graph original(runtime); original.build(storage);
        clone_graph copy(original, !runtime), foreign(runtime); foreign.build(storage);
        require(cuGraphClone(nullptr, original.value) == CUDA_ERROR_INVALID_VALUE, "Driver null clone output");
        require(cudaGraphClone(nullptr, reinterpret_cast<cudaGraph_t>(original.value)) == cudaErrorInvalidValue, "Runtime null clone output");
        require(cuGraphNodeFindInClone(nullptr, original.nodes[0], copy.value) == CUDA_ERROR_INVALID_VALUE, "Driver null mapping output");
        require(cudaGraphNodeFindInClone(nullptr, reinterpret_cast<cudaGraphNode_t>(original.nodes[0]), reinterpret_cast<cudaGraph_t>(copy.value))
            == cudaErrorInvalidValue, "Runtime null mapping output");
        CUgraphNode driver{}; cudaGraphNode_t runtime_node{};
        require(cuGraphNodeFindInClone(&driver, foreign.nodes[0], copy.value) == CUDA_ERROR_INVALID_VALUE, "Driver foreign original node");
        require(cudaGraphNodeFindInClone(&runtime_node, reinterpret_cast<cudaGraphNode_t>(foreign.nodes[0]), reinterpret_cast<cudaGraph_t>(copy.value))
            == cudaErrorInvalidValue, "Runtime foreign original node");
        copy.run(storage, !runtime, 227, 0x35, 43, 137, 0xab);
    }
}
std::string clone_case = "all";
void execute_graph_clone() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) {
        if (clone_case == "all" || clone_case == name) { execute(); selected = true; }
    };
    run("correspondence", graph_clone_correspondence); run("edits", graph_clone_edits);
    run("lifetime", graph_clone_lifetime); run("validation", graph_clone_validation);
    require(selected, "unknown graph_clone case: " + clone_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::clone_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_graph_clone);
}
