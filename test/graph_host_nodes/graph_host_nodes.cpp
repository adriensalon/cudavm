#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <cstring>

namespace cudavm {
void check_graph_host_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define GRAPH_HOST_RUNTIME(call) ::cudavm::check_graph_host_runtime((call), #call)
struct graph_host_pinned {
    void* value{};
    explicit graph_host_pinned(size_t size) { CONFORMANCE_CUDA(cuMemAllocHost(&value, size)); }
    ~graph_host_pinned() { if (value) cuMemFreeHost(value); }
};
struct graph_host_storage {
    static constexpr size_t payload = 512, device_size = payload + 32, output_size = device_size + 32;
    graph_host_pinned source{device_size}, output{output_size};
    device_buffer device{device_size};
    void reset() {
        std::memset(source.value, 0xa5, device_size);
        std::memset(output.value, 0xc3, output_size);
        CONFORMANCE_CUDA(cuMemsetD8(device.value, 0x91, device_size));
    }
};
struct graph_host_state {
    graph_host_storage& storage;
    unsigned seed = 0, expected_marker = 0, publishes = 0, inspections = 0, marker = 0;
    bool failed = false;
    void reset(unsigned value, unsigned expected) {
        seed = value; expected_marker = expected; publishes = inspections = marker = 0; failed = false;
    }
    static unsigned char pattern(unsigned seed, unsigned marker, size_t index) {
        return static_cast<unsigned char>(seed + marker + index * 17 + (index >> 3));
    }
    void publish(unsigned selected) {
        failed |= publishes != 0 || inspections != 0;
        ++publishes; marker = selected;
        auto* bytes = static_cast<unsigned char*>(storage.source.value);
        for (size_t i = 0; i < graph_host_storage::payload; ++i) bytes[i + 16] = pattern(seed, selected, i);
    }
    static void CUDA_CB primary(void* argument) { static_cast<graph_host_state*>(argument)->publish(0x11); }
    static void CUDA_CB alternate(void* argument) { static_cast<graph_host_state*>(argument)->publish(0x33); }
    static void CUDA_CB inspect(void* argument) {
        auto& state = *static_cast<graph_host_state*>(argument);
        state.failed |= state.publishes != 1 || state.inspections != 0 || state.marker != state.expected_marker;
        ++state.inspections;
        const auto* source = static_cast<const unsigned char*>(state.storage.source.value);
        const auto* output = static_cast<const unsigned char*>(state.storage.output.value);
        for (size_t i = 0; i < graph_host_storage::device_size; ++i) {
            const unsigned char expected = i >= 16 && i < 16 + graph_host_storage::payload
                ? pattern(state.seed, state.expected_marker, i - 16) : 0xa5;
            state.failed |= source[i] != expected;
        }
        for (size_t i = 0; i < graph_host_storage::output_size; ++i) {
            unsigned char expected = 0xc3;
            if (i >= 16 && i < 16 + graph_host_storage::device_size) expected = 0x91;
            if (i >= 32 && i < 32 + graph_host_storage::payload) expected = pattern(state.seed, state.expected_marker, i - 32);
            state.failed |= output[i] != expected;
        }
    }
};
struct graph_host_stream {
    CUstream value{};
    graph_host_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~graph_host_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct host_node_graph {
    CUgraph value{};
    CUgraphExec executable{};
    CUgraphNode root{}, publish{}, inspect{}, terminal{};
    bool runtime;
    explicit host_node_graph(bool use_runtime) : runtime(use_runtime) {
        if (runtime) {
            cudaGraph_t graph{}; GRAPH_HOST_RUNTIME(cudaGraphCreate(&graph, 0));
            value = reinterpret_cast<CUgraph>(graph);
        } else CONFORMANCE_CUDA(cuGraphCreate(&value, 0));
    }
    ~host_node_graph() {
        if (executable) cuGraphExecDestroy(executable);
        if (value) cuGraphDestroy(value);
    }
    cudaGraph_t runtime_value() const { return reinterpret_cast<cudaGraph_t>(value); }
    cudaGraphExec_t runtime_executable() const { return reinterpret_cast<cudaGraphExec_t>(executable); }
    CUgraphNode empty(CUgraphNode dependency) {
        CUgraphNode result{};
        if (runtime) {
            cudaGraphNode_t node{}, prior = reinterpret_cast<cudaGraphNode_t>(dependency);
            GRAPH_HOST_RUNTIME(cudaGraphAddEmptyNode(&node, runtime_value(), dependency ? &prior : nullptr, dependency ? 1 : 0));
            result = reinterpret_cast<CUgraphNode>(node);
        } else CONFORMANCE_CUDA(cuGraphAddEmptyNode(&result, value, dependency ? &dependency : nullptr, dependency ? 1 : 0));
        return result;
    }
    CUgraphNode host(CUgraphNode dependency, CUhostFn function, void* argument) {
        CUgraphNode result{};
        if (runtime) {
            cudaGraphNode_t node{}, prior = reinterpret_cast<cudaGraphNode_t>(dependency);
            cudaHostNodeParams parameters{function, argument};
            GRAPH_HOST_RUNTIME(cudaGraphAddHostNode(&node, runtime_value(), &prior, 1, &parameters));
            result = reinterpret_cast<CUgraphNode>(node);
        } else {
            CUDA_HOST_NODE_PARAMS parameters{function, argument};
            CONFORMANCE_CUDA(cuGraphAddHostNode(&result, value, &dependency, 1, &parameters));
        }
        return result;
    }
    void set(CUgraphNode node, CUhostFn function, void* argument, bool use_runtime, bool live) {
        if (use_runtime) {
            cudaHostNodeParams parameters{function, argument};
            auto handle = reinterpret_cast<cudaGraphNode_t>(node);
            if (live) GRAPH_HOST_RUNTIME(cudaGraphExecHostNodeSetParams(runtime_executable(), handle, &parameters));
            else GRAPH_HOST_RUNTIME(cudaGraphHostNodeSetParams(handle, &parameters));
        } else {
            CUDA_HOST_NODE_PARAMS parameters{function, argument};
            if (live) CONFORMANCE_CUDA(cuGraphExecHostNodeSetParams(executable, node, &parameters));
            else CONFORMANCE_CUDA(cuGraphHostNodeSetParams(node, &parameters));
        }
    }
    void expect(CUgraphNode node, CUhostFn function, void* argument) {
        struct driver_output { uint64_t before = 0xdeadbeef; CUDA_HOST_NODE_PARAMS params{}; uint64_t after = 0x89abcdef; } a;
        struct runtime_output { uint64_t before = 0xdeadbeef; cudaHostNodeParams params{}; uint64_t after = 0x89abcdef; } b;
        CONFORMANCE_CUDA(cuGraphHostNodeGetParams(node, &a.params));
        GRAPH_HOST_RUNTIME(cudaGraphHostNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &b.params));
        require(a.before == 0xdeadbeef && a.after == 0x89abcdef && b.before == 0xdeadbeef && b.after == 0x89abcdef,
            "host-node parameter output widths");
        require(a.params.fn == function && a.params.userData == argument
            && b.params.fn == function && b.params.userData == argument, "both facades retain exact function/data pointers");
    }
    CUgraphNode copy(CUgraphNode dependency, const CUDA_MEMCPY3D& parameters) {
        CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        CUgraphNode result{};
        CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&result, value, &dependency, 1, &parameters, context));
        return result;
    }
    void build(graph_host_storage& storage, graph_host_state& argument) {
        root = empty(nullptr);
        publish = host(root, graph_host_state::primary, &argument);
        CUDA_MEMCPY3D upload{};
        upload.srcMemoryType = CU_MEMORYTYPE_HOST;
        upload.srcHost = static_cast<unsigned char*>(storage.source.value) + 16;
        upload.dstMemoryType = CU_MEMORYTYPE_DEVICE; upload.dstDevice = storage.device.value + 16;
        upload.srcPitch = upload.dstPitch = upload.WidthInBytes = graph_host_storage::payload;
        upload.srcHeight = upload.dstHeight = upload.Height = upload.Depth = 1;
        auto uploaded = copy(publish, upload);
        CUDA_MEMCPY3D download{};
        download.srcMemoryType = CU_MEMORYTYPE_DEVICE; download.srcDevice = storage.device.value;
        download.dstMemoryType = CU_MEMORYTYPE_HOST;
        download.dstHost = static_cast<unsigned char*>(storage.output.value) + 16;
        download.srcPitch = download.dstPitch = download.WidthInBytes = graph_host_storage::device_size;
        download.srcHeight = download.dstHeight = download.Height = download.Depth = 1;
        auto downloaded = copy(uploaded, download);
        inspect = host(downloaded, graph_host_state::inspect, &argument);
        terminal = empty(inspect);
        for (auto node : {root, terminal}) {
            CUgraphNodeType type{}; CONFORMANCE_CUDA(cuGraphNodeGetType(node, &type));
            require(type == CU_GRAPH_NODE_TYPE_EMPTY, "empty-node type");
        }
        expect(publish, graph_host_state::primary, &argument);
        expect(inspect, graph_host_state::inspect, &argument);
    }
    void instantiate() {
        if (runtime) {
            cudaGraphExec_t result{};
            GRAPH_HOST_RUNTIME(cudaGraphInstantiateWithFlags(&result, runtime_value(), 0));
            executable = reinterpret_cast<CUgraphExec>(result);
        } else CONFORMANCE_CUDA(cuGraphInstantiate(&executable, value, 0));
    }
    void run(CUstream stream) {
        if (runtime) {
            GRAPH_HOST_RUNTIME(cudaGraphLaunch(runtime_executable(), reinterpret_cast<cudaStream_t>(stream)));
            CONFORMANCE_CUDA(cuStreamSynchronize(stream));
        } else {
            CONFORMANCE_CUDA(cuGraphLaunch(executable, stream));
            GRAPH_HOST_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
        }
    }
    void destroy_source() {
        if (runtime) CONFORMANCE_CUDA(cuGraphDestroy(value));
        else GRAPH_HOST_RUNTIME(cudaGraphDestroy(runtime_value()));
        value = nullptr;
    }
    void destroy_executable() {
        if (runtime) CONFORMANCE_CUDA(cuGraphExecDestroy(executable));
        else GRAPH_HOST_RUNTIME(cudaGraphExecDestroy(runtime_executable()));
        executable = nullptr;
    }
};
void graph_host_run(host_node_graph& graph, graph_host_storage& storage, graph_host_stream& stream,
                    graph_host_state& active, graph_host_state& inactive, unsigned seed, unsigned marker) {
    storage.reset(); active.reset(seed, marker); inactive.reset(seed + 93, marker ^ 0x22);
    graph.run(stream.value);
    require(active.publishes == 1 && active.inspections == 1 && !active.failed,
        "host-node exact delivery, ordering, selected function/data and guarded transfers");
    require(inactive.publishes == 0 && inactive.inspections == 0 && !inactive.failed,
        "unselected host argument remains unused");
}
void graph_host_ordering(bool retire) {
    for (bool runtime : {false, true}) {
        graph_host_storage storage;
        graph_host_state a{storage}, b{storage};
        host_node_graph graph(runtime);
        graph_host_stream stream;
        graph.build(storage, a); graph.instantiate();
        if (retire) graph.destroy_source();
        for (unsigned seed : {0u, 29u, 250u, 0xffffffffu}) graph_host_run(graph, storage, stream, a, b, seed, 0x11);
        graph.destroy_executable();
        if (!retire) graph.destroy_source();
    }
}
void graph_host_parameters() {
    for (bool runtime : {false, true}) {
        graph_host_storage storage;
        graph_host_state a{storage}, b{storage};
        host_node_graph graph(runtime);
        graph_host_stream stream;
        graph.build(storage, a); graph.instantiate();
        graph.set(graph.publish, graph_host_state::alternate, &b, !runtime, false);
        graph.set(graph.inspect, graph_host_state::inspect, &b, runtime, false);
        graph.expect(graph.publish, graph_host_state::alternate, &b);
        graph.expect(graph.inspect, graph_host_state::inspect, &b);
        graph_host_run(graph, storage, stream, a, b, 31, 0x11);
        graph.set(graph.publish, graph_host_state::alternate, &b, runtime, true);
        graph.set(graph.inspect, graph_host_state::inspect, &b, !runtime, true);
        graph_host_run(graph, storage, stream, b, a, 67, 0x33);
        graph.set(graph.publish, graph_host_state::primary, &a, runtime, false);
        graph.set(graph.inspect, graph_host_state::inspect, &a, !runtime, false);
        graph_host_run(graph, storage, stream, b, a, 131, 0x33);
        graph.expect(graph.publish, graph_host_state::primary, &a);
        graph.set(graph.publish, graph_host_state::primary, &a, !runtime, true);
        graph.set(graph.inspect, graph_host_state::inspect, &a, runtime, true);
        graph_host_run(graph, storage, stream, a, b, 197, 0x11);
        graph.destroy_executable(); graph.destroy_source();
    }
}
void graph_host_validation() {
    for (bool runtime : {false, true}) {
        graph_host_storage storage;
        graph_host_state a{storage}, b{storage};
        host_node_graph graph(runtime);
        graph_host_stream stream;
        graph.build(storage, a); graph.instantiate();
        CUDA_HOST_NODE_PARAMS invalid{nullptr, &b}, output{};
        cudaHostNodeParams runtime_invalid{nullptr, &b}, runtime_output{};
        auto host = reinterpret_cast<cudaGraphNode_t>(graph.publish);
        auto empty = reinterpret_cast<cudaGraphNode_t>(graph.root);
        require(cuGraphHostNodeGetParams(graph.root, &output) == CUDA_ERROR_INVALID_VALUE, "Driver host getter rejects empty node");
        require(cudaGraphHostNodeGetParams(empty, &runtime_output) == cudaErrorInvalidValue, "Runtime host getter rejects empty node");
        require(cuGraphHostNodeGetParams(graph.publish, nullptr) == CUDA_ERROR_INVALID_VALUE, "Driver null parameter output");
        require(cudaGraphHostNodeGetParams(host, nullptr) == cudaErrorInvalidValue, "Runtime null parameter output");
        require(cuGraphHostNodeSetParams(graph.publish, &invalid) == CUDA_ERROR_INVALID_VALUE, "Driver null host function");
        require(cudaGraphHostNodeSetParams(host, &runtime_invalid) == cudaErrorInvalidValue, "Runtime null host function");
        require(cuGraphHostNodeSetParams(graph.publish, nullptr) == CUDA_ERROR_INVALID_VALUE, "Driver null source parameters");
        require(cudaGraphHostNodeSetParams(host, nullptr) == cudaErrorInvalidValue, "Runtime null source parameters");
        require(cuGraphExecHostNodeSetParams(graph.executable, graph.publish, &invalid) == CUDA_ERROR_INVALID_VALUE,
            "Driver invalid executable host parameters");
        require(cudaGraphExecHostNodeSetParams(graph.runtime_executable(), host, &runtime_invalid) == cudaErrorInvalidValue,
            "Runtime invalid executable host parameters");
        CUgraphNode rejected{}; cudaGraphNode_t runtime_rejected{};
        require(cuGraphAddHostNode(&rejected, graph.value, nullptr, 0, &invalid) == CUDA_ERROR_INVALID_VALUE,
            "Driver invalid host-node creation");
        require(cudaGraphAddHostNode(&runtime_rejected, graph.runtime_value(), nullptr, 0, &runtime_invalid) == cudaErrorInvalidValue,
            "Runtime invalid host-node creation");
        graph.expect(graph.publish, graph_host_state::primary, &a);
        graph_host_run(graph, storage, stream, a, b, 223, 0x11);
        graph.destroy_executable(); graph.destroy_source();
    }
}
std::string graph_host_case = "all";
void execute_graph_host_nodes() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) {
        if (graph_host_case == "all" || graph_host_case == name) { execute(); selected = true; }
    };
    run("ordering", [] { graph_host_ordering(false); }); run("retirement", [] { graph_host_ordering(true); });
    run("parameters", graph_host_parameters); run("validation", graph_host_validation);
    require(selected, "unknown graph_host_nodes case: " + graph_host_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::graph_host_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_graph_host_nodes);
}
