#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace cudavm {
void check_enable_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define ENABLE_RUNTIME(call) ::cudavm::check_enable_runtime((call), #call)
struct enable_pinned {
    void* value{};
    explicit enable_pinned(size_t bytes) { CONFORMANCE_CUDA(cuMemAllocHost(&value, bytes)); }
    ~enable_pinned() { if (value) cuMemFreeHost(value); }
};
struct enable_storage {
    static constexpr size_t payload = 512, device_size = payload + 32, output_size = device_size + 32;
    device_buffer destination{device_size}, source{device_size};
    enable_pinned output{output_size};
    std::vector<unsigned char> initial, input;
    void reset(unsigned seed) {
        initial.assign(device_size, 0x91); input.assign(device_size, 0x8c);
        for (size_t i = 0; i < payload; ++i) {
            initial[i + 16] = static_cast<unsigned char>(seed + i * 13 + (i >> 2));
            input[i + 16] = static_cast<unsigned char>(0xf0u - seed + i * 5 + (i >> 1));
        }
        CONFORMANCE_CUDA(cuMemcpyHtoD(destination.value, initial.data(), initial.size()));
        CONFORMANCE_CUDA(cuMemcpyHtoD(source.value, input.data(), input.size()));
        std::memset(output.value, 0xd3, output_size);
    }
    void expect(unsigned mask) const {
        auto bytes = initial;
        if (mask & 1) std::fill_n(bytes.begin() + 16 + 13, 109, 0x26);
        if (mask & 2) std::copy_n(input.begin() + 16 + 31, 157, bytes.begin() + 16 + 67);
        if (mask & 4) for (size_t i = 0; i < 148; ++i)
            bytes[16 + 200 + i] = static_cast<unsigned char>(0x89abcdefu >> ((i % 4) * 8));
        std::vector<unsigned char> expected(output_size, 0xd3);
        std::copy(bytes.begin(), bytes.end(), expected.begin() + 16);
        require(std::memcmp(output.value, expected.data(), expected.size()) == 0,
            "enabled operations and empty disabled dependencies preserve exact bytes and guards mask=" + std::to_string(mask));
        std::vector<unsigned char> preserved(input.size());
        CONFORMANCE_CUDA(cuMemcpyDtoH(preserved.data(), source.value, preserved.size()));
        require(preserved == input, "graph copy source and its neighbors remain intact");
    }
};
struct enable_stream {
    CUstream value{};
    enable_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~enable_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct enable_graph {
    CUgraph graph{};
    CUgraphExec primary{}, independent{};
    CUgraphNode root{};
    std::array<CUgraphNode, 3> nodes{};
    bool runtime;
    enable_graph(enable_storage& storage, bool use_runtime) : runtime(use_runtime) {
        if (runtime) {
            cudaGraph_t value{}; ENABLE_RUNTIME(cudaGraphCreate(&value, 0)); graph = reinterpret_cast<CUgraph>(value);
        } else CONFORMANCE_CUDA(cuGraphCreate(&graph, 0));
        try {
            CONFORMANCE_CUDA(cuGraphAddEmptyNode(&root, graph, nullptr, 0));
            CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
            for (unsigned i : {0u, 2u}) {
                const auto prior = i == 0 ? root : nodes[1];
                CUDA_MEMSET_NODE_PARAMS parameters{};
                parameters.dst = storage.destination.value + 16 + (i == 0 ? 13 : 200);
                parameters.value = i == 0 ? 0x26 : 0x89abcdef;
                parameters.elementSize = i == 0 ? 1 : 4; parameters.width = i == 0 ? 109 : 37; parameters.height = 1;
                if (runtime) {
                    cudaMemsetParams p{};
                    p.dst = reinterpret_cast<void*>(static_cast<uintptr_t>(parameters.dst));
                    p.value = parameters.value; p.elementSize = parameters.elementSize; p.width = parameters.width; p.height = 1;
                    cudaGraphNode_t node{}, dependency = reinterpret_cast<cudaGraphNode_t>(prior);
                    ENABLE_RUNTIME(cudaGraphAddMemsetNode(&node, reinterpret_cast<cudaGraph_t>(graph), &dependency, 1, &p));
                    nodes[i] = reinterpret_cast<CUgraphNode>(node);
                } else CONFORMANCE_CUDA(cuGraphAddMemsetNode(&nodes[i], graph, &prior, 1, &parameters, context));
                if (i == 0) {
                    if (runtime) {
                        cudaGraphNode_t node{}, dependency = reinterpret_cast<cudaGraphNode_t>(nodes[0]);
                        ENABLE_RUNTIME(cudaGraphAddMemcpyNode1D(&node, reinterpret_cast<cudaGraph_t>(graph), &dependency, 1,
                            reinterpret_cast<void*>(static_cast<uintptr_t>(storage.destination.value + 16 + 67)),
                            reinterpret_cast<void*>(static_cast<uintptr_t>(storage.source.value + 16 + 31)), 157, cudaMemcpyDeviceToDevice));
                        nodes[1] = reinterpret_cast<CUgraphNode>(node);
                    } else {
                        CUDA_MEMCPY3D p{};
                        p.srcMemoryType = p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                        p.srcDevice = storage.source.value + 16 + 31; p.dstDevice = storage.destination.value + 16 + 67;
                        p.srcPitch = p.dstPitch = p.WidthInBytes = 157;
                        p.srcHeight = p.dstHeight = p.Height = p.Depth = 1;
                        CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&nodes[1], graph, &nodes[0], 1, &p, context));
                    }
                }
            }
            CUDA_MEMCPY3D read{};
            read.srcMemoryType = CU_MEMORYTYPE_DEVICE; read.srcDevice = storage.destination.value;
            read.dstMemoryType = CU_MEMORYTYPE_HOST; read.dstHost = static_cast<unsigned char*>(storage.output.value) + 16;
            read.srcPitch = read.dstPitch = read.WidthInBytes = enable_storage::device_size;
            read.srcHeight = read.dstHeight = read.Height = read.Depth = 1;
            CUgraphNode result{};
            CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&result, graph, &nodes[2], 1, &read, context));
            CONFORMANCE_CUDA(cuGraphInstantiate(&primary, graph, 0));
            cudaGraphExec_t second{};
            ENABLE_RUNTIME(cudaGraphInstantiateWithFlags(&second, reinterpret_cast<cudaGraph_t>(graph), 0));
            independent = reinterpret_cast<CUgraphExec>(second);
        } catch (...) { close(); throw; }
    }
    ~enable_graph() { close(); }
    void close() {
        if (independent) { cuGraphExecDestroy(independent); independent = nullptr; }
        if (primary) { cuGraphExecDestroy(primary); primary = nullptr; }
        if (graph) { cuGraphDestroy(graph); graph = nullptr; }
    }
    void set(CUgraphExec executable, unsigned i, unsigned enabled, bool use_runtime) {
        if (use_runtime) ENABLE_RUNTIME(cudaGraphNodeSetEnabled(reinterpret_cast<cudaGraphExec_t>(executable),
            reinterpret_cast<cudaGraphNode_t>(nodes[i]), enabled));
        else CONFORMANCE_CUDA(cuGraphNodeSetEnabled(executable, nodes[i], enabled));
    }
    void expect(CUgraphExec executable, unsigned mask) const {
        for (unsigned i = 0; i < nodes.size(); ++i) {
            std::array<unsigned, 3> driver{0xdeadbeef, 0xa5a5a5a5, 0x89abcdef}, runtime_output = driver;
            CONFORMANCE_CUDA(cuGraphNodeGetEnabled(executable, nodes[i], &driver[1]));
            ENABLE_RUNTIME(cudaGraphNodeGetEnabled(reinterpret_cast<cudaGraphExec_t>(executable),
                reinterpret_cast<cudaGraphNode_t>(nodes[i]), &runtime_output[1]));
            const std::array<unsigned, 3> expected{0xdeadbeef, (mask >> i) & 1u, 0x89abcdef};
            require(driver == expected && runtime_output == expected, "exact normalized enabled state and output width");
        }
    }
    void run(CUgraphExec executable, enable_storage& storage, enable_stream& stream, unsigned mask, unsigned seed) {
        storage.reset(seed); expect(executable, mask);
        if (runtime) ENABLE_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaStream_t>(stream.value)));
        else CONFORMANCE_CUDA(cuGraphLaunch(executable, stream.value));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream.value));
        storage.expect(mask); expect(executable, mask);
    }
};
void graph_enable_defaults() {
    for (bool runtime : {false, true}) {
        enable_storage storage; enable_graph graph(storage, runtime); enable_stream stream;
        graph.run(graph.primary, storage, stream, 7, 0);
        graph.run(graph.independent, storage, stream, 7, 149);
    }
}
void graph_enable_toggles() {
    for (bool runtime : {false, true}) {
        enable_storage storage; enable_graph graph(storage, runtime); enable_stream stream;
        for (unsigned mask : {7u, 0u, 1u, 2u, 4u, 3u, 5u, 6u, 7u}) {
            for (unsigned i = 0; i < 3; ++i) graph.set(graph.primary, i, (mask >> i) & 1u, (i % 2) ? runtime : !runtime);
            graph.run(graph.primary, storage, stream, mask, mask * 31 + 17);
        }
    }
}
void graph_enable_nonzero() {
    for (bool runtime : {false, true}) {
        enable_storage storage; enable_graph graph(storage, runtime); enable_stream stream;
        for (unsigned enabled : {2u, 0x80000000u, std::numeric_limits<unsigned>::max()}) {
            for (unsigned i = 0; i < 3; ++i) {
                graph.set(graph.primary, i, 0, runtime);
                graph.run(graph.primary, storage, stream, 7u & ~(1u << i), enabled + i);
                graph.set(graph.primary, i, enabled, !runtime);
                graph.run(graph.primary, storage, stream, 7, enabled + i + 23);
            }
        }
    }
}
void graph_enable_independence() {
    for (bool runtime : {false, true}) {
        enable_storage storage; enable_graph graph(storage, runtime); enable_stream stream;
        for (unsigned mask : {0u, 5u, 7u}) {
            for (unsigned i = 0; i < 3; ++i) graph.set(graph.primary, i, (mask >> i) & 1u, runtime);
            graph.run(graph.primary, storage, stream, mask, 37 + mask);
            graph.run(graph.independent, storage, stream, 7, 113 + mask);
            for (unsigned i : {0u, 2u}) {
                CUDA_MEMSET_NODE_PARAMS params{};
                CONFORMANCE_CUDA(cuGraphMemsetNodeGetParams(graph.nodes[i], &params));
                require(params.dst == storage.destination.value + 16 + (i == 0 ? 13 : 200)
                    && params.value == (i == 0 ? 0x26u : 0x89abcdefu)
                    && params.width == (i == 0 ? 109u : 37u) && params.height == 1
                    && params.elementSize == (i == 0 ? 1u : 4u), "enable changes preserve source memset parameters");
            }
        }
    }
}
void graph_enable_validation() {
    for (bool runtime : {false, true}) {
        enable_storage storage; enable_graph graph(storage, runtime); enable_stream stream;
        CUgraph foreign{}; CONFORMANCE_CUDA(cuGraphCreate(&foreign, 0));
        struct graph_cleanup { CUgraph value; ~graph_cleanup() { cuGraphDestroy(value); } } cleanup{foreign};
        CUDA_MEMSET_NODE_PARAMS params{}; params.dst = storage.destination.value; params.elementSize = params.width = params.height = 1;
        CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        CUgraphNode foreign_node{};
        CONFORMANCE_CUDA(cuGraphAddMemsetNode(&foreign_node, foreign, nullptr, 0, &params, context));
        for (auto node : {graph.root, foreign_node}) {
            std::array<unsigned, 3> driver{0xdeadbeef, 123, 0x89abcdef}, runtime_output = driver;
            require(cuGraphNodeSetEnabled(graph.primary, node, 0) == CUDA_ERROR_INVALID_VALUE, "Driver rejects ineligible/foreign node setter");
            require(cudaGraphNodeSetEnabled(reinterpret_cast<cudaGraphExec_t>(graph.primary), reinterpret_cast<cudaGraphNode_t>(node), 0)
                == cudaErrorInvalidValue, "Runtime rejects ineligible/foreign node setter");
            require(cuGraphNodeGetEnabled(graph.primary, node, &driver[1]) == CUDA_ERROR_INVALID_VALUE, "Driver rejects ineligible/foreign node getter");
            require(cudaGraphNodeGetEnabled(reinterpret_cast<cudaGraphExec_t>(graph.primary), reinterpret_cast<cudaGraphNode_t>(node), &runtime_output[1])
                == cudaErrorInvalidValue, "Runtime rejects ineligible/foreign node getter");
            require(driver.front() == 0xdeadbeef && driver.back() == 0x89abcdef
                && runtime_output.front() == 0xdeadbeef && runtime_output.back() == 0x89abcdef, "rejected enabled-query output width");
        }
        require(cuGraphNodeGetEnabled(graph.primary, graph.nodes[0], nullptr) == CUDA_ERROR_INVALID_VALUE, "Driver null enabled output");
        require(cudaGraphNodeGetEnabled(reinterpret_cast<cudaGraphExec_t>(graph.primary), reinterpret_cast<cudaGraphNode_t>(graph.nodes[0]), nullptr)
            == cudaErrorInvalidValue, "Runtime null enabled output");
        graph.run(graph.primary, storage, stream, 7, 211);
    }
}
std::string enable_case = "all";
void execute_graph_node_enable() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) {
        if (enable_case == "all" || enable_case == name) { execute(); selected = true; }
    };
    run("defaults", graph_enable_defaults); run("toggles", graph_enable_toggles); run("nonzero", graph_enable_nonzero);
    run("independence", graph_enable_independence); run("validation", graph_enable_validation);
    require(selected, "unknown graph_node_enable case: " + enable_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::enable_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_graph_node_enable);
}
