#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <cstring>
#include <type_traits>

namespace cudavm {
std::string_view graph_conditional_case;
void graph_conditional_runtime(cudaError_t status) { require(status == cudaSuccess, std::string("conditional Runtime: ") + cudaGetErrorString(status)); }
template<class T, class U> T graph_conditional_cast(U value) { return reinterpret_cast<T>(value); }
template<class T> struct graph_conditional_output {
    std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after;
    graph_conditional_output() { before.fill(0x63); after.fill(0xa7); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto v) { return v == 0x63; })
            && std::all_of(after.begin(), after.end(), [](auto v) { return v == 0xa7; }), "conditional output exact extents and guards"); return value;
    }
};
struct graph_conditional_graph {
    CUgraph value{}; CUcontext context{};
    explicit graph_conditional_graph(bool runtime) {
        TEST_CUDA(cuCtxGetCurrent(&context));
        if (runtime) { cudaGraph_t result{}; graph_conditional_runtime(cudaGraphCreate(&result, 0)); value = graph_conditional_cast<CUgraph>(result); }
        else TEST_CUDA(cuGraphCreate(&value, 0));
    }
    ~graph_conditional_graph() { if (value) cuGraphDestroy(value); }
    CUgraphConditionalHandle condition(unsigned initial, bool runtime) {
        graph_conditional_output<CUgraphConditionalHandle> result;
        if (runtime) {
            graph_conditional_output<cudaGraphConditionalHandle> handle;
            graph_conditional_runtime(cudaGraphConditionalHandleCreate(&handle.value, graph_conditional_cast<cudaGraph_t>(value), initial, cudaGraphCondAssignDefault)); result.value = handle.get();
        } else TEST_CUDA(cuGraphConditionalHandleCreate(&result.value, value, context, initial, CU_GRAPH_COND_ASSIGN_DEFAULT));
        require(result.get(), "conditional handle publication"); return result.value;
    }
};
struct graph_conditional_node {
    CUgraphNode value{}; CUgraphConditionalHandle handle{}; CUgraphConditionalNodeType type{};
    std::vector<CUgraph> bodies;
    graph_conditional_node(CUgraph graph, CUgraphConditionalHandle condition, CUgraphConditionalNodeType kind, unsigned count, bool runtime) : handle(condition), type(kind) {
        graph_conditional_output<CUgraphNode> result;
        if (runtime) {
            cudaGraphNodeParams params{}; params.type = cudaGraphNodeTypeConditional; params.conditional.handle = condition;
            params.conditional.type = static_cast<cudaGraphConditionalNodeType>(kind); params.conditional.size = count;
            graph_conditional_output<cudaGraphNode_t> node;
            graph_conditional_runtime(cudaGraphAddNode(&node.value, graph_conditional_cast<cudaGraph_t>(graph), nullptr, nullptr, 0, &params));
            result.value = graph_conditional_cast<CUgraphNode>(node.get());
            require(params.conditional.phGraph_out, "Runtime publishes body array");
            for (unsigned i = 0; i < count; ++i) bodies.push_back(graph_conditional_cast<CUgraph>(params.conditional.phGraph_out[i]));
        } else {
            CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context));
            CUgraphNodeParams params{}; params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL; params.conditional = {condition, kind, count, nullptr, context};
            TEST_CUDA(cuGraphAddNode(&result.value, graph, nullptr, nullptr, 0, &params)); require(params.conditional.phGraph_out, "Driver publishes body array");
            bodies.assign(params.conditional.phGraph_out, params.conditional.phGraph_out + count);
        }
        value = result.get();
        require(std::all_of(bodies.begin(), bodies.end(), [](auto body) { return body; }), "all owned bodies are published");
        auto distinct = bodies; std::sort(distinct.begin(), distinct.end()); require(std::adjacent_find(distinct.begin(), distinct.end()) == distinct.end(), "owned body handles are distinct");
        query();
    }
    void query() const {
        graph_conditional_output<CUgraphNodeParams> driver; TEST_CUDA(cuGraphNodeGetParams(value, &driver.value)); auto p = driver.get();
        graph_conditional_output<cudaGraphNodeParams> runtime; graph_conditional_runtime(cudaGraphNodeGetParams(graph_conditional_cast<cudaGraphNode_t>(value), &runtime.value)); auto q = runtime.get();
        require(p.type == CU_GRAPH_NODE_TYPE_CONDITIONAL && q.type == cudaGraphNodeTypeConditional
            && p.conditional.handle == handle && q.conditional.handle == handle && p.conditional.type == type
            && static_cast<unsigned>(q.conditional.type) == static_cast<unsigned>(type) && p.conditional.size == bodies.size() && q.conditional.size == bodies.size(), "conditional node parameters through both facades");
        for (size_t i = 0; i < bodies.size(); ++i) require(p.conditional.phGraph_out[i] == bodies[i]
            && graph_conditional_cast<CUgraph>(q.conditional.phGraph_out[i]) == bodies[i], "public body handle identity");
    }
};
struct graph_conditional_storage {
    static constexpr size_t bytes = 288;
    device_buffer device{bytes}; unsigned char* host{};
    graph_conditional_storage() { void* pointer{}; TEST_CUDA(cuMemAllocHost(&pointer, bytes + 32)); host = static_cast<unsigned char*>(pointer); }
    ~graph_conditional_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    void fill(CUgraph body, unsigned value, bool runtime) {
        if (runtime) {
            cudaMemsetParams p{}; p.dst = reinterpret_cast<void*>(uintptr_t(device.value + 16)); p.value = value; p.elementSize = 1; p.width = bytes - 32; p.height = 1;
            cudaGraphNode_t node{}; graph_conditional_runtime(cudaGraphAddMemsetNode(&node, graph_conditional_cast<cudaGraph_t>(body), nullptr, 0, &p));
        } else {
            CUDA_MEMSET_NODE_PARAMS p{}; p.dst = device.value + 16; p.value = value; p.elementSize = 1; p.width = bytes - 32; p.height = 1;
            CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context)); CUgraphNode node{}; TEST_CUDA(cuGraphAddMemsetNode(&node, body, nullptr, 0, &p, context));
        }
    }
    void copy(CUgraph graph, CUgraphNode dependency, bool runtime) {
        if (runtime) {
            auto dep = graph_conditional_cast<cudaGraphNode_t>(dependency); cudaGraphNode_t node{};
            graph_conditional_runtime(cudaGraphAddMemcpyNode1D(&node, graph_conditional_cast<cudaGraph_t>(graph), &dep, 1, host + 16,
                reinterpret_cast<void*>(uintptr_t(device.value)), bytes, cudaMemcpyDeviceToHost));
        } else {
            CUDA_MEMCPY3D p{}; p.srcMemoryType = CU_MEMORYTYPE_DEVICE; p.srcDevice = device.value; p.dstMemoryType = CU_MEMORYTYPE_HOST; p.dstHost = host + 16;
            p.WidthInBytes = p.srcPitch = p.dstPitch = bytes; p.srcHeight = p.dstHeight = p.Height = p.Depth = 1;
            CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context)); CUgraphNode node{}; TEST_CUDA(cuGraphAddMemcpyNode(&node, graph, &dependency, 1, &p, context));
        }
    }
    void reset() { TEST_CUDA(cuMemsetD8(device.value, 0xa7, bytes)); std::fill_n(host, bytes + 32, 0x63); }
    void verify(unsigned value) {
        std::vector<unsigned char> actual(bytes); TEST_CUDA(cuMemcpyDtoH(actual.data(), device.value, bytes));
        for (size_t i = 0; i < bytes; ++i) {
            unsigned expected = i < 16 || i >= bytes - 16 ? 0xa7 : value;
            require(actual[i] == expected && host[i + 16] == expected, "conditional complete payload byte=" + std::to_string(i));
        }
        for (size_t i = 0; i < 16; ++i) require(host[i] == 0x63 && host[bytes + 16 + i] == 0x63, "conditional pinned output guards");
    }
};
void graph_conditional_replay(graph_conditional_graph& graph, graph_conditional_storage& storage, unsigned expected, bool runtime, bool retire = false) {
    CUgraphExec executable{}; CUstream stream{}; TEST_CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    struct cleanup { CUgraphExec& executable; CUstream stream; ~cleanup() { cuStreamSynchronize(stream); if (executable) cuGraphExecDestroy(executable); cuStreamDestroy(stream); } } owner{executable, stream};
    if (runtime) { cudaGraphExec_t result{}; graph_conditional_runtime(cudaGraphInstantiateWithFlags(&result, graph_conditional_cast<cudaGraph_t>(graph.value), 0)); executable = graph_conditional_cast<CUgraphExec>(result); }
    else TEST_CUDA(cuGraphInstantiate(&executable, graph.value, 0));
    if (retire) { TEST_CUDA(cuGraphDestroy(graph.value)); graph.value = nullptr; }
    for (unsigned replay = 0; replay < 3; ++replay) {
        storage.reset();
        if (runtime) {
            graph_conditional_runtime(cudaGraphUpload(graph_conditional_cast<cudaGraphExec_t>(executable), graph_conditional_cast<cudaStream_t>(stream)));
            graph_conditional_runtime(cudaGraphLaunch(graph_conditional_cast<cudaGraphExec_t>(executable), graph_conditional_cast<cudaStream_t>(stream)));
        } else { TEST_CUDA(cuGraphUpload(executable, stream)); TEST_CUDA(cuGraphLaunch(executable, stream)); }
        TEST_CUDA(cuStreamSynchronize(stream)); storage.verify(expected);
    }
}
void graph_conditional_branches() {
    unsigned count = graph_conditional_case == "if_else" ? 2 : 1;
    for (bool runtime : {false, true}) for (unsigned initial : {0u, 1u, 7u}) {
        graph_conditional_storage storage; graph_conditional_graph graph(runtime);
        auto handle = graph.condition(initial, !runtime); graph_conditional_node node(graph.value, handle, CU_GRAPH_COND_TYPE_IF, count, runtime);
        for (unsigned i = 0; i < count; ++i) storage.fill(node.bodies[i], 0x35 + i * 0x18, !runtime);
        storage.copy(graph.value, node.value, runtime);
        graph_conditional_replay(graph, storage, initial ? 0x35 : count == 2 ? 0x4d : 0xa7, !runtime);
    }
}
void graph_conditional_switches() {
    for (bool runtime : {false, true}) for (unsigned initial : {0u, 1u, 2u, 3u, 7u}) {
        graph_conditional_storage storage; graph_conditional_graph graph(runtime); auto handle = graph.condition(initial, runtime);
        graph_conditional_node node(graph.value, handle, CU_GRAPH_COND_TYPE_SWITCH, 3, !runtime);
        for (unsigned i = 0; i < 3; ++i) storage.fill(node.bodies[i], 0x31 + i * 0x17, runtime);
        storage.copy(graph.value, node.value, !runtime); graph_conditional_replay(graph, storage, initial < 3 ? 0x31 + initial * 0x17 : 0xa7, runtime);
    }
}
void graph_conditional_nested() {
    for (bool runtime : {false, true}) for (unsigned initial : {0u, 1u}) {
        graph_conditional_storage storage; graph_conditional_graph graph(runtime); auto outer = graph.condition(initial, runtime), inner = graph.condition(1, !runtime);
        graph_conditional_node first(graph.value, outer, CU_GRAPH_COND_TYPE_IF, 1, runtime);
        graph_conditional_node second(first.bodies[0], inner, CU_GRAPH_COND_TYPE_SWITCH, 2, !runtime);
        storage.fill(second.bodies[0], 0x19, runtime); storage.fill(second.bodies[1], 0x59, !runtime);
        storage.copy(graph.value, first.value, !runtime); graph_conditional_replay(graph, storage, initial ? 0x59 : 0xa7, runtime);
    }
}
void graph_conditional_ownership() {
    for (bool runtime : {false, true}) {
        graph_conditional_storage storage; graph_conditional_graph graph(runtime); auto handle = graph.condition(1, runtime);
        graph_conditional_node node(graph.value, handle, CU_GRAPH_COND_TYPE_IF, 1, !runtime); storage.fill(node.bodies[0], 0x71, runtime); node.query();
        graph_conditional_graph foreign(!runtime); graph_conditional_output<CUgraphNode> rejected;
        CUgraphNodeParams params{}; params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL; params.conditional = {handle, CU_GRAPH_COND_TYPE_IF, 1, nullptr, graph.context};
        require(cuGraphAddNode(&rejected.value, foreign.value, nullptr, nullptr, 0, &params) == CUDA_ERROR_INVALID_VALUE && !rejected.get(), "foreign-tree conditional handle");
        require(cuGraphDestroy(node.bodies[0]) == CUDA_ERROR_INVALID_VALUE, "body graph cannot be independently destroyed");
        CUgraphExec executable{}; require(cuGraphInstantiate(&executable, node.bodies[0], 0) == CUDA_ERROR_INVALID_VALUE && !executable, "body graph cannot be independently instantiated");
        CUgraph clone{}; require(cuGraphClone(&clone, graph.value) == CUDA_ERROR_NOT_SUPPORTED && !clone, "conditional parent cannot be cloned");
        storage.copy(graph.value, node.value, runtime); graph_conditional_replay(graph, storage, 0x71, !runtime, true);
    }
}
void graph_conditional_validation() {
    graph_conditional_graph graph(false), foreign(true); graph_conditional_output<CUgraphConditionalHandle> handle;
    require(cuGraphConditionalHandleCreate(&handle.value, graph.value, graph.context, 1, 2) == CUDA_ERROR_INVALID_VALUE && !handle.get(), "invalid handle flags preserve output");
    require(cudaGraphConditionalHandleCreate(nullptr, graph_conditional_cast<cudaGraph_t>(graph.value), 1, cudaGraphCondAssignDefault) == cudaErrorInvalidValue, "null Runtime conditional handle output");
    CUgraphConditionalHandle condition = 7; graph_conditional_output<CUgraphNode> node;
    CUgraphNodeParams p{}; p.type = CU_GRAPH_NODE_TYPE_CONDITIONAL; p.conditional = {condition, CU_GRAPH_COND_TYPE_IF, 1, nullptr, graph.context};
    require(cuGraphAddNode(&node.value, foreign.value, nullptr, nullptr, 0, &p) == CUDA_ERROR_INVALID_VALUE && !node.get(), "foreign-tree conditional handle");
    auto invalid = p; invalid.conditional.size = 0;
    require(cuGraphAddNode(&node.value, graph.value, nullptr, nullptr, 0, &invalid) == CUDA_ERROR_INVALID_VALUE && !node.get(), "invalid conditional body count");
    invalid = p; CUgraph body{}; invalid.conditional.phGraph_out = &body;
    require(cuGraphAddNode(&node.value, graph.value, nullptr, nullptr, 0, &invalid) == CUDA_ERROR_INVALID_VALUE && !node.get(), "body array must be output");
    invalid = p; invalid.reserved0[0] = 1;
    require(cuGraphAddNode(&node.value, graph.value, nullptr, nullptr, 0, &invalid) == CUDA_ERROR_INVALID_VALUE && !node.get(), "conditional reserved field");
    size_t count{}; TEST_CUDA(cuGraphGetNodes(graph.value, nullptr, &count)); require(!count, "failed conditional additions leave an empty graph");
    graph_conditional_storage storage; storage.fill(graph.value, 0x3d, true);
    CUgraphNode fill{}; count = 1; TEST_CUDA(cuGraphGetNodes(graph.value, &fill, &count)); require(count == 1 && fill, "valid ordinary node repairs the rejected graph");
    storage.copy(graph.value, fill, true); graph_conditional_replay(graph, storage, 0x3d, false);
}
void graph_conditional_discovery() {
    static_assert(std::is_same_v<decltype(&cuGraphConditionalHandleCreate), PFN_cuGraphConditionalHandleCreate_v12030>);
    void* pointer{}; CUdriverProcAddressQueryResult query{};
    TEST_CUDA(cuGetProcAddress("cuGraphConditionalHandleCreate", &pointer, 12029, 0, &query)); require(!pointer && query == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "conditional introduction boundary");
    TEST_CUDA(cuGetProcAddress("cuGraphConditionalHandleCreate", &pointer, 12030, 0, &query)); require(pointer && query == CU_GET_PROC_ADDRESS_SUCCESS, "conditional introduction lookup");
    void* other{}; cudaDriverEntryPointQueryResult runtime{}; graph_conditional_runtime(cudaGetDriverEntryPointByVersion("cuGraphConditionalHandleCreate", &other, 12030, 0, &runtime));
    require(other == pointer && runtime == cudaDriverEntryPointSuccess, "Runtime discovers conditional Driver procedure");
}
void execute_graph_conditionals() {
    if (graph_conditional_case != "validation") {
        auto version = native_cuda_version();
        auto minimum = graph_conditional_case == "if_else" || graph_conditional_case == "nested" || graph_conditional_case == "switches" ? 12080 : 12030;
        require_capability(version >= minimum, "native conditional profile requires CUDA driver " + std::to_string(minimum) + "; installed=" + std::to_string(version));
    }
    if (graph_conditional_case == "branches" || graph_conditional_case == "if_else") graph_conditional_branches();
    else if (graph_conditional_case == "switches") graph_conditional_switches();
    else if (graph_conditional_case == "nested") graph_conditional_nested();
    else if (graph_conditional_case == "ownership") graph_conditional_ownership();
    else if (graph_conditional_case == "validation") graph_conditional_validation();
    else throw std::runtime_error("unknown graph conditional case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_test(argc, argv, cudavm::graph_conditional_discovery);
    if (argc == 3) { cudavm::graph_conditional_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_graph_conditionals);
}
