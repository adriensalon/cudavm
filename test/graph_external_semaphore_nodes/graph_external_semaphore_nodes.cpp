#include "test_framework.hpp"
#include <cuda/cudavm_cuda.h>
#include <cudaTypedefs.h>
#include <cstring>
#include <type_traits>
#include "directx_fences.hpp"

namespace cudavm {
std::string_view graph_external_case;
void graph_external_runtime(cudaError_t status) { require(status == cudaSuccess, std::string("graph external semaphore Runtime: ") + cudaGetErrorString(status)); }
template<class T, class U> T graph_external_cast(U value) { return reinterpret_cast<T>(value); }
template<class T> struct graph_external_output {
    std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after;
    graph_external_output() { before.fill(0x63); after.fill(0xa7); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto v) { return v == 0x63; })
            && std::all_of(after.begin(), after.end(), [](auto v) { return v == 0xa7; }), "semaphore graph output extents and guards"); return value;
    }
};
struct graph_external_graph {
    CUgraph value{};
    explicit graph_external_graph(bool runtime) {
        if (runtime) { cudaGraph_t graph{}; graph_external_runtime(cudaGraphCreate(&graph, 0)); value = graph_external_cast<CUgraph>(graph); }
        else TEST_CUDA(cuGraphCreate(&value, 0));
    }
    ~graph_external_graph() { if (value) cuGraphDestroy(value); }
};
struct graph_external_executable {
    CUgraphExec value{};
    graph_external_executable(CUgraph graph, bool runtime) {
        if (runtime) { cudaGraphExec_t result{}; graph_external_runtime(cudaGraphInstantiateWithFlags(&result, graph_external_cast<cudaGraph_t>(graph), 0)); value = graph_external_cast<CUgraphExec>(result); }
        else TEST_CUDA(cuGraphInstantiate(&value, graph, 0));
    }
    ~graph_external_executable() { if (value) { cuCtxSynchronize(); cuGraphExecDestroy(value); } }
};
struct graph_external_stream {
    CUstream value{};
    graph_external_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~graph_external_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
template<bool Signal> CUgraphNode graph_external_node(CUgraph graph, CUgraphNode dependency, CUexternalSemaphore semaphore, uint64_t fence, bool runtime) {
    graph_external_output<CUgraphNode> result;
    if (runtime) {
        auto handle = graph_external_cast<cudaExternalSemaphore_t>(semaphore); auto dep = graph_external_cast<cudaGraphNode_t>(dependency);
        cudaGraphNode_t node{};
        if constexpr (Signal) {
            cudaExternalSemaphoreSignalParams operation{}; operation.params.fence.value = fence;
            cudaExternalSemaphoreSignalNodeParams params{&handle, &operation, 1};
            graph_external_runtime(cudaGraphAddExternalSemaphoresSignalNode(&node, graph_external_cast<cudaGraph_t>(graph), dependency ? &dep : nullptr, dependency ? 1 : 0, &params));
        } else {
            cudaExternalSemaphoreWaitParams operation{}; operation.params.fence.value = fence;
            cudaExternalSemaphoreWaitNodeParams params{&handle, &operation, 1};
            graph_external_runtime(cudaGraphAddExternalSemaphoresWaitNode(&node, graph_external_cast<cudaGraph_t>(graph), dependency ? &dep : nullptr, dependency ? 1 : 0, &params));
        }
        result.value = graph_external_cast<CUgraphNode>(node);
    } else if constexpr (Signal) {
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS operation{}; operation.params.fence.value = fence;
        CUDA_EXT_SEM_SIGNAL_NODE_PARAMS params{&semaphore, &operation, 1};
        TEST_CUDA(cuGraphAddExternalSemaphoresSignalNode(&result.value, graph, dependency ? &dependency : nullptr, dependency ? 1 : 0, &params));
    } else {
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS operation{}; operation.params.fence.value = fence;
        CUDA_EXT_SEM_WAIT_NODE_PARAMS params{&semaphore, &operation, 1};
        TEST_CUDA(cuGraphAddExternalSemaphoresWaitNode(&result.value, graph, dependency ? &dependency : nullptr, dependency ? 1 : 0, &params));
    }
    return result.get();
}
template<bool Signal> void graph_external_set(CUgraphNode node, CUexternalSemaphore semaphore, uint64_t fence, bool runtime, CUgraphExec executable = nullptr) {
    if (runtime) {
        auto handle = graph_external_cast<cudaExternalSemaphore_t>(semaphore); auto target = graph_external_cast<cudaGraphNode_t>(node);
        if constexpr (Signal) {
            cudaExternalSemaphoreSignalParams operation{}; operation.params.fence.value = fence;
            cudaExternalSemaphoreSignalNodeParams params{&handle, &operation, 1};
            if (executable) graph_external_runtime(cudaGraphExecExternalSemaphoresSignalNodeSetParams(graph_external_cast<cudaGraphExec_t>(executable), target, &params));
            else graph_external_runtime(cudaGraphExternalSemaphoresSignalNodeSetParams(target, &params));
        } else {
            cudaExternalSemaphoreWaitParams operation{}; operation.params.fence.value = fence;
            cudaExternalSemaphoreWaitNodeParams params{&handle, &operation, 1};
            if (executable) graph_external_runtime(cudaGraphExecExternalSemaphoresWaitNodeSetParams(graph_external_cast<cudaGraphExec_t>(executable), target, &params));
            else graph_external_runtime(cudaGraphExternalSemaphoresWaitNodeSetParams(target, &params));
        }
    } else if constexpr (Signal) {
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS operation{}; operation.params.fence.value = fence;
        CUDA_EXT_SEM_SIGNAL_NODE_PARAMS params{&semaphore, &operation, 1};
        if (executable) TEST_CUDA(cuGraphExecExternalSemaphoresSignalNodeSetParams(executable, node, &params));
        else TEST_CUDA(cuGraphExternalSemaphoresSignalNodeSetParams(node, &params));
    } else {
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS operation{}; operation.params.fence.value = fence;
        CUDA_EXT_SEM_WAIT_NODE_PARAMS params{&semaphore, &operation, 1};
        if (executable) TEST_CUDA(cuGraphExecExternalSemaphoresWaitNodeSetParams(executable, node, &params));
        else TEST_CUDA(cuGraphExternalSemaphoresWaitNodeSetParams(node, &params));
    }
}
template<bool Signal> void graph_external_query(CUgraphNode node, CUexternalSemaphore handle, uint64_t fence) {
    const auto check = [&](const auto& params) {
        require(params.numExtSems == 1 && graph_external_cast<CUexternalSemaphore>(params.extSemArray[0]) == handle
            && params.paramsArray[0].params.fence.value == fence && !params.paramsArray[0].flags, "owned semaphore graph descriptor snapshot");
    };
    if constexpr (Signal) {
        graph_external_output<CUDA_EXT_SEM_SIGNAL_NODE_PARAMS> driver; TEST_CUDA(cuGraphExternalSemaphoresSignalNodeGetParams(node, &driver.value)); check(driver.get());
        graph_external_output<cudaExternalSemaphoreSignalNodeParams> runtime; graph_external_runtime(cudaGraphExternalSemaphoresSignalNodeGetParams(graph_external_cast<cudaGraphNode_t>(node), &runtime.value)); check(runtime.get());
    } else {
        graph_external_output<CUDA_EXT_SEM_WAIT_NODE_PARAMS> driver; TEST_CUDA(cuGraphExternalSemaphoresWaitNodeGetParams(node, &driver.value)); check(driver.get());
        graph_external_output<cudaExternalSemaphoreWaitNodeParams> runtime; graph_external_runtime(cudaGraphExternalSemaphoresWaitNodeGetParams(graph_external_cast<cudaGraphNode_t>(node), &runtime.value)); check(runtime.get());
    }
}
struct graph_external_storage {
    static constexpr size_t bytes = 1056;
    device_buffer device{bytes}; unsigned char* host{};
    graph_external_storage() { void* pointer{}; TEST_CUDA(cuMemAllocHost(&pointer, bytes + 32)); host = static_cast<unsigned char*>(pointer); }
    ~graph_external_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    void reset() { TEST_CUDA(cuMemsetD8(device.value, 0xa7, bytes)); std::fill_n(host, bytes + 32, 0x63); }
    void verify() {
        std::vector<unsigned char> actual(bytes); TEST_CUDA(cuMemcpyDtoH(actual.data(), device.value, bytes));
        for (size_t i = 0; i < bytes; ++i) {
            auto expected = static_cast<unsigned char>(i < 16 || i >= bytes - 16 ? 0xa7 : 0x35);
            require(actual[i] == expected && host[i + 16] == expected, "external semaphore graph complete payload byte=" + std::to_string(i));
        }
        for (size_t i = 0; i < 16; ++i) require(host[i] == 0x63 && host[bytes + 16 + i] == 0x63, "pinned output guards");
    }
    CUgraphNode nodes(CUgraph graph, CUgraphNode dependency, bool runtime) {
        CUgraphNode fill{}, copy{}; CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context));
        if (runtime) {
            cudaMemsetParams p{}; p.dst = reinterpret_cast<void*>(uintptr_t(device.value + 16)); p.value = 0x35; p.elementSize = 1; p.width = bytes - 32; p.height = 1;
            auto dep = graph_external_cast<cudaGraphNode_t>(dependency); cudaGraphNode_t first{}, last{};
            graph_external_runtime(cudaGraphAddMemsetNode(&first, graph_external_cast<cudaGraph_t>(graph), &dep, 1, &p));
            graph_external_runtime(cudaGraphAddMemcpyNode1D(&last, graph_external_cast<cudaGraph_t>(graph), &first, 1, host + 16,
                reinterpret_cast<void*>(uintptr_t(device.value)), bytes, cudaMemcpyDeviceToHost)); copy = graph_external_cast<CUgraphNode>(last);
        } else {
            CUDA_MEMSET_NODE_PARAMS p{}; p.dst = device.value + 16; p.value = 0x35; p.elementSize = 1; p.width = bytes - 32; p.height = 1;
            TEST_CUDA(cuGraphAddMemsetNode(&fill, graph, &dependency, 1, &p, context));
            CUDA_MEMCPY3D read{}; read.srcMemoryType = CU_MEMORYTYPE_DEVICE; read.srcDevice = device.value;
            read.dstMemoryType = CU_MEMORYTYPE_HOST; read.dstHost = host + 16; read.WidthInBytes = read.srcPitch = read.dstPitch = bytes;
            read.srcHeight = read.dstHeight = read.Height = read.Depth = 1;
            TEST_CUDA(cuGraphAddMemcpyNode(&copy, graph, &fill, 1, &read, context));
        }
        return copy;
    }
};
#if defined(_WIN32)
struct graph_external_import {
    CUexternalSemaphore value{};
    graph_external_import(HANDLE handle, bool runtime) {
        graph_external_output<CUexternalSemaphore> result;
        if (runtime) {
            cudaExternalSemaphoreHandleDesc p{}; p.type = cudaExternalSemaphoreHandleTypeD3D12Fence; p.handle.win32.handle = handle;
            cudaExternalSemaphore_t semaphore{}; graph_external_runtime(cudaImportExternalSemaphore(&semaphore, &p)); result.value = graph_external_cast<CUexternalSemaphore>(semaphore);
        } else {
            CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC p{}; p.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE; p.handle.win32.handle = handle;
            TEST_CUDA(cuImportExternalSemaphore(&result.value, &p));
        }
        value = result.get();
    }
    ~graph_external_import() { if (value) cuDestroyExternalSemaphore(value); }
};
void graph_external_execute() {
    cudavmInteropCapabilities caps{1, sizeof(caps)}; require(cudavmGetInteropCapabilities(&caps) == CUDA_SUCCESS, "public interop capability query");
    require_capability(caps.semaphore_handle_types & (uint64_t(1) << CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE), "D3D12 fence import requires advertised support");
    for (bool runtime : {false, true}) {
        graph_external_fences external; graph_external_import input(external.input_handle, runtime), output(external.output_handle, !runtime);
        graph_external_storage storage; graph_external_stream stream; graph_external_graph graph(runtime);
        CUgraphNode wait{}, signal{};
        if (graph_external_case == "capture") {
            TEST_CUDA(cuStreamBeginCapture(stream.value, CU_STREAM_CAPTURE_MODE_GLOBAL));
            CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS incoming{}; incoming.params.fence.value = 1;
            TEST_CUDA(cuWaitExternalSemaphoresAsync(&input.value, &incoming, 1, stream.value));
            graph_external_runtime(cudaMemsetAsync(reinterpret_cast<void*>(uintptr_t(storage.device.value + 16)), 0x35, storage.bytes - 32, graph_external_cast<cudaStream_t>(stream.value)));
            TEST_CUDA(cuMemcpyDtoHAsync(storage.host + 16, storage.device.value, storage.bytes, stream.value));
            cudaExternalSemaphoreSignalParams outgoing{}; outgoing.params.fence.value = 2; auto handle = graph_external_cast<cudaExternalSemaphore_t>(output.value);
            graph_external_runtime(cudaSignalExternalSemaphoresAsync(&handle, &outgoing, 1, graph_external_cast<cudaStream_t>(stream.value)));
            TEST_CUDA(cuGraphDestroy(graph.value)); graph.value = nullptr; TEST_CUDA(cuStreamEndCapture(stream.value, &graph.value));
        } else {
            wait = graph_external_node<false>(graph.value, nullptr, input.value, 1, runtime);
            auto copy = storage.nodes(graph.value, wait, !runtime);
            signal = graph_external_node<true>(graph.value, copy, output.value, 2, !runtime);
            graph_external_query<false>(wait, input.value, 1); graph_external_query<true>(signal, output.value, 2);
        }
        graph_external_executable original(graph.value, runtime);
        std::unique_ptr<graph_external_executable> changed;
        if (graph_external_case == "parameters" || graph_external_case == "executable") {
            if (graph_external_case == "executable") changed = std::make_unique<graph_external_executable>(graph.value, !runtime);
            auto executable = changed ? changed->value : nullptr;
            graph_external_set<false>(wait, input.value, 3, !runtime, executable);
            graph_external_set<true>(signal, output.value, 4, runtime, executable);
            graph_external_query<false>(wait, input.value, changed ? 1 : 3); graph_external_query<true>(signal, output.value, changed ? 2 : 4);
            if (!changed) changed = std::make_unique<graph_external_executable>(graph.value, !runtime);
        }
        const auto run = [&](CUgraphExec executable, uint64_t incoming, uint64_t outgoing, bool retire) {
            storage.reset(); external.prepare(incoming, outgoing);
            if (runtime) graph_external_runtime(cudaGraphLaunch(graph_external_cast<cudaGraphExec_t>(executable), graph_external_cast<cudaStream_t>(stream.value)));
            else TEST_CUDA(cuGraphLaunch(executable, stream.value));
            // Only early completion is an error here. No minimum GPU speed is
            // assumed; the positive completion deadline starts after release.
            auto inspect_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
            do {
                require(cuStreamQuery(stream.value) == CUDA_ERROR_NOT_READY, "external input keeps graph stream pending");
                require(external.output->GetCompletedValue() < outgoing && external.completion->GetCompletedValue() < external.epoch, "independent consumer cannot pass an unsignaled graph output");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            } while (std::chrono::steady_clock::now() < inspect_until);
            if (retire) {
                TEST_CUDA(cuGraphDestroy(graph.value)); graph.value = nullptr;
                TEST_CUDA(cuGraphExecDestroy(original.value)); original.value = nullptr;
            }
            external.release(); external.await();
            TEST_CUDA(cuStreamSynchronize(stream.value));
            require(external.output->GetCompletedValue() == outgoing, "exact externally observed graph signal value"); storage.verify();
        };
        run(original.value, 1, 2, graph_external_case == "retirement");
        if (changed) run(changed->value, 3, 4, false);
    }
}
#else
void graph_external_execute() { require_capability(false, "D3D12 external graph fixture requires Windows"); }
#endif
void graph_external_validation() {
    graph_external_graph graph(false); graph_external_output<CUgraphNode> node;
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS signal{}; signal.numExtSems = 1;
    require(cuGraphAddExternalSemaphoresSignalNode(&node.value, graph.value, nullptr, 0, &signal) == CUDA_ERROR_INVALID_VALUE && !node.get(), "missing signal arrays preserve output");
    cudaExternalSemaphoreWaitNodeParams wait{}; wait.numExtSems = 1; cudaGraphNode_t result{};
    require(cudaGraphAddExternalSemaphoresWaitNode(&result, graph_external_cast<cudaGraph_t>(graph.value), nullptr, 0, &wait) == cudaErrorInvalidValue && !result, "missing Runtime wait arrays");
    CUexternalSemaphore invalid = graph_external_cast<CUexternalSemaphore>(uintptr_t(7)); CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS operation{};
    signal.extSemArray = &invalid; signal.paramsArray = &operation;
    require(cuGraphAddExternalSemaphoresSignalNode(&node.value, graph.value, nullptr, 0, &signal) == CUDA_ERROR_INVALID_HANDLE && !node.get(), "invalid semaphore preserves node output");
    require(cuGraphAddExternalSemaphoresSignalNode(nullptr, graph.value, nullptr, 0, &signal) == CUDA_ERROR_INVALID_VALUE, "null node output");
    TEST_CUDA(cuGraphAddEmptyNode(&node.value, graph.value, nullptr, 0));
    require(cuGraphExternalSemaphoresSignalNodeGetParams(node.get(), &signal) == CUDA_ERROR_INVALID_VALUE, "wrong node signal query");
    signal = {}; CUgraphNode empty{}; TEST_CUDA(cuGraphAddExternalSemaphoresSignalNode(&empty, graph.value, nullptr, 0, &signal));
    auto bad = signal; bad.numExtSems = 1;
    require(cuGraphExternalSemaphoresSignalNodeSetParams(empty, &bad) == CUDA_ERROR_INVALID_VALUE, "failed setter rejects missing arrays");
    TEST_CUDA(cuGraphExternalSemaphoresSignalNodeGetParams(empty, &signal)); require(!signal.numExtSems, "failed setter preserves valid empty descriptor");
    require(cuGraphExternalSemaphoresWaitNodeGetParams(empty, nullptr) == CUDA_ERROR_INVALID_VALUE, "null wait query output");
}
void graph_external_discovery() {
    static_assert(std::is_same_v<decltype(&cuGraphAddExternalSemaphoresSignalNode), PFN_cuGraphAddExternalSemaphoresSignalNode_v11020>);
    static_assert(std::is_same_v<decltype(&cuGraphAddExternalSemaphoresWaitNode), PFN_cuGraphAddExternalSemaphoresWaitNode_v11020>);
    for (auto name : {"cuGraphAddExternalSemaphoresSignalNode", "cuGraphAddExternalSemaphoresWaitNode", "cuGraphExternalSemaphoresSignalNodeGetParams", "cuGraphExternalSemaphoresWaitNodeGetParams",
                      "cuGraphExternalSemaphoresSignalNodeSetParams", "cuGraphExternalSemaphoresWaitNodeSetParams", "cuGraphExecExternalSemaphoresSignalNodeSetParams", "cuGraphExecExternalSemaphoresWaitNodeSetParams"}) {
        void* pointer{}; CUdriverProcAddressQueryResult query{};
        TEST_CUDA(cuGetProcAddress(name, &pointer, 11019, 0, &query)); require(!pointer && query == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "graph semaphore introduction boundary");
        TEST_CUDA(cuGetProcAddress(name, &pointer, 11020, 0, &query)); require(pointer && query == CU_GET_PROC_ADDRESS_SUCCESS, "graph semaphore introduction lookup");
        void* other{}; cudaDriverEntryPointQueryResult runtime{}; graph_external_runtime(cudaGetDriverEntryPointByVersion(name, &other, 11020, 0, &runtime));
        require(other == pointer && runtime == cudaDriverEntryPointSuccess, "Runtime discovers Driver graph semaphore procedure");
    }
}
void execute_graph_external() { if (graph_external_case == "validation") graph_external_validation(); else graph_external_execute(); }
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_test(argc, argv, cudavm::graph_external_discovery);
    if (argc == 3) { cudavm::graph_external_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_graph_external);
}
