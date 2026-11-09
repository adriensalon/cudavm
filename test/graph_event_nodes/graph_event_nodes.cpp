#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>

namespace cudavm {
void check_graph_event_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define GRAPH_EVENT_RUNTIME(call) ::cudavm::check_graph_event_runtime((call), #call)
struct graph_event_storage {
    device_buffer device{557}; void* output{};
    graph_event_storage() { CONFORMANCE_CUDA(cuMemAllocHost(&output, 601)); }
    ~graph_event_storage() { if (output) { cuCtxSynchronize(); cuMemFreeHost(output); } }
    void reset() { CONFORMANCE_CUDA(cuMemsetD8(device.value, 0x91, 557)); clear_output(); }
    void clear_output() { std::memset(output, 0xd3, 601); }
    void expect(bool produced, bool copied = true) const {
        std::vector<unsigned char> expected(601, 0xd3);
        if (copied) {
            std::fill_n(expected.begin() + 19, 557, 0x91);
            if (produced) std::fill_n(expected.begin() + 42, 389, 0x71);
        }
        require(std::memcmp(output, expected.data(), expected.size()) == 0, "graph event visibility, untouched bytes and readback guards");
    }
    void read(CUstream stream) const { CONFORMANCE_CUDA(cuMemcpyDtoHAsync(static_cast<unsigned char*>(output) + 19, device.value, 557, stream)); }
    void produce(CUstream stream) const { CONFORMANCE_CUDA(cuMemsetD8Async(device.value + 23, 0x71, 389, stream)); }
};
struct graph_event_stream {
    CUstream value{};
    graph_event_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~graph_event_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct graph_event_handle {
    CUevent value{};
    explicit graph_event_handle(bool runtime, bool timing = false) {
        const unsigned flags = timing ? 0 : CU_EVENT_DISABLE_TIMING;
        if (runtime) { cudaEvent_t event{}; GRAPH_EVENT_RUNTIME(cudaEventCreateWithFlags(&event, flags)); value = reinterpret_cast<CUevent>(event); }
        else CONFORMANCE_CUDA(cuEventCreate(&value, flags));
    }
    ~graph_event_handle() { if (value) cuEventDestroy(value); }
    void record(CUstream stream, bool runtime) const {
        if (runtime) GRAPH_EVENT_RUNTIME(cudaEventRecord(reinterpret_cast<cudaEvent_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuEventRecord(value, stream));
    }
    void destroy(bool runtime) {
        if (runtime) GRAPH_EVENT_RUNTIME(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(value)));
        else CONFORMANCE_CUDA(cuEventDestroy(value)); value = nullptr;
    }
};
struct graph_event_gate {
    std::mutex mutex; std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<graph_event_gate*>(argument);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "graph event gate entry deadline"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void expect_held() { std::lock_guard lock(mutex); require(entered && !released && !expired, "event record completes independently of later host gate"); }
};
struct graph_event_gate_cleanup {
    graph_event_gate& gate;
    ~graph_event_gate_cleanup() { gate.release(); cuCtxSynchronize(); }
};
struct graph_event_graph {
    CUgraph value{}; CUcontext context{};
    explicit graph_event_graph(bool runtime) {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        if (runtime) { cudaGraph_t graph{}; GRAPH_EVENT_RUNTIME(cudaGraphCreate(&graph, 0)); value = reinterpret_cast<CUgraph>(graph); }
        else CONFORMANCE_CUDA(cuGraphCreate(&value, 0));
    }
    ~graph_event_graph() { if (value) cuGraphDestroy(value); }
    CUgraphNode event(CUevent event, bool record, bool runtime, CUgraphNode dependency = nullptr) const {
        CUgraphNode node{};
        if (runtime) {
            auto dep = reinterpret_cast<cudaGraphNode_t>(dependency); cudaGraphNode_t result{};
            if (record) GRAPH_EVENT_RUNTIME(cudaGraphAddEventRecordNode(&result, reinterpret_cast<cudaGraph_t>(value), dependency ? &dep : nullptr, dependency ? 1 : 0, reinterpret_cast<cudaEvent_t>(event)));
            else GRAPH_EVENT_RUNTIME(cudaGraphAddEventWaitNode(&result, reinterpret_cast<cudaGraph_t>(value), dependency ? &dep : nullptr, dependency ? 1 : 0, reinterpret_cast<cudaEvent_t>(event)));
            node = reinterpret_cast<CUgraphNode>(result);
        } else if (record) CONFORMANCE_CUDA(cuGraphAddEventRecordNode(&node, value, dependency ? &dependency : nullptr, dependency ? 1 : 0, event));
        else CONFORMANCE_CUDA(cuGraphAddEventWaitNode(&node, value, dependency ? &dependency : nullptr, dependency ? 1 : 0, event));
        return node;
    }
    CUgraphNode fill(const graph_event_storage& storage) const {
        CUDA_MEMSET_NODE_PARAMS p{}; p.dst = storage.device.value + 23; p.value = 0x71; p.elementSize = 1; p.width = 389; p.height = 1;
        CUgraphNode node{}; CONFORMANCE_CUDA(cuGraphAddMemsetNode(&node, value, nullptr, 0, &p, context)); return node;
    }
    CUgraphNode read(const graph_event_storage& storage, CUgraphNode dependency) const {
        CUDA_MEMCPY3D p{}; p.srcMemoryType = CU_MEMORYTYPE_DEVICE; p.srcDevice = storage.device.value;
        p.dstMemoryType = CU_MEMORYTYPE_HOST; p.dstHost = static_cast<unsigned char*>(storage.output) + 19;
        p.srcPitch = p.dstPitch = p.WidthInBytes = 557; p.srcHeight = p.dstHeight = p.Height = p.Depth = 1;
        CUgraphNode node{}; CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&node, value, &dependency, 1, &p, context)); return node;
    }
};
struct graph_event_executable {
    CUgraphExec value{};
    graph_event_executable(CUgraph graph, bool runtime) {
        if (runtime) { cudaGraphExec_t result{}; GRAPH_EVENT_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), 0)); value = reinterpret_cast<CUgraphExec>(result); }
        else CONFORMANCE_CUDA(cuGraphInstantiate(&value, graph, 0));
    }
    ~graph_event_executable() { if (value) { cuCtxSynchronize(); cuGraphExecDestroy(value); } }
    void launch(CUstream stream, bool runtime) const {
        if (runtime) GRAPH_EVENT_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuGraphLaunch(value, stream));
    }
};
void graph_event_expect(CUgraphNode node, CUevent expected, bool record) {
    struct guarded_event { uint64_t prefix = 0x123456789abcdef0; CUevent value{}; uint64_t suffix = 0xfedcba9876543210; } driver;
    cudaEvent_t runtime{};
    if (record) {
        CONFORMANCE_CUDA(cuGraphEventRecordNodeGetEvent(node, &driver.value));
        GRAPH_EVENT_RUNTIME(cudaGraphEventRecordNodeGetEvent(reinterpret_cast<cudaGraphNode_t>(node), &runtime));
    } else {
        CONFORMANCE_CUDA(cuGraphEventWaitNodeGetEvent(node, &driver.value));
        GRAPH_EVENT_RUNTIME(cudaGraphEventWaitNodeGetEvent(reinterpret_cast<cudaGraphNode_t>(node), &runtime));
    }
    require(driver.value == expected && reinterpret_cast<CUevent>(runtime) == expected && driver.prefix == 0x123456789abcdef0
        && driver.suffix == 0xfedcba9876543210, "graph event selection and exact-width output guards");
}
void graph_event_set(CUgraphNode node, CUevent event, bool record, bool runtime, CUgraphExec executable = nullptr) {
    if (runtime) {
        auto n = reinterpret_cast<cudaGraphNode_t>(node); auto e = reinterpret_cast<cudaEvent_t>(event); auto x = reinterpret_cast<cudaGraphExec_t>(executable);
        if (executable && record) GRAPH_EVENT_RUNTIME(cudaGraphExecEventRecordNodeSetEvent(x, n, e));
        else if (executable) GRAPH_EVENT_RUNTIME(cudaGraphExecEventWaitNodeSetEvent(x, n, e));
        else if (record) GRAPH_EVENT_RUNTIME(cudaGraphEventRecordNodeSetEvent(n, e));
        else GRAPH_EVENT_RUNTIME(cudaGraphEventWaitNodeSetEvent(n, e));
    } else if (executable && record) CONFORMANCE_CUDA(cuGraphExecEventRecordNodeSetEvent(executable, node, event));
    else if (executable) CONFORMANCE_CUDA(cuGraphExecEventWaitNodeSetEvent(executable, node, event));
    else if (record) CONFORMANCE_CUDA(cuGraphEventRecordNodeSetEvent(node, event));
    else CONFORMANCE_CUDA(cuGraphEventWaitNodeSetEvent(node, event));
}
void graph_event_nodes_recording() {
    for (bool runtime : {false, true}) {
        graph_event_gate gate; graph_event_storage storage; graph_event_stream producer, consumer;
        graph_event_handle event(runtime); graph_event_graph graph(runtime);
        const auto fill = graph.fill(storage), read = graph.read(storage, fill), record = graph.event(event.value, true, runtime, read);
        graph_event_expect(record, event.value, true);
        CUDA_HOST_NODE_PARAMS p{graph_event_gate::call, &gate}; CUgraphNode host{};
        CONFORMANCE_CUDA(cuGraphAddHostNode(&host, graph.value, &record, 1, &p));
        graph_event_executable executable(graph.value, !runtime);
        graph_event_gate_cleanup cleanup{gate};
        storage.reset(); executable.launch(producer.value, runtime); gate.await();
        CONFORMANCE_CUDA(cuEventSynchronize(event.value)); gate.expect_held();
        require(cuStreamQuery(producer.value) == CUDA_ERROR_NOT_READY, "graph remains pending after its event record completes");
        storage.expect(true);
        // The readback is mandated before the held callback. Release it before
        // requesting any independent GPU progress, as host-function rules require.
        gate.release(); producer.synchronize(); storage.clear_output();
        if (runtime) GRAPH_EVENT_RUNTIME(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(consumer.value), reinterpret_cast<cudaEvent_t>(event.value), 0));
        else CONFORMANCE_CUDA(cuStreamWaitEvent(consumer.value, event.value, 0));
        storage.read(consumer.value); consumer.synchronize(); storage.expect(true);
        require(!gate.expired, "record-node gate deadline");
    }
}
void graph_event_waiting(bool retirement) {
    for (bool runtime : {false, true}) {
        graph_event_gate gate; graph_event_storage storage; graph_event_stream producer, consumer;
        graph_event_handle event(runtime); graph_event_graph graph(runtime);
        const auto wait = graph.event(event.value, false, runtime); graph.read(storage, wait);
        graph_event_expect(wait, event.value, false); graph_event_executable executable(graph.value, !runtime);
        graph_event_gate_cleanup cleanup{gate};
        storage.reset(); executable.launch(consumer.value, runtime); consumer.synchronize(); storage.expect(false);
        storage.clear_output();
        CONFORMANCE_CUDA(cuLaunchHostFunc(producer.value, graph_event_gate::call, &gate)); gate.await();
        storage.produce(producer.value); event.record(producer.value, !runtime);
        require(cuEventQuery(event.value) == CUDA_ERROR_NOT_READY, "external producer record is pending");
        executable.launch(consumer.value, !runtime);
        require(cuStreamQuery(consumer.value) == CUDA_ERROR_NOT_READY, "graph event wait gates readback"); storage.expect(false, false);
        if (retirement) event.destroy(runtime);
        gate.release(); consumer.synchronize(); producer.synchronize(); storage.expect(true);
        require(!gate.expired, "graph wait/retirement gate deadline");
    }
}
void graph_event_nodes_waiting() { graph_event_waiting(false); }
void graph_event_nodes_retirement() { graph_event_waiting(true); }
void graph_event_record_selection(bool runtime, bool executable_setter) {
    graph_event_storage storage; graph_event_stream stream;
    graph_event_handle first(runtime, true), second(!runtime, true); graph_event_graph graph(runtime);
    const auto fill = graph.fill(storage), record = graph.event(first.value, true, runtime, fill); graph.read(storage, record);
    graph_event_executable original(graph.value, !runtime);
    if (!executable_setter) graph_event_set(record, second.value, true, runtime);
    graph_event_executable changed(graph.value, runtime);
    if (executable_setter) graph_event_set(record, second.value, true, runtime, changed.value);
    graph_event_expect(record, executable_setter ? first.value : second.value, true);
    storage.reset();
    original.launch(stream.value, runtime); stream.synchronize(); storage.expect(true);
    float elapsed = -1;
    CONFORMANCE_CUDA(cuEventElapsedTime(&elapsed, first.value, first.value));
    require(elapsed == 0, "original executable records its retained event");
    require(cuEventElapsedTime(&elapsed, second.value, second.value) == CUDA_ERROR_INVALID_HANDLE,
        "replacement event remains unrecorded by the independent executable");
    storage.clear_output(); changed.launch(stream.value, !runtime); stream.synchronize(); storage.expect(true);
    CONFORMANCE_CUDA(cuEventElapsedTime(&elapsed, second.value, second.value));
    require(elapsed == 0, "changed executable records its replacement event");
}
void graph_event_wait_selection(bool runtime, bool executable_setter) {
    graph_event_gate gate; graph_event_storage storage; graph_event_stream producer, consumer;
    graph_event_handle first(runtime), second(!runtime); graph_event_graph graph(runtime);
    const auto wait = graph.event(first.value, false, runtime); graph.read(storage, wait);
    graph_event_executable original(graph.value, !runtime);
    if (!executable_setter) graph_event_set(wait, second.value, false, runtime);
    graph_event_executable changed(graph.value, runtime);
    if (executable_setter) graph_event_set(wait, second.value, false, runtime, changed.value);
    graph_event_expect(wait, executable_setter ? first.value : second.value, false);
    graph_event_gate_cleanup cleanup{gate};
    storage.reset();
    original.launch(consumer.value, runtime); consumer.synchronize(); storage.expect(false);
    CONFORMANCE_CUDA(cuLaunchHostFunc(producer.value, graph_event_gate::call, &gate)); gate.await();
    storage.produce(producer.value); second.record(producer.value, !runtime);
    storage.clear_output(); changed.launch(consumer.value, !runtime);
    require(cuStreamQuery(consumer.value) == CUDA_ERROR_NOT_READY, "replacement wait selects the pending external generation"); storage.expect(false, false);
    gate.release(); consumer.synchronize(); producer.synchronize(); storage.expect(true);
    require(!gate.expired, "wait event replacement gate deadline");
}
void graph_event_nodes_parameters() {
    for (bool runtime : {false, true}) { graph_event_record_selection(runtime, false); graph_event_wait_selection(runtime, false); }
}
void graph_event_nodes_executable() {
    for (bool runtime : {false, true}) { graph_event_record_selection(runtime, true); graph_event_wait_selection(runtime, true); }
}
void graph_event_nodes_validation() {
    graph_event_handle event(false); graph_event_graph graph(true);
    const auto record = graph.event(event.value, true, false), wait = graph.event(event.value, false, true);
    require(cuGraphEventRecordNodeGetEvent(record, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver record event output");
    require(cudaGraphEventRecordNodeGetEvent(reinterpret_cast<cudaGraphNode_t>(record), nullptr) == cudaErrorInvalidValue, "null Runtime record event output");
    require(cuGraphEventWaitNodeGetEvent(wait, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver wait event output");
    require(cudaGraphEventWaitNodeGetEvent(reinterpret_cast<cudaGraphNode_t>(wait), nullptr) == cudaErrorInvalidValue, "null Runtime wait event output");
    CUevent driver{}; cudaEvent_t runtime{};
    require(cuGraphEventRecordNodeGetEvent(wait, &driver) == CUDA_ERROR_INVALID_VALUE, "wait node is not a record node");
    require(cuGraphEventWaitNodeGetEvent(record, &driver) == CUDA_ERROR_INVALID_VALUE, "record node is not a wait node");
    require(cudaGraphEventRecordNodeGetEvent(reinterpret_cast<cudaGraphNode_t>(wait), &runtime) == cudaErrorInvalidValue, "Runtime rejects record query on wait node");
    require(cudaGraphEventWaitNodeGetEvent(reinterpret_cast<cudaGraphNode_t>(record), &runtime) == cudaErrorInvalidValue, "Runtime rejects wait query on record node");
    require(cuGraphEventRecordNodeSetEvent(wait, event.value) == CUDA_ERROR_INVALID_VALUE, "record setter rejects wait node");
    require(cuGraphEventWaitNodeSetEvent(record, event.value) == CUDA_ERROR_INVALID_VALUE, "wait setter rejects record node");
    require(cudaGraphEventRecordNodeSetEvent(reinterpret_cast<cudaGraphNode_t>(wait), reinterpret_cast<cudaEvent_t>(event.value)) == cudaErrorInvalidValue, "Runtime record setter rejects wait node");
    require(cudaGraphEventWaitNodeSetEvent(reinterpret_cast<cudaGraphNode_t>(record), reinterpret_cast<cudaEvent_t>(event.value)) == cudaErrorInvalidValue, "Runtime wait setter rejects record node");
    require(cuGraphAddEventRecordNode(nullptr, graph.value, nullptr, 0, event.value) == CUDA_ERROR_INVALID_VALUE, "null record-node output");
    require(cudaGraphAddEventWaitNode(nullptr, reinterpret_cast<cudaGraph_t>(graph.value), nullptr, 0, reinterpret_cast<cudaEvent_t>(event.value)) == cudaErrorInvalidValue, "null Runtime wait-node output");
    graph_event_expect(record, event.value, true); graph_event_expect(wait, event.value, false);
}
std::string graph_event_case = "all";
void execute_graph_event_nodes() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (graph_event_case == "all" || graph_event_case == name) { execute(); selected = true; } };
    run("recording", graph_event_nodes_recording); run("waiting", graph_event_nodes_waiting); run("parameters", graph_event_nodes_parameters);
    run("executable", graph_event_nodes_executable); run("retirement", graph_event_nodes_retirement); run("validation", graph_event_nodes_validation);
    require(selected, "unknown graph_event_nodes case: " + graph_event_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::graph_event_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_graph_event_nodes);
}
