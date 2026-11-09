#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>

namespace cudavm {
void check_capture_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define CAPTURE_RUNTIME(call) ::cudavm::check_capture_runtime((call), #call)
template<class T> struct capture_guarded {
    uint64_t prefix = 0x0123456789abcdef; T value{}; uint64_t suffix = 0xfedcba9876543210;
    void check() const { require(prefix == 0x0123456789abcdef && suffix == 0xfedcba9876543210, "capture output guards"); }
};
struct capture_stream {
    CUstream value{};
    explicit capture_stream(unsigned flags = CU_STREAM_NON_BLOCKING) { CONFORMANCE_CUDA(cuStreamCreate(&value, flags)); }
    ~capture_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct capture_event {
    CUevent value{};
    capture_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~capture_event() { if (value) cuEventDestroy(value); }
};
struct capture_graph {
    CUgraph value{}; CUgraphExec executable{};
    ~capture_graph() { if (executable) cuGraphExecDestroy(executable); if (value) cuGraphDestroy(value); }
    void instantiate(bool runtime) {
        if (runtime) { cudaGraphExec_t result{}; CAPTURE_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(value), 0)); executable = reinterpret_cast<CUgraphExec>(result); }
        else CONFORMANCE_CUDA(cuGraphInstantiate(&executable, value, 0));
    }
    void retire() { CONFORMANCE_CUDA(cuGraphDestroy(value)); value = nullptr; }
    void run(CUstream stream, bool runtime) const {
        if (runtime) CAPTURE_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuGraphLaunch(executable, stream));
        CONFORMANCE_CUDA(cuStreamSynchronize(stream));
    }
};
struct capture_snapshot { CUstreamCaptureStatus status{}; cuuint64_t id{}; CUgraph graph{}; std::vector<CUgraphNode> dependencies; };
struct capture_route {
    bool runtime;
    static cudaStream_t stream(CUstream value) { return reinterpret_cast<cudaStream_t>(value); }
    static void* pointer(CUdeviceptr value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }
    CUresult begin(CUstream value, CUstreamCaptureMode mode) const {
        return runtime ? static_cast<CUresult>(cudaStreamBeginCapture(stream(value), static_cast<cudaStreamCaptureMode>(mode))) : cuStreamBeginCapture(value, mode);
    }
    CUresult end(CUstream value, CUgraph* graph) const {
        if (!runtime) return cuStreamEndCapture(value, graph);
        cudaGraph_t output = reinterpret_cast<cudaGraph_t>(*graph);
        const auto result = cudaStreamEndCapture(stream(value), &output); *graph = reinterpret_cast<CUgraph>(output); return static_cast<CUresult>(result);
    }
    CUstreamCaptureStatus status(CUstream value) const {
        capture_guarded<CUstreamCaptureStatus> output;
        if (runtime) {
            capture_guarded<cudaStreamCaptureStatus> result;
            CAPTURE_RUNTIME(cudaStreamIsCapturing(stream(value), &result.value)); result.check(); output.value = static_cast<CUstreamCaptureStatus>(result.value);
        } else CONFORMANCE_CUDA(cuStreamIsCapturing(value, &output.value));
        output.check(); return output.value;
    }
    capture_snapshot info(CUstream value) const {
        capture_guarded<CUstreamCaptureStatus> status; capture_guarded<cuuint64_t> id; capture_guarded<CUgraph> graph;
        capture_guarded<const CUgraphNode*> nodes; capture_guarded<const CUgraphEdgeData*> edges; capture_guarded<size_t> count;
        if (runtime) {
            capture_guarded<cudaStreamCaptureStatus> rt_status; capture_guarded<unsigned long long> rt_id;
            capture_guarded<cudaGraph_t> rt_graph; capture_guarded<const cudaGraphNode_t*> rt_nodes; capture_guarded<const cudaGraphEdgeData*> rt_edges;
            CAPTURE_RUNTIME(cudaStreamGetCaptureInfo(stream(value), &rt_status.value, &rt_id.value, &rt_graph.value, &rt_nodes.value, &rt_edges.value, &count.value));
            rt_status.check(); rt_id.check(); rt_graph.check(); rt_nodes.check(); rt_edges.check();
            status.value = static_cast<CUstreamCaptureStatus>(rt_status.value); id.value = rt_id.value; graph.value = reinterpret_cast<CUgraph>(rt_graph.value);
            nodes.value = reinterpret_cast<const CUgraphNode*>(rt_nodes.value); edges.value = reinterpret_cast<const CUgraphEdgeData*>(rt_edges.value);
        } else CONFORMANCE_CUDA(cuStreamGetCaptureInfo(value, &status.value, &id.value, &graph.value, &nodes.value, &edges.value, &count.value));
        status.check(); id.check(); graph.check(); nodes.check(); edges.check(); count.check();
        capture_snapshot result{status.value};
        // Other outputs are unspecified outside active capture.
        if (result.status == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
            result.id = id.value; result.graph = graph.value;
            require(result.id && result.graph && (!count.value || (nodes.value && edges.value)), "active capture identity and frontier outputs");
            for (size_t i = 0; i < count.value; ++i) {
                const auto& edge = edges.value[i];
                require(edge.from_port == 0 && edge.to_port == 0 && edge.type == CU_GRAPH_DEPENDENCY_TYPE_DEFAULT,
                    "ordinary captured operations expose full-completion dependencies");
                result.dependencies.push_back(nodes.value[i]);
            }
        }
        return result;
    }
    void host(CUstream value, CUhostFn function, void* data) const {
        if (runtime) CAPTURE_RUNTIME(cudaLaunchHostFunc(stream(value), function, data)); else CONFORMANCE_CUDA(cuLaunchHostFunc(value, function, data));
    }
    void fill(CUstream value, CUdeviceptr address, unsigned char byte, size_t size) const {
        if (runtime) CAPTURE_RUNTIME(cudaMemsetAsync(pointer(address), byte, size, stream(value))); else CONFORMANCE_CUDA(cuMemsetD8Async(address, byte, size, value));
    }
    void upload(CUstream value, CUdeviceptr device, const void* host, size_t size) const {
        if (runtime) CAPTURE_RUNTIME(cudaMemcpyAsync(pointer(device), host, size, cudaMemcpyHostToDevice, stream(value))); else CONFORMANCE_CUDA(cuMemcpyHtoDAsync(device, host, size, value));
    }
    void copy(CUstream value, CUdeviceptr destination, CUdeviceptr source, size_t size) const {
        if (runtime) CAPTURE_RUNTIME(cudaMemcpyAsync(pointer(destination), pointer(source), size, cudaMemcpyDeviceToDevice, stream(value))); else CONFORMANCE_CUDA(cuMemcpyDtoDAsync(destination, source, size, value));
    }
    void read(CUstream value, void* host, CUdeviceptr device, size_t size) const {
        if (runtime) CAPTURE_RUNTIME(cudaMemcpyAsync(host, pointer(device), size, cudaMemcpyDeviceToHost, stream(value))); else CONFORMANCE_CUDA(cuMemcpyDtoHAsync(host, device, size, value));
    }
    void record(CUevent event, CUstream value) const {
        if (runtime) CAPTURE_RUNTIME(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), stream(value))); else CONFORMANCE_CUDA(cuEventRecord(event, value));
    }
    CUresult wait(CUstream value, CUevent event) const {
        return runtime ? static_cast<CUresult>(cudaStreamWaitEvent(stream(value), reinterpret_cast<cudaEvent_t>(event), 0)) : cuStreamWaitEvent(value, event, 0);
    }
    CUresult update(CUstream value, std::vector<CUgraphNode> nodes, unsigned flags) const {
        if (!runtime) return cuStreamUpdateCaptureDependencies(value, nodes.data(), nullptr, nodes.size(), flags);
        std::vector<cudaGraphNode_t> converted; for (auto node : nodes) converted.push_back(reinterpret_cast<cudaGraphNode_t>(node));
        return static_cast<CUresult>(cudaStreamUpdateCaptureDependencies(stream(value), converted.data(), nullptr, converted.size(), flags));
    }
};
struct capture_session {
    capture_route route; CUstream stream; bool active = true;
    capture_session(capture_route selected, CUstream value, CUstreamCaptureMode mode) : route(selected), stream(value) { CONFORMANCE_CUDA(route.begin(stream, mode)); }
    ~capture_session() { if (active) { CUgraph discarded{}; if (route.end(stream, &discarded) == CUDA_SUCCESS && discarded) cuGraphDestroy(discarded); } }
    CUresult end(CUgraph* output) { const auto result = route.end(stream, output); active = false; return result; }
};
struct capture_storage {
    static constexpr size_t bytes = 613, host_bytes = 659;
    device_buffer source{bytes}, destination{bytes}; void* input{}; void* output{}; unsigned calls = 0;
    capture_storage() {
        CONFORMANCE_CUDA(cuMemAllocHost(&input, host_bytes));
        try { CONFORMANCE_CUDA(cuMemAllocHost(&output, host_bytes)); } catch (...) { cuMemFreeHost(input); throw; }
    }
    ~capture_storage() { cuCtxSynchronize(); if (output) cuMemFreeHost(output); if (input) cuMemFreeHost(input); }
    static void CUDA_CB callback(void* data) { ++static_cast<capture_storage*>(data)->calls; }
    std::vector<unsigned char> input_bytes(unsigned seed) const {
        std::vector<unsigned char> result(host_bytes, 0xc3);
        for (size_t i = 17; i < 17 + 419; ++i) result[i] = static_cast<unsigned char>(seed + 37 * i + (i >> 3));
        return result;
    }
    void reset(unsigned seed) {
        CONFORMANCE_CUDA(cuMemsetD8(source.value, 0xa7, bytes)); CONFORMANCE_CUDA(cuMemsetD8(destination.value, 0x91, bytes));
        const auto data = input_bytes(seed); std::memcpy(input, data.data(), data.size()); std::memset(output, 0xd3, host_bytes); calls = 0;
    }
    void first(const capture_route& route, CUstream stream) { route.upload(stream, source.value + 31, static_cast<unsigned char*>(input) + 17, 419); }
    void middle(const capture_route& route, CUstream stream) {
        route.copy(stream, destination.value + 13, source.value + 31, 419); route.fill(stream, destination.value + 87, 0x53, 97);
    }
    void last(const capture_route& route, CUstream stream) {
        route.read(stream, static_cast<unsigned char*>(output) + 19, destination.value, bytes); route.host(stream, callback, this);
    }
    void expect(unsigned seed, bool executed) const {
        const auto host_input = input_bytes(seed); std::vector<unsigned char> a(bytes, 0xa7), b(bytes, 0x91), host_output(host_bytes, 0xd3), actual(bytes);
        if (executed) {
            std::copy_n(host_input.begin() + 17, 419, a.begin() + 31); std::copy_n(host_input.begin() + 17, 419, b.begin() + 13);
            std::fill_n(b.begin() + 87, 97, 0x53); std::copy(b.begin(), b.end(), host_output.begin() + 19);
        }
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), source.value, bytes)); require(actual == a, "capture source bytes and device guards");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), destination.value, bytes)); require(actual == b, "capture destination bytes and device guards");
        require(std::memcmp(input, host_input.data(), host_bytes) == 0 && std::memcmp(output, host_output.data(), host_bytes) == 0 && calls == unsigned(executed),
            "capture/replay exact host payload, guards, input preservation and callback count");
    }
};
void capture_transitions() {
    std::vector<cuuint64_t> identities;
    for (bool runtime : {false, true}) for (auto mode : {CU_STREAM_CAPTURE_MODE_GLOBAL, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CU_STREAM_CAPTURE_MODE_RELAXED}) {
        capture_route route{runtime}; capture_stream stream; capture_graph graph;
        require(route.status(stream.value) == CU_STREAM_CAPTURE_STATUS_NONE && route.info(stream.value).status == CU_STREAM_CAPTURE_STATUS_NONE, "initial none state");
        capture_session session(route, stream.value, mode); const auto first = route.info(stream.value), repeated = route.info(stream.value);
        require(first.status == CU_STREAM_CAPTURE_STATUS_ACTIVE && first.dependencies.empty() && first.id == repeated.id && first.graph == repeated.graph,
            "empty active capture has stable identity, graph and empty frontier");
        require(std::find(identities.begin(), identities.end(), first.id) == identities.end(), "capture identity is process unique"); identities.push_back(first.id);
        CONFORMANCE_CUDA(session.end(&graph.value)); require(graph.value == first.graph && route.status(stream.value) == CU_STREAM_CAPTURE_STATUS_NONE, "end returns queried graph and clears status");
        size_t count = 99; CONFORMANCE_CUDA(cuGraphGetNodes(graph.value, nullptr, &count)); require(count == 0, "empty capture has no nodes");
        graph.instantiate(!runtime); graph.run(stream.value, runtime);
    }
}
void capture_replay(bool joined) {
    for (bool runtime : {false, true}) for (auto mode : {CU_STREAM_CAPTURE_MODE_GLOBAL, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CU_STREAM_CAPTURE_MODE_RELAXED}) {
        capture_route route{runtime}; capture_storage storage; capture_stream origin, branch, launch; capture_event fork, join; capture_graph graph;
        storage.reset(29); capture_session session(route, origin.value, mode); const auto initial = route.info(origin.value);
        storage.first(route, origin.value); const auto first = route.info(origin.value);
        require(first.id == initial.id && first.graph == initial.graph && first.dependencies.size() == 1, "upload advances capture frontier");
        if (joined) {
            route.record(fork.value, origin.value); CONFORMANCE_CUDA(route.wait(branch.value, fork.value));
            const auto member = route.info(branch.value); require(member.id == initial.id && member.graph == initial.graph && member.dependencies == first.dependencies, "event wait joins shared capture and frontier");
            storage.middle(route, branch.value); route.record(join.value, branch.value); CONFORMANCE_CUDA(route.wait(origin.value, join.value));
        } else storage.middle(route, origin.value);
        storage.last(route, origin.value); const auto last = route.info(origin.value);
        require(last.dependencies.size() == 1 && last.dependencies != first.dependencies, "final host operation advances capture frontier");
        CONFORMANCE_CUDA(session.end(&graph.value)); require(graph.value == initial.graph && route.status(branch.value) == CU_STREAM_CAPTURE_STATUS_NONE, "end detaches every member");
        size_t count{}; CONFORMANCE_CUDA(cuGraphGetNodes(graph.value, nullptr, &count)); require(count == 5, "ordinary event records/waits encode edges without event nodes");
        storage.expect(29, false); graph.instantiate(!runtime); graph.retire();
        // Dependency-only capture events need not survive graph execution.
        CONFORMANCE_CUDA(cuEventDestroy(fork.value)); fork.value = nullptr; CONFORMANCE_CUDA(cuEventDestroy(join.value)); join.value = nullptr;
        for (unsigned seed : {0u, 71u, 255u, 0xffffffffu}) { storage.reset(seed); graph.run(launch.value, runtime); storage.expect(seed, true); }
    }
}
void capture_dependencies() {
    for (bool runtime : {false, true}) {
        capture_route route{runtime}; capture_storage storage; capture_stream stream; capture_graph graph; storage.reset(43);
        CONFORMANCE_CUDA(cuGraphCreate(&graph.value, 0)); CUgraphNode root{}; CONFORMANCE_CUDA(cuGraphAddEmptyNode(&root, graph.value, nullptr, 0));
        if (runtime) { auto dependency = reinterpret_cast<cudaGraphNode_t>(root); CAPTURE_RUNTIME(cudaStreamBeginCaptureToGraph(capture_route::stream(stream.value), reinterpret_cast<cudaGraph_t>(graph.value), &dependency, nullptr, 1, cudaStreamCaptureModeThreadLocal)); }
        else CONFORMANCE_CUDA(cuStreamBeginCaptureToGraph(stream.value, graph.value, &root, nullptr, 1, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        // The source graph remains caller-owned even if capture is aborted.
        struct cleanup { capture_route route; CUstream stream; bool active = true; ~cleanup() { if (active) { CUgraph output{}; route.end(stream, &output); } } } end{route, stream.value};
        auto initial = route.info(stream.value); require(initial.graph == graph.value && initial.dependencies == std::vector<CUgraphNode>{root}, "capture-to-graph starts at caller dependencies");
        storage.first(route, stream.value); const auto upload = route.info(stream.value); CUgraphNode inserted{};
        CONFORMANCE_CUDA(cuGraphAddEmptyNode(&inserted, graph.value, upload.dependencies.data(), upload.dependencies.size()));
        CONFORMANCE_CUDA(route.update(stream.value, {inserted}, CU_STREAM_SET_CAPTURE_DEPENDENCIES));
        require(route.info(stream.value).dependencies == std::vector<CUgraphNode>{inserted}, "SET replaces capture frontier");
        CONFORMANCE_CUDA(route.update(stream.value, {root}, CU_STREAM_ADD_CAPTURE_DEPENDENCIES));
        auto added = route.info(stream.value); require(added.dependencies.size() == 2 && std::find(added.dependencies.begin(), added.dependencies.end(), root) != added.dependencies.end(), "ADD extends frontier");
        // Inputs are copied: the temporary caller vectors above have already died.
        storage.middle(route, stream.value); storage.last(route, stream.value); CUgraph output{};
        CONFORMANCE_CUDA(route.end(stream.value, &output)); end.active = false; require(output == graph.value, "capture-to-graph returns original caller graph");
        graph.instantiate(!runtime); storage.expect(43, false); graph.run(stream.value, runtime); storage.expect(43, true);
    }
}
void capture_invalidation() {
    for (bool runtime : {false, true}) for (bool query : {false, true}) {
        capture_route route{runtime}; capture_storage storage; capture_stream stream; storage.reset(61);
        capture_session session(route, stream.value, CU_STREAM_CAPTURE_MODE_GLOBAL); storage.first(route, stream.value); route.host(stream.value, capture_storage::callback, &storage);
        const auto invalid = runtime ? static_cast<CUresult>(query ? cudaStreamQuery(capture_route::stream(stream.value)) : cudaStreamSynchronize(capture_route::stream(stream.value)))
                                     : (query ? cuStreamQuery(stream.value) : cuStreamSynchronize(stream.value));
        require(invalid == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "query/synchronize of active capture is prohibited");
        require(route.status(stream.value) == CU_STREAM_CAPTURE_STATUS_INVALIDATED && route.info(stream.value).status == CU_STREAM_CAPTURE_STATUS_INVALIDATED, "invalidated capture remains queryable");
        require(route.update(stream.value, {}, 0) == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, "invalidated stream rejects further work");
        capture_guarded<CUgraph> output; output.value = reinterpret_cast<CUgraph>(uintptr_t{1});
        require(session.end(&output.value) == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED && output.value == nullptr, "invalidated end rejects graph and clears output"); output.check();
        require(route.status(stream.value) == CU_STREAM_CAPTURE_STATUS_NONE, "failed end releases capture state"); storage.expect(61, false);
        capture_graph graph; capture_session again(route, stream.value, CU_STREAM_CAPTURE_MODE_GLOBAL);
        storage.first(route, stream.value); storage.middle(route, stream.value); storage.last(route, stream.value); CONFORMANCE_CUDA(again.end(&graph.value));
        graph.instantiate(!runtime); graph.run(stream.value, runtime); storage.expect(61, true);
    }
}
void capture_isolation() {
    for (bool runtime : {false, true}) for (unsigned scenario : {0u, 1u, 2u}) {
        capture_route route{runtime}; capture_stream origin, branch; capture_event event; capture_storage storage; storage.reset(83);
        if (scenario == 0) { route.record(event.value, branch.value); CONFORMANCE_CUDA(cuStreamSynchronize(branch.value)); }
        capture_session first(route, origin.value, CU_STREAM_CAPTURE_MODE_GLOBAL); route.host(origin.value, capture_storage::callback, &storage);
        if (scenario == 0) {
            require(route.wait(origin.value, event.value) == CUDA_ERROR_STREAM_CAPTURE_ISOLATION, "capture cannot depend on uncaptured event");
        } else {
            route.record(event.value, origin.value);
            if (scenario == 1) {
                CONFORMANCE_CUDA(route.wait(branch.value, event.value)); route.host(branch.value, capture_storage::callback, &storage);
            } else {
                capture_session second(route, branch.value, CU_STREAM_CAPTURE_MODE_GLOBAL);
                require(route.wait(branch.value, event.value) == CUDA_ERROR_STREAM_CAPTURE_MERGE, "independent captures cannot merge");
                require(route.status(origin.value) == CU_STREAM_CAPTURE_STATUS_INVALIDATED && route.status(branch.value) == CU_STREAM_CAPTURE_STATUS_INVALIDATED, "merge invalidates both sequences");
                CUgraph discarded{}; require(second.end(&discarded) == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED && !discarded, "second invalid capture discarded");
            }
        }
        CUgraph discarded{}; require(first.end(&discarded) == (scenario == 1 ? CUDA_ERROR_STREAM_CAPTURE_UNJOINED : CUDA_ERROR_STREAM_CAPTURE_INVALIDATED) && !discarded,
            "isolated/merged/unjoined capture returns no executable graph");
        require(route.status(origin.value) == CU_STREAM_CAPTURE_STATUS_NONE && route.status(branch.value) == CU_STREAM_CAPTURE_STATUS_NONE, "failed end detaches all streams"); storage.expect(83, false);
        capture_graph valid; capture_session again(route, branch.value, CU_STREAM_CAPTURE_MODE_GLOBAL); CONFORMANCE_CUDA(again.end(&valid.value)); valid.instantiate(!runtime); valid.run(origin.value, runtime);
    }
}
void capture_legacy() {
    for (bool runtime : {false, true}) {
        capture_route route{runtime}; capture_stream blocking(CU_STREAM_DEFAULT); capture_graph graph;
        require(route.begin(CU_STREAM_LEGACY, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "legacy capture unsupported");
        capture_session session(route, blocking.value, CU_STREAM_CAPTURE_MODE_GLOBAL);
        CUstreamCaptureStatus driver_status{}; cudaStreamCaptureStatus runtime_status{};
        require(cuStreamIsCapturing(CU_STREAM_LEGACY, &driver_status) == CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, "Driver legacy status detects implicit dependency");
        require(route.status(blocking.value) == CU_STREAM_CAPTURE_STATUS_ACTIVE, "legacy status query does not invalidate blocking capture");
        require(cudaStreamIsCapturing(cudaStreamLegacy, &runtime_status) == cudaErrorStreamCaptureImplicit, "Runtime legacy status detects implicit dependency");
        require(cuStreamGetCaptureInfo(CU_STREAM_LEGACY, &driver_status, nullptr, nullptr, nullptr, nullptr, nullptr) == CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
            "Driver legacy capture-info query detects implicit dependency");
        require(cudaStreamGetCaptureInfo(cudaStreamLegacy, &runtime_status) == cudaErrorStreamCaptureImplicit,
            "Runtime legacy capture-info query detects implicit dependency");
        require(route.info(blocking.value).status == CU_STREAM_CAPTURE_STATUS_ACTIVE, "Runtime legacy query also preserves capture");
        CONFORMANCE_CUDA(session.end(&graph.value)); graph.instantiate(!runtime); graph.run(blocking.value, runtime);
        capture_graph per_thread; capture_session pt(route, CU_STREAM_PER_THREAD, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL); CONFORMANCE_CUDA(pt.end(&per_thread.value));
        per_thread.instantiate(!runtime); per_thread.run(CU_STREAM_PER_THREAD, runtime);
    }
}
void capture_threading() {
    for (bool runtime : {false, true}) {
        capture_route route{runtime}; capture_stream stream; capture_storage storage; capture_graph graph; storage.reset(107);
        CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        capture_session session(route, stream.value, CU_STREAM_CAPTURE_MODE_RELAXED); storage.first(route, stream.value); storage.middle(route, stream.value); storage.last(route, stream.value);
        CUresult result = CUDA_ERROR_UNKNOWN; std::exception_ptr failure;
        std::thread worker([&] { try { CONFORMANCE_CUDA(cuCtxSetCurrent(context)); result = session.end(&graph.value); CONFORMANCE_CUDA(cuCtxSetCurrent(nullptr)); } catch (...) { failure = std::current_exception(); } });
        worker.join(); if (failure) std::rethrow_exception(failure); CONFORMANCE_CUDA(result);
        require(route.status(stream.value) == CU_STREAM_CAPTURE_STATUS_NONE, "relaxed capture can end on a different bound host thread");
        graph.instantiate(!runtime); graph.run(stream.value, runtime); storage.expect(107, true);
        // Exchange mode is thread-local and interoperable across both public facades.
        CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED; CONFORMANCE_CUDA(cuThreadExchangeStreamCaptureMode(&mode));
        const auto original = mode; CUstreamCaptureMode observed = CU_STREAM_CAPTURE_MODE_GLOBAL;
        std::thread independent([&] { observed = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL; result = cuThreadExchangeStreamCaptureMode(&observed); }); independent.join(); CONFORMANCE_CUDA(result);
        require(observed == CU_STREAM_CAPTURE_MODE_GLOBAL, "new host thread starts with independent global capture mode");
        cudaStreamCaptureMode rt_mode = cudaStreamCaptureModeThreadLocal; CAPTURE_RUNTIME(cudaThreadExchangeStreamCaptureMode(&rt_mode));
        require(rt_mode == cudaStreamCaptureModeRelaxed, "Runtime exchange observes Driver thread mode");
        mode = original; CONFORMANCE_CUDA(cuThreadExchangeStreamCaptureMode(&mode)); require(mode == CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, "Driver restores previous mode from Runtime exchange");
    }
}
void capture_validation() {
    for (bool runtime : {false, true}) {
        capture_route route{runtime}; capture_stream stream;
        require(route.begin(stream.value, static_cast<CUstreamCaptureMode>(99)) == CUDA_ERROR_INVALID_VALUE, "invalid capture mode");
        require(route.update(stream.value, {}, 0) == CUDA_ERROR_ILLEGAL_STATE, "dependency update requires capture");
        require(cuStreamIsCapturing(stream.value, nullptr) == CUDA_ERROR_INVALID_VALUE && cudaStreamIsCapturing(capture_route::stream(stream.value), nullptr) == cudaErrorInvalidValue, "null capture status outputs");
        require(cuThreadExchangeStreamCaptureMode(nullptr) == CUDA_ERROR_INVALID_VALUE && cudaThreadExchangeStreamCaptureMode(nullptr) == cudaErrorInvalidValue, "null exchange mode pointers");
        capture_graph graph; capture_session session(route, stream.value, CU_STREAM_CAPTURE_MODE_GLOBAL); const auto initial = route.info(stream.value);
        require(route.begin(stream.value, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_ERROR_ILLEGAL_STATE, "capture cannot begin twice on the same stream");
        require(route.update(stream.value, {}, 0xff) == CUDA_ERROR_INVALID_VALUE, "invalid capture dependency flags");
        require(cuStreamEndCapture(stream.value, nullptr) == CUDA_ERROR_INVALID_VALUE && cudaStreamEndCapture(capture_route::stream(stream.value), nullptr) == cudaErrorInvalidValue, "null end output is rejected without ending capture");
        const auto after = route.info(stream.value); require(after.status == CU_STREAM_CAPTURE_STATUS_ACTIVE && initial.id == after.id && initial.graph == after.graph && after.dependencies.empty(), "argument rejections preserve active capture");
        CONFORMANCE_CUDA(session.end(&graph.value)); graph.instantiate(!runtime); graph.run(stream.value, runtime);
    }
}
std::string capture_case = "all";
void execute_stream_capture() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (capture_case == "all" || capture_case == name) { std::cout << "case=" << name << std::endl; execute(); selected = true; } };
    run("transitions", capture_transitions); run("replay", [] { capture_replay(false); }); run("joins", [] { capture_replay(true); });
    run("dependencies", capture_dependencies); run("invalidation", capture_invalidation); run("isolation", capture_isolation);
    run("legacy", capture_legacy); run("threading", capture_threading); run("validation", capture_validation);
    require(selected, "unknown stream_capture case: " + capture_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::capture_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_stream_capture);
}
