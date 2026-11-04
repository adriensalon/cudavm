#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>

namespace cudavm {
void check_lifetime_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define LIFETIME_RUNTIME(call) ::cudavm::check_lifetime_runtime((call), #call)
struct lifetime_state {
    std::mutex mutex; std::condition_variable changed;
    unsigned calls = 0, completions = 0;
    uint64_t tag = 0;
    bool invalid = false;
    static void CUDA_CB publish(void* argument) {
        auto& state = *static_cast<lifetime_state*>(argument);
        std::lock_guard lock(state.mutex); ++state.completions;
    }
};
struct lifetime_payload {
    uint64_t prefix = 0x123456789abcdef0;
    std::shared_ptr<lifetime_state> state;
    uint64_t tag = 0, suffix = 0xfedcba9876543210;
};
struct lifetime_object {
    CUuserObject value{};
    unsigned caller_count;
    std::shared_ptr<lifetime_state> state = std::make_shared<lifetime_state>();
    static void CUDA_CB destroy(void* argument) {
        std::unique_ptr<lifetime_payload> payload(static_cast<lifetime_payload*>(argument));
        auto state = payload->state;
        {
            std::lock_guard lock(state->mutex);
            state->invalid |= payload->prefix != 0x123456789abcdef0 || payload->suffix != 0xfedcba9876543210
                || payload->tag != state->tag || state->completions != 1;
            ++state->calls;
        }
        state->changed.notify_all();
    }
    lifetime_object(unsigned count, bool runtime, uint64_t tag) : caller_count(count) {
        state->tag = tag;
        auto payload = std::make_unique<lifetime_payload>(); payload->state = state; payload->tag = tag;
        if (runtime) {
            cudaUserObject_t object{};
            LIFETIME_RUNTIME(cudaUserObjectCreate(&object, payload.get(), destroy, count, cudaUserObjectNoDestructorSync));
            value = reinterpret_cast<CUuserObject>(object);
        } else CONFORMANCE_CUDA(cuUserObjectCreate(&value, payload.get(), destroy, count, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC));
        payload.release();
    }
    ~lifetime_object() { if (caller_count) cuUserObjectRelease(value, caller_count); }
    void retain(unsigned count, bool runtime) {
        if (runtime) LIFETIME_RUNTIME(cudaUserObjectRetain(reinterpret_cast<cudaUserObject_t>(value), count));
        else CONFORMANCE_CUDA(cuUserObjectRetain(value, count)); caller_count += count;
    }
    void release(unsigned count, bool runtime) {
        require(count <= caller_count, "fixture releases only its owned caller references");
        if (runtime) LIFETIME_RUNTIME(cudaUserObjectRelease(reinterpret_cast<cudaUserObject_t>(value), count));
        else CONFORMANCE_CUDA(cuUserObjectRelease(value, count)); caller_count -= count;
    }
    void attach(CUgraph graph, unsigned count, bool move, bool runtime) {
        require(!move || count <= caller_count, "fixture transfers only owned caller references");
        if (runtime) LIFETIME_RUNTIME(cudaGraphRetainUserObject(reinterpret_cast<cudaGraph_t>(graph), reinterpret_cast<cudaUserObject_t>(value), count, move ? cudaGraphUserObjectMove : 0));
        else CONFORMANCE_CUDA(cuGraphRetainUserObject(graph, value, count, move ? CU_GRAPH_USER_OBJECT_MOVE : 0));
        if (move) caller_count -= count;
    }
    void detach(CUgraph graph, unsigned count, bool runtime) const {
        if (runtime) LIFETIME_RUNTIME(cudaGraphReleaseUserObject(reinterpret_cast<cudaGraph_t>(graph), reinterpret_cast<cudaUserObject_t>(value), count));
        else CONFORMANCE_CUDA(cuGraphReleaseUserObject(graph, value, count));
    }
    void alive() const { std::lock_guard lock(state->mutex); require(state->calls == 0, "user object remains alive through its last retained use"); }
    void prove_retained(bool runtime) { retain(1, runtime); release(1, !runtime); alive(); }
    void await() const {
        std::unique_lock lock(state->mutex);
        require(state->changed.wait_for(lock, std::chrono::seconds(5), [&] { return state->calls != 0; }), "user-object destructor completion deadline");
        require(state->calls == 1 && !state->invalid && state->completions == 1,
            "exactly one intact user-object destructor runs after the accepted graph's completion marker");
    }
};
constexpr size_t lifetime_words = 112, lifetime_guard = 8, lifetime_slot_words = 128, lifetime_slots = 4;
struct lifetime_storage {
    device_buffer device{lifetime_words * sizeof(uint32_t)};
    void* host{};
    std::vector<uint32_t> initial;
    lifetime_storage() : initial(lifetime_words) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = 0xa9a90000u + uint32_t(i) * 0x10101u;
        CONFORMANCE_CUDA(cuMemAllocHost(&host, lifetime_slots * lifetime_slot_words * sizeof(uint32_t)));
        try { CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, initial.data(), initial.size() * sizeof(uint32_t))); }
        catch (...) { cuMemFreeHost(host); host = nullptr; throw; }
        std::memset(host, 0xd3, lifetime_slots * lifetime_slot_words * sizeof(uint32_t));
    }
    ~lifetime_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    uint32_t* slot(unsigned index) const { require(index < lifetime_slots, "valid lifetime readback slot"); return static_cast<uint32_t*>(host) + index * lifetime_slot_words; }
    void expect(unsigned index, uint32_t value) const {
        std::vector<uint32_t> expected(lifetime_slot_words, 0xd3d3d3d3);
        std::copy(initial.begin(), initial.end(), expected.begin() + lifetime_guard);
        for (unsigned i = 0; i < 91; ++i) expected[2 * lifetime_guard + i] = static_cast<uint32_t>(uint64_t(value) + uint64_t(i) * 7);
        expect_words({slot(index), lifetime_slot_words}, expected, "retained graph kernel/readback and complete guards");
    }
    void untouched(unsigned first = 0) const {
        for (unsigned i = first; i < lifetime_slots; ++i) for (size_t j = 0; j < lifetime_slot_words; ++j)
            require(slot(i)[j] == 0xd3d3d3d3, "pending or unused lifetime readback slot remains untouched");
    }
};
struct lifetime_stream {
    CUstream value{};
    lifetime_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~lifetime_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct lifetime_gate {
    std::mutex mutex; std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<lifetime_gate*>(argument);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "lifetime gate entry deadline"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void held() { std::lock_guard lock(mutex); require(entered && !released && !expired, "graph retirement does not wait for gated accepted work"); }
};
struct lifetime_gate_cleanup {
    lifetime_gate& gate;
    ~lifetime_gate_cleanup() { gate.release(); cuCtxSynchronize(); }
};
struct lifetime_graph {
    CUgraph value{}; CUcontext context{};
    CUgraphNode kernel{}, read{}, marker{};
    explicit lifetime_graph(bool runtime) {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        if (runtime) { cudaGraph_t graph{}; LIFETIME_RUNTIME(cudaGraphCreate(&graph, 0)); value = reinterpret_cast<CUgraph>(graph); }
        else CONFORMANCE_CUDA(cuGraphCreate(&value, 0));
    }
    lifetime_graph(const lifetime_graph& original, bool runtime) {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        if (runtime) { cudaGraph_t graph{}; LIFETIME_RUNTIME(cudaGraphClone(&graph, reinterpret_cast<cudaGraph_t>(original.value))); value = reinterpret_cast<CUgraph>(graph); }
        else CONFORMANCE_CUDA(cuGraphClone(&value, original.value));
    }
    lifetime_graph(const lifetime_graph&) = delete;
    lifetime_graph& operator=(const lifetime_graph&) = delete;
    ~lifetime_graph() { if (value) cuGraphDestroy(value); }
    void destroy(bool runtime) {
        if (runtime) LIFETIME_RUNTIME(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(value)));
        else CONFORMANCE_CUDA(cuGraphDestroy(value));
        value = nullptr;
    }
    void child(CUgraph source, bool runtime) const {
        if (runtime) {
            cudaGraphNode_t node{}; LIFETIME_RUNTIME(cudaGraphAddChildGraphNode(&node, reinterpret_cast<cudaGraph_t>(value), nullptr, 0, reinterpret_cast<cudaGraph_t>(source)));
        } else { CUgraphNode node{}; CONFORMANCE_CUDA(cuGraphAddChildGraphNode(&node, value, nullptr, 0, source)); }
    }
    void configure(CUfunction function, lifetime_storage& storage, unsigned slot, uint32_t color, lifetime_state* state, bool runtime) {
        auto output = storage.device.value + lifetime_guard * sizeof(uint32_t);
        std::array<void*, 2> arguments{&output, &color};
        if (runtime) {
            cudaGraphNodeParams p{}; p.type = cudaGraphNodeTypeKernel;
            p.kernel.functionType = cudaKernelFunctionTypeFunction; p.kernel.cuFunc = reinterpret_cast<cudaFunction_t>(function);
            p.kernel.gridDim = dim3(3); p.kernel.blockDim = dim3(32); p.kernel.kernelParams = arguments.data();
            if (kernel) LIFETIME_RUNTIME(cudaGraphNodeSetParams(reinterpret_cast<cudaGraphNode_t>(kernel), &p));
            else {
                cudaGraphNode_t node{}; LIFETIME_RUNTIME(cudaGraphAddNode(&node, reinterpret_cast<cudaGraph_t>(value), nullptr, nullptr, 0, &p)); kernel = reinterpret_cast<CUgraphNode>(node);
            }
        } else {
            CUDA_KERNEL_NODE_PARAMS p{}; p.func = function; p.gridDimX = 3; p.gridDimY = p.gridDimZ = 1;
            p.blockDimX = 32; p.blockDimY = p.blockDimZ = 1; p.kernelParams = arguments.data();
            if (kernel) CONFORMANCE_CUDA(cuGraphKernelNodeSetParams(kernel, &p));
            else CONFORMANCE_CUDA(cuGraphAddKernelNode(&kernel, value, nullptr, 0, &p));
        }
        const size_t bytes = lifetime_words * sizeof(uint32_t);
        auto destination = storage.slot(slot) + lifetime_guard;
        if (runtime) {
            cudaMemcpy3DParms p{}; p.srcPtr = {reinterpret_cast<void*>(static_cast<uintptr_t>(storage.device.value)), bytes, bytes, 1};
            p.dstPtr = {destination, bytes, bytes, 1}; p.extent = {bytes, 1, 1}; p.kind = cudaMemcpyDeviceToHost;
            if (read) LIFETIME_RUNTIME(cudaGraphMemcpyNodeSetParams(reinterpret_cast<cudaGraphNode_t>(read), &p));
            else {
                cudaGraphNode_t node{}; const auto dependency = reinterpret_cast<cudaGraphNode_t>(kernel);
                LIFETIME_RUNTIME(cudaGraphAddMemcpyNode(&node, reinterpret_cast<cudaGraph_t>(value), &dependency, 1, &p)); read = reinterpret_cast<CUgraphNode>(node);
            }
            cudaHostNodeParams host{lifetime_state::publish, state};
            if (marker) LIFETIME_RUNTIME(cudaGraphHostNodeSetParams(reinterpret_cast<cudaGraphNode_t>(marker), &host));
            else {
                cudaGraphNode_t node{}; const auto dependency = reinterpret_cast<cudaGraphNode_t>(read);
                LIFETIME_RUNTIME(cudaGraphAddHostNode(&node, reinterpret_cast<cudaGraph_t>(value), &dependency, 1, &host)); marker = reinterpret_cast<CUgraphNode>(node);
            }
        } else {
            CUDA_MEMCPY3D p{}; p.srcMemoryType = CU_MEMORYTYPE_DEVICE; p.srcDevice = storage.device.value;
            p.dstMemoryType = CU_MEMORYTYPE_HOST; p.dstHost = destination;
            p.srcPitch = p.dstPitch = p.WidthInBytes = bytes; p.srcHeight = p.dstHeight = p.Height = p.Depth = 1;
            if (read) CONFORMANCE_CUDA(cuGraphMemcpyNodeSetParams(read, &p));
            else CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&read, value, &kernel, 1, &p, context));
            CUDA_HOST_NODE_PARAMS host{lifetime_state::publish, state};
            if (marker) CONFORMANCE_CUDA(cuGraphHostNodeSetParams(marker, &host));
            else CONFORMANCE_CUDA(cuGraphAddHostNode(&marker, value, &read, 1, &host));
        }
    }
};
struct lifetime_executable {
    CUgraphExec value{};
    lifetime_executable(CUgraph graph, bool runtime) {
        if (runtime) { cudaGraphExec_t executable{}; LIFETIME_RUNTIME(cudaGraphInstantiateWithFlags(&executable, reinterpret_cast<cudaGraph_t>(graph), 0)); value = reinterpret_cast<CUgraphExec>(executable); }
        else CONFORMANCE_CUDA(cuGraphInstantiate(&value, graph, 0));
    }
    ~lifetime_executable() { if (value) cuGraphExecDestroy(value); }
    void destroy(bool runtime) {
        if (runtime) LIFETIME_RUNTIME(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(value)));
        else CONFORMANCE_CUDA(cuGraphExecDestroy(value));
        value = nullptr;
    }
    void launch(CUstream stream, bool runtime) const {
        if (runtime) LIFETIME_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuGraphLaunch(value, stream));
    }
    void update(CUgraph source, bool runtime) const {
        if (runtime) {
            cudaGraphExecUpdateResultInfo info{};
            LIFETIME_RUNTIME(cudaGraphExecUpdate(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaGraph_t>(source), &info));
            require(info.result == cudaGraphExecUpdateSuccess && !info.errorNode && !info.errorFromNode, "successful Runtime lifetime update diagnostics");
        } else {
            CUgraphExecUpdateResultInfo info{}; CONFORMANCE_CUDA(cuGraphExecUpdate(value, source, &info));
            require(info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS && !info.errorNode && !info.errorFromNode, "successful Driver lifetime update diagnostics");
        }
    }
};
struct lifetime_event {
    CUevent value{};
    lifetime_event() { CONFORMANCE_CUDA(cuEventCreate(&value, CU_EVENT_DISABLE_TIMING)); }
    ~lifetime_event() { if (value) cuEventDestroy(value); }
    void record(CUstream stream) const { CONFORMANCE_CUDA(cuEventRecord(value, stream)); }
    void synchronize() const { CONFORMANCE_CUDA(cuEventSynchronize(value)); }
};
void lifetime_references(CUfunction function, bool runtime) {
    lifetime_object object(2, runtime, 0x1111222233334444);
    lifetime_storage storage; lifetime_stream stream; lifetime_graph graph(runtime);
    graph.configure(function, storage, 0, 0xffffffd1, object.state.get(), runtime);
    object.retain(3, !runtime); object.release(4, runtime); object.alive();
    object.attach(graph.value, 3, false, !runtime); object.release(1, !runtime);
    object.detach(graph.value, 2, runtime); object.prove_retained(runtime);
    lifetime_executable executable(graph.value, !runtime);
    object.detach(graph.value, 1, !runtime); object.prove_retained(!runtime);
    executable.launch(stream.value, runtime); stream.synchronize(); storage.expect(0, 0xffffffd1); storage.untouched(1);
    object.alive(); executable.destroy(runtime); object.await();
}
void lifetime_clones(CUfunction function, bool runtime) {
    lifetime_object object(1, runtime, 0x2222333344445555);
    lifetime_storage storage; lifetime_stream stream; lifetime_graph source(runtime);
    source.configure(function, storage, 0, 0x80000013, object.state.get(), runtime);
    object.attach(source.value, 1, true, !runtime);
    lifetime_graph clone(source, !runtime); source.destroy(!runtime); object.prove_retained(runtime);
    lifetime_executable executable(clone.value, runtime); clone.destroy(runtime); object.prove_retained(!runtime);
    executable.launch(stream.value, !runtime); stream.synchronize(); storage.expect(0, 0x80000013); storage.untouched(1);
    executable.destroy(!runtime); object.await();
}
void lifetime_children(CUfunction function, bool runtime) {
    lifetime_object object(1, runtime, 0x3333444455556666);
    lifetime_storage storage; lifetime_stream stream; lifetime_graph leaf(runtime), middle(!runtime), parent(runtime);
    leaf.configure(function, storage, 0, 0xabcdef01, object.state.get(), !runtime);
    object.attach(leaf.value, 1, true, runtime);
    middle.child(leaf.value, !runtime); parent.child(middle.value, runtime);
    leaf.destroy(!runtime); middle.destroy(runtime); object.prove_retained(runtime);
    lifetime_executable executable(parent.value, !runtime); parent.destroy(!runtime); object.prove_retained(!runtime);
    executable.launch(stream.value, runtime); stream.synchronize(); storage.expect(0, 0xabcdef01); storage.untouched(1);
    executable.destroy(runtime); object.await();
}
void lifetime_queued(CUfunction function, bool runtime) {
    lifetime_object object(1, runtime, 0x4444555566667777);
    lifetime_gate gate; lifetime_storage storage; lifetime_stream stream; lifetime_event done;
    lifetime_graph source(runtime); source.configure(function, storage, 0, 0x10203040, object.state.get(), runtime);
    object.attach(source.value, 1, true, !runtime); lifetime_executable executable(source.value, !runtime);
    lifetime_gate_cleanup cleanup{gate};
    CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, lifetime_gate::call, &gate)); gate.await();
    executable.launch(stream.value, runtime); done.record(stream.value);
    source.destroy(!runtime); executable.destroy(runtime);
    object.prove_retained(runtime); storage.untouched(); gate.held();
    gate.release(); done.synchronize(); stream.synchronize(); storage.expect(0, 0x10203040); storage.untouched(1);
    object.await(); require(!gate.expired, "pending graph retirement gate deadline");
}
void lifetime_updates(CUfunction function, bool runtime) {
    lifetime_object old_object(1, runtime, 0x5555666677778888), new_object(1, !runtime, 0x6666777788889999);
    lifetime_gate gate; lifetime_storage storage; lifetime_stream stream; lifetime_event done;
    lifetime_graph source(runtime); source.configure(function, storage, 0, 0xfffffff1, old_object.state.get(), runtime);
    old_object.attach(source.value, 1, true, !runtime); lifetime_executable executable(source.value, runtime);
    lifetime_gate_cleanup cleanup{gate};
    CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, lifetime_gate::call, &gate)); gate.await();
    executable.launch(stream.value, !runtime);
    new_object.attach(source.value, 1, true, runtime); old_object.detach(source.value, 1, !runtime);
    source.configure(function, storage, 1, 0x13579bdf, new_object.state.get(), !runtime);
    executable.update(source.value, runtime);
    executable.launch(stream.value, runtime); done.record(stream.value);
    source.destroy(runtime); executable.destroy(!runtime);
    old_object.prove_retained(runtime); new_object.prove_retained(!runtime); storage.untouched(); gate.held();
    gate.release(); done.synchronize(); stream.synchronize();
    storage.expect(0, 0xfffffff1); storage.expect(1, 0x13579bdf); storage.untouched(2);
    old_object.await(); new_object.await(); require(!gate.expired, "accepted graph update lifetime gate deadline");
}
void lifetime_validation(CUfunction function, bool runtime) {
    lifetime_object object(1, runtime, 0x777788889999aaaa);
    lifetime_storage storage; lifetime_stream stream; lifetime_graph source(runtime);
    source.configure(function, storage, 0, 0x2468ace0, object.state.get(), !runtime);
    require(cuUserObjectRetain(object.value, 0) == CUDA_ERROR_INVALID_VALUE, "zero Driver user-object retain count");
    require(cuUserObjectRelease(object.value, 0) == CUDA_ERROR_INVALID_VALUE, "zero Driver user-object release count");
    require(cudaUserObjectRetain(reinterpret_cast<cudaUserObject_t>(object.value), 0) == cudaErrorInvalidValue, "zero Runtime user-object retain count");
    require(cudaUserObjectRelease(reinterpret_cast<cudaUserObject_t>(object.value), 0) == cudaErrorInvalidValue, "zero Runtime user-object release count");
    require(cuGraphRetainUserObject(source.value, object.value, 0, 0) == CUDA_ERROR_INVALID_VALUE, "zero Driver graph reference count");
    require(cudaGraphRetainUserObject(reinterpret_cast<cudaGraph_t>(source.value), reinterpret_cast<cudaUserObject_t>(object.value), 0, 0) == cudaErrorInvalidValue, "zero Runtime graph reference count");
    require(cuGraphRetainUserObject(source.value, object.value, 1, 2) == CUDA_ERROR_INVALID_VALUE, "unknown Driver graph-retain flags");
    require(cudaGraphRetainUserObject(reinterpret_cast<cudaGraph_t>(source.value), reinterpret_cast<cudaUserObject_t>(object.value), 1, 2) == cudaErrorInvalidValue, "unknown Runtime graph-retain flags");
    object.attach(source.value, 1, true, runtime);
    require(cuGraphReleaseUserObject(source.value, object.value, 0) == CUDA_ERROR_INVALID_VALUE, "zero Driver graph-release count");
    require(cudaGraphReleaseUserObject(reinterpret_cast<cudaGraph_t>(source.value), reinterpret_cast<cudaUserObject_t>(object.value), 0) == cudaErrorInvalidValue, "zero Runtime graph-release count");
    object.prove_retained(!runtime); lifetime_executable executable(source.value, !runtime); source.destroy(!runtime);
    executable.launch(stream.value, runtime); stream.synchronize(); storage.expect(0, 0x2468ace0); storage.untouched(1);
    executable.destroy(runtime); object.await();
}
void execute_graph_lifetime(const loaded_program& program) {
    const auto function = program.function("lifetime_write");
    const std::array<std::pair<const char*, void (*)(CUfunction, bool)>, 6> cases{{
        {"references", lifetime_references}, {"clones", lifetime_clones}, {"children", lifetime_children},
        {"queued", lifetime_queued}, {"updates", lifetime_updates}, {"validation", lifetime_validation}}};
    for (const auto& [name, execute] : cases) for (bool runtime : {false, true}) {
        std::cout << "graph_lifetime=" << name << " api=" << (runtime ? "runtime" : "driver") << std::endl;
        execute(function, runtime);
    }
}
}
int main(int argc, char** argv) { return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_graph_lifetime); }
