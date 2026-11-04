#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>

namespace cudavm {
void check_update_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define UPDATE_RUNTIME(call) ::cudavm::check_update_runtime((call), #call)
constexpr size_t update_words = 112, update_guard = 8, update_lanes = 96, update_slot_words = 128, update_slots = 4;
void update_apply(std::vector<uint32_t>& state, unsigned count, uint32_t value, bool twist) {
    for (size_t i = 0; i < std::min<size_t>(count, update_lanes); ++i)
        state[i + update_guard] = twist ? value ^ static_cast<uint32_t>(uint64_t(i) * 13 + 0xa5a55a5a)
                                      : static_cast<uint32_t>(uint64_t(value) + uint64_t(i) * 7);
}
struct update_storage {
    device_buffer device{update_words * sizeof(uint32_t)}, sink{update_words * sizeof(uint32_t)};
    void* host{};
    std::vector<uint32_t> initial;
    explicit update_storage(unsigned seed) : initial(update_words) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = 0xcafe0000u + uint32_t(i) * 0x10203u + seed;
        CONFORMANCE_CUDA(cuMemAllocHost(&host, update_slots * update_slot_words * sizeof(uint32_t)));
        try {
            CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, initial.data(), initial.size() * sizeof(uint32_t)));
            CONFORMANCE_CUDA(cuMemcpyHtoD(sink.value, initial.data(), initial.size() * sizeof(uint32_t)));
        } catch (...) { cuMemFreeHost(host); host = nullptr; throw; }
        std::memset(host, 0xd3, update_slots * update_slot_words * sizeof(uint32_t));
    }
    ~update_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    uint32_t* slot(unsigned index) const {
        require(index < update_slots, "valid graph update readback slot");
        return static_cast<uint32_t*>(host) + index * update_slot_words;
    }
    void* destination(unsigned index) const { return slot(index) + update_guard; }
    void expect(unsigned index, std::span<const uint32_t> state) const {
        std::vector<uint32_t> expected(update_slot_words, 0xd3d3d3d3);
        std::copy(state.begin(), state.end(), expected.begin() + update_guard);
        expect_words({slot(index), update_slot_words}, expected, "updated graph readback and complete guards");
    }
    void untouched(unsigned first = 0) const {
        for (unsigned i = first; i < update_slots; ++i) for (size_t j = 0; j < update_slot_words; ++j)
            require(slot(i)[j] == 0xd3d3d3d3, "pending or unused update slot remains untouched");
    }
    void expect_device(std::span<const uint32_t> expected) const {
        std::vector<uint32_t> actual(update_words);
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), device.value, actual.size() * sizeof(uint32_t)));
        expect_words(actual, expected, "updated device data, inactive lanes and allocation guards");
        CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), sink.value, actual.size() * sizeof(uint32_t)));
        expect_words(actual, initial, "rejected copy destination remains untouched");
    }
};
struct update_stream {
    CUstream value{};
    update_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~update_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct update_graph {
    CUgraph value{}; CUcontext context{};
    CUgraphNode root{}, kernel{}, read{}, tail{};
    update_graph(CUfunction function, update_storage& storage, unsigned count, uint32_t color, bool runtime) {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        if (runtime) { cudaGraph_t result{}; UPDATE_RUNTIME(cudaGraphCreate(&result, 0)); value = reinterpret_cast<CUgraph>(result); }
        else CONFORMANCE_CUDA(cuGraphCreate(&value, 0));
        try {
            root = empty({}, runtime); set_kernel(function, storage, count, color, runtime);
            set_read(storage, 0, runtime); tail = empty(std::array{read, root}, runtime);
        } catch (...) { cuGraphDestroy(value); value = nullptr; throw; }
    }
    ~update_graph() { if (value) cuGraphDestroy(value); }
    CUgraphNode empty(std::span<const CUgraphNode> dependencies, bool runtime) const {
        CUgraphNode result{};
        if (runtime) {
            std::vector<cudaGraphNode_t> converted;
            for (auto node : dependencies) converted.push_back(reinterpret_cast<cudaGraphNode_t>(node));
            cudaGraphNode_t node{}; UPDATE_RUNTIME(cudaGraphAddEmptyNode(&node, reinterpret_cast<cudaGraph_t>(value), converted.data(), converted.size()));
            result = reinterpret_cast<CUgraphNode>(node);
        } else CONFORMANCE_CUDA(cuGraphAddEmptyNode(&result, value, dependencies.data(), dependencies.size()));
        return result;
    }
    void edge(CUgraphNode from, CUgraphNode to, bool add, bool runtime) const {
        if (runtime) {
            auto source = reinterpret_cast<cudaGraphNode_t>(from), target = reinterpret_cast<cudaGraphNode_t>(to);
            if (add) UPDATE_RUNTIME(cudaGraphAddDependencies(reinterpret_cast<cudaGraph_t>(value), &source, &target, nullptr, 1));
            else UPDATE_RUNTIME(cudaGraphRemoveDependencies(reinterpret_cast<cudaGraph_t>(value), &source, &target, nullptr, 1));
        } else if (add) CONFORMANCE_CUDA(cuGraphAddDependencies(value, &from, &to, nullptr, 1));
        else CONFORMANCE_CUDA(cuGraphRemoveDependencies(value, &from, &to, nullptr, 1));
    }
    void destroy_node(CUgraphNode node, bool runtime) const {
        if (runtime) UPDATE_RUNTIME(cudaGraphDestroyNode(reinterpret_cast<cudaGraphNode_t>(node)));
        else CONFORMANCE_CUDA(cuGraphDestroyNode(node));
    }
    void set_kernel(CUfunction function, update_storage& storage, unsigned count, uint32_t color, bool runtime,
                    CUgraphExec executable = nullptr, unsigned block = 32) {
        auto pointer = storage.device.value + update_guard * sizeof(uint32_t);
        std::array<void*, 3> arguments{&pointer, &count, &color};
        if (runtime) {
            cudaGraphNodeParams p{}; p.type = cudaGraphNodeTypeKernel;
            p.kernel.functionType = cudaKernelFunctionTypeFunction; p.kernel.cuFunc = reinterpret_cast<cudaFunction_t>(function);
            p.kernel.gridDim = dim3(96 / block); p.kernel.blockDim = dim3(block); p.kernel.kernelParams = arguments.data();
            if (executable) UPDATE_RUNTIME(cudaGraphExecNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaGraphNode_t>(kernel), &p));
            else if (kernel) UPDATE_RUNTIME(cudaGraphNodeSetParams(reinterpret_cast<cudaGraphNode_t>(kernel), &p));
            else {
                cudaGraphNode_t result{}, dependency = reinterpret_cast<cudaGraphNode_t>(root);
                UPDATE_RUNTIME(cudaGraphAddNode(&result, reinterpret_cast<cudaGraph_t>(value), &dependency, nullptr, 1, &p)); kernel = reinterpret_cast<CUgraphNode>(result);
            }
        } else {
            CUDA_KERNEL_NODE_PARAMS p{}; p.func = function; p.gridDimX = 96 / block; p.gridDimY = p.gridDimZ = 1;
            p.blockDimX = block; p.blockDimY = p.blockDimZ = 1; p.kernelParams = arguments.data();
            if (executable) CONFORMANCE_CUDA(cuGraphExecKernelNodeSetParams(executable, kernel, &p));
            else if (kernel) CONFORMANCE_CUDA(cuGraphKernelNodeSetParams(kernel, &p));
            else CONFORMANCE_CUDA(cuGraphAddKernelNode(&kernel, value, &root, 1, &p));
        }
        pointer = 0; count = 0; color = 0; // Accepted nodes own argument bytes, not this stack frame.
    }
    void set_read(update_storage& storage, unsigned slot, bool runtime, CUgraphExec executable = nullptr,
                  bool device_destination = false, bool reject = false) {
        const size_t bytes = update_words * sizeof(uint32_t);
        if (runtime) {
            cudaMemcpy3DParms p{}; p.srcPtr = {reinterpret_cast<void*>(static_cast<uintptr_t>(storage.device.value)), bytes, bytes, 1};
            p.dstPtr = {device_destination ? reinterpret_cast<void*>(static_cast<uintptr_t>(storage.sink.value)) : storage.destination(slot), bytes, bytes, 1};
            p.extent = {bytes, 1, 1}; p.kind = device_destination ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
            if (executable) {
                auto result = cudaGraphExecMemcpyNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaGraphNode_t>(read), &p);
                if (reject) require(result == cudaErrorInvalidValue, "Runtime executable copy rejects changed endpoint memory type");
                else UPDATE_RUNTIME(result);
            } else if (read) UPDATE_RUNTIME(cudaGraphMemcpyNodeSetParams(reinterpret_cast<cudaGraphNode_t>(read), &p));
            else {
                cudaGraphNode_t result{}, dependency = reinterpret_cast<cudaGraphNode_t>(kernel);
                UPDATE_RUNTIME(cudaGraphAddMemcpyNode(&result, reinterpret_cast<cudaGraph_t>(value), &dependency, 1, &p)); read = reinterpret_cast<CUgraphNode>(result);
            }
        } else {
            CUDA_MEMCPY3D p{}; p.srcMemoryType = CU_MEMORYTYPE_DEVICE; p.srcDevice = storage.device.value;
            p.dstMemoryType = device_destination ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
            if (device_destination) p.dstDevice = storage.sink.value; else p.dstHost = storage.destination(slot);
            p.srcPitch = p.dstPitch = p.WidthInBytes = bytes; p.srcHeight = p.dstHeight = p.Height = p.Depth = 1;
            if (executable) {
                auto result = cuGraphExecMemcpyNodeSetParams(executable, read, &p, context);
                if (reject) require(result == CUDA_ERROR_INVALID_VALUE, "Driver executable copy rejects changed endpoint memory type");
                else CONFORMANCE_CUDA(result);
            } else if (read) CONFORMANCE_CUDA(cuGraphMemcpyNodeSetParams(read, &p));
            else CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&read, value, &kernel, 1, &p, context));
        }
    }
};
template<class T> struct update_guarded {
    uint64_t prefix = 0x123456789abcdef0; T value{}; uint64_t suffix = 0xfedcba9876543210;
    void check() const { require(prefix == 0x123456789abcdef0 && suffix == 0xfedcba9876543210, "graph update diagnostic output guards"); }
};
struct update_diagnostic { CUgraphExecUpdateResult result; CUgraphNode node{}, from{}; };
struct update_executable {
    CUgraphExec value{};
    update_executable(CUgraph graph, bool runtime, unsigned long long flags = 0) {
        if (runtime) { cudaGraphExec_t result{}; UPDATE_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), flags)); value = reinterpret_cast<CUgraphExec>(result); }
        else CONFORMANCE_CUDA(cuGraphInstantiate(&value, graph, flags));
    }
    ~update_executable() { if (value) cuGraphExecDestroy(value); }
    void launch(CUstream stream, bool runtime) const {
        if (runtime) UPDATE_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuGraphLaunch(value, stream));
    }
    update_diagnostic update(CUgraph graph, bool runtime, bool success) const {
        if (runtime) {
            update_guarded<cudaGraphExecUpdateResultInfo> info;
            info.value.result = cudaGraphExecUpdateError;
            auto result = cudaGraphExecUpdate(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaGraph_t>(graph), &info.value);
            require(result == (success ? cudaSuccess : cudaErrorGraphExecUpdateFailure), "Runtime whole update status=" + std::to_string(result) + " result=" + std::to_string(info.value.result)); info.check();
            return {static_cast<CUgraphExecUpdateResult>(info.value.result), reinterpret_cast<CUgraphNode>(info.value.errorNode), reinterpret_cast<CUgraphNode>(info.value.errorFromNode)};
        }
        update_guarded<CUgraphExecUpdateResultInfo> info;
        info.value.result = CU_GRAPH_EXEC_UPDATE_ERROR;
        auto result = cuGraphExecUpdate(value, graph, &info.value);
        require(result == (success ? CUDA_SUCCESS : CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE), "Driver whole update status=" + std::to_string(result) + " result=" + std::to_string(info.value.result)); info.check();
        return {info.value.result, info.value.errorNode, info.value.errorFromNode};
    }
    void accept(CUgraph graph, bool runtime) const { require(update(graph, runtime, true).result == CU_GRAPH_EXEC_UPDATE_SUCCESS, "successful whole update result"); }
};
struct update_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<update_gate*>(argument); std::unique_lock lock(gate.mutex);
        gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "graph update gate entry deadline"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void held() { std::lock_guard lock(mutex); require(entered && !released && !expired, "updates complete while earlier accepted work remains pending"); }
};
struct update_gate_cleanup { update_gate& gate; ~update_gate_cleanup() { gate.release(); cuCtxSynchronize(); } };
using update_functions = std::array<CUfunction, 2>;
void update_nodes(const update_functions& functions, bool runtime) {
    update_storage original(11), changed(37); update_stream stream;
    update_graph source(functions[0], original, 65, 0x12345678, runtime); update_executable executable(source.value, !runtime);
    auto expected = changed.initial;
    for (unsigned stage = 0; stage < 3; ++stage) {
        unsigned count = stage == 0 ? 0 : stage == 1 ? 79 : 113;
        uint32_t color = stage == 2 ? 0xfffffff1 : 0x2468ace0;
        source.set_kernel(functions[stage == 1], changed, count, color, runtime, executable.value, stage == 1 ? 16 : 32);
        source.set_read(changed, stage, !runtime, executable.value); executable.launch(stream.value, runtime); stream.synchronize();
        update_apply(expected, count, color, stage == 1); changed.expect(stage, expected);
    }
    update_executable unchanged(source.value, runtime); unchanged.launch(stream.value, !runtime); stream.synchronize();
    auto baseline = original.initial; update_apply(baseline, 65, 0x12345678, false); original.expect(0, baseline);
    original.untouched(1); changed.untouched(3); original.expect_device(baseline); changed.expect_device(expected);
}
void update_pending(const update_functions& functions, bool runtime) {
    update_gate gate; update_storage original(19), changed(53); update_stream stream;
    update_graph source(functions[0], original, 65, 0x11112222, runtime);
    update_graph candidate(functions[1], changed, 79, 0xabcdef01, !runtime); update_executable executable(source.value, runtime);
    update_gate_cleanup cleanup{gate};
    CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, update_gate::call, &gate)); gate.await(); executable.launch(stream.value, runtime);
    executable.accept(candidate.value, runtime); executable.launch(stream.value, !runtime);
    candidate.set_kernel(functions[0], changed, 1, 0xfffffffe, runtime); candidate.set_read(changed, 1, !runtime);
    executable.accept(candidate.value, !runtime); executable.launch(stream.value, runtime);
    // Later source edits are not executable edits. Original node handles still name executable nodes after whole updates.
    candidate.set_kernel(functions[1], original, 0, 7, !runtime); candidate.set_read(original, 1, runtime);
    source.set_kernel(functions[0], changed, 96, 0x76543210, runtime, executable.value, 16);
    source.set_read(changed, 2, !runtime, executable.value); executable.launch(stream.value, !runtime);
    original.untouched(); changed.untouched(); gate.held(); gate.release(); stream.synchronize();
    auto old_expected = original.initial; update_apply(old_expected, 65, 0x11112222, false); original.expect(0, old_expected);
    auto expected = changed.initial; update_apply(expected, 79, 0xabcdef01, true); changed.expect(0, expected);
    update_apply(expected, 1, 0xfffffffe, false); changed.expect(1, expected);
    update_apply(expected, 96, 0x76543210, false); changed.expect(2, expected);
    original.untouched(1); changed.untouched(3); original.expect_device(old_expected); changed.expect_device(expected);
    require(!gate.expired, "accepted graph update snapshot gate deadline");
}
void update_rejections(const update_functions& functions, bool runtime) {
    for (unsigned kind = 0; kind < 5; ++kind) {
        std::cout << " rejection=" << kind << std::endl;
        update_storage original(71), changed(97); update_stream stream;
        update_graph source(functions[0], original, 65, 0x13579bdf, runtime);
        update_graph candidate(functions[1], changed, 79, 0x2468ace0, !runtime); update_executable executable(source.value, !runtime);
        CUgraphNode extra{};
        if (kind == 0) extra = candidate.empty(std::array{candidate.tail}, runtime);
        if (kind == 1) {
            candidate.destroy_node(candidate.read, runtime); candidate.edge(candidate.root, candidate.tail, false, !runtime);
            candidate.read = candidate.empty(std::array{candidate.kernel}, !runtime);
            candidate.edge(candidate.read, candidate.tail, true, runtime); candidate.edge(candidate.root, candidate.tail, true, !runtime);
        }
        if (kind == 2) candidate.set_read(changed, 0, runtime, nullptr, true);
        if (kind == 3) candidate.edge(candidate.root, candidate.tail, false, runtime);
        if (kind == 4) {
            candidate.edge(candidate.root, candidate.tail, false, runtime);
            candidate.edge(candidate.kernel, candidate.tail, true, !runtime);
        }
        auto diagnostic = executable.update(candidate.value, runtime, false);
        auto result = kind == 1 ? CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED : kind == 2 ? CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED : CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED;
        require(diagnostic.result == result, "whole update returns the documented rejection category");
        if (kind == 0) require(!diagnostic.node, "different node count reports no individual error node");
        if (kind == 1 || kind == 2) require(diagnostic.node == candidate.read, "type/parameter rejection identifies the candidate copy position");
        if (kind == 3) require(diagnostic.node == candidate.tail, "dependency-count rejection identifies the candidate node");
        if (kind == 4) require(diagnostic.node == candidate.tail && diagnostic.from == candidate.kernel, "edge mismatch identifies candidate destination and mismatched dependency");
        executable.launch(stream.value, !runtime); stream.synchronize();
        auto baseline = original.initial; update_apply(baseline, 65, 0x13579bdf, false); original.expect(0, baseline);
        changed.untouched(); changed.expect_device(changed.initial); original.expect_device(baseline);
        // Repair the same candidate and prove rejection did not poison future valid updates.
        if (kind == 0) candidate.destroy_node(extra, !runtime);
        if (kind == 1) {
            candidate.destroy_node(candidate.read, !runtime); candidate.edge(candidate.root, candidate.tail, false, runtime);
            candidate.read = nullptr; candidate.set_read(changed, 0, runtime);
            candidate.edge(candidate.read, candidate.tail, true, !runtime); candidate.edge(candidate.root, candidate.tail, true, runtime);
        }
        if (kind == 2) candidate.set_read(changed, 0, !runtime);
        if (kind == 3) candidate.edge(candidate.root, candidate.tail, true, !runtime);
        if (kind == 4) {
            candidate.edge(candidate.kernel, candidate.tail, false, runtime);
            candidate.edge(candidate.root, candidate.tail, true, !runtime);
        }
        executable.accept(candidate.value, !runtime); executable.launch(stream.value, runtime); stream.synchronize();
        auto expected = changed.initial; update_apply(expected, 79, 0x2468ace0, true); changed.expect(0, expected);
        original.expect(0, baseline); original.untouched(1); changed.untouched(1); changed.expect_device(expected); original.expect_device(baseline);
    }
}
void update_priority(CUgraphNode node, int priority, bool runtime) {
    if (runtime) { cudaKernelNodeAttrValue p{}; p.priority = priority; UPDATE_RUNTIME(cudaGraphKernelNodeSetAttribute(reinterpret_cast<cudaGraphNode_t>(node), cudaKernelNodeAttributePriority, &p)); }
    else { CUkernelNodeAttrValue p{}; p.priority = priority; CONFORMANCE_CUDA(cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &p)); }
}
void update_attributes(const update_functions& functions, bool runtime) {
    update_storage original(101), changed(131); update_stream stream;
    update_graph source(functions[0], original, 65, 17, runtime), candidate(functions[1], changed, 79, 23, !runtime);
    update_priority(source.kernel, 17, runtime); update_priority(candidate.kernel, 18, !runtime);
    update_executable executable(source.value, runtime, CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY);
    auto diagnostic = executable.update(candidate.value, runtime, false);
    require(diagnostic.result == CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED && diagnostic.node == candidate.kernel, "node-priority update compares requested values before device clamping");
    executable.launch(stream.value, !runtime); stream.synchronize();
    auto baseline = original.initial; update_apply(baseline, 65, 17, false); original.expect(0, baseline); changed.untouched(); changed.expect_device(changed.initial);
    update_priority(candidate.kernel, 17, runtime); executable.accept(candidate.value, !runtime); executable.launch(stream.value, runtime); stream.synchronize();
    auto expected = changed.initial; update_apply(expected, 79, 23, true); changed.expect(0, expected);
    original.untouched(1); changed.untouched(1); original.expect_device(baseline); changed.expect_device(expected);
}
void update_invalid_nodes(const update_functions& functions, bool runtime) {
    update_storage storage(149); update_stream stream; update_graph source(functions[0], storage, 65, 29, runtime);
    update_executable executable(source.value, !runtime);
    source.set_read(storage, 0, runtime, executable.value, true, true);
    // A correctly formed kernel update aimed at an empty node must leave the executable intact.
    {
        bool rejected = false;
        auto pointer = storage.device.value + update_guard * sizeof(uint32_t); unsigned count = 96, color = 0;
        std::array<void*, 3> arguments{&pointer, &count, &color};
        if (runtime) {
            cudaGraphNodeParams p{}; p.type = cudaGraphNodeTypeKernel; p.kernel.functionType = cudaKernelFunctionTypeFunction;
            p.kernel.cuFunc = reinterpret_cast<cudaFunction_t>(functions[1]); p.kernel.gridDim = dim3(3); p.kernel.blockDim = dim3(32); p.kernel.kernelParams = arguments.data();
            rejected = cudaGraphExecNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable.value), reinterpret_cast<cudaGraphNode_t>(source.root), &p) == cudaErrorInvalidValue;
            require(cudaGraphNodeSetParams(reinterpret_cast<cudaGraphNode_t>(source.root), &p) == cudaErrorInvalidValue, "Runtime source kernel setter rejects an empty node");
        } else {
            CUDA_KERNEL_NODE_PARAMS p{}; p.func = functions[1]; p.gridDimX = 3; p.gridDimY = p.gridDimZ = 1;
            p.blockDimX = 32; p.blockDimY = p.blockDimZ = 1; p.kernelParams = arguments.data();
            rejected = cuGraphExecKernelNodeSetParams(executable.value, source.root, &p) == CUDA_ERROR_INVALID_VALUE;
            require(cuGraphKernelNodeSetParams(source.root, &p) == CUDA_ERROR_INVALID_VALUE, "Driver source kernel setter rejects an empty node");
            CUDA_KERNEL_NODE_PARAMS_v1 legacy{}; legacy.func = p.func;
            legacy.gridDimX = 3; legacy.gridDimY = legacy.gridDimZ = 1;
            legacy.blockDimX = 32; legacy.blockDimY = legacy.blockDimZ = 1; legacy.kernelParams = arguments.data();
            void* address{};
            CONFORMANCE_CUDA(cuGetProcAddress("cuGraphExecKernelNodeSetParams", &address, 10010, CU_GET_PROC_ADDRESS_DEFAULT, nullptr));
            auto exec_set = reinterpret_cast<CUresult (CUDAAPI*)(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS_v1*)>(address);
            require(exec_set(executable.value, source.root, &legacy) == CUDA_ERROR_INVALID_VALUE, "legacy executable kernel setter rejects an empty node");
            CONFORMANCE_CUDA(cuGetProcAddress("cuGraphKernelNodeSetParams", &address, 10000, CU_GET_PROC_ADDRESS_DEFAULT, nullptr));
            auto node_set = reinterpret_cast<CUresult (CUDAAPI*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS_v1*)>(address);
            require(node_set(source.root, &legacy) == CUDA_ERROR_INVALID_VALUE, "legacy source kernel setter rejects an empty node");
        }
        require(rejected, "executable kernel setter rejects a node of the wrong type");
    }
    CUgraphNodeType root_type{}; CONFORMANCE_CUDA(cuGraphNodeGetType(source.root, &root_type));
    require(root_type == CU_GRAPH_NODE_TYPE_EMPTY, "rejected source kernel setters preserve the empty-node type");
    executable.launch(stream.value, !runtime); stream.synchronize();
    auto expected = storage.initial; update_apply(expected, 65, 29, false); storage.expect(0, expected); storage.untouched(1); storage.expect_device(expected);
    update_executable fresh(source.value, runtime); fresh.launch(stream.value, !runtime); stream.synchronize();
    storage.expect(0, expected); storage.untouched(1); storage.expect_device(expected);
    source.set_kernel(functions[1], storage, 1, 0xabcdef01, runtime, executable.value);
    source.set_read(storage, 1, !runtime, executable.value); executable.launch(stream.value, runtime); stream.synchronize();
    update_apply(expected, 1, 0xabcdef01, true); storage.expect(1, expected); storage.untouched(2); storage.expect_device(expected);
}
void execute_graph_update(const loaded_program& program) {
    const update_functions functions{program.function("update_write"), program.function("update_twist")};
    const std::array<std::pair<const char*, void (*)(const update_functions&, bool)>, 5> cases{{
        {"nodes", update_nodes}, {"pending", update_pending}, {"rejections", update_rejections},
        {"attributes", update_attributes}, {"invalid_nodes", update_invalid_nodes}}};
    for (const auto& [name, execute] : cases) for (bool runtime : {false, true}) {
        std::cout << "graph_update=" << name << " api=" << (runtime ? "runtime" : "driver") << std::endl; execute(functions, runtime);
    }
}
}
int main(int argc, char** argv) { return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_graph_update); }
