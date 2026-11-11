#include "test_fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <thread>

namespace cudavm {
void check_replay_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define REPLAY_RUNTIME(call) ::cudavm::check_replay_runtime((call), #call)
constexpr size_t replay_guard_words = 8, replay_lanes = 96, replay_state_words = replay_lanes + 2 * replay_guard_words;
constexpr size_t replay_slot_words = replay_state_words + 2 * replay_guard_words, replay_slots = 8;
constexpr uint32_t replay_host_guard = 0xd3d3d3d3;
void replay_apply(std::vector<uint32_t>& state, unsigned count, uint32_t add) {
    for (unsigned i = 0; i < std::min(unsigned(replay_lanes), count); ++i) {
        const uint64_t next = uint64_t(state[replay_guard_words + i]) * 3 + add + uint64_t(i) * 7;
        state[replay_guard_words + i] = static_cast<uint32_t>(next);
    }
}
struct replay_storage {
    device_buffer device{replay_state_words * sizeof(uint32_t)};
    void* host{};
    std::vector<uint32_t> initial;
    explicit replay_storage(unsigned seed) : initial(replay_state_words) {
        TEST_CUDA(cuMemAllocHost(&host, replay_slots * replay_slot_words * sizeof(uint32_t)));
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = 0xcafe0000u + uint32_t(i) * 0x20201u;
        for (size_t i = 0; i < replay_lanes; ++i) initial[i + replay_guard_words] = 0xfffffff0u + uint32_t(i) * 0x01020305u + seed * 0x103u;
        TEST_CUDA(cuMemcpyHtoD(device.value, initial.data(), initial.size() * sizeof(uint32_t)));
        std::memset(host, 0xd3, replay_slots * replay_slot_words * sizeof(uint32_t));
    }
    ~replay_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    uint32_t* slot(unsigned index) const {
        require(index < replay_slots, "valid replay readback slot");
        return static_cast<uint32_t*>(host) + index * replay_slot_words;
    }
    void* destination(unsigned index) const { return slot(index) + replay_guard_words; }
    void expect(unsigned index, std::span<const uint32_t> state, const std::string& context) const {
        std::vector<uint32_t> wanted(replay_slot_words, replay_host_guard);
        require(state.size() == replay_state_words, "complete replay state oracle");
        std::copy(state.begin(), state.end(), wanted.begin() + replay_guard_words);
        expect_words({slot(index), replay_slot_words}, wanted, context + " slot=" + std::to_string(index));
    }
    void expect_unused(unsigned first) const {
        for (unsigned i = first; i < replay_slots; ++i) for (size_t j = 0; j < replay_slot_words; ++j)
            require(slot(i)[j] == replay_host_guard, "unused replay slots retain every initialized byte");
    }
    void expect_device(std::span<const uint32_t> expected) const {
        std::vector<uint32_t> actual(replay_state_words);
        TEST_CUDA(cuMemcpyDtoH(actual.data(), device.value, actual.size() * sizeof(uint32_t)));
        expect_words(actual, expected, "final replay state and device allocation guards");
    }
};
struct replay_stream {
    CUstream value{};
    replay_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~replay_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { TEST_CUDA(cuStreamSynchronize(value)); }
};
struct replay_gate {
    std::mutex mutex; std::condition_variable changed;
    bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<replay_gate*>(argument);
        std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(15), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "replay gate entry deadline"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void expect_held() { std::lock_guard lock(mutex); require(entered && !released && !expired, "accepted replay remains gated during host edits"); }
};
struct replay_gate_cleanup {
    replay_gate& gate;
    ~replay_gate_cleanup() { gate.release(); cuCtxSynchronize(); }
};
struct replay_graph {
    CUgraph value{};
    CUgraphNode kernel{}, read{};
    CUfunction function{};
    CUcontext context{};
    replay_graph(CUfunction entry, replay_storage& storage, unsigned count, uint32_t add, bool runtime) : function(entry) {
        TEST_CUDA(cuCtxGetCurrent(&context));
        if (runtime) { cudaGraph_t graph{}; REPLAY_RUNTIME(cudaGraphCreate(&graph, 0)); value = reinterpret_cast<CUgraph>(graph); }
        else TEST_CUDA(cuGraphCreate(&value, 0));
        try { configure(storage, 0, count, add, runtime); }
        catch (...) { cuGraphDestroy(value); value = nullptr; throw; }
    }
    ~replay_graph() { if (value) cuGraphDestroy(value); }
    void configure(replay_storage& storage, unsigned slot, unsigned count, uint32_t add, bool runtime, CUgraphExec executable = nullptr) {
        auto pointer = storage.device.value + replay_guard_words * sizeof(uint32_t);
        std::array<void*, 3> arguments{&pointer, &count, &add};
        CUDA_KERNEL_NODE_PARAMS params{}; params.func = function; params.gridDimX = 3; params.gridDimY = params.gridDimZ = 1;
        params.blockDimX = 32; params.blockDimY = params.blockDimZ = 1; params.kernelParams = arguments.data();
        if (runtime) {
            cudaGraphNodeParams p{}; p.type = cudaGraphNodeTypeKernel;
            p.kernel.functionType = cudaKernelFunctionTypeFunction; p.kernel.cuFunc = reinterpret_cast<cudaFunction_t>(function);
            p.kernel.gridDim = dim3(3); p.kernel.blockDim = dim3(32); p.kernel.kernelParams = arguments.data();
            if (executable) REPLAY_RUNTIME(cudaGraphExecNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaGraphNode_t>(kernel), &p));
            else if (kernel) REPLAY_RUNTIME(cudaGraphNodeSetParams(reinterpret_cast<cudaGraphNode_t>(kernel), &p));
            else {
                cudaGraphNode_t node{}; REPLAY_RUNTIME(cudaGraphAddNode(&node, reinterpret_cast<cudaGraph_t>(value), nullptr, nullptr, 0, &p)); kernel = reinterpret_cast<CUgraphNode>(node);
            }
        } else if (executable) TEST_CUDA(cuGraphExecKernelNodeSetParams(executable, kernel, &params));
        else if (kernel) TEST_CUDA(cuGraphKernelNodeSetParams(kernel, &params));
        else TEST_CUDA(cuGraphAddKernelNode(&kernel, value, nullptr, 0, &params));
        const size_t bytes = replay_state_words * sizeof(uint32_t);
        if (runtime) {
            cudaMemcpy3DParms p{}; p.srcPtr = {reinterpret_cast<void*>(static_cast<uintptr_t>(storage.device.value)), bytes, bytes, 1};
            p.dstPtr = {storage.destination(slot), bytes, bytes, 1}; p.extent = {bytes, 1, 1}; p.kind = cudaMemcpyDeviceToHost;
            if (executable) REPLAY_RUNTIME(cudaGraphExecMemcpyNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaGraphNode_t>(read), &p));
            else if (read) REPLAY_RUNTIME(cudaGraphMemcpyNodeSetParams(reinterpret_cast<cudaGraphNode_t>(read), &p));
            else {
                cudaGraphNode_t node{}; const auto dependency = reinterpret_cast<cudaGraphNode_t>(kernel);
                REPLAY_RUNTIME(cudaGraphAddMemcpyNode(&node, reinterpret_cast<cudaGraph_t>(value), &dependency, 1, &p)); read = reinterpret_cast<CUgraphNode>(node);
            }
        } else {
            CUDA_MEMCPY3D p{}; p.srcMemoryType = CU_MEMORYTYPE_DEVICE; p.srcDevice = storage.device.value;
            p.dstMemoryType = CU_MEMORYTYPE_HOST; p.dstHost = storage.destination(slot);
            p.srcPitch = p.dstPitch = p.WidthInBytes = bytes; p.srcHeight = p.dstHeight = p.Height = p.Depth = 1;
            if (executable) TEST_CUDA(cuGraphExecMemcpyNodeSetParams(executable, read, &p, context));
            else if (read) TEST_CUDA(cuGraphMemcpyNodeSetParams(read, &p));
            else TEST_CUDA(cuGraphAddMemcpyNode(&read, value, &kernel, 1, &p, context));
        }
        // No launch may retain pointers into this caller-owned argument storage.
        count = 0; add = 0; pointer = 0;
    }
};
struct replay_executable {
    CUgraphExec value{};
    replay_executable(CUgraph graph, bool runtime) {
        if (runtime) { cudaGraphExec_t result{}; REPLAY_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), 0)); value = reinterpret_cast<CUgraphExec>(result); }
        else TEST_CUDA(cuGraphInstantiate(&value, graph, 0));
    }
    ~replay_executable() { if (value) { cuCtxSynchronize(); cuGraphExecDestroy(value); } }
    void launch(CUstream stream, bool runtime) const {
        if (runtime) REPLAY_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else TEST_CUDA(cuGraphLaunch(value, stream));
    }
};
void replay_repeated(CUfunction function, bool runtime) {
    for (unsigned count : {0u, 1u, 63u, 96u, 113u}) {
        replay_storage storage(count); replay_stream stream;
        replay_graph graph(function, storage, count, 0xffffffedu, runtime); replay_executable executable(graph.value, !runtime);
        auto expected = storage.initial;
        for (unsigned generation = 0; generation < 7; ++generation) {
            executable.launch(stream.value, (generation & 1) != runtime); replay_apply(expected, count, 0xffffffedu);
            stream.synchronize(); storage.expect(0, expected, "repeated generation=" + std::to_string(generation));
        }
        storage.expect_unused(1); storage.expect_device(expected);
    }
}
void replay_serialization(CUfunction function, bool runtime) {
    replay_gate gate; replay_storage storage(17); replay_stream first, second;
    replay_graph graph(function, storage, 91, 0x12345678, runtime); replay_executable executable(graph.value, runtime);
    replay_gate_cleanup cleanup{gate};
    TEST_CUDA(cuLaunchHostFunc(first.value, replay_gate::call, &gate)); gate.await();
    executable.launch(first.value, runtime); executable.launch(second.value, !runtime);
    require(cuStreamQuery(first.value) == CUDA_ERROR_NOT_READY && cuStreamQuery(second.value) == CUDA_ERROR_NOT_READY,
        "same executable launches remain ordered behind its first gated stream");
    storage.expect_unused(0); gate.expect_held(); gate.release();
    second.synchronize(); first.synchronize();
    auto expected = storage.initial; replay_apply(expected, 91, 0x12345678); replay_apply(expected, 91, 0x12345678);
    storage.expect(0, expected, "same executable serialized across streams"); storage.expect_unused(1); storage.expect_device(expected);
    require(!gate.expired, "serialized replay gate deadline");
}
void replay_snapshots(CUfunction function, bool runtime) {
    constexpr std::array<unsigned, 6> counts{93, 0, 1, 65, 96, 17};
    constexpr std::array<uint32_t, 6> addends{0xfffffff0, 7, 0x80000000, 0xffffffff, 0x11223344, 0};
    replay_gate gate; replay_storage storage(23); replay_stream first, second;
    replay_graph graph(function, storage, counts[0], addends[0], runtime); replay_executable executable(graph.value, !runtime);
    replay_gate_cleanup cleanup{gate};
    TEST_CUDA(cuLaunchHostFunc(first.value, replay_gate::call, &gate)); gate.await();
    std::array<std::vector<uint32_t>, counts.size()> snapshots;
    auto expected = storage.initial;
    for (unsigned generation = 0; generation < counts.size(); ++generation) {
        if (generation) graph.configure(storage, generation, counts[generation], addends[generation], (generation & 1) != runtime, executable.value);
        executable.launch((generation & 1) ? second.value : first.value, (generation & 1) == runtime);
        replay_apply(expected, counts[generation], addends[generation]); snapshots[generation] = expected;
    }
    storage.expect_unused(0); gate.expect_held(); gate.release(); first.synchronize(); second.synchronize();
    for (unsigned generation = 0; generation < snapshots.size(); ++generation)
        storage.expect(generation, snapshots[generation], "queued kernel/copy parameter snapshot");
    storage.expect_unused(unsigned(snapshots.size())); storage.expect_device(expected);
    require(!gate.expired, "queued replay snapshot gate deadline");
}
void replay_source_edits(CUfunction function, bool runtime) {
    replay_gate gate; replay_storage before(31), after(73); replay_stream first, second;
    replay_graph source(function, before, 89, 0x31, runtime); replay_executable original(source.value, !runtime);
    replay_gate_cleanup cleanup{gate};
    TEST_CUDA(cuLaunchHostFunc(first.value, replay_gate::call, &gate)); gate.await();
    original.launch(first.value, runtime);
    source.configure(after, 0, 67, 0xabcdef01, !runtime);
    // This executable lives through gate cleanup even when later work fails.
    // Its destructor must not wait before a held host callback is released.
    CUgraphExec changed{};
    if (runtime) { cudaGraphExec_t result{}; REPLAY_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(source.value), 0)); changed = reinterpret_cast<CUgraphExec>(result); }
    else TEST_CUDA(cuGraphInstantiate(&changed, source.value, 0));
    struct changed_cleanup {
        CUgraphExec value; replay_gate& gate;
        ~changed_cleanup() { gate.release(); cuCtxSynchronize(); cuGraphExecDestroy(value); }
    } retire{changed, gate};
    for (unsigned generation = 0; generation < 5; ++generation) {
        original.launch(first.value, generation & 1);
        if (generation & 1) REPLAY_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(changed), reinterpret_cast<cudaStream_t>(second.value)));
        else TEST_CUDA(cuGraphLaunch(changed, second.value));
    }
    before.expect_unused(0); gate.expect_held(); gate.release(); first.synchronize(); second.synchronize();
    auto old_expected = before.initial, new_expected = after.initial;
    for (unsigned generation = 0; generation < 6; ++generation) replay_apply(old_expected, 89, 0x31);
    for (unsigned generation = 0; generation < 5; ++generation) replay_apply(new_expected, 67, 0xabcdef01);
    before.expect(0, old_expected, "pending executable retains old source allocation/arguments");
    after.expect(0, new_expected, "later executable uses edited source allocation/arguments");
    before.expect_unused(1); after.expect_unused(1); before.expect_device(old_expected); after.expect_device(new_expected);
    require(!gate.expired, "source edit replay gate deadline");
}
struct replay_thread_start {
    std::mutex mutex; std::condition_variable changed;
    unsigned ready = 0; bool released = false;
    void enter() {
        std::unique_lock lock(mutex); ++ready; changed.notify_all();
        require(changed.wait_for(lock, std::chrono::seconds(10), [&] { return released; }), "replay host worker release deadline");
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return ready == 2; }), "replay host workers entry deadline"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
void replay_concurrent(CUfunction function, bool runtime) {
    replay_storage first(53), second(97); replay_stream left, right;
    replay_graph a(function, first, 79, 0x13579bdf, runtime), b(function, second, 95, 0x2468ace0, !runtime);
    replay_executable x(a.value, runtime), y(b.value, !runtime);
    CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context));
    replay_thread_start start;
    std::array<std::exception_ptr, 2> errors;
    const auto work = [&](unsigned index, const replay_executable& executable, CUstream stream) {
        struct unbind { ~unbind() { cuCtxSetCurrent(nullptr); } } unbind;
        try {
            TEST_CUDA(cuCtxSetCurrent(context)); start.enter();
            for (unsigned generation = 0; generation < 11; ++generation) executable.launch(stream, (generation & 1) == runtime);
        } catch (...) { errors[index] = std::current_exception(); }
    };
    std::jthread one([&] { work(0, x, left.value); }), two([&] { work(1, y, right.value); });
    struct release_workers { replay_thread_start& start; ~release_workers() { start.release(); } } cleanup{start};
    start.await(); start.release(); one.join(); two.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    left.synchronize(); right.synchronize();
    auto expected_a = first.initial, expected_b = second.initial;
    for (unsigned generation = 0; generation < 11; ++generation) { replay_apply(expected_a, 79, 0x13579bdf); replay_apply(expected_b, 95, 0x2468ace0); }
    first.expect(0, expected_a, "concurrent independent executable A"); second.expect(0, expected_b, "concurrent independent executable B");
    first.expect_unused(1); second.expect_unused(1); first.expect_device(expected_a); second.expect_device(expected_b);
    CUcontext current{}; TEST_CUDA(cuCtxGetCurrent(&current)); require(current == context, "worker replay submissions preserve the main context binding");
}
void execute_graph_replay(const loaded_program& program) {
    const auto function = program.function("replay_step");
    const std::array<std::pair<const char*, void (*)(CUfunction, bool)>, 5> cases{{
        {"repeated", replay_repeated}, {"serialization", replay_serialization}, {"snapshots", replay_snapshots},
        {"source_edits", replay_source_edits}, {"concurrent", replay_concurrent}}};
    for (const auto& [name, execute] : cases) for (bool runtime : {false, true}) {
        std::cout << "graph_replay=" << name << " api=" << (runtime ? "runtime" : "driver") << std::endl;
        execute(function, runtime);
    }
}
}
int main(int argc, char** argv) { return cudavm::run_test(argc, argv, cudavm::fixtures, cudavm::execute_graph_replay); }
