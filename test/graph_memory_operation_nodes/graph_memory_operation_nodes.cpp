#include "test_framework.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>

namespace cudavm {
template<class T> struct memory_node_guarded {
    uint64_t prefix = 0x123456789abcdef0; T value{}; uint64_t suffix = 0xfedcba9876543210;
    void check() const { require(prefix == 0x123456789abcdef0 && suffix == 0xfedcba9876543210, "memory-node output record guards"); }
};
constexpr size_t memory_node_bytes = 293, memory_node_host_bytes = 337, memory_node_host_offset = 19;
constexpr uint32_t memory_node_fill = 0x41414141;
struct memory_node_storage {
    device_buffer device{memory_node_bytes}; void* host{}; std::vector<unsigned char> initial;
    memory_node_storage() : initial(memory_node_bytes) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = static_cast<unsigned char>(i * 37 + 13);
        CONFORMANCE_CUDA(cuMemAllocHost(&host, memory_node_host_bytes));
        try { reset(); } catch (...) { cuMemFreeHost(host); host = nullptr; throw; }
    }
    ~memory_node_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    void reset() { CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, initial.data(), initial.size())); std::memset(host, 0xd3, memory_node_host_bytes); }
    std::vector<unsigned char> expected(unsigned variant, bool operations = true) const {
        auto bytes = initial; std::fill_n(bytes.begin() + 32, 16, 0x41);
        if (operations) {
            const uint32_t narrow = 0xfedcba90u + variant;
            const uint64_t wide = 0x81234567abcdef00ull + variant;
            std::memcpy(bytes.data() + 64 + variant * 32, &narrow, sizeof(narrow));
            std::memcpy(bytes.data() + 72 + variant * 32, &wide, sizeof(wide));
        }
        return bytes;
    }
    void expect(std::span<const unsigned char> bytes) const {
        std::vector<unsigned char> wanted(memory_node_host_bytes, 0xd3);
        std::copy(bytes.begin(), bytes.end(), wanted.begin() + memory_node_host_offset);
        require(std::memcmp(host, wanted.data(), wanted.size()) == 0, "memory-node payload, neighbors and complete device/host guards");
    }
    void untouched() const {
        require(std::all_of(static_cast<unsigned char*>(host), static_cast<unsigned char*>(host) + memory_node_host_bytes,
            [](unsigned char byte) { return byte == 0xd3; }), "memory-node upload does not execute readback");
        std::vector<unsigned char> actual(memory_node_bytes); CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), device.value, actual.size()));
        require(actual == initial, "memory-node construction/upload/invalid edits do not modify device storage");
    }
};
CUstreamBatchMemOpParams memory_node_write(CUdeviceptr address, uint64_t value, bool wide) {
    CUstreamBatchMemOpParams p{}; p.operation = wide ? CU_STREAM_MEM_OP_WRITE_VALUE_64 : CU_STREAM_MEM_OP_WRITE_VALUE_32;
    p.writeValue.address = address; if (wide) p.writeValue.value64 = value; else p.writeValue.value = static_cast<uint32_t>(value); return p;
}
CUstreamBatchMemOpParams memory_node_wait(CUdeviceptr address, uint64_t value, bool wide, unsigned predicate = CU_STREAM_WAIT_VALUE_EQ) {
    CUstreamBatchMemOpParams p{}; p.operation = wide ? CU_STREAM_MEM_OP_WAIT_VALUE_64 : CU_STREAM_MEM_OP_WAIT_VALUE_32;
    p.waitValue.address = address; if (wide) p.waitValue.value64 = value; else p.waitValue.value = static_cast<uint32_t>(value); p.waitValue.flags = predicate; return p;
}
std::vector<CUstreamBatchMemOpParams> memory_node_operations(const memory_node_storage& storage, unsigned variant = 0, unsigned predicate = CU_STREAM_WAIT_VALUE_EQ) {
    const auto address = storage.device.value + 64 + variant * 32;
    auto barrier = CUstreamBatchMemOpParams{}; barrier.operation = CU_STREAM_MEM_OP_BARRIER; barrier.memoryBarrier.flags = CU_STREAM_MEMORY_BARRIER_TYPE_SYS;
    auto gpu_barrier = barrier; gpu_barrier.memoryBarrier.flags = CU_STREAM_MEMORY_BARRIER_TYPE_GPU;
    return {memory_node_wait(storage.device.value + 32, memory_node_fill, false),
        memory_node_write(address, 0xfedcba90u + variant, false), memory_node_wait(address, 0xfedcba90u + variant, false, predicate), barrier,
        memory_node_write(address + 8, 0x81234567abcdef00ull + variant, true), memory_node_wait(address + 8, 0x81234567abcdef00ull + variant, true, predicate), gpu_barrier};
}
CUDA_BATCH_MEM_OP_NODE_PARAMS memory_node_params(std::vector<CUstreamBatchMemOpParams>& ops) {
    CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context)); return {context, static_cast<unsigned>(ops.size()), ops.data(), 0};
}
CUgraphNodeParams memory_node_generic(const CUDA_BATCH_MEM_OP_NODE_PARAMS& p) {
    CUgraphNodeParams result{}; result.type = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP; result.memOp = {p.ctx, p.count, p.paramArray, p.flags}; return result;
}
CUresult memory_node_set(CUgraphNode node, const CUDA_BATCH_MEM_OP_NODE_PARAMS& p, bool generic, CUgraphExec executable = nullptr) {
    if (generic) { auto params = memory_node_generic(p); return executable ? cuGraphExecNodeSetParams(executable, node, &params) : cuGraphNodeSetParams(node, &params); }
    return executable ? cuGraphExecBatchMemOpNodeSetParams(executable, node, &p) : cuGraphBatchMemOpNodeSetParams(node, &p);
}
void memory_node_query(CUgraphNode node, const CUDA_BATCH_MEM_OP_NODE_PARAMS& expected) {
    memory_node_guarded<CUgraphNodeType> type; CONFORMANCE_CUDA(cuGraphNodeGetType(node, &type.value)); type.check();
    require(type.value == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP, "memory-operation node type");
    memory_node_guarded<CUDA_BATCH_MEM_OP_NODE_PARAMS> typed;
    CONFORMANCE_CUDA(cuGraphBatchMemOpNodeGetParams(node, &typed.value)); typed.check();
    memory_node_guarded<CUgraphNodeParams> generic;
    CONFORMANCE_CUDA(cuGraphNodeGetParams(node, &generic.value)); generic.check();
    require(generic.value.type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP, "generic memory-operation node type");
    const auto check = [&](const auto& p) {
        require(p.ctx == expected.ctx && p.count == expected.count && p.flags == expected.flags && (!p.count || p.paramArray), "copied memory-operation context/count/flags");
        for (unsigned i = 0; i < p.count; ++i) {
            const auto& a = p.paramArray[i]; const auto& b = expected.paramArray[i]; require(a.operation == b.operation, "copied memory-operation tag/order");
            if (b.operation == CU_STREAM_MEM_OP_BARRIER) require(a.memoryBarrier.flags == b.memoryBarrier.flags, "copied memory barrier scope");
            else if (b.operation == CU_STREAM_MEM_OP_WRITE_VALUE_32 || b.operation == CU_STREAM_MEM_OP_WRITE_VALUE_64)
                require(a.writeValue.address == b.writeValue.address && a.writeValue.flags == b.writeValue.flags
                    && (b.operation == CU_STREAM_MEM_OP_WRITE_VALUE_32 ? a.writeValue.value == b.writeValue.value : a.writeValue.value64 == b.writeValue.value64), "copied write address/value/flags");
            else require(a.waitValue.address == b.waitValue.address && a.waitValue.flags == b.waitValue.flags
                && (b.operation == CU_STREAM_MEM_OP_WAIT_VALUE_32 ? a.waitValue.value == b.waitValue.value : a.waitValue.value64 == b.waitValue.value64), "copied wait address/value/predicate");
        }
    };
    check(typed.value); check(generic.value.memOp);
}
struct memory_node_stream {
    CUstream value{}; memory_node_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~memory_node_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct memory_node_graph {
    CUgraph value{}; CUgraphNode fill{}, node{}, read{};
    memory_node_graph() { CONFORMANCE_CUDA(cuGraphCreate(&value, 0)); }
    ~memory_node_graph() { if (value) cuGraphDestroy(value); }
    void build(memory_node_storage& storage, std::vector<CUstreamBatchMemOpParams>& ops, bool generic) {
        auto params = memory_node_params(ops);
        CUDA_MEMSET_NODE_PARAMS producer{}; producer.dst = storage.device.value + 32; producer.value = memory_node_fill; producer.elementSize = 4; producer.width = 4; producer.height = 1;
        CONFORMANCE_CUDA(cuGraphAddMemsetNode(&fill, value, nullptr, 0, &producer, params.ctx));
        if (generic) { auto p = memory_node_generic(params); CONFORMANCE_CUDA(cuGraphAddNode(&node, value, &fill, nullptr, 1, &p)); }
        else CONFORMANCE_CUDA(cuGraphAddBatchMemOpNode(&node, value, &fill, 1, &params));
        CUDA_MEMCPY3D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_DEVICE; copy.srcDevice = storage.device.value;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = static_cast<unsigned char*>(storage.host) + memory_node_host_offset;
        copy.srcPitch = copy.dstPitch = copy.WidthInBytes = memory_node_bytes; copy.srcHeight = copy.dstHeight = copy.Height = copy.Depth = 1;
        CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&read, value, &node, 1, &copy, params.ctx));
    }
    void clone(const memory_node_graph& source) {
        CONFORMANCE_CUDA(cuGraphDestroy(value)); value = nullptr; CONFORMANCE_CUDA(cuGraphClone(&value, source.value));
        CONFORMANCE_CUDA(cuGraphNodeFindInClone(&node, source.node, value));
    }
};
struct memory_node_executable {
    CUgraphExec value{};
    explicit memory_node_executable(CUgraph graph) { CONFORMANCE_CUDA(cuGraphInstantiate(&value, graph, 0)); }
    ~memory_node_executable() { if (value) cuGraphExecDestroy(value); }
    void launch(CUstream stream) const { CONFORMANCE_CUDA(cuGraphLaunch(value, stream)); }
};
void memory_node_ordering(bool generic) {
    memory_node_storage storage; memory_node_stream stream;
    for (unsigned variant : {0u, 1u, 2u}) {
        storage.reset(); memory_node_graph graph; auto ops = memory_node_operations(storage, variant), expected_ops = ops;
        graph.build(storage, ops, generic); std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{});
        auto expected = memory_node_params(expected_ops); memory_node_query(graph.node, expected);
        memory_node_executable executable(graph.value); CONFORMANCE_CUDA(cuGraphUpload(executable.value, stream.value)); stream.synchronize(); storage.untouched();
        for (unsigned repeat = 0; repeat < 3; ++repeat) { storage.reset(); executable.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(variant)); }
    }
    for (unsigned count : {0u, 1u, 255u}) {
        storage.reset(); memory_node_graph graph; std::vector<CUstreamBatchMemOpParams> ops(count, memory_node_write(storage.device.value + 64, 0xfedcba90, false));
        graph.build(storage, ops, generic); auto expected_ops = ops; std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{});
        auto expected_params = memory_node_params(expected_ops); memory_node_query(graph.node, expected_params);
        memory_node_executable executable(graph.value); executable.launch(stream.value); stream.synchronize();
        auto expected = storage.expected(0, false); if (count) { const uint32_t narrow = 0xfedcba90; std::memcpy(expected.data() + 64, &narrow, sizeof(narrow)); } storage.expect(expected);
    }
}
void memory_node_parameters(bool generic) {
    memory_node_storage storage; memory_node_stream stream; memory_node_graph source, clone;
    auto ops = memory_node_operations(storage); source.build(storage, ops, generic); clone.clone(source); memory_node_executable original(source.value), cloned(clone.value);
    auto first = memory_node_operations(storage); auto expected_first = memory_node_params(first);
    ops = memory_node_operations(storage, 1); auto params = memory_node_params(ops); CONFORMANCE_CUDA(memory_node_set(source.node, params, generic)); std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{});
    auto second = memory_node_operations(storage, 1); auto expected_second = memory_node_params(second);
    memory_node_query(source.node, expected_second); memory_node_query(clone.node, expected_first); memory_node_executable changed(source.value);
    for (auto [executable, variant] : {std::pair{original.value, 0u}, std::pair{cloned.value, 0u}, std::pair{changed.value, 1u}}) {
        storage.reset(); CONFORMANCE_CUDA(cuGraphLaunch(executable, stream.value)); stream.synchronize(); storage.expect(storage.expected(variant));
    }
    CONFORMANCE_CUDA(cuGraphDestroy(source.value)); source.value = nullptr; memory_node_query(clone.node, expected_first);
    ops = memory_node_operations(storage, 2); params = memory_node_params(ops); CONFORMANCE_CUDA(memory_node_set(clone.node, params, !generic)); std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{});
    memory_node_executable edited(clone.value); storage.reset(); edited.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(2));
    storage.reset(); cloned.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(0));
}
void memory_node_executable_updates(bool generic) {
    memory_node_storage storage; memory_node_stream stream; memory_node_graph graph;
    auto ops = memory_node_operations(storage); graph.build(storage, ops, generic); memory_node_executable executable(graph.value);
    auto original_ops = memory_node_operations(storage); auto original = memory_node_params(original_ops);
    ops = memory_node_operations(storage, 1, CU_STREAM_WAIT_VALUE_GEQ); auto params = memory_node_params(ops);
    CONFORMANCE_CUDA(memory_node_set(graph.node, params, generic, executable.value)); std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{});
    memory_node_query(graph.node, original);
    for (unsigned invalid = 0; invalid < 4; ++invalid) {
        ops = memory_node_operations(storage, 2); params = memory_node_params(ops);
        if (invalid == 0) --params.count;
        if (invalid == 1) ops[1] = memory_node_write(storage.device.value + 128, 9, true);
        if (invalid == 2) ops[3].memoryBarrier.flags = CU_STREAM_MEMORY_BARRIER_TYPE_GPU;
        if (invalid == 3) ops[1].writeValue.flags = CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER;
        require(memory_node_set(graph.node, params, generic, executable.value) == CUDA_ERROR_INVALID_VALUE, "incompatible executable memory-operation edits reject");
        memory_node_query(graph.node, original); storage.reset(); executable.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(1));
    }
    ops = memory_node_operations(storage, 2); params = memory_node_params(ops); CONFORMANCE_CUDA(memory_node_set(graph.node, params, !generic, executable.value));
    std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{}); storage.reset(); executable.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(2));
    memory_node_executable source(graph.value); storage.reset(); source.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(0));
}
struct memory_node_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<memory_node_gate*>(argument); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "memory-node pending gate entry"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void check() { std::lock_guard lock(mutex); require(entered && !expired, "memory-node gate did not expire"); }
};
struct memory_node_gate_cleanup { memory_node_gate& gate; ~memory_node_gate_cleanup() { gate.release(); cuCtxSynchronize(); } };
void memory_node_pending(bool generic) {
    memory_node_storage storage; memory_node_stream stream; memory_node_graph graph; auto ops = memory_node_operations(storage); graph.build(storage, ops, generic);
    memory_node_executable executable(graph.value); memory_node_gate gate; memory_node_gate_cleanup cleanup{gate};
    CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, memory_node_gate::call, &gate)); gate.await(); executable.launch(stream.value);
    ops = memory_node_operations(storage, 1); auto params = memory_node_params(ops); CONFORMANCE_CUDA(memory_node_set(graph.node, params, generic, executable.value));
    std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{}); require(cuStreamQuery(stream.value) == CUDA_ERROR_NOT_READY, "accepted memory-node launch remains pending");
    gate.release(); stream.synchronize(); gate.check(); storage.expect(storage.expected(0));
    storage.reset(); executable.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(1));
    storage.reset(); memory_node_gate retiring; memory_node_gate_cleanup retire_cleanup{retiring};
    CONFORMANCE_CUDA(cuLaunchHostFunc(stream.value, memory_node_gate::call, &retiring)); retiring.await(); executable.launch(stream.value);
    CONFORMANCE_CUDA(cuGraphExecDestroy(executable.value)); executable.value = nullptr; CONFORMANCE_CUDA(cuGraphDestroy(graph.value)); graph.value = nullptr;
    retiring.release(); stream.synchronize(); retiring.check(); storage.expect(storage.expected(1));
}
void memory_node_validation(bool generic) {
    memory_node_storage storage; memory_node_stream stream; memory_node_graph graph; auto ops = memory_node_operations(storage); graph.build(storage, ops, generic);
    auto original_ops = ops; auto original = memory_node_params(original_ops); CUgraphNode empty{}; CONFORMANCE_CUDA(cuGraphAddEmptyNode(&empty, graph.value, nullptr, 0));
    for (unsigned invalid = 0; invalid < 11; ++invalid) {
        ops = memory_node_operations(storage, 1); auto params = memory_node_params(ops);
        if (invalid == 0) params.paramArray = nullptr;
        if (invalid == 1) { ops.resize(256, ops[1]); params.paramArray = ops.data(); params.count = 256; }
        if (invalid == 2) params.flags = 1;
        if (invalid == 3) params.ctx = nullptr;
        if (invalid == 4) ops[1].writeValue.address = 0;
        if (invalid == 5) ops[1].writeValue.address = storage.device.value + 65;
        if (invalid == 6) ops[4].writeValue.address = storage.device.value + 284;
        if (invalid == 7) ops[4].writeValue.address = storage.device.value + 288;
        if (invalid == 8) ops[2].waitValue.flags = 4;
        if (invalid == 9) ops[3].memoryBarrier.flags = 2;
        if (invalid == 10) ops[1].operation = static_cast<CUstreamBatchMemOpType>(99);
        memory_node_guarded<CUgraphNode> added;
        auto p = memory_node_generic(params);
        const auto status = generic ? cuGraphAddNode(&added.value, graph.value, nullptr, nullptr, 0, &p) : cuGraphAddBatchMemOpNode(&added.value, graph.value, nullptr, 0, &params);
        require(status == CUDA_ERROR_INVALID_VALUE, "invalid memory-operation additions reject"); added.check();
        require(memory_node_set(graph.node, params, generic) == CUDA_ERROR_INVALID_VALUE, "invalid memory-operation source setters reject");
        memory_node_query(graph.node, original); size_t count{}; CONFORMANCE_CUDA(cuGraphGetNodes(graph.value, nullptr, &count)); require(count == 4, "invalid edits preserve graph membership"); storage.untouched();
    }
    require(cuGraphBatchMemOpNodeGetParams(graph.node, nullptr) == CUDA_ERROR_INVALID_VALUE, "null memory-operation getter output");
    require(cuGraphBatchMemOpNodeSetParams(graph.node, nullptr) == CUDA_ERROR_INVALID_VALUE, "null memory-operation setter input");
    memory_node_guarded<CUDA_BATCH_MEM_OP_NODE_PARAMS> out;
    require(cuGraphBatchMemOpNodeGetParams(empty, &out.value) == CUDA_ERROR_INVALID_VALUE, "wrong-type memory-operation getter"); out.check();
    require(memory_node_set(empty, original, generic) == CUDA_ERROR_INVALID_VALUE, "wrong-type memory-operation setter");
    memory_node_executable executable(graph.value);
    require(cuGraphExecBatchMemOpNodeSetParams(executable.value, graph.node, nullptr) == CUDA_ERROR_INVALID_VALUE, "null executable memory-operation parameters");
    executable.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(0));
    ops = memory_node_operations(storage, 2); auto params = memory_node_params(ops); CONFORMANCE_CUDA(memory_node_set(graph.node, params, generic));
    std::fill(ops.begin(), ops.end(), CUstreamBatchMemOpParams{}); memory_node_executable repaired(graph.value); storage.reset(); repaired.launch(stream.value); stream.synchronize(); storage.expect(storage.expected(2));
}
std::string memory_node_case;
void execute_memory_nodes() {
    const std::array<std::pair<const char*, void (*)(bool)>, 5> cases{{{"ordering", memory_node_ordering}, {"parameters", memory_node_parameters},
        {"executable", memory_node_executable_updates}, {"pending", memory_node_pending}, {"validation", memory_node_validation}}};
    bool found = false;
    for (auto [name, execute] : cases) if (memory_node_case == name) { found = true; for (bool generic : {false, true}) { std::cout << "graph_memory_operation_nodes=" << name << " generic=" << generic << std::endl; execute(generic); } }
    require(found, "unknown graph memory operation case");
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    cudavm::memory_node_case = argv[2]; return cudavm::run_conformance(2, argv, cudavm::execute_memory_nodes);
}
