#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>

namespace cudavm {
void check_allocation_node_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define ALLOCATION_NODE_RUNTIME(call) ::cudavm::check_allocation_node_runtime((call), #call)
template<class T> struct allocation_node_guarded {
    uint64_t prefix = 0x123456789abcdef0; T value{}; uint64_t suffix = 0xfedcba9876543210;
    void check() const { require(prefix == 0x123456789abcdef0 && suffix == 0xfedcba9876543210, "allocation-node public output guards"); }
};
struct allocation_node_route { bool runtime, generic; };
constexpr size_t allocation_node_bytes = 311, allocation_node_host_bytes = 355, allocation_node_host_offset = 19;
void* allocation_node_pointer(CUdeviceptr pointer) { return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)); }
std::vector<unsigned char> allocation_node_expected() {
    std::vector<unsigned char> result(allocation_node_bytes, 0x9b); std::fill_n(result.begin() + 17, 193, 0x47); return result;
}
struct allocation_node_host {
    void* value{};
    allocation_node_host() { TEST_CUDA(cuMemAllocHost(&value, allocation_node_host_bytes)); reset(); }
    ~allocation_node_host() { if (value) { cuCtxSynchronize(); cuMemFreeHost(value); } }
    void reset() { std::memset(value, 0xd3, allocation_node_host_bytes); }
    void expect(bool copied = true) const {
        std::vector<unsigned char> expected(allocation_node_host_bytes, 0xd3);
        if (copied) { auto bytes = allocation_node_expected(); std::copy(bytes.begin(), bytes.end(), expected.begin() + allocation_node_host_offset); }
        require(std::memcmp(value, expected.data(), expected.size()) == 0, "graph allocation complete payload, interior neighbors and host guards");
    }
};
struct allocation_node_stream {
    CUstream value{}; allocation_node_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~allocation_node_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { TEST_CUDA(cuStreamSynchronize(value)); }
};
CUDA_MEM_ALLOC_NODE_PARAMS allocation_node_parameters() {
    CUDA_MEM_ALLOC_NODE_PARAMS p{}; p.bytesize = allocation_node_bytes; p.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE; TEST_CUDA(cuCtxGetDevice(&p.poolProps.location.id)); return p;
}
CUresult allocation_node_add(CUgraph graph, CUgraphNode& node, CUDA_MEM_ALLOC_NODE_PARAMS& p, allocation_node_route route) {
    if (route.runtime) {
        cudaMemAllocNodeParams input{}; input.bytesize = p.bytesize;
        input.poolProps.allocType = static_cast<cudaMemAllocationType>(p.poolProps.allocType); input.poolProps.handleTypes = static_cast<cudaMemAllocationHandleType>(p.poolProps.handleTypes);
        input.poolProps.location = {static_cast<cudaMemLocationType>(p.poolProps.location.type), p.poolProps.location.id};
        static_assert(sizeof(input.poolProps.reserved) >= sizeof(p.poolProps.reserved)); std::memcpy(input.poolProps.reserved, p.poolProps.reserved, sizeof(p.poolProps.reserved));
        std::vector<cudaMemAccessDesc> access;
        if (p.accessDescs) for (size_t i = 0; i < p.accessDescCount; ++i) access.push_back({{static_cast<cudaMemLocationType>(p.accessDescs[i].location.type), p.accessDescs[i].location.id}, static_cast<cudaMemAccessFlags>(p.accessDescs[i].flags)});
        input.accessDescs = access.empty() ? nullptr : access.data(); input.accessDescCount = p.accessDescCount;
        allocation_node_guarded<cudaGraphNode_t> output; cudaError_t status;
        if (route.generic) {
            cudaGraphNodeParams params{}; params.type = cudaGraphNodeTypeMemAlloc;
            params.alloc = {input.poolProps, input.accessDescs, input.accessDescCount, input.bytesize, nullptr};
            status = cudaGraphAddNode(&output.value, reinterpret_cast<cudaGraph_t>(graph), nullptr, nullptr, 0, &params);
            if (status == cudaSuccess) p.dptr = reinterpret_cast<uintptr_t>(params.alloc.dptr);
        } else {
            status = cudaGraphAddMemAllocNode(&output.value, reinterpret_cast<cudaGraph_t>(graph), nullptr, 0, &input);
            if (status == cudaSuccess) p.dptr = reinterpret_cast<uintptr_t>(input.dptr);
        }
        output.check(); if (status == cudaSuccess) node = reinterpret_cast<CUgraphNode>(output.value); return static_cast<CUresult>(status);
    }
    allocation_node_guarded<CUgraphNode> output; CUresult status;
    if (route.generic) {
        CUgraphNodeParams params{}; params.type = CU_GRAPH_NODE_TYPE_MEM_ALLOC; std::memcpy(&params.alloc, &p, sizeof(p));
        status = cuGraphAddNode(&output.value, graph, nullptr, nullptr, 0, &params); if (status == CUDA_SUCCESS) p.dptr = params.alloc.dptr;
    } else status = cuGraphAddMemAllocNode(&output.value, graph, nullptr, 0, &p);
    output.check(); if (status == CUDA_SUCCESS) node = output.value; return status;
}
CUresult allocation_node_free(CUgraph graph, CUgraphNode& node, CUdeviceptr pointer, CUgraphNode dependency, allocation_node_route route) {
    if (route.runtime) {
        allocation_node_guarded<cudaGraphNode_t> output; auto dep = reinterpret_cast<cudaGraphNode_t>(dependency); cudaError_t status;
        if (route.generic) {
            cudaGraphNodeParams p{}; p.type = cudaGraphNodeTypeMemFree; p.free.dptr = allocation_node_pointer(pointer);
            status = cudaGraphAddNode(&output.value, reinterpret_cast<cudaGraph_t>(graph), dependency ? &dep : nullptr, nullptr, dependency ? 1 : 0, &p);
        } else status = cudaGraphAddMemFreeNode(&output.value, reinterpret_cast<cudaGraph_t>(graph), dependency ? &dep : nullptr, dependency ? 1 : 0, allocation_node_pointer(pointer));
        output.check(); if (status == cudaSuccess) node = reinterpret_cast<CUgraphNode>(output.value); return static_cast<CUresult>(status);
    }
    allocation_node_guarded<CUgraphNode> output; CUresult status;
    if (route.generic) {
        CUgraphNodeParams p{}; p.type = CU_GRAPH_NODE_TYPE_MEM_FREE; p.free.dptr = pointer;
        status = cuGraphAddNode(&output.value, graph, dependency ? &dependency : nullptr, nullptr, dependency ? 1 : 0, &p);
    } else status = cuGraphAddMemFreeNode(&output.value, graph, dependency ? &dependency : nullptr, dependency ? 1 : 0, pointer);
    output.check(); if (status == CUDA_SUCCESS) node = output.value; return status;
}
void allocation_node_query(CUgraphNode node, CUdeviceptr address, CUdevice device, bool access) {
    const auto check = [&](const auto& p) {
        require(p.bytesize == allocation_node_bytes && static_cast<int>(p.poolProps.allocType) == int(CU_MEM_ALLOCATION_TYPE_PINNED)
            && static_cast<int>(p.poolProps.handleTypes) == int(CU_MEM_HANDLE_TYPE_NONE)
            && static_cast<int>(p.poolProps.location.type) == int(CU_MEM_LOCATION_TYPE_DEVICE) && p.poolProps.location.id == device,
            "exact allocation size and pool metadata");
        require(p.accessDescCount == (access ? 1u : 0u), "exact copied allocation descriptor count");
        if (access) require(p.accessDescs && p.accessDescs[0].location.id == device
            && static_cast<int>(p.accessDescs[0].location.type) == int(CU_MEM_LOCATION_TYPE_DEVICE)
            && static_cast<int>(p.accessDescs[0].flags) == int(CU_MEM_ACCESS_FLAGS_PROT_READWRITE), "allocation owns a copied access descriptor");
    };
    allocation_node_guarded<CUDA_MEM_ALLOC_NODE_PARAMS> driver; TEST_CUDA(cuGraphMemAllocNodeGetParams(node, &driver.value)); driver.check(); check(driver.value);
    require(driver.value.dptr == address, "typed Driver graph allocation address");
    allocation_node_guarded<CUgraphNodeParams> generic; TEST_CUDA(cuGraphNodeGetParams(node, &generic.value)); generic.check();
    require(generic.value.type == CU_GRAPH_NODE_TYPE_MEM_ALLOC && generic.value.alloc.dptr == address, "generic Driver graph allocation identity"); check(generic.value.alloc);
    allocation_node_guarded<cudaMemAllocNodeParams> runtime; ALLOCATION_NODE_RUNTIME(cudaGraphMemAllocNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &runtime.value)); runtime.check(); check(runtime.value);
    require(runtime.value.dptr == allocation_node_pointer(address), "typed Runtime graph allocation address");
    allocation_node_guarded<cudaGraphNodeParams> runtime_generic; ALLOCATION_NODE_RUNTIME(cudaGraphNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &runtime_generic.value)); runtime_generic.check();
    require(runtime_generic.value.type == cudaGraphNodeTypeMemAlloc && runtime_generic.value.alloc.dptr == allocation_node_pointer(address), "generic Runtime graph allocation identity"); check(runtime_generic.value.alloc);
}
void allocation_node_free_query(CUgraphNode node, CUdeviceptr address) {
    allocation_node_guarded<CUdeviceptr> driver; TEST_CUDA(cuGraphMemFreeNodeGetParams(node, &driver.value)); driver.check(); require(driver.value == address, "Driver graph free address");
    allocation_node_guarded<void*> runtime; ALLOCATION_NODE_RUNTIME(cudaGraphMemFreeNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &runtime.value)); runtime.check(); require(runtime.value == allocation_node_pointer(address), "Runtime graph free address");
    allocation_node_guarded<CUgraphNodeParams> generic; TEST_CUDA(cuGraphNodeGetParams(node, &generic.value)); generic.check(); require(generic.value.type == CU_GRAPH_NODE_TYPE_MEM_FREE && generic.value.free.dptr == address, "generic Driver free metadata");
    allocation_node_guarded<cudaGraphNodeParams> other; ALLOCATION_NODE_RUNTIME(cudaGraphNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &other.value)); other.check(); require(other.value.type == cudaGraphNodeTypeMemFree && other.value.free.dptr == allocation_node_pointer(address), "generic Runtime free metadata");
}
struct allocation_node_graph {
    CUgraph value{}; CUgraphNode allocation{}, fill{}, patch{}, read{}, free{}; CUdeviceptr address{}; CUdevice device{}; bool access{};
    explicit allocation_node_graph(bool runtime) {
        TEST_CUDA(cuCtxGetDevice(&device));
        if (runtime) { cudaGraph_t result{}; ALLOCATION_NODE_RUNTIME(cudaGraphCreate(&result, 0)); value = reinterpret_cast<CUgraph>(result); }
        else TEST_CUDA(cuGraphCreate(&value, 0));
    }
    ~allocation_node_graph() { if (value) cuGraphDestroy(value); }
    void destroy() { TEST_CUDA(cuGraphDestroy(value)); value = nullptr; }
    void build(allocation_node_host& host, allocation_node_route route, bool internal_free = true, bool descriptor = false, unsigned missing_edge = 0) {
        access = descriptor; auto p = allocation_node_parameters(); CUmemAccessDesc description{p.poolProps.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
        if (descriptor) { p.accessDescs = &description; p.accessDescCount = 1; }
        TEST_CUDA(allocation_node_add(value, allocation, p, route)); address = p.dptr; require(address, "allocation node returns a nonnull address");
        description.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE; p = {}; allocation_node_query(allocation, address, device, descriptor);
        const auto add_fill = [&](CUgraphNode& node, CUdeviceptr pointer, size_t count, unsigned byte, CUgraphNode dependency) {
            if (route.runtime) {
                cudaMemsetParams params{allocation_node_pointer(pointer), 0, byte, 1, count, 1}; auto dep = reinterpret_cast<cudaGraphNode_t>(dependency); cudaGraphNode_t result{};
                ALLOCATION_NODE_RUNTIME(cudaGraphAddMemsetNode(&result, reinterpret_cast<cudaGraph_t>(value), dependency ? &dep : nullptr, dependency ? 1 : 0, &params)); node = reinterpret_cast<CUgraphNode>(result);
            } else {
                CUDA_MEMSET_NODE_PARAMS params{}; params.dst = pointer; params.value = byte; params.elementSize = 1; params.width = count; params.height = 1; CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context));
                TEST_CUDA(cuGraphAddMemsetNode(&node, value, dependency ? &dependency : nullptr, dependency ? 1 : 0, &params, context));
            }
        };
        add_fill(fill, address, allocation_node_bytes, 0x9b, missing_edge == 1 ? nullptr : allocation);
        add_fill(patch, address + 17, 193, 0x47, fill);
        if (route.runtime) {
            cudaGraphNode_t result{}; auto dep = reinterpret_cast<cudaGraphNode_t>(patch);
            ALLOCATION_NODE_RUNTIME(cudaGraphAddMemcpyNode1D(&result, reinterpret_cast<cudaGraph_t>(value), &dep, 1,
                static_cast<unsigned char*>(host.value) + allocation_node_host_offset, allocation_node_pointer(address), allocation_node_bytes, cudaMemcpyDeviceToHost)); read = reinterpret_cast<CUgraphNode>(result);
        } else {
            CUDA_MEMCPY3D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_DEVICE; copy.srcDevice = address; copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = static_cast<unsigned char*>(host.value) + allocation_node_host_offset;
            copy.srcPitch = copy.dstPitch = copy.WidthInBytes = allocation_node_bytes; copy.srcHeight = copy.dstHeight = copy.Height = copy.Depth = 1; CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context));
            TEST_CUDA(cuGraphAddMemcpyNode(&read, value, &patch, 1, &copy, context));
        }
        if (internal_free) { TEST_CUDA(allocation_node_free(value, free, address, missing_edge == 2 ? allocation : read, route)); allocation_node_free_query(free, address); }
    }
};
struct allocation_node_executable {
    CUgraphExec value{};
    allocation_node_executable(CUgraph graph, bool runtime, unsigned long long flags = 0) {
        if (runtime) { cudaGraphExec_t result{}; ALLOCATION_NODE_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), flags)); value = reinterpret_cast<CUgraphExec>(result); }
        else TEST_CUDA(cuGraphInstantiate(&value, graph, flags));
    }
    ~allocation_node_executable() { if (value) cuGraphExecDestroy(value); }
    void destroy() { TEST_CUDA(cuGraphExecDestroy(value)); value = nullptr; }
    void launch(CUstream stream, bool runtime) const {
        if (runtime) ALLOCATION_NODE_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else TEST_CUDA(cuGraphLaunch(value, stream));
    }
};
uint64_t allocation_node_stat(CUdevice device, CUgraphMem_attribute attribute) {
    allocation_node_guarded<uint64_t> driver, runtime; driver.value = runtime.value = ~uint64_t(0);
    TEST_CUDA(cuDeviceGetGraphMemAttribute(device, attribute, &driver.value)); driver.check();
    ALLOCATION_NODE_RUNTIME(cudaDeviceGetGraphMemAttribute(device, static_cast<cudaGraphMemAttributeType>(attribute), &runtime.value)); runtime.check();
    require(driver.value == runtime.value, "both graph memory statistic surfaces agree"); return driver.value;
}
void allocation_node_trim(CUdevice device, bool runtime) {
    if (runtime) ALLOCATION_NODE_RUNTIME(cudaDeviceGraphMemTrim(device)); else TEST_CUDA(cuDeviceGraphMemTrim(device));
}
void allocation_node_footprint(CUdevice device) {
    auto used = allocation_node_stat(device, CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT), reserved = allocation_node_stat(device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT);
    require(used >= allocation_node_bytes && reserved >= used, "graph memory accounting retains allocation extent with backend granularity");
    require(allocation_node_stat(device, CU_GRAPH_MEM_ATTR_USED_MEM_HIGH) >= used && allocation_node_stat(device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH) >= reserved, "graph memory high watermarks bound current usage");
}
void allocation_node_idle(CUdevice device) {
    require(allocation_node_stat(device, CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT) == 0 && allocation_node_stat(device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT) == 0, "trimming releases all idle graph backing");
}
void allocation_node_live_bytes(CUdeviceptr pointer) {
    std::vector<unsigned char> bytes(allocation_node_bytes); TEST_CUDA(cuMemcpyDtoH(bytes.data(), pointer, bytes.size())); require(bytes == allocation_node_expected(), "live graph allocation preserves every byte outside its graph");
}
void allocation_node_replay(allocation_node_route route) {
    allocation_node_host host; allocation_node_stream stream;
    for (bool access : {false, true}) {
        host.reset();
        allocation_node_graph graph(route.runtime); graph.build(host, route, true, access); allocation_node_executable executable(graph.value, !route.runtime);
        require(cuMemsetD8(graph.address, 0, 1) == CUDA_ERROR_INVALID_VALUE, "reserved graph address is unavailable before allocation executes");
        TEST_CUDA(cuGraphUpload(executable.value, stream.value)); stream.synchronize(); host.expect(false);
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            host.reset(); executable.launch(stream.value, route.runtime); stream.synchronize(); host.expect(); allocation_node_footprint(graph.device);
            allocation_node_trim(graph.device, route.runtime); allocation_node_idle(graph.device); allocation_node_query(graph.allocation, graph.address, graph.device, access); allocation_node_free_query(graph.free, graph.address);
        }
    }
}
void allocation_node_live(allocation_node_route route) {
    allocation_node_host host; allocation_node_stream stream; allocation_node_graph graph(route.runtime); graph.build(host, route, false); allocation_node_executable executable(graph.value, route.runtime);
    executable.launch(stream.value, !route.runtime); stream.synchronize(); host.expect(); allocation_node_footprint(graph.device);
    const auto reserved = allocation_node_stat(graph.device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT); allocation_node_trim(graph.device, route.runtime);
    require(allocation_node_stat(graph.device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT) == reserved, "trimming preserves live graph memory"); allocation_node_live_bytes(graph.address);
    uint64_t zero{}; TEST_CUDA(cuDeviceSetGraphMemAttribute(graph.device, CU_GRAPH_MEM_ATTR_USED_MEM_HIGH, &zero));
    ALLOCATION_NODE_RUNTIME(cudaDeviceSetGraphMemAttribute(graph.device, cudaGraphMemAttrReservedMemHigh, &zero));
    require(allocation_node_stat(graph.device, CU_GRAPH_MEM_ATTR_USED_MEM_HIGH) == allocation_node_stat(graph.device, CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT)
        && allocation_node_stat(graph.device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH) == reserved, "reset watermarks start at the live footprint");
    executable.destroy(); graph.destroy(); allocation_node_trim(graph.device, !route.runtime); allocation_node_live_bytes(graph.address);
    if (route.runtime) ALLOCATION_NODE_RUNTIME(cudaFree(allocation_node_pointer(graph.address))); else TEST_CUDA(cuMemFree(graph.address));
    allocation_node_trim(graph.device, route.runtime); allocation_node_idle(graph.device);
}
void allocation_node_auto_free(allocation_node_route route) {
    allocation_node_host host; allocation_node_stream stream;
    for (bool automatic : {false, true}) {
        allocation_node_graph graph(route.runtime); graph.build(host, route, false); allocation_node_executable executable(graph.value, route.runtime, automatic ? CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH : 0);
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            host.reset(); executable.launch(stream.value, !route.runtime); stream.synchronize(); host.expect(); allocation_node_live_bytes(graph.address);
            if (!automatic) require(cuGraphLaunch(executable.value, stream.value) == CUDA_ERROR_INVALID_VALUE, "unfreed graph allocation rejects another ordinary launch");
            if (!automatic || repeat == 2) { TEST_CUDA(cuMemFree(graph.address)); allocation_node_trim(graph.device, route.runtime); allocation_node_idle(graph.device); }
        }
    }
}
struct allocation_node_gate {
    std::mutex mutex; std::condition_variable changed; bool entered = false, released = false, expired = false;
    static void CUDA_CB call(void* argument) {
        auto& gate = *static_cast<allocation_node_gate*>(argument); std::unique_lock lock(gate.mutex); gate.entered = true; gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate.released; })) gate.expired = true;
    }
    void await() { std::unique_lock lock(mutex); require(changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "pending graph allocation gate entry"); }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
    void check() { std::lock_guard lock(mutex); require(entered && !expired, "pending graph allocation gate deadline"); }
};
struct allocation_node_gate_cleanup { allocation_node_gate& gate; ~allocation_node_gate_cleanup() { gate.release(); cuCtxSynchronize(); } };
void allocation_node_pending(allocation_node_route route) {
    allocation_node_host host; allocation_node_stream stream;
    for (bool upload : {false, true}) {
        host.reset(); allocation_node_graph graph(route.runtime); graph.build(host, route); allocation_node_executable executable(graph.value, route.runtime);
        allocation_node_gate gate; allocation_node_gate_cleanup cleanup{gate}; TEST_CUDA(cuLaunchHostFunc(stream.value, allocation_node_gate::call, &gate)); gate.await();
        if (upload) TEST_CUDA(cuGraphUpload(executable.value, stream.value)); else executable.launch(stream.value, !route.runtime);
        auto reserved = allocation_node_stat(graph.device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT); require(reserved >= allocation_node_bytes, "accepted graph work retains allocation backing");
        if (!upload) { executable.destroy(); graph.destroy(); }
        allocation_node_trim(graph.device, route.runtime); require(allocation_node_stat(graph.device, CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT) == reserved, "trimming preserves pending graph backing");
        require(cuStreamQuery(stream.value) == CUDA_ERROR_NOT_READY, "pending graph allocation is gated");
        gate.release(); stream.synchronize(); gate.check(); host.expect(!upload); allocation_node_trim(graph.device, !route.runtime); allocation_node_idle(graph.device);
        if (upload) { executable.launch(stream.value, route.runtime); stream.synchronize(); host.expect(); allocation_node_trim(graph.device, route.runtime); allocation_node_idle(graph.device); }
    }
}
void allocation_node_restrictions(allocation_node_route route) {
    allocation_node_host host; allocation_node_stream stream;
    {
        allocation_node_graph graph(route.runtime); graph.build(host, route); allocation_node_executable executable(graph.value, route.runtime);
        allocation_node_guarded<CUgraph> clone; require(cuGraphClone(&clone.value, graph.value) == CUDA_ERROR_NOT_SUPPORTED, "allocation graphs cannot be cloned"); clone.check();
        allocation_node_guarded<CUgraphExec> second; require(cuGraphInstantiate(&second.value, graph.value, 0) == CUDA_ERROR_NOT_SUPPORTED, "allocation graph has one live executable"); second.check();
        require(cuGraphDestroyNode(graph.free) == CUDA_ERROR_INVALID_VALUE, "allocation graph nodes cannot be removed");
        require(cuGraphRemoveDependencies(graph.value, &graph.allocation, &graph.fill, nullptr, 1) == CUDA_ERROR_INVALID_VALUE, "allocation graph lifetime edges cannot be removed");
        executable.launch(stream.value, !route.runtime); stream.synchronize(); host.expect(); allocation_node_trim(graph.device, route.runtime);
        executable.destroy(); allocation_node_executable replacement(graph.value, !route.runtime); host.reset(); replacement.launch(stream.value, route.runtime); stream.synchronize(); host.expect(); allocation_node_trim(graph.device, route.runtime);
    }
    for (unsigned missing : {1u, 2u}) {
        host.reset(); allocation_node_graph graph(route.runtime); graph.build(host, route, true, false, missing);
        allocation_node_guarded<CUgraphExec> failed; require(cuGraphInstantiate(&failed.value, graph.value, 0) == CUDA_ERROR_INVALID_VALUE, "unordered graph allocation use rejects instantiation"); failed.check(); host.expect(false);
        auto from = missing == 1 ? graph.allocation : graph.read, to = missing == 1 ? graph.fill : graph.free;
        TEST_CUDA(cuGraphAddDependencies(graph.value, &from, &to, nullptr, 1)); allocation_node_executable repaired(graph.value, route.runtime);
        repaired.launch(stream.value, !route.runtime); stream.synchronize(); host.expect(); allocation_node_trim(graph.device, route.runtime); allocation_node_idle(graph.device);
    }
}
void allocation_node_validation(allocation_node_route route) {
    allocation_node_host host; allocation_node_stream stream; allocation_node_graph graph(route.runtime);
    for (unsigned invalid = 0; invalid < 4; ++invalid) {
        auto p = allocation_node_parameters(); CUgraphNode node{};
        if (invalid == 0) p.bytesize = 0;
        if (invalid == 1) p.poolProps.handleTypes = CU_MEM_HANDLE_TYPE_WIN32;
        if (invalid == 2) p.poolProps.reserved[0] = 1;
        if (invalid == 3) p.accessDescCount = 1;
        require(allocation_node_add(graph.value, node, p, route) == CUDA_ERROR_INVALID_VALUE, "invalid allocation descriptions reject");
        size_t count{}; TEST_CUDA(cuGraphGetNodes(graph.value, nullptr, &count)); require(count == 0, "invalid allocations do not add nodes"); host.expect(false);
    }
    graph.build(host, route); device_buffer ordinary(allocation_node_bytes); CUgraphNode output{};
    for (auto pointer : {CUdeviceptr(0), ordinary.value, graph.address + 1, graph.address})
        require(allocation_node_free(graph.value, output, pointer, graph.read, route) == CUDA_ERROR_INVALID_VALUE, "invalid and duplicate graph frees reject");
    require(cuGraphMemAllocNodeGetParams(graph.allocation, nullptr) == CUDA_ERROR_INVALID_VALUE, "null allocation output");
    require(cudaGraphMemFreeNodeGetParams(reinterpret_cast<cudaGraphNode_t>(graph.free), nullptr) == cudaErrorInvalidValue, "null Runtime free output");
    allocation_node_guarded<CUDA_MEM_ALLOC_NODE_PARAMS> wrong; require(cuGraphMemAllocNodeGetParams(graph.free, &wrong.value) == CUDA_ERROR_INVALID_VALUE, "wrong-type allocation getter"); wrong.check();
    allocation_node_guarded<void*> pointer; require(cudaGraphMemFreeNodeGetParams(reinterpret_cast<cudaGraphNode_t>(graph.allocation), &pointer.value) == cudaErrorInvalidValue, "wrong-type Runtime free getter"); pointer.check();
    allocation_node_guarded<uint64_t> statistic; statistic.value = 0xabcdef1234567890;
    require(cuDeviceGetGraphMemAttribute(graph.device, static_cast<CUgraphMem_attribute>(99), &statistic.value) == CUDA_ERROR_INVALID_VALUE, "invalid graph statistic"); statistic.check(); require(statistic.value == 0xabcdef1234567890, "invalid graph statistic preserves output");
    require(cudaDeviceGetGraphMemAttribute(graph.device, cudaGraphMemAttrUsedMemCurrent, nullptr) == cudaErrorInvalidValue, "null Runtime graph statistic output");
    uint64_t zero{}, nonzero = 1;
    require(cuDeviceSetGraphMemAttribute(graph.device, CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT, &zero) == CUDA_ERROR_INVALID_VALUE, "current graph memory statistics are read-only");
    require(cudaDeviceSetGraphMemAttribute(graph.device, cudaGraphMemAttrReservedMemHigh, &nonzero) == cudaErrorInvalidValue, "high graph memory statistics accept only zero reset");
    allocation_node_executable executable(graph.value, route.runtime); executable.launch(stream.value, !route.runtime); stream.synchronize(); host.expect();
    allocation_node_trim(graph.device, route.runtime); allocation_node_idle(graph.device);
}
std::string allocation_node_case;
void execute_allocation_nodes() {
    const std::array<std::pair<const char*, void (*)(allocation_node_route)>, 6> cases{{{"replay", allocation_node_replay}, {"live", allocation_node_live},
        {"auto_free", allocation_node_auto_free}, {"pending", allocation_node_pending}, {"restrictions", allocation_node_restrictions}, {"validation", allocation_node_validation}}};
    bool found = false;
    for (auto [name, execute] : cases) if (allocation_node_case == name) { found = true; for (bool runtime : {false, true}) for (bool generic : {false, true}) {
        std::cout << "graph_allocation_nodes=" << name << " runtime=" << runtime << " generic=" << generic << std::endl; execute({runtime, generic});
    } }
    require(found, "unknown graph allocation case");
}
}
int main(int argc, char** argv) {
    if (argc != 3) return 1; cudavm::allocation_node_case = argv[2]; return cudavm::run_test(2, argv, cudavm::execute_allocation_nodes);
}
