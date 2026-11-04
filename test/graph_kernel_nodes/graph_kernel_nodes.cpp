#include "fixtures.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

namespace cudavm {
void check_kernel_node_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define KERNEL_NODE_RUNTIME(call) ::cudavm::check_kernel_node_runtime((call), #call)
template<class T> struct kernel_node_guarded {
    uint64_t prefix = 0x123456789abcdef0; T value{}; uint64_t suffix = 0xfedcba9876543210;
    void check() const { require(prefix == 0x123456789abcdef0 && suffix == 0xfedcba9876543210, "kernel-node public output guards"); }
};
template<class T> T kernel_node_procedure(const char* name, int version) {
    void* address{}; CUdriverProcAddressQueryResult status{};
    CONFORMANCE_CUDA(cuGetProcAddress(name, &address, version, CU_GET_PROC_ADDRESS_DEFAULT, &status));
    require(address && status == CU_GET_PROC_ADDRESS_SUCCESS, "legacy graph-kernel procedure lookup");
    return reinterpret_cast<T>(address);
}
enum class kernel_node_api { driver, generic_driver, legacy_driver, generic_runtime, library_driver, library_runtime, library_generic_runtime };
struct kernel_node_route {
    kernel_node_api api; const char* name; CUfunction function; CUkernel kernel;
    bool runtime() const { return api == kernel_node_api::generic_runtime || api == kernel_node_api::library_runtime || api == kernel_node_api::library_generic_runtime; }
    bool contextless() const { return api == kernel_node_api::library_driver || api == kernel_node_api::library_runtime || api == kernel_node_api::library_generic_runtime; }
};
struct kernel_node_shape { dim3 grid, block; };
const std::array<kernel_node_shape, 3> kernel_node_shapes{{{{2, 3, 2}, {4, 2, 2}}, {{3, 2, 1}, {4, 4, 2}}, {{1, 1, 2}, {8, 4, 3}}}};
constexpr size_t kernel_node_guard = 8, kernel_node_lanes = 192, kernel_node_words = 208, kernel_node_slot_words = 224, kernel_node_slots = 3;
void kernel_node_apply(std::vector<uint32_t>& state, const kernel_node_shape& shape, unsigned count, uint32_t seed) {
    const auto grid = shape.grid, block = shape.block;
    size_t index = 0; const uint64_t threads = uint64_t(block.x) * block.y * block.z;
    for (unsigned bz = 0; bz < grid.z; ++bz) for (unsigned by = 0; by < grid.y; ++by) for (unsigned bx = 0; bx < grid.x; ++bx)
        for (uint64_t lane = 0; lane < threads; ++lane, ++index) {
            require(index < kernel_node_lanes, "bounded independent kernel-node oracle");
            const uint64_t peer = threads - 1 - lane, tx = peer % block.x, ty = (peer / block.x) % block.y, tz = peer / (uint64_t(block.x) * block.y);
            if (index < count) state[index + kernel_node_guard] = static_cast<uint32_t>(uint64_t(seed) + tx * 3 + ty * 5 + tz * 7
                + uint64_t(bx) * 11 + uint64_t(by) * 13 + uint64_t(bz) * 17 + uint64_t(block.x) * 19 + uint64_t(block.y) * 23 + uint64_t(block.z) * 29);
        }
    require(index == kernel_node_lanes, "all selected shapes have the same invocation extent");
}
struct kernel_node_storage {
    device_buffer device{kernel_node_words * sizeof(uint32_t)}; void* host{}; std::vector<uint32_t> initial;
    explicit kernel_node_storage(unsigned seed) : initial(kernel_node_words) {
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = 0xcafe0000u + uint32_t(i) * 0x10203u + seed;
        CONFORMANCE_CUDA(cuMemAllocHost(&host, kernel_node_slots * kernel_node_slot_words * sizeof(uint32_t)));
        try { reset(); } catch (...) { cuMemFreeHost(host); host = nullptr; throw; }
    }
    ~kernel_node_storage() { if (host) { cuCtxSynchronize(); cuMemFreeHost(host); } }
    void reset() {
        CONFORMANCE_CUDA(cuMemcpyHtoD(device.value, initial.data(), initial.size() * sizeof(uint32_t)));
        std::memset(host, 0xd3, kernel_node_slots * kernel_node_slot_words * sizeof(uint32_t));
    }
    uint32_t* slot(unsigned index) const { require(index < kernel_node_slots, "valid kernel-node readback slot"); return static_cast<uint32_t*>(host) + index * kernel_node_slot_words; }
    void read(CUstream stream, unsigned index) const { CONFORMANCE_CUDA(cuMemcpyDtoHAsync(slot(index) + kernel_node_guard, device.value, kernel_node_words * sizeof(uint32_t), stream)); }
    void expect(unsigned index, std::span<const uint32_t> state) const {
        std::vector<uint32_t> expected(kernel_node_slot_words, 0xd3d3d3d3);
        std::copy(state.begin(), state.end(), expected.begin() + kernel_node_guard);
        expect_words({slot(index), kernel_node_slot_words}, expected, "kernel-node payload, inactive lanes and complete device/host guards");
    }
    void untouched(unsigned first) const {
        for (unsigned i = first; i < kernel_node_slots; ++i) for (size_t j = 0; j < kernel_node_slot_words; ++j)
            require(slot(i)[j] == 0xd3d3d3d3, "unused kernel-node readback storage remains untouched");
    }
};
struct kernel_node_input {
    struct packet { CUdeviceptr pointer; unsigned count, seed; } bytes;
    static_assert(sizeof(packet) == 16 && offsetof(packet, count) == 8 && offsetof(packet, seed) == 12);
    std::array<void*, 3> arguments;
    size_t size = sizeof(packet); std::array<void*, 5> extra;
    kernel_node_shape shape; unsigned shared; bool packed, simultaneous = false;
    kernel_node_input(kernel_node_storage& storage, kernel_node_shape geometry, unsigned count, uint32_t seed, bool use_extra)
        : bytes{storage.device.value + kernel_node_guard * sizeof(uint32_t), count, seed}, arguments{&bytes.pointer, &bytes.count, &bytes.seed},
          extra{CU_LAUNCH_PARAM_BUFFER_POINTER, &bytes, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END},
          shape(geometry), shared(geometry.block.x * geometry.block.y * geometry.block.z * sizeof(uint32_t) + 16), packed(use_extra) {}
    CUDA_KERNEL_NODE_PARAMS driver(const kernel_node_route& route) const {
        CUDA_KERNEL_NODE_PARAMS p{};
        if (route.contextless()) { p.kern = route.kernel; CONFORMANCE_CUDA(cuCtxGetCurrent(&p.ctx)); } else p.func = route.function;
        p.gridDimX = shape.grid.x; p.gridDimY = shape.grid.y; p.gridDimZ = shape.grid.z;
        p.blockDimX = shape.block.x; p.blockDimY = shape.block.y; p.blockDimZ = shape.block.z; p.sharedMemBytes = shared;
        if (packed) p.extra = const_cast<void**>(extra.data()); else p.kernelParams = const_cast<void**>(arguments.data());
        if (simultaneous) { p.kernelParams = const_cast<void**>(arguments.data()); p.extra = const_cast<void**>(extra.data()); }
        return p;
    }
    void expire() { bytes = {}; std::fill(arguments.begin(), arguments.end(), nullptr); std::fill(extra.begin(), extra.end(), nullptr); size = 0; }
};
int kernel_node_configure(CUgraph graph, CUgraphNode& node, const kernel_node_route& route, const kernel_node_input& input, bool create) {
    auto p = input.driver(route);
    if (route.runtime()) {
        kernel_node_guarded<cudaGraphNode_t> output; cudaError_t status;
        if (route.api == kernel_node_api::library_runtime) {
            cudaKernelNodeParams params{reinterpret_cast<void*>(route.kernel), input.shape.grid, input.shape.block, input.shared, p.kernelParams, p.extra};
            status = create ? cudaGraphAddKernelNode(&output.value, reinterpret_cast<cudaGraph_t>(graph), nullptr, 0, &params)
                            : cudaGraphKernelNodeSetParams(reinterpret_cast<cudaGraphNode_t>(node), &params);
        } else {
            cudaGraphNodeParams params{}; params.type = cudaGraphNodeTypeKernel;
            params.kernel.functionType = route.contextless() ? cudaKernelFunctionTypeKernel : cudaKernelFunctionTypeFunction;
            if (route.contextless()) params.kernel.kern = reinterpret_cast<cudaKernel_t>(route.kernel); else params.kernel.cuFunc = reinterpret_cast<cudaFunction_t>(route.function);
            params.kernel.gridDim = input.shape.grid; params.kernel.blockDim = input.shape.block; params.kernel.sharedMemBytes = input.shared;
            params.kernel.kernelParams = p.kernelParams; params.kernel.extra = p.extra;
            status = create ? cudaGraphAddNode(&output.value, reinterpret_cast<cudaGraph_t>(graph), nullptr, nullptr, 0, &params)
                            : cudaGraphNodeSetParams(reinterpret_cast<cudaGraphNode_t>(node), &params);
        }
        output.check(); if (create && status == cudaSuccess) node = reinterpret_cast<CUgraphNode>(output.value); return int(status);
    }
    kernel_node_guarded<CUgraphNode> output; CUresult status;
    if (route.api == kernel_node_api::legacy_driver) {
        CUDA_KERNEL_NODE_PARAMS_v1 params{}; std::memcpy(&params, &p, sizeof(params));
        if (create) status = kernel_node_procedure<CUresult (CUDAAPI*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_KERNEL_NODE_PARAMS_v1*)>("cuGraphAddKernelNode", 10000)(&output.value, graph, nullptr, 0, &params);
        else status = kernel_node_procedure<CUresult (CUDAAPI*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS_v1*)>("cuGraphKernelNodeSetParams", 10000)(node, &params);
    } else if (route.api == kernel_node_api::generic_driver) {
        CUgraphNodeParams params{}; params.type = CU_GRAPH_NODE_TYPE_KERNEL; std::memcpy(&params.kernel, &p, sizeof(p));
        status = create ? cuGraphAddNode(&output.value, graph, nullptr, nullptr, 0, &params) : cuGraphNodeSetParams(node, &params);
    } else status = create ? cuGraphAddKernelNode(&output.value, graph, nullptr, 0, &p) : cuGraphKernelNodeSetParams(node, &p);
    output.check(); if (create && status == CUDA_SUCCESS) node = output.value; return int(status);
}
struct kernel_node_graph {
    CUgraph value{}; CUgraphNode node{};
    explicit kernel_node_graph(bool runtime) {
        if (runtime) { cudaGraph_t result{}; KERNEL_NODE_RUNTIME(cudaGraphCreate(&result, 0)); value = reinterpret_cast<CUgraph>(result); }
        else CONFORMANCE_CUDA(cuGraphCreate(&value, 0));
    }
    ~kernel_node_graph() { if (value) cuGraphDestroy(value); }
    void configure(const kernel_node_route& route, kernel_node_input& input) {
        auto status = kernel_node_configure(value, node, route, input, !node);
        CONFORMANCE_CUDA(static_cast<CUresult>(status)); require(node, "valid kernel-node construction/setter publishes a node"); input.expire();
    }
};
void kernel_node_shape_check(dim3 grid, dim3 block, unsigned shared, const kernel_node_input& expected) {
    require(grid.x == expected.shape.grid.x && grid.y == expected.shape.grid.y && grid.z == expected.shape.grid.z
        && block.x == expected.shape.block.x && block.y == expected.shape.block.y && block.z == expected.shape.block.z && shared == expected.shared,
        "queried graph kernel preserves every geometry and dynamic-shared-memory field");
}
void kernel_node_arguments_check(void** arguments, void** extra, const kernel_node_input& expected) {
    kernel_node_input::packet actual{};
    if (expected.packed) {
        require(!arguments && extra && extra[4] == CU_LAUNCH_PARAM_END, "packed graph parameters preserve their argument form");
        const void* buffer = nullptr; size_t bytes = 0;
        for (unsigned i = 0; i < 4; i += 2) {
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) buffer = extra[i + 1];
            else { require(extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[i + 1], "valid returned packed-argument tag"); std::memcpy(&bytes, extra[i + 1], sizeof(bytes)); }
        }
        require(buffer && bytes == sizeof(actual), "exact copied packed-argument extent"); std::memcpy(&actual, buffer, sizeof(actual));
    } else {
        require(arguments && !extra && arguments[0] && arguments[1] && arguments[2], "pointer-list graph parameters preserve their argument form");
        std::memcpy(&actual.pointer, arguments[0], sizeof(actual.pointer)); std::memcpy(&actual.count, arguments[1], sizeof(actual.count)); std::memcpy(&actual.seed, arguments[2], sizeof(actual.seed));
    }
    require(actual.pointer == expected.bytes.pointer && actual.count == expected.bytes.count && actual.seed == expected.bytes.seed, "node-owned arguments survive caller storage retirement");
}
void kernel_node_query(CUgraphNode node, const kernel_node_route& route, const kernel_node_input& expected) {
    kernel_node_guarded<CUDA_KERNEL_NODE_PARAMS> typed; CONFORMANCE_CUDA(cuGraphKernelNodeGetParams(node, &typed.value)); typed.check();
    auto check_driver = [&](const CUDA_KERNEL_NODE_PARAMS& p) {
        require(route.contextless() ? !p.func && p.kern == route.kernel : p.func == route.function, "queried Driver kernel identity");
        kernel_node_shape_check({p.gridDimX, p.gridDimY, p.gridDimZ}, {p.blockDimX, p.blockDimY, p.blockDimZ}, p.sharedMemBytes, expected);
        kernel_node_arguments_check(p.kernelParams, p.extra, expected);
    };
    check_driver(typed.value);
    kernel_node_guarded<CUgraphNodeParams> generic; CONFORMANCE_CUDA(cuGraphNodeGetParams(node, &generic.value)); generic.check();
    require(generic.value.type == CU_GRAPH_NODE_TYPE_KERNEL, "generic Driver kernel node type");
    CUDA_KERNEL_NODE_PARAMS driver{}; std::memcpy(&driver, &generic.value.kernel, sizeof(driver)); check_driver(driver);
    kernel_node_guarded<CUDA_KERNEL_NODE_PARAMS_v1> legacy;
    CONFORMANCE_CUDA(kernel_node_procedure<CUresult (CUDAAPI*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS_v1*)>("cuGraphKernelNodeGetParams", 10000)(node, &legacy.value));
    legacy.check(); require(legacy.value.func == route.function, "legacy getter resolves the kernel's context-bound function");
    auto& old = legacy.value;
    kernel_node_shape_check({old.gridDimX, old.gridDimY, old.gridDimZ}, {old.blockDimX, old.blockDimY, old.blockDimZ}, old.sharedMemBytes, expected);
    kernel_node_arguments_check(old.kernelParams, old.extra, expected);
    kernel_node_guarded<cudaGraphNodeParams> runtime; KERNEL_NODE_RUNTIME(cudaGraphNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &runtime.value)); runtime.check();
    require(runtime.value.type == cudaGraphNodeTypeKernel, "generic Runtime kernel node type");
    const auto& p = runtime.value.kernel;
    require(route.contextless() ? p.functionType == cudaKernelFunctionTypeKernel && p.kern == reinterpret_cast<cudaKernel_t>(route.kernel)
                               : p.functionType == cudaKernelFunctionTypeFunction && p.cuFunc == reinterpret_cast<cudaFunction_t>(route.function), "generic Runtime kernel-handle identity/type");
    kernel_node_shape_check(p.gridDim, p.blockDim, p.sharedMemBytes, expected); kernel_node_arguments_check(p.kernelParams, p.extra, expected);
    if (route.contextless()) {
        kernel_node_guarded<cudaKernelNodeParams> runtime_typed;
        KERNEL_NODE_RUNTIME(cudaGraphKernelNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &runtime_typed.value)); runtime_typed.check();
        const auto& params = runtime_typed.value; require(params.func == reinterpret_cast<void*>(route.kernel), "typed Runtime library-kernel identity");
        kernel_node_shape_check(params.gridDim, params.blockDim, params.sharedMemBytes, expected); kernel_node_arguments_check(params.kernelParams, params.extra, expected);
    }
}
struct kernel_node_stream {
    CUstream value{}; kernel_node_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~kernel_node_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void synchronize() const { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct kernel_node_executable {
    CUgraphExec value{};
    kernel_node_executable(CUgraph graph, bool runtime, unsigned long long flags = 0) {
        if (runtime) { cudaGraphExec_t result{}; KERNEL_NODE_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), flags)); value = reinterpret_cast<CUgraphExec>(result); }
        else CONFORMANCE_CUDA(cuGraphInstantiate(&value, graph, flags));
    }
    ~kernel_node_executable() { if (value) cuGraphExecDestroy(value); }
    void read(const kernel_node_storage& storage, const kernel_node_stream& stream, unsigned slot, bool runtime) const {
        if (runtime) KERNEL_NODE_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaStream_t>(stream.value)));
        else CONFORMANCE_CUDA(cuGraphLaunch(value, stream.value)); storage.read(stream.value, slot);
    }
};
void kernel_node_construction(const kernel_node_route& route) {
    kernel_node_storage storage(11); kernel_node_stream stream;
    for (bool packed : {false, true}) for (unsigned count : {0u, 1u, 97u, 192u, 211u}) {
        storage.reset(); kernel_node_graph graph(route.runtime());
        kernel_node_input input(storage, kernel_node_shapes[0], count, 0xfffffff0, packed), expected(storage, kernel_node_shapes[0], count, 0xfffffff0, packed);
        graph.configure(route, input); kernel_node_query(graph.node, route, expected);
        kernel_node_executable executable(graph.value, !route.runtime()); executable.read(storage, stream, 0, route.runtime()); stream.synchronize();
        auto wanted = storage.initial; kernel_node_apply(wanted, expected.shape, count, expected.bytes.seed); storage.expect(0, wanted); storage.untouched(1);
    }
}
void kernel_node_parameters(const kernel_node_route& route) {
    for (bool packed : {false, true}) {
        kernel_node_storage original(23), changed(47); kernel_node_stream stream; kernel_node_graph graph(route.runtime());
        kernel_node_input first(original, kernel_node_shapes[0], 97, 0x12345678, packed), expected_first(original, kernel_node_shapes[0], 97, 0x12345678, packed);
        graph.configure(route, first); kernel_node_query(graph.node, route, expected_first); kernel_node_executable before(graph.value, route.runtime());
        kernel_node_input second(changed, kernel_node_shapes[1], 181, 0xfedcba98, !packed), expected_second(changed, kernel_node_shapes[1], 181, 0xfedcba98, !packed);
        graph.configure(route, second); kernel_node_query(graph.node, route, expected_second); kernel_node_executable after(graph.value, !route.runtime());
        before.read(original, stream, 0, !route.runtime()); after.read(changed, stream, 0, route.runtime()); stream.synchronize();
        auto a = original.initial, b = changed.initial; kernel_node_apply(a, expected_first.shape, 97, expected_first.bytes.seed); kernel_node_apply(b, expected_second.shape, 181, expected_second.bytes.seed);
        original.expect(0, a); changed.expect(0, b); original.untouched(1); changed.untouched(1);
    }
}
void kernel_node_clones(const kernel_node_route& route) {
    kernel_node_storage original(59), changed(83); kernel_node_stream stream; kernel_node_graph source(route.runtime()), clone(!route.runtime());
    kernel_node_input first(original, kernel_node_shapes[2], 157, 0x13579bdf, true), expected_first(original, kernel_node_shapes[2], 157, 0x13579bdf, true);
    source.configure(route, first); CONFORMANCE_CUDA(cuGraphDestroy(clone.value)); clone.value = nullptr;
    if (route.runtime()) {
        cudaGraph_t result{}; KERNEL_NODE_RUNTIME(cudaGraphClone(&result, reinterpret_cast<cudaGraph_t>(source.value))); clone.value = reinterpret_cast<CUgraph>(result);
        cudaGraphNode_t node{}; KERNEL_NODE_RUNTIME(cudaGraphNodeFindInClone(&node, reinterpret_cast<cudaGraphNode_t>(source.node), result)); clone.node = reinterpret_cast<CUgraphNode>(node);
    } else { CONFORMANCE_CUDA(cuGraphClone(&clone.value, source.value)); CONFORMANCE_CUDA(cuGraphNodeFindInClone(&clone.node, source.node, clone.value)); }
    kernel_node_query(clone.node, route, expected_first);
    kernel_node_input second(changed, kernel_node_shapes[1], 1, 0x2468ace0, false), expected_second(changed, kernel_node_shapes[1], 1, 0x2468ace0, false);
    source.configure(route, second); kernel_node_query(source.node, route, expected_second); kernel_node_query(clone.node, route, expected_first);
    kernel_node_executable before(clone.value, route.runtime()), after(source.value, !route.runtime());
    before.read(original, stream, 0, !route.runtime()); after.read(changed, stream, 0, route.runtime()); stream.synchronize();
    CONFORMANCE_CUDA(cuGraphDestroy(source.value)); source.value = nullptr;
    kernel_node_query(clone.node, route, expected_first); before.read(original, stream, 1, route.runtime()); stream.synchronize();
    auto a = original.initial, b = changed.initial; kernel_node_apply(a, expected_first.shape, 157, expected_first.bytes.seed); kernel_node_apply(b, expected_second.shape, 1, expected_second.bytes.seed);
    original.expect(0, a); original.expect(1, a); changed.expect(0, b); original.untouched(2); changed.untouched(1);
}
void kernel_node_priority(CUgraphNode node, int priority, bool runtime) {
    if (runtime) { cudaKernelNodeAttrValue p{}; p.priority = priority; KERNEL_NODE_RUNTIME(cudaGraphKernelNodeSetAttribute(reinterpret_cast<cudaGraphNode_t>(node), cudaKernelNodeAttributePriority, &p)); }
    else { CUkernelNodeAttrValue p{}; p.priority = priority; CONFORMANCE_CUDA(cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &p)); }
}
void kernel_node_priority_check(CUgraphNode node, int request) {
    int least{}, greatest{}; CONFORMANCE_CUDA(cuCtxGetStreamPriorityRange(&least, &greatest)); int expected = std::clamp(request, greatest, least);
    kernel_node_guarded<CUkernelNodeAttrValue> driver; driver.value.priority = std::numeric_limits<int>::max();
    CONFORMANCE_CUDA(cuGraphKernelNodeGetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &driver.value)); driver.check(); require(driver.value.priority == expected, "Driver priority query uses device clamping");
    kernel_node_guarded<cudaKernelNodeAttrValue> runtime; runtime.value.priority = std::numeric_limits<int>::max();
    KERNEL_NODE_RUNTIME(cudaGraphKernelNodeGetAttribute(reinterpret_cast<cudaGraphNode_t>(node), cudaKernelNodeAttributePriority, &runtime.value)); runtime.check(); require(runtime.value.priority == expected, "Runtime priority query uses device clamping");
}
void kernel_node_attributes(const kernel_node_route& route) {
    kernel_node_storage storage(101); kernel_node_stream stream;
    kernel_node_graph source(route.runtime()), target(!route.runtime()), zero(route.runtime());
    kernel_node_input initial(storage, kernel_node_shapes[0], 97, 17, false), initial_expected(storage, kernel_node_shapes[0], 97, 17, false);
    source.configure(route, initial); kernel_node_priority(source.node, 17, route.runtime());
    kernel_node_input input(storage, kernel_node_shapes[0], 97, 17, false); target.configure(route, input);
    kernel_node_input template_input(storage, kernel_node_shapes[0], 97, 17, false); zero.configure(route, template_input);
    kernel_node_executable executable(target.value, route.runtime(), CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY);
    const auto copy = [&](CUgraphNode from) {
        if (route.runtime()) KERNEL_NODE_RUNTIME(cudaGraphKernelNodeCopyAttributes(reinterpret_cast<cudaGraphNode_t>(target.node), reinterpret_cast<cudaGraphNode_t>(from)));
        else CONFORMANCE_CUDA(cuGraphKernelNodeCopyAttributes(target.node, from));
    };
    const auto reject = [&] {
        kernel_node_guarded<CUgraphExecUpdateResultInfo> info; info.value.result = CU_GRAPH_EXEC_UPDATE_ERROR;
        require(cuGraphExecUpdate(executable.value, target.value, &info.value) == CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE, "copied requested priority changes the executable's source attributes"); info.check();
        require(info.value.result == CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED && info.value.errorNode == target.node, "priority-copy snapshot rejection diagnostic");
    };
    copy(source.node); kernel_node_priority_check(target.node, 17); kernel_node_query(target.node, route, initial_expected); reject();
    kernel_node_input changed(storage, kernel_node_shapes[1], 173, 0xffffffe0, true), changed_expected(storage, kernel_node_shapes[1], 173, 0xffffffe0, true);
    target.configure(route, changed); kernel_node_priority_check(target.node, 17); reject();
    executable.read(storage, stream, 0, !route.runtime()); stream.synchronize();
    auto expected = storage.initial; kernel_node_apply(expected, initial_expected.shape, 97, 17); storage.expect(0, expected);
    copy(zero.node); kernel_node_priority_check(target.node, 0); kernel_node_priority_check(source.node, 17); kernel_node_query(source.node, route, initial_expected); kernel_node_query(target.node, route, changed_expected);
    kernel_node_guarded<CUgraphExecUpdateResultInfo> info; info.value.result = CU_GRAPH_EXEC_UPDATE_ERROR;
    CONFORMANCE_CUDA(cuGraphExecUpdate(executable.value, target.value, &info.value)); info.check(); require(info.value.result == CU_GRAPH_EXEC_UPDATE_SUCCESS, "restoring copied attributes permits a valid update");
    executable.read(storage, stream, 1, route.runtime()); stream.synchronize(); kernel_node_apply(expected, changed_expected.shape, 173, changed_expected.bytes.seed);
    storage.expect(1, expected); storage.untouched(2);
}
void kernel_node_validation(const kernel_node_route& route) {
    kernel_node_storage storage(127); kernel_node_stream stream; kernel_node_graph graph(route.runtime());
    kernel_node_input original(storage, kernel_node_shapes[2], 129, 0xabcdef01, false), expected_original(storage, kernel_node_shapes[2], 129, 0xabcdef01, false);
    graph.configure(route, original); CUgraphNode empty{}; CONFORMANCE_CUDA(cuGraphAddEmptyNode(&empty, graph.value, nullptr, 0));
    for (bool add : {false, true}) for (bool both : {false, true}) {
        kernel_node_input invalid(storage, kernel_node_shapes[1], 1, 7, false);
        if (both) invalid.simultaneous = true; else invalid.shape.block.y = 0;
        CUgraphNode attempted = graph.node;
        require(kernel_node_configure(graph.value, attempted, route, invalid, add) == int(CUDA_ERROR_INVALID_VALUE),
            both ? "simultaneous graph-kernel argument forms reject without mutation" : "zero graph-kernel dimensions reject without mutation");
        kernel_node_query(graph.node, route, expected_original); size_t count = 0; CONFORMANCE_CUDA(cuGraphGetNodes(graph.value, nullptr, &count)); require(count == 2, "rejected additions leave graph membership intact");
    }
    require(cuGraphKernelNodeGetParams(graph.node, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver kernel output");
    require(cudaGraphNodeGetParams(reinterpret_cast<cudaGraphNode_t>(graph.node), nullptr) == cudaErrorInvalidValue, "null generic Runtime output");
    kernel_node_guarded<CUDA_KERNEL_NODE_PARAMS> output;
    require(cuGraphKernelNodeGetParams(empty, &output.value) == CUDA_ERROR_INVALID_VALUE, "wrong-type Driver getter"); output.check();
    kernel_node_guarded<cudaKernelNodeParams> runtime_output;
    require(cudaGraphKernelNodeGetParams(reinterpret_cast<cudaGraphNode_t>(empty), &runtime_output.value) == cudaErrorInvalidValue, "wrong-type Runtime getter"); runtime_output.check();
    kernel_node_executable before(graph.value, route.runtime()); before.read(storage, stream, 0, !route.runtime()); stream.synchronize();
    auto expected = storage.initial; kernel_node_apply(expected, expected_original.shape, 129, expected_original.bytes.seed); storage.expect(0, expected);
    kernel_node_input valid(storage, kernel_node_shapes[1], 1, 37, true), expected_valid(storage, kernel_node_shapes[1], 1, 37, true);
    graph.configure(route, valid); kernel_node_query(graph.node, route, expected_valid); kernel_node_executable after(graph.value, !route.runtime());
    after.read(storage, stream, 1, route.runtime()); stream.synchronize(); kernel_node_apply(expected, expected_valid.shape, 1, 37); storage.expect(1, expected); storage.untouched(2);
}
void execute_graph_kernel_nodes(const loaded_program& program) {
    const auto function = program.function("kernel_node_write"); const auto kernel = program.library_kernel("kernel_node_write");
    std::vector<kernel_node_route> routes{{kernel_node_api::driver, "driver", function, kernel}, {kernel_node_api::generic_driver, "driver_generic", function, kernel},
        {kernel_node_api::legacy_driver, "driver_legacy", function, kernel}, {kernel_node_api::generic_runtime, "runtime_generic", function, kernel}};
    if (kernel) for (auto route : {kernel_node_route{kernel_node_api::library_driver, "driver_library", function, kernel},
        kernel_node_route{kernel_node_api::library_runtime, "runtime_library", function, kernel}, kernel_node_route{kernel_node_api::library_generic_runtime, "runtime_library_generic", function, kernel}}) routes.push_back(route);
    const std::array<std::pair<const char*, void (*)(const kernel_node_route&)>, 5> cases{{
        {"construction", kernel_node_construction}, {"parameters", kernel_node_parameters}, {"clones", kernel_node_clones},
        {"attributes", kernel_node_attributes}, {"validation", kernel_node_validation}}};
    for (const auto& [name, execute] : cases) for (const auto& route : routes) {
        std::cout << "graph_kernel_nodes=" << name << " route=" << route.name << std::endl; execute(route);
    }
}
}
int main(int argc, char** argv) { return cudavm::run_conformance(argc, argv, cudavm::fixtures, cudavm::execute_graph_kernel_nodes); }
