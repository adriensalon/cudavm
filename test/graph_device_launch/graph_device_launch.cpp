#include "test_framework.hpp"
#include <cudaTypedefs.h>
#include <cstring>
#include <type_traits>

namespace cudavm {
std::string_view graph_device_case;
void graph_device_runtime(cudaError_t status) { require(status == cudaSuccess, std::string("device graph Runtime: ") + cudaGetErrorString(status)); }
template<class T, class U> T graph_device_cast(U value) { return reinterpret_cast<T>(value); }
template<class T> struct graph_device_output {
    std::array<unsigned char, 16> before; T value{}; std::array<unsigned char, 16> after;
    graph_device_output() { before.fill(0x63); after.fill(0xa7); }
    T get() const {
        require(std::all_of(before.begin(), before.end(), [](auto v) { return v == 0x63; })
            && std::all_of(after.begin(), after.end(), [](auto v) { return v == 0xa7; }), "device graph output exact extents and guards"); return value;
    }
};
struct graph_device_graph {
    CUgraph value{}; CUcontext context{};
    graph_device_graph() { CONFORMANCE_CUDA(cuCtxGetCurrent(&context)); CONFORMANCE_CUDA(cuGraphCreate(&value, 0)); }
    ~graph_device_graph() { if (value) cuGraphDestroy(value); }
    void fill(CUdeviceptr pointer, size_t bytes, bool runtime) {
        if (runtime) {
            cudaMemsetParams p{}; p.dst = reinterpret_cast<void*>(uintptr_t(pointer)); p.value = 0x35; p.elementSize = 1; p.width = bytes; p.height = 1;
            cudaGraphNode_t node{}; graph_device_runtime(cudaGraphAddMemsetNode(&node, graph_device_cast<cudaGraph_t>(value), nullptr, 0, &p));
        } else {
            CUDA_MEMSET_NODE_PARAMS p{}; p.dst = pointer; p.value = 0x35; p.elementSize = 1; p.width = bytes; p.height = 1;
            CUgraphNode node{}; CONFORMANCE_CUDA(cuGraphAddMemsetNode(&node, value, nullptr, 0, &p, context));
        }
    }
};
struct graph_device_stream {
    CUstream value{};
    graph_device_stream() { CONFORMANCE_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~graph_device_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
    void wait() { CONFORMANCE_CUDA(cuStreamSynchronize(value)); }
};
struct graph_device_executable {
    CUgraphExec value{};
    graph_device_executable(CUgraph graph, unsigned mode, CUstream stream, bool upload = false) {
        unsigned long long flags = CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH | (upload ? CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD : 0);
        graph_device_output<CUgraphExec> result;
        if (mode == 0) CONFORMANCE_CUDA(cuGraphInstantiate(&result.value, graph, flags));
        else if (mode == 1) {
            graph_device_output<cudaGraphExec_t> output; graph_device_runtime(cudaGraphInstantiateWithFlags(&output.value, graph_device_cast<cudaGraph_t>(graph), flags)); result.value = graph_device_cast<CUgraphExec>(output.get());
        } else if (mode == 2) {
            CUDA_GRAPH_INSTANTIATE_PARAMS params{}; params.flags = flags; params.hUploadStream = stream;
            CONFORMANCE_CUDA(cuGraphInstantiateWithParams(&result.value, graph, &params));
            require(params.result_out == CUDA_GRAPH_INSTANTIATE_SUCCESS && !params.hErrNode_out, "device graph instantiation diagnostics");
        } else {
            graph_device_output<cudaGraphExec_t> output; cudaGraphInstantiateParams params{}; params.flags = flags; params.uploadStream = graph_device_cast<cudaStream_t>(stream);
            graph_device_runtime(cudaGraphInstantiateWithParams(&output.value, graph_device_cast<cudaGraph_t>(graph), &params)); result.value = graph_device_cast<CUgraphExec>(output.get());
            require(params.result_out == cudaGraphInstantiateSuccess && !params.errNode_out, "Runtime device graph instantiation diagnostics");
        }
        value = result.get(); require(value, "device graph executable publication");
    }
    ~graph_device_executable() { if (value) { cuCtxSynchronize(); cuGraphExecDestroy(value); } }
    void flags() {
        graph_device_output<cuuint64_t> driver; CONFORMANCE_CUDA(cuGraphExecGetFlags(value, &driver.value));
        graph_device_output<unsigned long long> runtime; graph_device_runtime(cudaGraphExecGetFlags(graph_device_cast<cudaGraphExec_t>(value), &runtime.value));
        require(driver.get() == CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH && runtime.get() == cudaGraphInstantiateFlagDeviceLaunch, "persistent device graph flag through both facades");
    }
    void upload(CUstream stream, bool runtime) {
        if (runtime) graph_device_runtime(cudaGraphUpload(graph_device_cast<cudaGraphExec_t>(value), graph_device_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuGraphUpload(value, stream));
    }
    void launch(CUstream stream, bool runtime) {
        if (runtime) graph_device_runtime(cudaGraphLaunch(graph_device_cast<cudaGraphExec_t>(value), graph_device_cast<cudaStream_t>(stream)));
        else CONFORMANCE_CUDA(cuGraphLaunch(value, stream));
    }
};
struct graph_device_storage {
    static constexpr size_t bytes = 1056;
    device_buffer device{bytes};
    void reset(unsigned seed) { CONFORMANCE_CUDA(cuMemsetD8(device.value, static_cast<unsigned char>(seed), bytes)); }
    void verify(unsigned seed, bool executed) {
        graph_device_output<std::array<unsigned char, bytes>> result;
        CONFORMANCE_CUDA(cuMemcpyDtoH(result.value.data(), device.value, bytes)); auto actual = result.get();
        for (size_t i = 0; i < bytes; ++i) require(actual[i] == (executed && i >= 16 && i < bytes - 16 ? 0x35 : seed), "device graph complete bytes and preserved guards at=" + std::to_string(i));
    }
};
void graph_device_preparation() {
    for (unsigned mode = 0; mode < 4; ++mode) {
        graph_device_storage storage; graph_device_graph graph; graph_device_stream stream;
        graph.fill(storage.device.value + 16, storage.bytes - 32, mode & 1); storage.reset(0xa7);
        graph_device_executable executable(graph.value, mode, stream.value, graph_device_case == "upload" && mode >= 2);
        stream.wait(); executable.flags(); storage.verify(0xa7, false);
        for (bool runtime : {false, true}) {
            executable.upload(stream.value, runtime); stream.wait(); storage.verify(0xa7, false);
        }
        if (graph_device_case == "lifetime") { CONFORMANCE_CUDA(cuGraphDestroy(graph.value)); graph.value = nullptr; }
        for (unsigned seed : {0u, 0x7du, 0xffu}) {
            storage.reset(seed); executable.launch(stream.value, mode & 1); stream.wait(); storage.verify(seed, true);
        }
        if (graph_device_case == "lifetime") {
            storage.reset(0xa7); executable.launch(stream.value, !(mode & 1));
            CONFORMANCE_CUDA(cuGraphExecDestroy(executable.value)); executable.value = nullptr; stream.wait(); storage.verify(0xa7, true);
        }
    }
}
constexpr char graph_device_identity_ptx[] = R"ptx(.version 8.0
.target sm_75
.address_size 64
.visible .entry graph_identity(.param .u64 output) {
 .reg .u64 %p, %g;
 ld.param.u64 %p, [output]; mov.u64 %g, %current_graph_exec;
 st.global.u64 [%p], %g; ret;
}
)ptx";
void graph_device_identity() {
    for (bool library : {false, true}) for (unsigned mode : {0u, 1u}) {
        loaded_program program(graph_device_identity_ptx, library); auto function = program.function("graph_identity");
        device_buffer memory(64); graph_device_graph graph; graph_device_stream stream;
        CUdeviceptr output = memory.value + 16; void* arguments[]{&output};
        CONFORMANCE_CUDA(cuMemsetD8(memory.value, 0xa7, 64));
        CONFORMANCE_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream.value, arguments, nullptr)); stream.wait();
        std::array<uint64_t, 8> result{}; CONFORMANCE_CUDA(cuMemcpyDtoH(result.data(), memory.value, 64));
        for (size_t i = 0; i < result.size(); ++i)
            require(result[i] == (i == 2 ? 0 : UINT64_C(0xa7a7a7a7a7a7a7a7)), "separate kernel has no current graph and preserves complete guards");
        CUDA_KERNEL_NODE_PARAMS p{}; p.func = function; p.gridDimX = p.gridDimY = p.gridDimZ = p.blockDimX = p.blockDimY = p.blockDimZ = 1; p.kernelParams = arguments;
        CUgraphNode node{}; CONFORMANCE_CUDA(cuGraphAddKernelNode(&node, graph.value, nullptr, 0, &p));
        graph_device_executable executable(graph.value, mode, stream.value); executable.upload(stream.value, true); stream.wait();
        CONFORMANCE_CUDA(cuMemsetD8(memory.value, 0xa7, 64)); executable.launch(stream.value, mode & 1); stream.wait();
        CONFORMANCE_CUDA(cuMemcpyDtoH(result.data(), memory.value, 64));
        for (size_t i = 0; i < result.size(); ++i) require(result[i] == (i == 2 ? reinterpret_cast<uintptr_t>(executable.value) : UINT64_C(0xa7a7a7a7a7a7a7a7)), "device-visible graph identity and complete output guards");
    }
}
void graph_device_reject(CUgraph graph, unsigned mode, unsigned long long flags,
                         CUgraphInstantiateResult diagnostic = CUDA_GRAPH_INSTANTIATE_ERROR, CUgraphNode error = nullptr) {
    graph_device_output<CUgraphExec> driver;
    graph_device_output<cudaGraphExec_t> runtime;
    if (mode == 0) require(cuGraphInstantiate(&driver.value, graph, flags) == CUDA_ERROR_INVALID_VALUE, "Driver rejects invalid device graph");
    else if (mode == 1) require(cudaGraphInstantiateWithFlags(&runtime.value, graph_device_cast<cudaGraph_t>(graph), flags) == cudaErrorInvalidValue, "Runtime rejects invalid device graph");
    else if (mode == 2) {
        graph_device_output<CUDA_GRAPH_INSTANTIATE_PARAMS> output; output.value.flags = flags;
        require(cuGraphInstantiateWithParams(&driver.value, graph, &output.value) == CUDA_ERROR_INVALID_VALUE, "Driver parameter instantiation rejects invalid device graph");
        auto params = output.get();
        require(params.flags == flags && params.result_out == diagnostic && params.hErrNode_out == error, "Driver rejection diagnostics and preserved inputs");
    } else {
        graph_device_output<cudaGraphInstantiateParams> output; output.value.flags = flags;
        require(cudaGraphInstantiateWithParams(&runtime.value, graph_device_cast<cudaGraph_t>(graph), &output.value) == cudaErrorInvalidValue, "Runtime parameter instantiation rejects invalid device graph");
        auto params = output.get();
        require(params.flags == flags && params.result_out == static_cast<cudaGraphInstantiateResult>(diagnostic)
            && params.errNode_out == graph_device_cast<cudaGraphNode_t>(error), "Runtime rejection diagnostics and preserved inputs");
    }
    require(!driver.get() && !runtime.get(), "rejected instantiation preserves executable outputs and guards");
}
void graph_device_validation() {
    for (unsigned mode = 0; mode < 4; ++mode) {
        graph_device_graph graph;
        constexpr auto flags = CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH;
        graph_device_reject(graph.value, mode, flags | CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        graph_device_reject(graph.value, mode, flags);
        CUgraphNode node{}; CONFORMANCE_CUDA(cuGraphAddEmptyNode(&node, graph.value, nullptr, 0));
        graph_device_reject(graph.value, mode, flags, CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED, node);
        CONFORMANCE_CUDA(cuGraphDestroyNode(node));
        CUDA_HOST_NODE_PARAMS params{[](void*) {}, nullptr};
        CONFORMANCE_CUDA(cuGraphAddHostNode(&node, graph.value, nullptr, 0, &params));
        graph_device_reject(graph.value, mode, flags, CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED, node);
        CONFORMANCE_CUDA(cuGraphDestroyNode(node));
        graph_device_storage storage; graph_device_stream stream;
        image_test_storage array(CU_RESOURCE_TYPE_ARRAY, CU_AD_FORMAT_UNSIGNED_INT8, cudaChannelFormatKindUnsigned, 8, 1, 64, 1, mode & 1);
        array.upload(); storage.reset(0xa7);
        CUDA_MEMCPY3D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_ARRAY; copy.srcArray = array.array;
        copy.dstMemoryType = CU_MEMORYTYPE_DEVICE; copy.dstDevice = storage.device.value; copy.dstPitch = copy.WidthInBytes = 64;
        copy.Height = copy.Depth = 1;
        CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&node, graph.value, nullptr, 0, &copy, graph.context));
        graph_device_reject(graph.value, mode, flags, CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED, node);
        CONFORMANCE_CUDA(cuGraphDestroyNode(node)); array.verify(); storage.verify(0xa7, false);
        graph.fill(storage.device.value + 16, storage.bytes - 32, mode & 1);
        graph_device_executable repaired(graph.value, mode, stream.value);
        storage.reset(0xa7); repaired.upload(stream.value, !(mode & 1)); repaired.launch(stream.value, mode & 1); stream.wait(); storage.verify(0xa7, true);
    }
}
void graph_device_discovery() {
    static_assert(std::is_same_v<decltype(&cuGraphUpload), PFN_cuGraphUpload_v11010>);
    for (auto name : {"cuGraphInstantiateWithFlags", "cuGraphUpload"}) {
        int version = std::string_view(name) == "cuGraphUpload" ? 11010 : 11040;
        void* pointer{}; CUdriverProcAddressQueryResult query{};
        CONFORMANCE_CUDA(cuGetProcAddress(name, &pointer, version - 1, 0, &query)); require(!pointer && query == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, "device graph preparation procedure introduction boundary");
        CONFORMANCE_CUDA(cuGetProcAddress(name, &pointer, version, 0, &query)); require(pointer && query == CU_GET_PROC_ADDRESS_SUCCESS, "device graph preparation procedure lookup");
    }
}
void execute_graph_device() {
    auto version = native_cuda_version(); require_capability(version >= 12000, "native device graph preparation requires CUDA driver 12.0");
    CUdevice device{}; int unified{}; CONFORMANCE_CUDA(cuCtxGetDevice(&device)); CONFORMANCE_CUDA(cuDeviceGetAttribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
    require_capability(unified, "device graph preparation requires unified addressing");
    if (graph_device_case == "identity") {
        int major{}, minor{};
        CONFORMANCE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        CONFORMANCE_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        require_capability(major * 10 + minor >= 75, "graph identity PTX fixture requires SM75");
        graph_device_identity();
    }
    else if (graph_device_case == "validation") graph_device_validation();
    else if (graph_device_case == "upload" || graph_device_case == "replay" || graph_device_case == "lifetime") graph_device_preparation();
    else throw std::runtime_error("unknown device graph case");
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "discovery") return cudavm::run_host_conformance(argc, argv, cudavm::graph_device_discovery);
    if (argc == 3) { cudavm::graph_device_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_graph_device);
}
