#include "test_framework.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>

namespace cudavm {
void check_instantiation_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define INSTANTIATION_RUNTIME(call) ::cudavm::check_instantiation_runtime((call), #call)
struct instantiation_storage {
    device_buffer device{557}; void* output{}; unsigned callbacks = 0;
    instantiation_storage() { TEST_CUDA(cuMemAllocHost(&output, 601)); }
    ~instantiation_storage() { if (output) { cuCtxSynchronize(); cuMemFreeHost(output); } }
    static void CUDA_CB call(void* argument) { ++static_cast<instantiation_storage*>(argument)->callbacks; }
    void reset(unsigned seed) {
        TEST_CUDA(cuMemsetD8(device.value, static_cast<unsigned char>(seed), 557)); std::memset(output, 0xd3, 601); callbacks = 0;
    }
    void expect(unsigned seed, unsigned fill, bool executed) const {
        std::vector<unsigned char> expected_device(557, static_cast<unsigned char>(seed)), actual_device(557), expected_host(601, 0xd3);
        if (executed) {
            std::fill_n(expected_device.begin() + 23, 389, static_cast<unsigned char>(fill));
            std::copy(expected_device.begin(), expected_device.end(), expected_host.begin() + 19);
        }
        TEST_CUDA(cuMemcpyDtoH(actual_device.data(), device.value, actual_device.size()));
        require(actual_device == expected_device && std::memcmp(output, expected_host.data(), expected_host.size()) == 0 && callbacks == unsigned(executed),
            "instantiation/upload/launch has exact device bytes, host guards and callback count");
    }
};
struct instantiation_graph {
    CUgraph value{}; CUgraphNode fill{};
    explicit instantiation_graph(bool runtime) {
        if (runtime) { cudaGraph_t result{}; INSTANTIATION_RUNTIME(cudaGraphCreate(&result, 0)); value = reinterpret_cast<CUgraph>(result); }
        else TEST_CUDA(cuGraphCreate(&value, 0));
    }
    ~instantiation_graph() { if (value) cuGraphDestroy(value); }
    void destroy(bool runtime) {
        if (runtime) INSTANTIATION_RUNTIME(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(value)));
        else TEST_CUDA(cuGraphDestroy(value)); value = nullptr;
    }
    void set_fill(const instantiation_storage& storage, unsigned value, bool runtime) const {
        CUDA_MEMSET_NODE_PARAMS p{}; p.dst = storage.device.value + 23; p.value = value; p.elementSize = 1; p.width = 389; p.height = 1;
        if (runtime) {
            cudaMemsetParams params{reinterpret_cast<void*>(static_cast<uintptr_t>(p.dst)), 0, value, 1, 389, 1};
            INSTANTIATION_RUNTIME(cudaGraphMemsetNodeSetParams(reinterpret_cast<cudaGraphNode_t>(fill), &params));
        } else TEST_CUDA(cuGraphMemsetNodeSetParams(fill, &p));
    }
    void build(instantiation_storage& storage) {
        CUcontext context{}; TEST_CUDA(cuCtxGetCurrent(&context));
        CUDA_MEMSET_NODE_PARAMS p{}; p.dst = storage.device.value + 23; p.value = 0x35; p.elementSize = 1; p.width = 389; p.height = 1;
        TEST_CUDA(cuGraphAddMemsetNode(&fill, value, nullptr, 0, &p, context));
        CUDA_HOST_NODE_PARAMS callback{instantiation_storage::call, &storage}; CUgraphNode host{}, read{};
        TEST_CUDA(cuGraphAddHostNode(&host, value, &fill, 1, &callback));
        CUDA_MEMCPY3D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_DEVICE; copy.srcDevice = storage.device.value;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = static_cast<unsigned char*>(storage.output) + 19;
        copy.srcPitch = copy.dstPitch = copy.WidthInBytes = 557; copy.srcHeight = copy.dstHeight = copy.Height = copy.Depth = 1;
        TEST_CUDA(cuGraphAddMemcpyNode(&read, value, &host, 1, &copy, context));
    }
};
struct instantiation_stream {
    CUstream value{};
    instantiation_stream() { TEST_CUDA(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~instantiation_stream() { if (value) { cuStreamSynchronize(value); cuStreamDestroy(value); } }
};
struct instantiated_graph {
    CUgraphExec value{};
    instantiated_graph(CUgraph graph, unsigned mode, unsigned long long flags = 0, CUstream upload_stream = nullptr) {
        if (mode == 0) TEST_CUDA(cuGraphInstantiate(&value, graph, flags));
        else if (mode == 1) {
            cudaGraphExec_t result{}; INSTANTIATION_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), flags)); value = reinterpret_cast<CUgraphExec>(result);
        } else if (mode == 2) {
            CUDA_GRAPH_INSTANTIATE_PARAMS p{}; p.flags = flags; p.hUploadStream = upload_stream;
            TEST_CUDA(cuGraphInstantiateWithParams(&value, graph, &p));
            require(p.result_out == CUDA_GRAPH_INSTANTIATE_SUCCESS && !p.hErrNode_out, "Driver successful instantiation diagnostics");
        } else {
            cudaGraphInstantiateParams p{}; p.flags = flags; p.uploadStream = reinterpret_cast<cudaStream_t>(upload_stream);
            cudaGraphExec_t result{}; INSTANTIATION_RUNTIME(cudaGraphInstantiateWithParams(&result, reinterpret_cast<cudaGraph_t>(graph), &p)); value = reinterpret_cast<CUgraphExec>(result);
            require(p.result_out == cudaGraphInstantiateSuccess && !p.errNode_out, "Runtime successful instantiation diagnostics");
        }
    }
    ~instantiated_graph() { if (value) { cuCtxSynchronize(); cuGraphExecDestroy(value); } }
    void expect_flags(unsigned long long expected) const {
        struct guarded_flags { unsigned long long prefix = 0x123456789abcdef0, value = ~0ull, suffix = 0xfedcba9876543210; } driver, runtime;
        TEST_CUDA(cuGraphExecGetFlags(value, &driver.value));
        INSTANTIATION_RUNTIME(cudaGraphExecGetFlags(reinterpret_cast<cudaGraphExec_t>(value), &runtime.value));
        for (const auto& p : {driver, runtime}) require(p.value == expected && p.prefix == 0x123456789abcdef0 && p.suffix == 0xfedcba9876543210, "persistent executable flags and exact-width guards");
    }
    void upload(CUstream stream, bool runtime) const {
        if (runtime) INSTANTIATION_RUNTIME(cudaGraphUpload(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else TEST_CUDA(cuGraphUpload(value, stream));
        TEST_CUDA(cuStreamSynchronize(stream));
    }
    void launch(bool runtime, CUstream stream = nullptr) const {
        if (runtime) INSTANTIATION_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), reinterpret_cast<cudaStream_t>(stream)));
        else TEST_CUDA(cuGraphLaunch(value, stream));
        TEST_CUDA(cuStreamSynchronize(stream));
    }
};
void graph_instantiation_empty() {
    for (unsigned mode = 0; mode < 4; ++mode) {
        instantiation_graph source(mode & 1); instantiated_graph executable(source.value, mode);
        executable.expect_flags(0);
        for (unsigned repeat = 0; repeat < 3; ++repeat) executable.launch(repeat & 1);
    }
}
void graph_instantiation_flags() {
    for (unsigned mode = 0; mode < 4; ++mode)
    for (unsigned long long flags : {0ull, 1ull, 8ull, 9ull}) {
        instantiation_storage storage; instantiation_graph source(mode & 1); source.build(storage);
        instantiated_graph executable(source.value, mode, flags); executable.expect_flags(flags);
        storage.reset(0x91); executable.launch(!(mode & 1)); storage.expect(0x91, 0x35, true);
    }
}
void graph_instantiation_snapshot() {
    for (unsigned mode = 0; mode < 4; ++mode) {
        instantiation_storage storage; instantiation_graph source(mode & 1); source.build(storage);
        instantiated_graph original(source.value, mode);
        source.set_fill(storage, 0x99, mode & 1);
        instantiated_graph changed(source.value, (mode + 1) % 4);
        for (unsigned seed : {0u, 0x7du, 0xffu}) {
            storage.reset(seed); original.launch(true); storage.expect(seed, 0x35, true);
            storage.reset(seed); changed.launch(false); storage.expect(seed, 0x99, true);
        }
    }
}
void graph_instantiation_upload() {
    for (unsigned mode = 0; mode < 4; ++mode) {
        instantiation_storage storage; instantiation_graph source(mode & 1); source.build(storage);
        instantiation_stream stream; storage.reset(0x91);
        const auto flags = mode < 2 ? 0ull : static_cast<unsigned long long>(CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD | CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY);
        instantiated_graph executable(source.value, mode, flags, stream.value);
        TEST_CUDA(cuStreamSynchronize(stream.value)); storage.expect(0x91, 0, false);
        const auto persistent = flags & ~static_cast<unsigned long long>(CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD);
        executable.expect_flags(persistent);
        for (bool runtime : {false, true}) {
            executable.upload(stream.value, runtime); storage.expect(0x91, 0, false); executable.expect_flags(persistent);
        }
        executable.launch(!(mode & 1), stream.value); storage.expect(0x91, 0x35, true);
        storage.reset(0x7d); executable.upload(stream.value, true); storage.expect(0x7d, 0, false);
        executable.launch(mode & 1, stream.value); storage.expect(0x7d, 0x35, true);
    }
}
void graph_instantiation_lifetime() {
    for (unsigned mode = 0; mode < 4; ++mode) {
        instantiation_storage storage; instantiation_graph source(mode & 1); source.build(storage);
        instantiated_graph executable(source.value, mode);
        source.destroy(!(mode & 1));
        executable.expect_flags(0);
        for (unsigned seed : {0u, 0x7du, 0xffu}) {
            storage.reset(seed); executable.launch(mode & 1); storage.expect(seed, 0x35, true);
        }
    }
}
void graph_instantiation_validation() {
    instantiation_graph source(false); CUgraphExec driver{}; cudaGraphExec_t runtime{};
    CUDA_GRAPH_INSTANTIATE_PARAMS p{}; cudaGraphInstantiateParams q{};
    require(cuGraphInstantiate(nullptr, source.value, 0) == CUDA_ERROR_INVALID_VALUE, "null Driver executable output");
    require(cudaGraphInstantiateWithFlags(nullptr, reinterpret_cast<cudaGraph_t>(source.value), 0) == cudaErrorInvalidValue, "null Runtime executable output");
    require(cuGraphInstantiateWithParams(&driver, source.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver instantiation parameters");
    require(cudaGraphInstantiateWithParams(&runtime, reinterpret_cast<cudaGraph_t>(source.value), nullptr) == cudaErrorInvalidValue, "null Runtime instantiation parameters");
    require(cuGraphInstantiateWithParams(nullptr, source.value, &p) == CUDA_ERROR_INVALID_VALUE, "null WithParams Driver executable output");
    require(cudaGraphInstantiateWithParams(nullptr, reinterpret_cast<cudaGraph_t>(source.value), &q) == cudaErrorInvalidValue, "null WithParams Runtime executable output");
    constexpr auto unknown_flags = 1ull << 63;
    require(cuGraphInstantiate(&driver, source.value, unknown_flags) == CUDA_ERROR_INVALID_VALUE, "Driver unknown instantiation flags");
    require(cudaGraphInstantiateWithFlags(&runtime, reinterpret_cast<cudaGraph_t>(source.value), unknown_flags) == cudaErrorInvalidValue, "Runtime unknown instantiation flags");
    p.flags = q.flags = unknown_flags;
    require(cuGraphInstantiateWithParams(&driver, source.value, &p) == CUDA_ERROR_INVALID_VALUE && p.result_out == CUDA_GRAPH_INSTANTIATE_ERROR, "Driver unknown flags have failure diagnostics");
    require(cudaGraphInstantiateWithParams(&runtime, reinterpret_cast<cudaGraph_t>(source.value), &q) == cudaErrorInvalidValue && q.result_out == cudaGraphInstantiateError, "Runtime unknown flags have failure diagnostics");
    CUgraphNode a{}, b{};
    TEST_CUDA(cuGraphAddEmptyNode(&a, source.value, nullptr, 0)); TEST_CUDA(cuGraphAddEmptyNode(&b, source.value, &a, 1));
    TEST_CUDA(cuGraphAddDependencies(source.value, &b, &a, nullptr, 1));
    p = {}; q = {};
    require(cuGraphInstantiateWithParams(&driver, source.value, &p) == CUDA_ERROR_INVALID_VALUE && p.result_out == CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE
        && (p.hErrNode_out == a || p.hErrNode_out == b), "Driver cyclic graph reports structural failure and a cycle node");
    require(cudaGraphInstantiateWithParams(&runtime, reinterpret_cast<cudaGraph_t>(source.value), &q) == cudaErrorInvalidValue && q.result_out == cudaGraphInstantiateInvalidStructure
        && (reinterpret_cast<CUgraphNode>(q.errNode_out) == a || reinterpret_cast<CUgraphNode>(q.errNode_out) == b), "Runtime cyclic graph reports structural failure and a cycle node");
    TEST_CUDA(cuGraphRemoveDependencies(source.value, &b, &a, nullptr, 1));
    instantiated_graph executable(source.value, 3); executable.launch(false);
    require(cuGraphExecGetFlags(executable.value, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver flags output");
    require(cudaGraphExecGetFlags(reinterpret_cast<cudaGraphExec_t>(executable.value), nullptr) == cudaErrorInvalidValue, "null Runtime flags output");
}
std::string instantiation_case = "all";
void execute_graph_instantiation() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (instantiation_case == "all" || instantiation_case == name) { execute(); selected = true; } };
    run("empty", graph_instantiation_empty); run("flags", graph_instantiation_flags); run("snapshot", graph_instantiation_snapshot);
    run("upload", graph_instantiation_upload); run("lifetime", graph_instantiation_lifetime); run("validation", graph_instantiation_validation);
    require(selected, "unknown graph_instantiation case: " + instantiation_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::instantiation_case = argv[2]; --argc; }
    return cudavm::run_test(argc, argv, cudavm::execute_graph_instantiation);
}
