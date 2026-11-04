#include "conformance.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <limits>

namespace cudavm {
void check_graph_memory_runtime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define GRAPH_MEMORY_RUNTIME(call) ::cudavm::check_graph_memory_runtime((call), #call)
struct graph_memory_host {
    static constexpr size_t bytes = 4128;
    void* value{};
    graph_memory_host() { CONFORMANCE_CUDA(cuMemAllocHost(&value, bytes)); }
    ~graph_memory_host() { if (value) { cuCtxSynchronize(); cuMemFreeHost(value); } }
    unsigned char* data() const { return static_cast<unsigned char*>(value); }
    void reset(unsigned char value) { std::memset(data(), value, bytes); }
    void expect(std::span<const unsigned char> expected) const {
        require(expected.size() == bytes && std::memcmp(data(), expected.data(), bytes) == 0, "graph memory host payload and all initialized guards");
    }
};
cudaMemsetParams graph_memory_runtime_fill(const CUDA_MEMSET_NODE_PARAMS& p) {
    return {reinterpret_cast<void*>(static_cast<uintptr_t>(p.dst)), p.pitch, p.value, p.elementSize, p.width, p.height};
}
cudaMemcpy3DParms graph_memory_runtime_copy(const CUDA_MEMCPY3D& p) {
    cudaMemcpy3DParms result{};
    result.srcPtr = {p.srcMemoryType == CU_MEMORYTYPE_HOST ? const_cast<void*>(p.srcHost) : reinterpret_cast<void*>(static_cast<uintptr_t>(p.srcDevice)), p.srcPitch, p.srcPitch, p.srcHeight};
    result.dstPtr = {p.dstMemoryType == CU_MEMORYTYPE_HOST ? p.dstHost : reinterpret_cast<void*>(static_cast<uintptr_t>(p.dstDevice)), p.dstPitch, p.dstPitch, p.dstHeight};
    result.srcPos = {p.srcXInBytes, p.srcY, p.srcZ}; result.dstPos = {p.dstXInBytes, p.dstY, p.dstZ};
    result.extent = {p.WidthInBytes, p.Height, p.Depth};
    result.kind = p.srcMemoryType == CU_MEMORYTYPE_HOST ? cudaMemcpyHostToDevice
        : p.dstMemoryType == CU_MEMORYTYPE_HOST ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
    return result;
}
struct graph_memory_graph {
    CUgraph value{}; CUcontext context{};
    explicit graph_memory_graph(bool runtime) {
        CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        if (runtime) { cudaGraph_t result{}; GRAPH_MEMORY_RUNTIME(cudaGraphCreate(&result, 0)); value = reinterpret_cast<CUgraph>(result); }
        else CONFORMANCE_CUDA(cuGraphCreate(&value, 0));
    }
    ~graph_memory_graph() { if (value) cuGraphDestroy(value); }
    CUgraphNode fill(const CUDA_MEMSET_NODE_PARAMS& params, bool runtime, CUgraphNode dependency = nullptr) {
        CUgraphNode node{};
        if (runtime) {
            auto p = graph_memory_runtime_fill(params); auto dep = reinterpret_cast<cudaGraphNode_t>(dependency); cudaGraphNode_t result{};
            GRAPH_MEMORY_RUNTIME(cudaGraphAddMemsetNode(&result, reinterpret_cast<cudaGraph_t>(value), dependency ? &dep : nullptr, dependency ? 1 : 0, &p));
            node = reinterpret_cast<CUgraphNode>(result);
        } else CONFORMANCE_CUDA(cuGraphAddMemsetNode(&node, value, dependency ? &dependency : nullptr, dependency ? 1 : 0, &params, context));
        return node;
    }
    CUgraphNode copy(const CUDA_MEMCPY3D& params, bool runtime, CUgraphNode dependency = nullptr) {
        CUgraphNode node{};
        if (runtime) {
            auto p = graph_memory_runtime_copy(params); auto dep = reinterpret_cast<cudaGraphNode_t>(dependency); cudaGraphNode_t result{};
            GRAPH_MEMORY_RUNTIME(cudaGraphAddMemcpyNode(&result, reinterpret_cast<cudaGraph_t>(value), dependency ? &dep : nullptr, dependency ? 1 : 0, &p));
            node = reinterpret_cast<CUgraphNode>(result);
        } else CONFORMANCE_CUDA(cuGraphAddMemcpyNode(&node, value, dependency ? &dependency : nullptr, dependency ? 1 : 0, &params, context));
        return node;
    }
};
struct graph_memory_executable {
    CUgraphExec value{};
    graph_memory_executable(CUgraph graph, bool runtime) {
        if (runtime) { cudaGraphExec_t result{}; GRAPH_MEMORY_RUNTIME(cudaGraphInstantiateWithFlags(&result, reinterpret_cast<cudaGraph_t>(graph), 0)); value = reinterpret_cast<CUgraphExec>(result); }
        else CONFORMANCE_CUDA(cuGraphInstantiate(&value, graph, 0));
    }
    ~graph_memory_executable() { if (value) { cuCtxSynchronize(); cuGraphExecDestroy(value); } }
    void launch(bool runtime) {
        if (runtime) GRAPH_MEMORY_RUNTIME(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(value), nullptr));
        else CONFORMANCE_CUDA(cuGraphLaunch(value, nullptr));
        CONFORMANCE_CUDA(cuCtxSynchronize());
    }
};
CUDA_MEMSET_NODE_PARAMS graph_memory_fill(CUdeviceptr device, size_t offset, unsigned element, size_t width, size_t height, size_t pitch, unsigned value) {
    CUDA_MEMSET_NODE_PARAMS p{}; p.dst = device + offset; p.elementSize = element; p.width = width; p.height = height; p.pitch = pitch; p.value = value; return p;
}
CUDA_MEMCPY3D graph_memory_read(CUdeviceptr device, void* host, size_t bytes) {
    CUDA_MEMCPY3D p{}; p.srcMemoryType = CU_MEMORYTYPE_DEVICE; p.srcDevice = device;
    p.dstMemoryType = CU_MEMORYTYPE_HOST; p.dstHost = host;
    p.srcPitch = p.dstPitch = p.WidthInBytes = bytes; p.srcHeight = p.dstHeight = p.Height = p.Depth = 1; return p;
}
void graph_memory_expect_fill(CUgraphNode node, const CUDA_MEMSET_NODE_PARAMS& expected) {
    struct output { uint64_t prefix = 0x123456789abcdef0; CUDA_MEMSET_NODE_PARAMS value{}; uint64_t suffix = 0xfedcba9876543210; } driver;
    CONFORMANCE_CUDA(cuGraphMemsetNodeGetParams(node, &driver.value));
    cudaMemsetParams runtime{}; GRAPH_MEMORY_RUNTIME(cudaGraphMemsetNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &runtime));
    const auto check = [&](const auto& p, CUdeviceptr destination) {
        require(destination == expected.dst && p.pitch == expected.pitch && p.value == expected.value && p.elementSize == expected.elementSize
            && p.width == expected.width && p.height == expected.height, "graph fill parameter snapshot through both facades");
    };
    check(driver.value, driver.value.dst); check(runtime, reinterpret_cast<uintptr_t>(runtime.dst));
    require(driver.prefix == 0x123456789abcdef0 && driver.suffix == 0xfedcba9876543210, "fill parameter output guards");
}
void graph_memory_expect_copy(CUgraphNode node, const CUDA_MEMCPY3D& expected) {
    CUDA_MEMCPY3D p{}; CONFORMANCE_CUDA(cuGraphMemcpyNodeGetParams(node, &p));
    require(p.srcMemoryType == expected.srcMemoryType && p.dstMemoryType == expected.dstMemoryType
        && (p.srcMemoryType == CU_MEMORYTYPE_HOST ? p.srcHost == expected.srcHost : p.srcDevice == expected.srcDevice)
        && (p.dstMemoryType == CU_MEMORYTYPE_HOST ? p.dstHost == expected.dstHost : p.dstDevice == expected.dstDevice)
        && p.srcPitch == expected.srcPitch && p.dstPitch == expected.dstPitch && p.srcHeight == expected.srcHeight && p.dstHeight == expected.dstHeight
        && p.srcXInBytes == expected.srcXInBytes && p.srcY == expected.srcY && p.srcZ == expected.srcZ
        && p.dstXInBytes == expected.dstXInBytes && p.dstY == expected.dstY && p.dstZ == expected.dstZ
        && p.WidthInBytes == expected.WidthInBytes && p.Height == expected.Height && p.Depth == expected.Depth, "Driver copy parameter snapshot");
    cudaMemcpy3DParms actual{}; GRAPH_MEMORY_RUNTIME(cudaGraphMemcpyNodeGetParams(reinterpret_cast<cudaGraphNode_t>(node), &actual));
    const auto wanted = graph_memory_runtime_copy(expected);
    require(actual.srcPtr.ptr == wanted.srcPtr.ptr && actual.dstPtr.ptr == wanted.dstPtr.ptr && !actual.srcArray && !actual.dstArray
        && actual.srcPtr.pitch == wanted.srcPtr.pitch && actual.dstPtr.pitch == wanted.dstPtr.pitch
        && actual.srcPtr.ysize == wanted.srcPtr.ysize && actual.dstPtr.ysize == wanted.dstPtr.ysize
        && actual.srcPos.x == wanted.srcPos.x && actual.srcPos.y == wanted.srcPos.y && actual.srcPos.z == wanted.srcPos.z
        && actual.dstPos.x == wanted.dstPos.x && actual.dstPos.y == wanted.dstPos.y && actual.dstPos.z == wanted.dstPos.z
        && actual.extent.width == wanted.extent.width && actual.extent.height == wanted.extent.height && actual.extent.depth == wanted.extent.depth
        && actual.kind == wanted.kind, "Runtime copy endpoint classification and geometry snapshot");
}
void graph_memory_set_fill(CUgraphNode node, const CUDA_MEMSET_NODE_PARAMS& p, bool runtime, CUgraphExec executable = nullptr) {
    if (runtime) {
        auto params = graph_memory_runtime_fill(p);
        if (executable) GRAPH_MEMORY_RUNTIME(cudaGraphExecMemsetNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaGraphNode_t>(node), &params));
        else GRAPH_MEMORY_RUNTIME(cudaGraphMemsetNodeSetParams(reinterpret_cast<cudaGraphNode_t>(node), &params));
    } else if (executable) {
        CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        CONFORMANCE_CUDA(cuGraphExecMemsetNodeSetParams(executable, node, &p, context));
    } else CONFORMANCE_CUDA(cuGraphMemsetNodeSetParams(node, &p));
}
void graph_memory_set_copy(CUgraphNode node, const CUDA_MEMCPY3D& p, bool runtime, CUgraphExec executable = nullptr, bool convenience = false) {
    if (runtime) {
        auto params = graph_memory_runtime_copy(p);
        if (convenience && executable) GRAPH_MEMORY_RUNTIME(cudaGraphExecMemcpyNodeSetParams1D(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaGraphNode_t>(node), params.dstPtr.ptr, params.srcPtr.ptr, p.WidthInBytes, params.kind));
        else if (convenience) GRAPH_MEMORY_RUNTIME(cudaGraphMemcpyNodeSetParams1D(reinterpret_cast<cudaGraphNode_t>(node), params.dstPtr.ptr, params.srcPtr.ptr, p.WidthInBytes, params.kind));
        else if (executable) GRAPH_MEMORY_RUNTIME(cudaGraphExecMemcpyNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable), reinterpret_cast<cudaGraphNode_t>(node), &params));
        else GRAPH_MEMORY_RUNTIME(cudaGraphMemcpyNodeSetParams(reinterpret_cast<cudaGraphNode_t>(node), &params));
    } else if (executable) {
        CUcontext context{}; CONFORMANCE_CUDA(cuCtxGetCurrent(&context));
        CONFORMANCE_CUDA(cuGraphExecMemcpyNodeSetParams(executable, node, &p, context));
    } else CONFORMANCE_CUDA(cuGraphMemcpyNodeSetParams(node, &p));
}
void graph_memory_run_fill(graph_memory_executable& executable, CUdeviceptr device, graph_memory_host& output,
                          const CUDA_MEMSET_NODE_PARAMS& fill, const CUDA_MEMCPY3D& read, unsigned seed, bool runtime) {
    CONFORMANCE_CUDA(cuMemsetD8(device, static_cast<unsigned char>(seed), 4096)); output.reset(0xd3);
    std::vector<unsigned char> expected_device(4096, static_cast<unsigned char>(seed)), expected_host(graph_memory_host::bytes, 0xd3);
    const auto start = size_t(fill.dst - device), pitch = fill.height == 1 ? fill.width * fill.elementSize : fill.pitch;
    for (size_t row = 0; row < fill.height; ++row) for (size_t element = 0; element < fill.width; ++element)
        for (unsigned byte = 0; byte < fill.elementSize; ++byte)
            expected_device[start + row * pitch + element * fill.elementSize + byte] = static_cast<unsigned char>(fill.value >> (8 * byte));
    const auto source = size_t(read.srcDevice - device), destination = size_t(static_cast<unsigned char*>(read.dstHost) - output.data());
    for (size_t i = 0; i < read.WidthInBytes; ++i) expected_host[destination + i] = expected_device[source + i];
    executable.launch(runtime); output.expect(expected_host);
    std::vector<unsigned char> actual_device(4096); CONFORMANCE_CUDA(cuMemcpyDtoH(actual_device.data(), device, actual_device.size()));
    require(actual_device == expected_device, "graph typed fill preserves device padding and every untouched byte");
}
void graph_memory_nodes_fills() {
    for (bool runtime : {false, true}) for (unsigned element : {1u, 2u, 4u})
    for (size_t width : {size_t(1), size_t(17), size_t(53)}) for (size_t height : {size_t(1), size_t(4)}) {
        device_buffer device(4096); graph_memory_host output; graph_memory_graph graph(runtime);
        auto fill = graph_memory_fill(device.value, 64, element, width, height, 256, 0x89abcdef);
        const auto expected = fill; const auto node = graph.fill(fill, runtime); fill = {};
        graph_memory_expect_fill(node, expected);
        const auto read = graph_memory_read(device.value, output.data() + 16, 4096);
        const auto copy = graph.copy(read, !runtime, node); graph_memory_expect_copy(copy, read);
        graph_memory_executable executable(graph.value, !runtime);
        graph_memory_run_fill(executable, device.value, output, expected, read, 0x91, runtime);
    }
}
void graph_memory_nodes_copies() {
    for (bool runtime : {false, true}) {
        device_buffer first(4096), second(4096); graph_memory_host input, output;
        std::vector<unsigned char> source(graph_memory_host::bytes);
        for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<unsigned char>((i * 73 + 19) ^ (i >> 3));
        std::memcpy(input.value, source.data(), source.size());
        graph_memory_graph graph(runtime);
        CUDA_MEMCPY3D upload{};
        upload.srcMemoryType = CU_MEMORYTYPE_HOST; upload.srcHost = input.data() + 16;
        upload.srcPitch = 83; upload.srcHeight = 5; upload.srcXInBytes = 5; upload.srcY = upload.srcZ = 1;
        upload.dstMemoryType = CU_MEMORYTYPE_DEVICE; upload.dstDevice = first.value + 64;
        upload.dstPitch = 101; upload.dstHeight = 7; upload.dstXInBytes = 7; upload.dstY = 1;
        upload.WidthInBytes = 37; upload.Height = 3; upload.Depth = 2;
        const auto a = graph.copy(upload, runtime); graph_memory_expect_copy(a, upload);
        CUDA_MEMCPY3D transfer{};
        transfer.srcMemoryType = transfer.dstMemoryType = CU_MEMORYTYPE_DEVICE; transfer.srcDevice = first.value + 64;
        transfer.srcPitch = 101; transfer.srcHeight = 7; transfer.srcXInBytes = 7; transfer.srcY = 1;
        transfer.dstDevice = second.value + 128; transfer.dstPitch = 113; transfer.dstHeight = 6;
        transfer.dstXInBytes = 11; transfer.dstY = 2; transfer.dstZ = 1;
        transfer.WidthInBytes = 37; transfer.Height = 3; transfer.Depth = 2;
        const auto b = graph.copy(transfer, !runtime, a); graph_memory_expect_copy(b, transfer);
        CUDA_MEMCPY3D read{};
        read.srcMemoryType = CU_MEMORYTYPE_DEVICE; read.srcDevice = second.value + 128;
        read.srcPitch = 113; read.srcHeight = 6; read.srcXInBytes = 11; read.srcY = 2; read.srcZ = 1;
        read.dstMemoryType = CU_MEMORYTYPE_HOST; read.dstHost = output.data() + 16;
        read.dstPitch = 89; read.dstHeight = 8; read.dstXInBytes = 3; read.dstY = 1;
        read.WidthInBytes = 37; read.Height = 3; read.Depth = 2;
        const auto c = graph.copy(read, runtime, b); graph_memory_expect_copy(c, read);
        graph_memory_executable executable(graph.value, !runtime);
        for (unsigned seed : {0u, 0x91u, 0xffu}) {
            CONFORMANCE_CUDA(cuMemsetD8(first.value, static_cast<unsigned char>(seed), 4096));
            CONFORMANCE_CUDA(cuMemsetD8(second.value, static_cast<unsigned char>(seed ^ 0x5a), 4096)); output.reset(0xd3);
            std::vector<unsigned char> expected_a(4096, static_cast<unsigned char>(seed)), expected_b(4096, static_cast<unsigned char>(seed ^ 0x5a)), expected_output(graph_memory_host::bytes, 0xd3);
            for (size_t z = 0; z < 2; ++z) for (size_t y = 0; y < 3; ++y) for (size_t x = 0; x < 37; ++x) {
                const auto byte = source[16 + (z + 1) * 5 * 83 + (y + 1) * 83 + 5 + x];
                expected_a[64 + z * 7 * 101 + (y + 1) * 101 + 7 + x] = byte;
                expected_b[128 + (z + 1) * 6 * 113 + (y + 2) * 113 + 11 + x] = byte;
                expected_output[16 + z * 8 * 89 + (y + 1) * 89 + 3 + x] = byte;
            }
            executable.launch(runtime); input.expect(source); output.expect(expected_output);
            std::vector<unsigned char> actual(4096);
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), first.value, actual.size())); require(actual == expected_a, "graph HtoD rows, slices and untouched source storage");
            CONFORMANCE_CUDA(cuMemcpyDtoH(actual.data(), second.value, actual.size())); require(actual == expected_b, "graph DtoD rows, slices and device padding");
        }
    }
}
void graph_memory_nodes_parameters() {
    for (bool runtime : {false, true}) {
        device_buffer device(4096); graph_memory_host output; graph_memory_graph graph(runtime);
        const auto original_fill = graph_memory_fill(device.value, 64, 1, 173, 1, 0, 0x35);
        const auto original_read = graph_memory_read(device.value, output.data() + 16, 557);
        const auto fill = graph.fill(original_fill, runtime), read = graph.copy(original_read, !runtime, fill);
        graph_memory_executable original(graph.value, runtime);
        auto changed_fill = graph_memory_fill(device.value, 128, 2, 67, 1, 0, 0x89abcdef);
        auto changed_read = graph_memory_read(device.value + 96, output.data() + 47, 389);
        graph_memory_set_fill(fill, changed_fill, !runtime); graph_memory_set_copy(read, changed_read, runtime);
        graph_memory_expect_fill(fill, changed_fill); graph_memory_expect_copy(read, changed_read);
        graph_memory_executable changed(graph.value, !runtime);
        const auto retained_fill = changed_fill; const auto retained_read = changed_read; changed_fill = {}; changed_read = {};
        graph_memory_expect_fill(fill, retained_fill); graph_memory_expect_copy(read, retained_read);
        graph_memory_run_fill(original, device.value, output, original_fill, original_read, 0x91, !runtime);
        graph_memory_run_fill(changed, device.value, output, retained_fill, retained_read, 0x7d, runtime);
    }
}
void graph_memory_nodes_executable() {
    for (bool runtime : {false, true}) {
        device_buffer device(4096); graph_memory_host output; graph_memory_graph graph(runtime);
        const auto original_fill = graph_memory_fill(device.value, 64, 1, 173, 1, 0, 0x35);
        const auto original_read = graph_memory_read(device.value, output.data() + 16, 557);
        const auto fill = graph.fill(original_fill, runtime), read = graph.copy(original_read, !runtime, fill);
        graph_memory_executable changed(graph.value, runtime), independent(graph.value, !runtime);
        const auto updated_fill = graph_memory_fill(device.value, 128, 1, 173, 1, 0, 0x71);
        const auto updated_read = graph_memory_read(device.value + 23, output.data() + 47, 557);
        graph_memory_set_fill(fill, updated_fill, runtime, changed.value);
        graph_memory_set_copy(read, updated_read, !runtime, changed.value);
        graph_memory_expect_fill(fill, original_fill); graph_memory_expect_copy(read, original_read);
        graph_memory_run_fill(changed, device.value, output, updated_fill, updated_read, 0x91, runtime);
        graph_memory_run_fill(independent, device.value, output, original_fill, original_read, 0x7d, !runtime);
        const auto convenience_read = graph_memory_read(device.value + 41, output.data() + 73, 389);
        graph_memory_set_copy(read, convenience_read, true, changed.value, true);
        graph_memory_run_fill(changed, device.value, output, updated_fill, convenience_read, 0xff, false);
        graph_memory_set_copy(read, convenience_read, true, nullptr, true);
        graph_memory_executable subsequent(graph.value, false);
        graph_memory_run_fill(subsequent, device.value, output, original_fill, convenience_read, 0, true);
    }
}
void graph_memory_nodes_validation() {
    device_buffer device(4096); graph_memory_host output; graph_memory_graph graph(false);
    const auto fill_params = graph_memory_fill(device.value, 64, 2, 71, 1, 0, 0x89abcdef);
    const auto copy_params = graph_memory_read(device.value, output.data() + 16, 557);
    const auto fill = graph.fill(fill_params, false), copy = graph.copy(copy_params, true, fill);
    require(cuGraphMemsetNodeSetParams(fill, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver fill parameters");
    require(cudaGraphMemsetNodeSetParams(reinterpret_cast<cudaGraphNode_t>(fill), nullptr) == cudaErrorInvalidValue, "null Runtime fill parameters");
    require(cuGraphMemcpyNodeSetParams(copy, nullptr) == CUDA_ERROR_INVALID_VALUE, "null Driver copy parameters");
    require(cudaGraphMemcpyNodeSetParams(reinterpret_cast<cudaGraphNode_t>(copy), nullptr) == cudaErrorInvalidValue, "null Runtime copy parameters");
    CUDA_MEMSET_NODE_PARAMS queried_fill{}; CUDA_MEMCPY3D queried_copy{};
    require(cuGraphMemsetNodeGetParams(copy, &queried_fill) == CUDA_ERROR_INVALID_VALUE, "copy node cannot be queried as fill");
    require(cuGraphMemcpyNodeGetParams(fill, &queried_copy) == CUDA_ERROR_INVALID_VALUE, "fill node cannot be queried as copy");
    require(cuGraphMemsetNodeGetParams(fill, nullptr) == CUDA_ERROR_INVALID_VALUE, "null fill query output");
    require(cudaGraphMemcpyNodeGetParams(reinterpret_cast<cudaGraphNode_t>(copy), nullptr) == cudaErrorInvalidValue, "null Runtime copy query output");
    for (unsigned variant = 0; variant < 8; ++variant) {
        auto p = fill_params;
        if (variant == 0) p.elementSize = 3;
        else if (variant == 1) p.width = 0;
        else if (variant == 2) p.height = 0;
        else if (variant == 3) ++p.dst;
        else if (variant == 4) { p.height = 3; p.pitch = p.width * p.elementSize - 2; }
        else if (variant == 5) { p.height = 3; p.pitch = 257; }
        else if (variant == 6) p.dst = device.value + 4090;
        else p.width = std::numeric_limits<size_t>::max();
        require(cuGraphMemsetNodeSetParams(fill, &p) == CUDA_ERROR_INVALID_VALUE, "invalid Driver fill geometry, alignment or bounds");
        auto runtime = graph_memory_runtime_fill(p);
        require(cudaGraphMemsetNodeSetParams(reinterpret_cast<cudaGraphNode_t>(fill), &runtime) == cudaErrorInvalidValue, "invalid Runtime fill geometry, alignment or bounds");
        graph_memory_expect_fill(fill, fill_params);
    }
    for (unsigned variant = 0; variant < 5; ++variant) {
        auto p = copy_params;
        if (variant == 0) p.WidthInBytes = 0;
        else if (variant == 1) p.Height = 0;
        else if (variant == 2) p.Depth = 0;
        else if (variant == 3) p.srcXInBytes = p.srcPitch;
        else p.srcDevice = device.value + 4090;
        require(cuGraphMemcpyNodeSetParams(copy, &p) == CUDA_ERROR_INVALID_VALUE, "invalid Driver copy extent or bounds");
        auto runtime = graph_memory_runtime_copy(p);
        const auto result = cudaGraphMemcpyNodeSetParams(reinterpret_cast<cudaGraphNode_t>(copy), &runtime);
        require(result == cudaErrorInvalidValue, "invalid Runtime copy extent or bounds variant=" + std::to_string(variant) + " result=" + cudaGetErrorName(result));
        graph_memory_expect_copy(copy, copy_params);
    }
    auto invalid_direction = graph_memory_runtime_copy(copy_params); invalid_direction.kind = static_cast<cudaMemcpyKind>(127);
    require(cudaGraphMemcpyNodeSetParams(reinterpret_cast<cudaGraphNode_t>(copy), &invalid_direction) == cudaErrorInvalidMemcpyDirection, "invalid Runtime copy direction");
    require(cuGraphMemsetNodeSetParams(copy, &fill_params) == CUDA_ERROR_INVALID_VALUE, "fill setter rejects copy node");
    require(cuGraphMemcpyNodeSetParams(fill, &copy_params) == CUDA_ERROR_INVALID_VALUE, "copy setter rejects fill node");
    graph_memory_expect_fill(fill, fill_params); graph_memory_expect_copy(copy, copy_params);
    graph_memory_executable executable(graph.value, true);
    auto invalid_pitch = graph_memory_runtime_copy(copy_params); invalid_pitch.srcPos.x = invalid_pitch.srcPtr.pitch;
    cudaGraphNode_t rejected{};
    require(cudaGraphAddMemcpyNode(&rejected, reinterpret_cast<cudaGraph_t>(graph.value), nullptr, 0, &invalid_pitch) == cudaErrorInvalidValue,
        "graph copy creation reports invalid value for invalid pitch");
    require(cudaGraphExecMemcpyNodeSetParams(reinterpret_cast<cudaGraphExec_t>(executable.value), reinterpret_cast<cudaGraphNode_t>(copy), &invalid_pitch) == cudaErrorInvalidValue,
        "executable graph copy setter reports invalid value for invalid pitch");
    cudaGraphNodeParams generic{}; generic.type = cudaGraphNodeTypeMemcpy; generic.memcpy.copyParams = invalid_pitch;
    require(cudaGraphAddNode(&rejected, reinterpret_cast<cudaGraph_t>(graph.value), nullptr, nullptr, 0, &generic) == cudaErrorInvalidValue,
        "generic graph copy creation reports invalid value for invalid pitch");
    graph_memory_run_fill(executable, device.value, output, fill_params, copy_params, 0x91, false);
}
std::string graph_memory_case = "all";
void execute_graph_memory_nodes() {
    bool selected = false;
    const auto run = [&](const char* name, auto execute) { if (graph_memory_case == "all" || graph_memory_case == name) { execute(); selected = true; } };
    run("fills", graph_memory_nodes_fills); run("copies", graph_memory_nodes_copies); run("parameters", graph_memory_nodes_parameters);
    run("executable", graph_memory_nodes_executable); run("validation", graph_memory_nodes_validation);
    require(selected, "unknown graph_memory_nodes case: " + graph_memory_case);
}
}
int main(int argc, char** argv) {
    if (argc == 3) { cudavm::graph_memory_case = argv[2]; --argc; }
    return cudavm::run_conformance(argc, argv, cudavm::execute_graph_memory_nodes);
}
