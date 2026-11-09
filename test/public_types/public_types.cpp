#include "test_framework.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc/nvrtc.h>
#include <nvvm/nvvm.h>
#include <nvfatbin/nvFatbin.h>
#include <nvptxcompiler/nvPTXCompiler.h>
#define NVJITLINK_NO_INLINE
#include <nvjitlink/nvJitLink.h>
#include <type_traits>
#include <utility>

namespace cudavm {
#define PUBLIC_VALUE(expression, expected) ::cudavm::require(static_cast<uint64_t>(expression) == uint64_t(expected), #expression ": pinned public value")
void types_errors() {
    PUBLIC_VALUE(CUDA_SUCCESS, 0); PUBLIC_VALUE(CUDA_ERROR_INVALID_VALUE, 1); PUBLIC_VALUE(CUDA_ERROR_OUT_OF_MEMORY, 2);
    PUBLIC_VALUE(CUDA_ERROR_NOT_INITIALIZED, 3); PUBLIC_VALUE(CUDA_ERROR_INVALID_DEVICE, 101); PUBLIC_VALUE(CUDA_ERROR_INVALID_CONTEXT, 201);
    PUBLIC_VALUE(CUDA_ERROR_INVALID_HANDLE, 400); PUBLIC_VALUE(CUDA_ERROR_NOT_READY, 600); PUBLIC_VALUE(CUDA_ERROR_NOT_SUPPORTED, 801);
    PUBLIC_VALUE(cudaSuccess, 0); PUBLIC_VALUE(cudaErrorInvalidValue, 1); PUBLIC_VALUE(cudaErrorMemoryAllocation, 2);
    PUBLIC_VALUE(cudaErrorInitializationError, 3); PUBLIC_VALUE(cudaErrorInvalidDevice, 101); PUBLIC_VALUE(cudaErrorInvalidResourceHandle, 400);
    PUBLIC_VALUE(cudaErrorNotReady, 600); PUBLIC_VALUE(cudaErrorNotSupported, 801);
    PUBLIC_VALUE(NVRTC_SUCCESS, 0); PUBLIC_VALUE(NVRTC_ERROR_INVALID_INPUT, 3); PUBLIC_VALUE(NVRTC_ERROR_INVALID_PROGRAM, 4);
    PUBLIC_VALUE(NVRTC_ERROR_INVALID_OPTION, 5); PUBLIC_VALUE(NVRTC_ERROR_COMPILATION, 6); PUBLIC_VALUE(NVRTC_ERROR_CANCELLED, 16);
    PUBLIC_VALUE(NVVM_SUCCESS, 0); PUBLIC_VALUE(NVVM_ERROR_IR_VERSION_MISMATCH, 3); PUBLIC_VALUE(NVVM_ERROR_INVALID_INPUT, 4);
    PUBLIC_VALUE(NVVM_ERROR_INVALID_IR, 6); PUBLIC_VALUE(NVVM_ERROR_INVALID_OPTION, 7); PUBLIC_VALUE(NVVM_ERROR_COMPILATION, 9);
    PUBLIC_VALUE(NVPTXCOMPILE_SUCCESS, 0); PUBLIC_VALUE(NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE, 1); PUBLIC_VALUE(NVPTXCOMPILE_ERROR_INVALID_INPUT, 2);
    PUBLIC_VALUE(NVPTXCOMPILE_ERROR_CANCELLED, 9); PUBLIC_VALUE(NVPTXCOMPILE_PARSE_ONLY_SUCCESS, 10);
    PUBLIC_VALUE(NVJITLINK_SUCCESS, 0); PUBLIC_VALUE(NVJITLINK_ERROR_MISSING_ARCH, 2); PUBLIC_VALUE(NVJITLINK_ERROR_INVALID_INPUT, 3);
    PUBLIC_VALUE(NVJITLINK_ERROR_FINALIZE, 9); PUBLIC_VALUE(NVJITLINK_ERROR_ARCH_MISMATCH, 13); PUBLIC_VALUE(NVJITLINK_ERROR_LTO_NOT_ENABLED, 18);
    PUBLIC_VALUE(NVFATBIN_SUCCESS, 0); PUBLIC_VALUE(NVFATBIN_ERROR_NULL_POINTER, 5); PUBLIC_VALUE(NVFATBIN_ERROR_INVALID_ARCH, 9);
    require(sizeof(CUresult) == 4 && sizeof(cudaError_t) == 4 && sizeof(nvrtcResult) == 4 && sizeof(nvvmResult) == 4
        && sizeof(nvPTXCompileResult) == 4 && sizeof(nvJitLinkResult) == 4 && sizeof(nvFatbinResult) == 4, "public status enums have four-byte host ABI width");
}
void types_memory() {
    PUBLIC_VALUE(CU_MEMORYTYPE_HOST, 1); PUBLIC_VALUE(CU_MEMORYTYPE_DEVICE, 2); PUBLIC_VALUE(CU_MEMORYTYPE_ARRAY, 3); PUBLIC_VALUE(CU_MEMORYTYPE_UNIFIED, 4);
    PUBLIC_VALUE(cudaMemoryTypeUnregistered, 0); PUBLIC_VALUE(cudaMemoryTypeHost, 1); PUBLIC_VALUE(cudaMemoryTypeDevice, 2); PUBLIC_VALUE(cudaMemoryTypeManaged, 3);
    PUBLIC_VALUE(cudaMemcpyHostToHost, 0); PUBLIC_VALUE(cudaMemcpyHostToDevice, 1); PUBLIC_VALUE(cudaMemcpyDeviceToHost, 2);
    PUBLIC_VALUE(cudaMemcpyDeviceToDevice, 3); PUBLIC_VALUE(cudaMemcpyDefault, 4);
    PUBLIC_VALUE(CU_AD_FORMAT_UNSIGNED_INT8, 1); PUBLIC_VALUE(CU_AD_FORMAT_UNSIGNED_INT16, 2); PUBLIC_VALUE(CU_AD_FORMAT_UNSIGNED_INT32, 3);
    PUBLIC_VALUE(CU_AD_FORMAT_SIGNED_INT8, 8); PUBLIC_VALUE(CU_AD_FORMAT_SIGNED_INT16, 9); PUBLIC_VALUE(CU_AD_FORMAT_SIGNED_INT32, 10);
    PUBLIC_VALUE(CU_AD_FORMAT_HALF, 16); PUBLIC_VALUE(CU_AD_FORMAT_FLOAT, 32);
    PUBLIC_VALUE(cudaChannelFormatKindSigned, 0); PUBLIC_VALUE(cudaChannelFormatKindUnsigned, 1); PUBLIC_VALUE(cudaChannelFormatKindFloat, 2); PUBLIC_VALUE(cudaChannelFormatKindNone, 3);
    PUBLIC_VALUE(CU_MEMHOSTALLOC_PORTABLE, 1); PUBLIC_VALUE(CU_MEMHOSTALLOC_DEVICEMAP, 2); PUBLIC_VALUE(CU_MEMHOSTALLOC_WRITECOMBINED, 4);
    PUBLIC_VALUE(cudaHostAllocPortable, 1); PUBLIC_VALUE(cudaHostAllocMapped, 2); PUBLIC_VALUE(cudaHostAllocWriteCombined, 4);
    PUBLIC_VALUE(CU_MEMHOSTREGISTER_PORTABLE, 1); PUBLIC_VALUE(CU_MEMHOSTREGISTER_DEVICEMAP, 2); PUBLIC_VALUE(CU_MEMHOSTREGISTER_IOMEMORY, 4); PUBLIC_VALUE(CU_MEMHOSTREGISTER_READ_ONLY, 8);
    PUBLIC_VALUE(cudaHostRegisterPortable, 1); PUBLIC_VALUE(cudaHostRegisterMapped, 2); PUBLIC_VALUE(cudaHostRegisterIoMemory, 4); PUBLIC_VALUE(cudaHostRegisterReadOnly, 8);
}
void types_streams() {
    PUBLIC_VALUE(CU_STREAM_DEFAULT, 0); PUBLIC_VALUE(CU_STREAM_NON_BLOCKING, 1); PUBLIC_VALUE(cudaStreamDefault, 0); PUBLIC_VALUE(cudaStreamNonBlocking, 1);
    PUBLIC_VALUE(CU_EVENT_DEFAULT, 0); PUBLIC_VALUE(CU_EVENT_BLOCKING_SYNC, 1); PUBLIC_VALUE(CU_EVENT_DISABLE_TIMING, 2); PUBLIC_VALUE(CU_EVENT_INTERPROCESS, 4);
    PUBLIC_VALUE(cudaEventDefault, 0); PUBLIC_VALUE(cudaEventBlockingSync, 1); PUBLIC_VALUE(cudaEventDisableTiming, 2); PUBLIC_VALUE(cudaEventInterprocess, 4);
    PUBLIC_VALUE(CU_STREAM_CAPTURE_STATUS_NONE, 0); PUBLIC_VALUE(CU_STREAM_CAPTURE_STATUS_ACTIVE, 1); PUBLIC_VALUE(CU_STREAM_CAPTURE_STATUS_INVALIDATED, 2);
    PUBLIC_VALUE(cudaStreamCaptureStatusNone, 0); PUBLIC_VALUE(cudaStreamCaptureStatusActive, 1); PUBLIC_VALUE(cudaStreamCaptureStatusInvalidated, 2);
    PUBLIC_VALUE(CU_STREAM_CAPTURE_MODE_GLOBAL, 0); PUBLIC_VALUE(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, 1); PUBLIC_VALUE(CU_STREAM_CAPTURE_MODE_RELAXED, 2);
    PUBLIC_VALUE(cudaStreamCaptureModeGlobal, 0); PUBLIC_VALUE(cudaStreamCaptureModeThreadLocal, 1); PUBLIC_VALUE(cudaStreamCaptureModeRelaxed, 2);
    require(reinterpret_cast<uintptr_t>(CU_STREAM_LEGACY) == 1 && reinterpret_cast<uintptr_t>(CU_STREAM_PER_THREAD) == 2
        && reinterpret_cast<uintptr_t>(cudaStreamLegacy) == 1 && reinterpret_cast<uintptr_t>(cudaStreamPerThread) == 2, "public default-stream sentinels retain exact handle encodings");
    PUBLIC_VALUE(CU_GET_PROC_ADDRESS_DEFAULT, 0); PUBLIC_VALUE(CU_GET_PROC_ADDRESS_LEGACY_STREAM, 1); PUBLIC_VALUE(CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, 2);
    PUBLIC_VALUE(cudaEnableDefault, 0); PUBLIC_VALUE(cudaEnableLegacyStream, 1); PUBLIC_VALUE(cudaEnablePerThreadDefaultStream, 2);
    PUBLIC_VALUE(CU_GET_PROC_ADDRESS_SUCCESS, 0); PUBLIC_VALUE(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, 1); PUBLIC_VALUE(CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, 2);
    PUBLIC_VALUE(cudaDriverEntryPointSuccess, 0); PUBLIC_VALUE(cudaDriverEntryPointSymbolNotFound, 1); PUBLIC_VALUE(cudaDriverEntryPointVersionNotSufficent, 2);
}
void types_graphs() {
    PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_KERNEL, 0); PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_MEMCPY, 1); PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_MEMSET, 2);
    PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_HOST, 3); PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_GRAPH, 4); PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_EMPTY, 5);
    PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_WAIT_EVENT, 6); PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_EVENT_RECORD, 7); PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_MEM_ALLOC, 10);
    PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_MEM_FREE, 11); PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_BATCH_MEM_OP, 12); PUBLIC_VALUE(CU_GRAPH_NODE_TYPE_CONDITIONAL, 13);
    PUBLIC_VALUE(cudaGraphNodeTypeKernel, 0); PUBLIC_VALUE(cudaGraphNodeTypeMemcpy, 1); PUBLIC_VALUE(cudaGraphNodeTypeMemset, 2);
    PUBLIC_VALUE(cudaGraphNodeTypeHost, 3); PUBLIC_VALUE(cudaGraphNodeTypeGraph, 4); PUBLIC_VALUE(cudaGraphNodeTypeEmpty, 5);
    PUBLIC_VALUE(cudaGraphNodeTypeWaitEvent, 6); PUBLIC_VALUE(cudaGraphNodeTypeEventRecord, 7); PUBLIC_VALUE(cudaGraphNodeTypeMemAlloc, 10);
    PUBLIC_VALUE(cudaGraphNodeTypeMemFree, 11); PUBLIC_VALUE(cudaGraphNodeTypeConditional, 13);
    PUBLIC_VALUE(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH, 1); PUBLIC_VALUE(CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD, 2);
    PUBLIC_VALUE(CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH, 4); PUBLIC_VALUE(CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, 8);
    PUBLIC_VALUE(cudaGraphInstantiateFlagAutoFreeOnLaunch, 1); PUBLIC_VALUE(cudaGraphInstantiateFlagUpload, 2);
    PUBLIC_VALUE(cudaGraphInstantiateFlagDeviceLaunch, 4); PUBLIC_VALUE(cudaGraphInstantiateFlagUseNodePriority, 8);
}
void types_handles() {
    require(std::is_integral_v<CUdevice> && std::is_signed_v<CUdevice> && sizeof(CUdevice) == 4, "Driver device ordinal is signed four-byte integer");
    require(std::is_integral_v<CUdeviceptr> && std::is_unsigned_v<CUdeviceptr> && sizeof(CUdeviceptr) == 8, "Driver device pointer bits are unsigned eight-byte integer");
    require(std::is_pointer_v<CUcontext> && std::is_pointer_v<CUstream> && std::is_pointer_v<CUevent>
        && std::is_pointer_v<CUmodule> && std::is_pointer_v<CUlibrary> && std::is_pointer_v<CUgraph>, "public Driver ownership handles are opaque pointer types");
    require(std::is_same_v<CUstream, cudaStream_t> && std::is_same_v<CUevent, cudaEvent_t>
        && std::is_same_v<CUgraph, cudaGraph_t> && std::is_same_v<CUgraphExec, cudaGraphExec_t>, "shared Driver/Runtime stream/event/graph handle aliases retain exact types");
    require(std::is_integral_v<CUtexObject> && std::is_integral_v<CUsurfObject> && sizeof(CUtexObject) == 8 && sizeof(CUsurfObject) == 8
        && std::is_same_v<CUtexObject, cudaTextureObject_t> && std::is_same_v<CUsurfObject, cudaSurfaceObject_t>, "image object handles retain exact integer aliases and widths");
    require(std::is_integral_v<CUmemGenericAllocationHandle> && sizeof(CUmemGenericAllocationHandle) == 8
        && std::is_integral_v<CUgraphConditionalHandle> && sizeof(CUgraphConditionalHandle) == 8, "generic-memory and conditional handles are eight-byte integer tokens");
    require(std::is_pointer_v<nvrtcProgram> && std::is_pointer_v<nvvmProgram> && std::is_pointer_v<nvFatbinHandle>
        && std::is_pointer_v<nvPTXCompilerHandle> && std::is_pointer_v<nvJitLinkHandle>, "compiler facade handles retain opaque pointer categories");
}
std::string_view public_types_case = "all";
void execute_public_types() {
    require(CUDA_VERSION == 13040 && CUDART_VERSION == 13040, "pinned public type baseline");
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"errors", types_errors}, {"memory", types_memory}, {"streams", types_streams}, {"graphs", types_graphs}, {"handles", types_handles}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (public_types_case == "all" || public_types_case == name) { selected = true; execute(); }
    require(selected, "unknown public_types case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::public_types_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_public_types);
}
