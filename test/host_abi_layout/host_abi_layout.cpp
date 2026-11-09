#include "test_framework.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace cudavm {
template<class T> void abi_shape(size_t size, size_t alignment, const char* name) {
    static_assert(std::is_standard_layout_v<T>);
    require(sizeof(T) == size && alignof(T) == alignment,
        std::string(name) + ": size/alignment " + std::to_string(sizeof(T)) + "/" + std::to_string(alignof(T))
        + ", expected " + std::to_string(size) + "/" + std::to_string(alignment));
}
#define ABI_SHAPE(type, size, alignment) ::cudavm::abi_shape<type>(size, alignment, #type)
#define ABI_FIELD(type, field, expected) ::cudavm::require(offsetof(type, field) == (expected), #type "." #field ": pinned member offset")
void abi_vectors() {
    ABI_SHAPE(CUdeviceptr, 8, 8); ABI_SHAPE(CUcontext, 8, 8); ABI_SHAPE(CUstream, 8, 8); ABI_SHAPE(CUevent, 8, 8);
    ABI_SHAPE(CUuuid, 16, 1); ABI_SHAPE(cudaUUID_t, 16, 1);
    ABI_SHAPE(char3, 3, 1); ABI_SHAPE(uchar4, 4, 4); ABI_SHAPE(short2, 4, 4); ABI_SHAPE(short3, 6, 2); ABI_SHAPE(short4, 8, 8);
    ABI_SHAPE(int2, 8, 8); ABI_SHAPE(int3, 12, 4); ABI_SHAPE(int4, 16, 16);
    ABI_SHAPE(uint3, 12, 4); ABI_SHAPE(float2, 8, 8); ABI_SHAPE(float3, 12, 4); ABI_SHAPE(float4, 16, 16);
    ABI_SHAPE(longlong2, 16, 16); ABI_SHAPE(double2, 16, 16); ABI_SHAPE(double3, 24, 8); ABI_SHAPE(double4, 32, 16);
    ABI_SHAPE(double4_16a, 32, 16); ABI_SHAPE(double4_32a, 32, 32);
    ABI_SHAPE(dim3, 12, 4);
    ABI_FIELD(dim3, x, 0); ABI_FIELD(dim3, y, 4); ABI_FIELD(dim3, z, 8);
    ABI_FIELD(float4, x, 0); ABI_FIELD(float4, y, 4); ABI_FIELD(float4, z, 8); ABI_FIELD(float4, w, 12);
    ABI_FIELD(double3, x, 0); ABI_FIELD(double3, y, 8); ABI_FIELD(double3, z, 16);
}
void abi_transfers() {
    ABI_SHAPE(CUDA_MEMCPY2D, 128, 8);
    ABI_FIELD(CUDA_MEMCPY2D, srcXInBytes, 0); ABI_FIELD(CUDA_MEMCPY2D, srcY, 8); ABI_FIELD(CUDA_MEMCPY2D, srcMemoryType, 16);
    ABI_FIELD(CUDA_MEMCPY2D, srcHost, 24); ABI_FIELD(CUDA_MEMCPY2D, srcDevice, 32); ABI_FIELD(CUDA_MEMCPY2D, srcArray, 40); ABI_FIELD(CUDA_MEMCPY2D, srcPitch, 48);
    ABI_FIELD(CUDA_MEMCPY2D, dstXInBytes, 56); ABI_FIELD(CUDA_MEMCPY2D, dstY, 64); ABI_FIELD(CUDA_MEMCPY2D, dstMemoryType, 72);
    ABI_FIELD(CUDA_MEMCPY2D, dstHost, 80); ABI_FIELD(CUDA_MEMCPY2D, dstDevice, 88); ABI_FIELD(CUDA_MEMCPY2D, dstArray, 96); ABI_FIELD(CUDA_MEMCPY2D, dstPitch, 104);
    ABI_FIELD(CUDA_MEMCPY2D, WidthInBytes, 112); ABI_FIELD(CUDA_MEMCPY2D, Height, 120);
    ABI_SHAPE(CUDA_MEMCPY3D, 200, 8);
    ABI_FIELD(CUDA_MEMCPY3D, srcXInBytes, 0); ABI_FIELD(CUDA_MEMCPY3D, srcY, 8); ABI_FIELD(CUDA_MEMCPY3D, srcZ, 16); ABI_FIELD(CUDA_MEMCPY3D, srcLOD, 24);
    ABI_FIELD(CUDA_MEMCPY3D, srcMemoryType, 32); ABI_FIELD(CUDA_MEMCPY3D, srcHost, 40); ABI_FIELD(CUDA_MEMCPY3D, srcDevice, 48); ABI_FIELD(CUDA_MEMCPY3D, srcArray, 56);
    ABI_FIELD(CUDA_MEMCPY3D, reserved0, 64); ABI_FIELD(CUDA_MEMCPY3D, srcPitch, 72); ABI_FIELD(CUDA_MEMCPY3D, srcHeight, 80);
    ABI_FIELD(CUDA_MEMCPY3D, dstXInBytes, 88); ABI_FIELD(CUDA_MEMCPY3D, dstY, 96); ABI_FIELD(CUDA_MEMCPY3D, dstZ, 104); ABI_FIELD(CUDA_MEMCPY3D, dstLOD, 112);
    ABI_FIELD(CUDA_MEMCPY3D, dstMemoryType, 120); ABI_FIELD(CUDA_MEMCPY3D, dstHost, 128); ABI_FIELD(CUDA_MEMCPY3D, dstDevice, 136); ABI_FIELD(CUDA_MEMCPY3D, dstArray, 144);
    ABI_FIELD(CUDA_MEMCPY3D, reserved1, 152); ABI_FIELD(CUDA_MEMCPY3D, dstPitch, 160); ABI_FIELD(CUDA_MEMCPY3D, dstHeight, 168);
    ABI_FIELD(CUDA_MEMCPY3D, WidthInBytes, 176); ABI_FIELD(CUDA_MEMCPY3D, Height, 184); ABI_FIELD(CUDA_MEMCPY3D, Depth, 192);
    ABI_SHAPE(cudaMemcpy3DParms, 160, 8);
    ABI_FIELD(cudaMemcpy3DParms, srcArray, 0); ABI_FIELD(cudaMemcpy3DParms, srcPos, 8); ABI_FIELD(cudaMemcpy3DParms, srcPtr, 32);
    ABI_FIELD(cudaMemcpy3DParms, dstArray, 64); ABI_FIELD(cudaMemcpy3DParms, dstPos, 72); ABI_FIELD(cudaMemcpy3DParms, dstPtr, 96);
    ABI_FIELD(cudaMemcpy3DParms, extent, 128); ABI_FIELD(cudaMemcpy3DParms, kind, 152);
    static_assert(std::is_same_v<CUDA_MEMCPY2D, CUDA_MEMCPY2D_v2> && std::is_same_v<CUDA_MEMCPY3D, CUDA_MEMCPY3D_v2>);
}
void abi_arrays() {
    ABI_SHAPE(CUDA_ARRAY_DESCRIPTOR, 24, 8);
    ABI_FIELD(CUDA_ARRAY_DESCRIPTOR, Width, 0); ABI_FIELD(CUDA_ARRAY_DESCRIPTOR, Height, 8); ABI_FIELD(CUDA_ARRAY_DESCRIPTOR, Format, 16); ABI_FIELD(CUDA_ARRAY_DESCRIPTOR, NumChannels, 20);
    ABI_SHAPE(CUDA_ARRAY3D_DESCRIPTOR, 40, 8);
    ABI_FIELD(CUDA_ARRAY3D_DESCRIPTOR, Width, 0); ABI_FIELD(CUDA_ARRAY3D_DESCRIPTOR, Height, 8); ABI_FIELD(CUDA_ARRAY3D_DESCRIPTOR, Depth, 16);
    ABI_FIELD(CUDA_ARRAY3D_DESCRIPTOR, Format, 24); ABI_FIELD(CUDA_ARRAY3D_DESCRIPTOR, NumChannels, 28); ABI_FIELD(CUDA_ARRAY3D_DESCRIPTOR, Flags, 32);
    ABI_SHAPE(cudaChannelFormatDesc, 20, 4);
    ABI_FIELD(cudaChannelFormatDesc, x, 0); ABI_FIELD(cudaChannelFormatDesc, y, 4); ABI_FIELD(cudaChannelFormatDesc, z, 8); ABI_FIELD(cudaChannelFormatDesc, w, 12); ABI_FIELD(cudaChannelFormatDesc, f, 16);
    ABI_SHAPE(cudaExtent, 24, 8); ABI_FIELD(cudaExtent, width, 0); ABI_FIELD(cudaExtent, height, 8); ABI_FIELD(cudaExtent, depth, 16);
    ABI_SHAPE(cudaPos, 24, 8); ABI_FIELD(cudaPos, x, 0); ABI_FIELD(cudaPos, y, 8); ABI_FIELD(cudaPos, z, 16);
    ABI_SHAPE(cudaPitchedPtr, 32, 8);
    ABI_FIELD(cudaPitchedPtr, ptr, 0); ABI_FIELD(cudaPitchedPtr, pitch, 8); ABI_FIELD(cudaPitchedPtr, xsize, 16); ABI_FIELD(cudaPitchedPtr, ysize, 24);
}
void abi_graphs() {
    ABI_SHAPE(CUDA_KERNEL_NODE_PARAMS_v1, 56, 8); ABI_SHAPE(CUDA_KERNEL_NODE_PARAMS_v2, 72, 8); ABI_SHAPE(CUDA_KERNEL_NODE_PARAMS_v3, 72, 8);
    ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v1, func, 0); ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v1, gridDimX, 8); ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v1, blockDimX, 20);
    ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v1, sharedMemBytes, 32); ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v1, kernelParams, 40); ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v1, extra, 48);
    ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v2, kern, 56); ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v2, ctx, 64);
    ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v3, kern, 56); ABI_FIELD(CUDA_KERNEL_NODE_PARAMS_v3, ctx, 64);
    ABI_SHAPE(cudaKernelNodeParams, 56, 8); ABI_SHAPE(cudaKernelNodeParamsV2, 72, 8);
    ABI_FIELD(cudaKernelNodeParams, func, 0); ABI_FIELD(cudaKernelNodeParams, gridDim, 8); ABI_FIELD(cudaKernelNodeParams, blockDim, 20);
    ABI_FIELD(cudaKernelNodeParams, sharedMemBytes, 32); ABI_FIELD(cudaKernelNodeParams, kernelParams, 40); ABI_FIELD(cudaKernelNodeParams, extra, 48);
    ABI_FIELD(cudaKernelNodeParamsV2, ctx, 56); ABI_FIELD(cudaKernelNodeParamsV2, functionType, 64);
    ABI_SHAPE(CUDA_MEMSET_NODE_PARAMS_v1, 40, 8); ABI_SHAPE(CUDA_MEMSET_NODE_PARAMS_v2, 48, 8);
    ABI_FIELD(CUDA_MEMSET_NODE_PARAMS_v1, dst, 0); ABI_FIELD(CUDA_MEMSET_NODE_PARAMS_v1, pitch, 8); ABI_FIELD(CUDA_MEMSET_NODE_PARAMS_v1, value, 16);
    ABI_FIELD(CUDA_MEMSET_NODE_PARAMS_v1, elementSize, 20); ABI_FIELD(CUDA_MEMSET_NODE_PARAMS_v1, width, 24); ABI_FIELD(CUDA_MEMSET_NODE_PARAMS_v1, height, 32);
    ABI_FIELD(CUDA_MEMSET_NODE_PARAMS_v2, ctx, 40);
    ABI_SHAPE(cudaMemsetParams, 40, 8); ABI_SHAPE(cudaMemsetParamsV2, 48, 8);
    ABI_FIELD(cudaMemsetParams, dst, 0); ABI_FIELD(cudaMemsetParams, pitch, 8); ABI_FIELD(cudaMemsetParams, value, 16);
    ABI_FIELD(cudaMemsetParams, elementSize, 20); ABI_FIELD(cudaMemsetParams, width, 24); ABI_FIELD(cudaMemsetParams, height, 32); ABI_FIELD(cudaMemsetParamsV2, ctx, 40);
    static_assert(std::is_same_v<CUDA_KERNEL_NODE_PARAMS, CUDA_KERNEL_NODE_PARAMS_v2>
        && std::is_same_v<CUDA_MEMSET_NODE_PARAMS, CUDA_MEMSET_NODE_PARAMS_v1>);
}
void abi_unions() {
    ABI_SHAPE(CUDA_HOST_NODE_PARAMS_v1, 16, 8); ABI_SHAPE(CUDA_HOST_NODE_PARAMS_v2, 32, 8);
    ABI_FIELD(CUDA_HOST_NODE_PARAMS_v1, fn, 0); ABI_FIELD(CUDA_HOST_NODE_PARAMS_v1, userData, 8);
    ABI_FIELD(CUDA_HOST_NODE_PARAMS_v2, fn, 0); ABI_FIELD(CUDA_HOST_NODE_PARAMS_v2, userData, 8); ABI_FIELD(CUDA_HOST_NODE_PARAMS_v2, syncMode, 16);
    ABI_FIELD(CUDA_HOST_NODE_PARAMS_v2, ctx, 24); ABI_FIELD(CUDA_HOST_NODE_PARAMS_v2, gCtx, 24);
    ABI_SHAPE(cudaHostNodeParams, 16, 8); ABI_SHAPE(cudaHostNodeParamsV2, 32, 8);
    ABI_FIELD(cudaHostNodeParamsV2, fn, 0); ABI_FIELD(cudaHostNodeParamsV2, userData, 8); ABI_FIELD(cudaHostNodeParamsV2, syncMode, 16); ABI_FIELD(cudaHostNodeParamsV2, ctx, 24);
    ABI_FIELD(cudaKernelNodeParamsV2, func, 0); ABI_FIELD(cudaKernelNodeParamsV2, kern, 0); ABI_FIELD(cudaKernelNodeParamsV2, cuFunc, 0);
    // The pinned reserved longs intentionally distinguish LLP64 and LP64.
    require(sizeof(long) == 4 || sizeof(long) == 8, "supported 64-bit host long width");
    ABI_SHAPE(cudaPointerAttributes, sizeof(long) == 4 ? 56 : 88, 8);
    ABI_FIELD(cudaPointerAttributes, type, 0); ABI_FIELD(cudaPointerAttributes, device, 4);
    ABI_FIELD(cudaPointerAttributes, devicePointer, 8); ABI_FIELD(cudaPointerAttributes, hostPointer, 16);
    ABI_FIELD(cudaPointerAttributes, localityDomainOrdinal, 24); ABI_FIELD(cudaPointerAttributes, unused, 24);
    ABI_FIELD(cudaPointerAttributes, reserved, sizeof(long) == 4 ? 28 : 32);
}
std::string_view host_abi_layout_case = "all";
void execute_host_abi_layout() {
    require(CUDA_VERSION == 13040 && CUDART_VERSION == 13040, "pinned CUDA 13.4 host layout profile");
    require(sizeof(void*) == 8 && sizeof(size_t) == 8 && sizeof(int) == 4, "64-bit host ABI profile");
    const std::array<std::pair<std::string_view, void (*)()>, 5> cases{{
        {"vectors", abi_vectors}, {"transfers", abi_transfers}, {"arrays", abi_arrays}, {"graphs", abi_graphs}, {"unions", abi_unions}}};
    bool selected = false;
    for (const auto& [name, execute] : cases) if (host_abi_layout_case == "all" || host_abi_layout_case == name) { selected = true; execute(); }
    require(selected, "unknown host_abi_layout case");
}
}
int main(int argc, char** argv) {
    if (argc == 2) cudavm::host_abi_layout_case = argv[1];
    return cudavm::run_host_conformance(argc, argv, cudavm::execute_host_abi_layout);
}
