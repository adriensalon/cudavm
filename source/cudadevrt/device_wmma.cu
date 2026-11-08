// NVRTC definitions for the pinned public m16n16k16 half/float WMMA declarations.
// Fragment element mapping is private to these operations. Half operand storage
// duplicates its eight logical elements to retain the pinned sixteen-element ABI.
#pragma once
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
// WMMA accumulation has its own precision contract, including unspecified
// rounding and subnormal handling. Keep this operation separate from scalar
// explicit-rounding intrinsics and their exact numerical fallback helpers.
extern "C" __device__ __attribute__((noinline)) float __cudavm_rtc_wmma_fma(float a, float b, float c) {
    return __builtin_elementwise_fma(a, b, c);
}
namespace cudavm {
__device__ __forceinline__ unsigned wmma_lane() {
    return (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) & 31u;
}
template<class Fragment> __device__ __forceinline__ void wmma_load_half(Fragment& fragment, const __half* pointer, unsigned stride, bool column) {
    __syncwarp();
    #pragma unroll 1
    for (unsigned i = 0; i < 16; ++i) {
        const unsigned index = wmma_lane() * 8 + i % 8, row = index / 16, col = index % 16;
        fragment.x[i] = pointer[column ? col * stride + row : row * stride + col];
    }
    __syncwarp();
}
__device__ __forceinline__ void wmma_load_float(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& fragment,
    const float* pointer, unsigned stride, bool column) {
    __syncwarp();
    #pragma unroll 1
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned index = wmma_lane() * 8 + i, row = index / 16, col = index % 16;
        fragment.x[i] = pointer[column ? col * stride + row : row * stride + col];
    }
    __syncwarp();
}
__device__ __forceinline__ void wmma_store_float(float* pointer,
    const nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& fragment, unsigned stride, bool column) {
    __syncwarp();
    #pragma unroll 1
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned index = wmma_lane() * 8 + i, row = index / 16, col = index % 16;
        pointer[column ? col * stride + row : row * stride + col] = fragment.x[i];
    }
    __syncwarp();
}
template<class A, class B> __device__ __forceinline__ void wmma_multiply(
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& destination,
    const A& a, const B& b, const nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& initial, bool saturate) {
    const unsigned lane = wmma_lane();
    float values[8];
    #pragma unroll 1
    for (unsigned i = 0; i < 8; ++i) values[i] = initial.x[i];
    #pragma unroll 1
    for (unsigned k = 0; k < 16; ++k) {
        const float left = __shfl_sync(0xffffffffu, __half2float(a.x[k % 8]), (lane / 2) * 2 + k / 8);
        #pragma unroll 1
        for (unsigned i = 0; i < 8; ++i) {
            const float right = __shfl_sync(0xffffffffu, __half2float(b.x[i]), k * 2 + lane % 2);
            values[i] = __cudavm_rtc_wmma_fma(left, right, values[i]);
        }
    }
    #pragma unroll 1
    for (unsigned i = 0; i < 8; ++i) {
        float value = values[i];
        if (saturate) {
            const unsigned bits = __float_as_uint(value), magnitude = bits & 0x7fffffffu;
            if (magnitude > 0x7f800000u) value = 0.0f;
            else if (magnitude == 0x7f800000u) value = __uint_as_float((bits & 0x80000000u) | 0x7f7fffffu);
        }
        destination.x[i] = value;
    }
}
}
namespace nvcuda { namespace wmma {
#define CUDAVM_WMMA_LOAD(use, layout, column) \
static __device__ __inline__ void load_matrix_sync(fragment<use, 16, 16, 16, __half, layout>& value, const __half* pointer, unsigned stride) { \
    cudavm::wmma_load_half(value, pointer, stride, column); }
CUDAVM_WMMA_LOAD(matrix_a, row_major, false)
CUDAVM_WMMA_LOAD(matrix_a, col_major, true)
CUDAVM_WMMA_LOAD(matrix_b, row_major, false)
CUDAVM_WMMA_LOAD(matrix_b, col_major, true)
#undef CUDAVM_WMMA_LOAD
static __device__ __inline__ void load_matrix_sync(fragment<accumulator, 16, 16, 16, float>& value, const float* pointer, unsigned stride, layout_t layout) {
    cudavm::wmma_load_float(value, pointer, stride, layout == mem_col_major);
}
static __device__ __inline__ void store_matrix_sync(float* pointer, const fragment<accumulator, 16, 16, 16, float>& value, unsigned stride, layout_t layout) {
    cudavm::wmma_store_float(pointer, value, stride, layout == mem_col_major);
}
#define CUDAVM_WMMA_MULTIPLY(a_layout, b_layout) \
static __device__ __inline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& destination, \
    const fragment<matrix_a, 16, 16, 16, __half, a_layout>& a, const fragment<matrix_b, 16, 16, 16, __half, b_layout>& b, \
    const fragment<accumulator, 16, 16, 16, float>& initial, bool saturate) { cudavm::wmma_multiply(destination, a, b, initial, saturate); }
CUDAVM_WMMA_MULTIPLY(row_major, row_major)
CUDAVM_WMMA_MULTIPLY(row_major, col_major)
CUDAVM_WMMA_MULTIPLY(col_major, row_major)
CUDAVM_WMMA_MULTIPLY(col_major, col_major)
#undef CUDAVM_WMMA_MULTIPLY
} }
#endif
