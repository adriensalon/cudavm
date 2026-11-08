#include <cuda_bf16.h>
#define BFLOAT16_MATH_KERNEL(NAME, FUNCTION) \
extern "C" __global__ void bfloat16_math_##NAME(unsigned* output, const unsigned* input, unsigned count) { \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const __nv_bfloat16 value = __ushort_as_bfloat16(static_cast<unsigned short>(input[i * 3])); \
    output[i] = __bfloat16_as_ushort(FUNCTION(value)); \
}
#if !defined(CUDAVM_BFLOAT16_MATH_PROFILE) || CUDAVM_BFLOAT16_MATH_PROFILE == 0
extern "C" __global__ void bfloat16_math_rounding(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const __nv_bfloat16 value = __ushort_as_bfloat16(static_cast<unsigned short>(input[i * 3]));
    output += i * 4;
    output[0] = __bfloat16_as_ushort(hfloor(value));
    output[1] = __bfloat16_as_ushort(hceil(value));
    output[2] = __bfloat16_as_ushort(htrunc(value));
    output[3] = __bfloat16_as_ushort(hrint(value));
}
#endif
#if !defined(CUDAVM_BFLOAT16_MATH_PROFILE) || CUDAVM_BFLOAT16_MATH_PROFILE == 1
BFLOAT16_MATH_KERNEL(sqrt, hsqrt)
#endif
#if !defined(CUDAVM_BFLOAT16_MATH_PROFILE) || CUDAVM_BFLOAT16_MATH_PROFILE == 2
BFLOAT16_MATH_KERNEL(reciprocal, hrcp)
#endif
#if !defined(CUDAVM_BFLOAT16_MATH_PROFILE) || CUDAVM_BFLOAT16_MATH_PROFILE == 3
BFLOAT16_MATH_KERNEL(rsqrt, hrsqrt)
#endif
#undef BFLOAT16_MATH_KERNEL

