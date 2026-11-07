#include <cuda_fp16.h>
#define HALF_MATH_KERNEL(NAME, FUNCTION) \
extern "C" __global__ void half_math_##NAME(unsigned* output, const unsigned* input, unsigned count) { \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const __half value = __ushort_as_half(static_cast<unsigned short>(input[i * 3])); \
    output[i] = __half_as_ushort(FUNCTION(value)); \
}
#if !defined(CUDAVM_HALF_MATH_PROFILE) || CUDAVM_HALF_MATH_PROFILE == 0
extern "C" __global__ void half_math_rounding(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const __half value = __ushort_as_half(static_cast<unsigned short>(input[i * 3]));
    output += i * 4;
    output[0] = __half_as_ushort(hfloor(value));
    output[1] = __half_as_ushort(hceil(value));
    output[2] = __half_as_ushort(htrunc(value));
    output[3] = __half_as_ushort(hrint(value));
}
#endif
#if !defined(CUDAVM_HALF_MATH_PROFILE) || CUDAVM_HALF_MATH_PROFILE == 1
HALF_MATH_KERNEL(sqrt, hsqrt)
#endif
#if !defined(CUDAVM_HALF_MATH_PROFILE) || CUDAVM_HALF_MATH_PROFILE == 2
HALF_MATH_KERNEL(reciprocal, hrcp)
#endif
#if !defined(CUDAVM_HALF_MATH_PROFILE) || CUDAVM_HALF_MATH_PROFILE == 3
HALF_MATH_KERNEL(rsqrt, hrsqrt)
#endif
#undef HALF_MATH_KERNEL
