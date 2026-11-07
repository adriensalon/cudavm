#include <cuda_fp16.h>
#define HALF_INPUTS \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const __half a = __ushort_as_half(static_cast<unsigned short>(input[i * 3])); \
    const __half b = __ushort_as_half(static_cast<unsigned short>(input[i * 3 + 1])); \
    const __half c = __ushort_as_half(static_cast<unsigned short>(input[i * 3 + 2]));
#if !defined(CUDAVM_HALF_PROFILE) || CUDAVM_HALF_PROFILE == 0
extern "C" __global__ void half_add_subtract(unsigned* output, const unsigned* input, unsigned count) {
    HALF_INPUTS
    output += i * 4;
    output[0] = __half_as_ushort(__hadd(a, b));
    output[1] = __half_as_ushort(__hsub(a, b));
    output[2] = __half_as_ushort(__hadd_rn(a, b));
    output[3] = __half_as_ushort(__hsub_rn(a, b));
}
#endif
#if !defined(CUDAVM_HALF_PROFILE) || CUDAVM_HALF_PROFILE == 1
extern "C" __global__ void half_multiply(unsigned* output, const unsigned* input, unsigned count) {
    HALF_INPUTS
    output[i * 2] = __half_as_ushort(__hmul(a, b));
    output[i * 2 + 1] = __half_as_ushort(__hmul_rn(a, b));
}
#endif
#if !defined(CUDAVM_HALF_PROFILE) || CUDAVM_HALF_PROFILE == 2
extern "C" __global__ void half_fused(unsigned* output, const unsigned* input, unsigned count) {
    HALF_INPUTS
    output[i * 2] = __half_as_ushort(__hfma(a, b, c));
    output[i * 2 + 1] = __half_as_ushort(__hadd_rn(__hmul_rn(a, b), c));
}
#endif
#if !defined(CUDAVM_HALF_PROFILE) || CUDAVM_HALF_PROFILE == 3
extern "C" __global__ void half_saturation(unsigned* output, const unsigned* input, unsigned count) {
    HALF_INPUTS
    output += i * 5;
    output[0] = __half_as_ushort(__hadd_sat(a, b));
    output[1] = __half_as_ushort(__hsub_sat(a, b));
    output[2] = __half_as_ushort(__hmul_sat(a, b));
    output[3] = __half_as_ushort(__hfma_sat(a, b, c));
    output[4] = __half_as_ushort(__hfma_relu(a, b, c));
}
#endif
#if !defined(CUDAVM_HALF_PROFILE) || CUDAVM_HALF_PROFILE == 4
extern "C" __global__ void half_operators(unsigned* output, const unsigned* input, unsigned count) {
    HALF_INPUTS
    output += i * 10;
    output[0] = __half_as_ushort(a + b);
    output[1] = __half_as_ushort(a - b);
    output[2] = __half_as_ushort(a * b);
    __half value = a; value += b; output[3] = __half_as_ushort(value);
    value = a; value -= b; output[4] = __half_as_ushort(value);
    value = a; value *= b; output[5] = __half_as_ushort(value);
    value = a; output[6] = __half_as_ushort(value++); output[7] = __half_as_ushort(value);
    value = a; output[8] = __half_as_ushort(value--); output[9] = __half_as_ushort(value);
}
#endif
#if !defined(CUDAVM_HALF_PROFILE) || CUDAVM_HALF_PROFILE == 5
extern "C" __global__ void half_divide(unsigned* output, const unsigned* input, unsigned count) {
    HALF_INPUTS
    output += i * 3;
    output[0] = __half_as_ushort(__hdiv(a, b));
    output[1] = __half_as_ushort(a / b);
    __half value = a; value /= b; output[2] = __half_as_ushort(value);
}
#endif
#undef HALF_INPUTS
