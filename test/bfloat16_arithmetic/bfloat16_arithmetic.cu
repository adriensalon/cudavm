#include <cuda_bf16.h>
static_assert(__CUDA_ARCH__ >= 800, "bfloat16 arithmetic fixture requires SM80");
#define BFLOAT16_INPUTS \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const __nv_bfloat16 a = __ushort_as_bfloat16(static_cast<unsigned short>(input[i * 3])); \
    const __nv_bfloat16 b = __ushort_as_bfloat16(static_cast<unsigned short>(input[i * 3 + 1])); \
    const __nv_bfloat16 c = __ushort_as_bfloat16(static_cast<unsigned short>(input[i * 3 + 2]));
#if !defined(CUDAVM_BFLOAT16_PROFILE) || CUDAVM_BFLOAT16_PROFILE == 0
extern "C" __global__ void bfloat16_add_subtract(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT16_INPUTS
    output += i * 4;
    output[0] = __bfloat16_as_ushort(__hadd(a, b));
    output[1] = __bfloat16_as_ushort(__hsub(a, b));
    output[2] = __bfloat16_as_ushort(__hadd_rn(a, b));
    output[3] = __bfloat16_as_ushort(__hsub_rn(a, b));
}
#endif
#if !defined(CUDAVM_BFLOAT16_PROFILE) || CUDAVM_BFLOAT16_PROFILE == 1
extern "C" __global__ void bfloat16_multiply(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT16_INPUTS
    output[i * 2] = __bfloat16_as_ushort(__hmul(a, b));
    output[i * 2 + 1] = __bfloat16_as_ushort(__hmul_rn(a, b));
}
#endif
#if !defined(CUDAVM_BFLOAT16_PROFILE) || CUDAVM_BFLOAT16_PROFILE == 2
extern "C" __global__ void bfloat16_fused(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT16_INPUTS
    output[i * 2] = __bfloat16_as_ushort(__hfma(a, b, c));
    output[i * 2 + 1] = __bfloat16_as_ushort(__hadd_rn(__hmul_rn(a, b), c));
}
#endif
#if !defined(CUDAVM_BFLOAT16_PROFILE) || CUDAVM_BFLOAT16_PROFILE == 3
extern "C" __global__ void bfloat16_saturation(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT16_INPUTS
    output += i * 5;
    output[0] = __bfloat16_as_ushort(__hadd_sat(a, b));
    output[1] = __bfloat16_as_ushort(__hsub_sat(a, b));
    output[2] = __bfloat16_as_ushort(__hmul_sat(a, b));
    output[3] = __bfloat16_as_ushort(__hfma_sat(a, b, c));
    output[4] = __bfloat16_as_ushort(__hfma_relu(a, b, c));
}
#endif
#if !defined(CUDAVM_BFLOAT16_PROFILE) || CUDAVM_BFLOAT16_PROFILE == 4
extern "C" __global__ void bfloat16_operators(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT16_INPUTS
    output += i * 10;
    output[0] = __bfloat16_as_ushort(a + b);
    output[1] = __bfloat16_as_ushort(a - b);
    output[2] = __bfloat16_as_ushort(a * b);
    __nv_bfloat16 value = a; value += b; output[3] = __bfloat16_as_ushort(value);
    value = a; value -= b; output[4] = __bfloat16_as_ushort(value);
    value = a; value *= b; output[5] = __bfloat16_as_ushort(value);
    value = a; output[6] = __bfloat16_as_ushort(value++); output[7] = __bfloat16_as_ushort(value);
    value = a; output[8] = __bfloat16_as_ushort(value--); output[9] = __bfloat16_as_ushort(value);
}
#endif
// Compile each division form independently to bound shader optimization cost.
#define BFLOAT16_DIVIDE_KERNEL(NAME, EXPRESSION) \
extern "C" __global__ void bfloat16_##NAME(unsigned* output, const unsigned* input, unsigned count) { \
    BFLOAT16_INPUTS \
    __nv_bfloat16 value = a; \
    output[i] = __bfloat16_as_ushort(EXPRESSION); \
}
#if !defined(CUDAVM_BFLOAT16_PROFILE) || CUDAVM_BFLOAT16_PROFILE == 5
BFLOAT16_DIVIDE_KERNEL(divide, __hdiv(a, b))
#endif
#if !defined(CUDAVM_BFLOAT16_PROFILE) || CUDAVM_BFLOAT16_PROFILE == 6
BFLOAT16_DIVIDE_KERNEL(divide_operator, a / b)
#endif
#if !defined(CUDAVM_BFLOAT16_PROFILE) || CUDAVM_BFLOAT16_PROFILE == 7
BFLOAT16_DIVIDE_KERNEL(divide_assign, value /= b)
#endif
#undef BFLOAT16_DIVIDE_KERNEL
#undef BFLOAT16_INPUTS
