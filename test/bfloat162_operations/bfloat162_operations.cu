#include <cuda_bf16.h>
static_assert(__CUDA_ARCH__ >= 800, "packed bfloat16 fixture requires SM80");
namespace cudavm {
__device__ inline __nv_bfloat162 bfloat162_slice_value(unsigned bits) {
    return __halves2bfloat162(__ushort_as_bfloat16(static_cast<unsigned short>(bits)), __ushort_as_bfloat16(static_cast<unsigned short>(bits >> 16)));
}
__device__ inline unsigned bfloat162_slice_bits(__nv_bfloat162 value) {
    return unsigned(__bfloat16_as_ushort(__low2bfloat16(value))) | (unsigned(__bfloat16_as_ushort(__high2bfloat16(value))) << 16);
}
}
#define BFLOAT162_INPUTS \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const __nv_bfloat162 a = cudavm::bfloat162_slice_value(input[i * 3]); \
    const __nv_bfloat162 b = cudavm::bfloat162_slice_value(input[i * 3 + 1]); \
    const __nv_bfloat162 c = cudavm::bfloat162_slice_value(input[i * 3 + 2]);
#define BFLOAT162_BITS(VALUE) cudavm::bfloat162_slice_bits(VALUE)
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 0
extern "C" __global__ void bfloat162_representation(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    output += i * 9;
    output[0] = BFLOAT162_BITS(a);
    output[1] = BFLOAT162_BITS(__lowhigh2highlow(a));
    output[2] = BFLOAT162_BITS(__lows2bfloat162(a, b));
    output[3] = BFLOAT162_BITS(__highs2bfloat162(a, b));
    output[4] = BFLOAT162_BITS(__low2bfloat162(a));
    output[5] = BFLOAT162_BITS(__high2bfloat162(a));
    output[6] = BFLOAT162_BITS(__bfloat162bfloat162(__low2bfloat16(a)));
    const __nv_bfloat162_raw raw = a;
    output[7] = unsigned(raw.x) | (unsigned(raw.y) << 16);
    const __nv_bfloat162_raw mixed{static_cast<unsigned short>(input[i * 3 + 1]), static_cast<unsigned short>(input[i * 3] >> 16)};
    output[8] = BFLOAT162_BITS(__nv_bfloat162(mixed));
}
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 1
extern "C" __global__ void bfloat162_widen(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    const float2 value = __bfloat1622float2(a);
    output += i * 4;
    output[0] = __float_as_uint(value.x); output[1] = __float_as_uint(value.y);
    output[2] = __float_as_uint(__low2float(a)); output[3] = __float_as_uint(__high2float(a));
}
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 2
extern "C" __global__ void bfloat162_narrow(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float a = __uint_as_float(input[i * 3]), b = __uint_as_float(input[i * 3 + 1]);
    output += i * 4;
    output[0] = BFLOAT162_BITS(__floats2bfloat162_rn(a, b));
    output[1] = BFLOAT162_BITS(__float22bfloat162_rn(make_float2(a, b)));
    output[2] = BFLOAT162_BITS(__float2bfloat162_rn(a)); output[3] = BFLOAT162_BITS(__float2bfloat162_rn(b));
}
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 3
extern "C" __global__ void bfloat162_arithmetic(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    output += i * 6;
    output[0] = BFLOAT162_BITS(__hadd2(a, b)); output[1] = BFLOAT162_BITS(__hsub2(a, b)); output[2] = BFLOAT162_BITS(__hmul2(a, b));
    output[3] = BFLOAT162_BITS(__hadd2_rn(a, b)); output[4] = BFLOAT162_BITS(__hsub2_rn(a, b)); output[5] = BFLOAT162_BITS(__hmul2_rn(a, b));
}
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 4
extern "C" __global__ void bfloat162_fused(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    output[i * 2] = BFLOAT162_BITS(__hfma2(a, b, c));
    output[i * 2 + 1] = BFLOAT162_BITS(__hadd2_rn(__hmul2_rn(a, b), c));
}
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 5
extern "C" __global__ void bfloat162_saturation(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    output += i * 5;
    output[0] = BFLOAT162_BITS(__hadd2_sat(a, b)); output[1] = BFLOAT162_BITS(__hsub2_sat(a, b)); output[2] = BFLOAT162_BITS(__hmul2_sat(a, b));
    output[3] = BFLOAT162_BITS(__hfma2_sat(a, b, c)); output[4] = BFLOAT162_BITS(__hfma2_relu(a, b, c));
}
#endif
#define BFLOAT162_DIVIDE_KERNEL(NAME, EXPRESSION) \
extern "C" __global__ void bfloat162_##NAME(unsigned* output, const unsigned* input, unsigned count) { \
    BFLOAT162_INPUTS \
    __nv_bfloat162 value = a; \
    output[i] = BFLOAT162_BITS(EXPRESSION); \
}
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 6
BFLOAT162_DIVIDE_KERNEL(divide, __h2div(a, b))
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 10
BFLOAT162_DIVIDE_KERNEL(divide_operator, a / b)
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 11
BFLOAT162_DIVIDE_KERNEL(divide_assign, value /= b)
#endif
#undef BFLOAT162_DIVIDE_KERNEL
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 7
extern "C" __global__ void bfloat162_operators(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    output += i * 6;
    output[0] = BFLOAT162_BITS(a + b); output[1] = BFLOAT162_BITS(a - b); output[2] = BFLOAT162_BITS(a * b);
    __nv_bfloat162 value = a; value += b; output[3] = BFLOAT162_BITS(value);
    value = a; value -= b; output[4] = BFLOAT162_BITS(value);
    value = a; value *= b; output[5] = BFLOAT162_BITS(value);
}
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 8
extern "C" __global__ void bfloat162_comparisons(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    output += i * 12;
    output[0] = BFLOAT162_BITS(__heq2(a, b)); output[1] = BFLOAT162_BITS(__hne2(a, b));
    output[2] = BFLOAT162_BITS(__hlt2(a, b)); output[3] = BFLOAT162_BITS(__hle2(a, b));
    output[4] = BFLOAT162_BITS(__hgt2(a, b)); output[5] = BFLOAT162_BITS(__hge2(a, b));
    output[6] = BFLOAT162_BITS(__hequ2(a, b)); output[7] = BFLOAT162_BITS(__hneu2(a, b));
    output[8] = BFLOAT162_BITS(__hltu2(a, b)); output[9] = BFLOAT162_BITS(__hleu2(a, b));
    output[10] = BFLOAT162_BITS(__hgtu2(a, b)); output[11] = BFLOAT162_BITS(__hgeu2(a, b));
}
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 9
extern "C" __global__ void bfloat162_predicates(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    output += i * 12;
    output[0] = __hbeq2(a, b); output[1] = __hbne2(a, b); output[2] = __hblt2(a, b); output[3] = __hble2(a, b);
    output[4] = __hbgt2(a, b); output[5] = __hbge2(a, b); output[6] = __hbequ2(a, b); output[7] = __hbneu2(a, b);
    output[8] = __hbltu2(a, b); output[9] = __hbleu2(a, b); output[10] = __hbgtu2(a, b); output[11] = __hbgeu2(a, b);
}
#endif
#if !defined(CUDAVM_BFLOAT162_PROFILE) || CUDAVM_BFLOAT162_PROFILE == 12
extern "C" __global__ void bfloat162_masks(unsigned* output, const unsigned* input, unsigned count) {
    BFLOAT162_INPUTS
    output += i * 12;
    output[0] = __heq2_mask(a, b); output[1] = __hne2_mask(a, b);
    output[2] = __hlt2_mask(a, b); output[3] = __hle2_mask(a, b);
    output[4] = __hgt2_mask(a, b); output[5] = __hge2_mask(a, b);
    output[6] = __hequ2_mask(a, b); output[7] = __hneu2_mask(a, b);
    output[8] = __hltu2_mask(a, b); output[9] = __hleu2_mask(a, b);
    output[10] = __hgtu2_mask(a, b); output[11] = __hgeu2_mask(a, b);
}
#endif
#undef BFLOAT162_INPUTS
#undef BFLOAT162_BITS

