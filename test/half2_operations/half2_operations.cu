#include <cuda_fp16.h>
namespace cudavm {
__device__ inline __half2 half2_slice_value(unsigned bits) {
    return __halves2half2(__ushort_as_half(static_cast<unsigned short>(bits)), __ushort_as_half(static_cast<unsigned short>(bits >> 16)));
}
__device__ inline unsigned half2_slice_bits(__half2 value) {
    return unsigned(__half_as_ushort(__low2half(value))) | (unsigned(__half_as_ushort(__high2half(value))) << 16);
}
}
#define HALF2_INPUTS \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const __half2 a = cudavm::half2_slice_value(input[i * 3]); \
    const __half2 b = cudavm::half2_slice_value(input[i * 3 + 1]); \
    const __half2 c = cudavm::half2_slice_value(input[i * 3 + 2]);
#define HALF2_BITS(VALUE) cudavm::half2_slice_bits(VALUE)
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 0
extern "C" __global__ void half2_representation(unsigned* output, const unsigned* input, unsigned count) {
    HALF2_INPUTS
    output += i * 9;
    output[0] = HALF2_BITS(a);
    output[1] = HALF2_BITS(__lowhigh2highlow(a));
    output[2] = HALF2_BITS(__lows2half2(a, b));
    output[3] = HALF2_BITS(__highs2half2(a, b));
    output[4] = HALF2_BITS(__low2half2(a));
    output[5] = HALF2_BITS(__high2half2(a));
    output[6] = HALF2_BITS(__half2half2(__low2half(a)));
    const __half2_raw raw = a;
    output[7] = unsigned(raw.x) | (unsigned(raw.y) << 16);
    const __half2_raw mixed{static_cast<unsigned short>(input[i * 3 + 1]), static_cast<unsigned short>(input[i * 3] >> 16)};
    output[8] = HALF2_BITS(__half2(mixed));
}
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 1
extern "C" __global__ void half2_widen(unsigned* output, const unsigned* input, unsigned count) {
    HALF2_INPUTS
    const float2 value = __half22float2(a);
    output += i * 4;
    output[0] = __float_as_uint(value.x); output[1] = __float_as_uint(value.y);
    output[2] = __float_as_uint(__low2float(a)); output[3] = __float_as_uint(__high2float(a));
}
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 2
extern "C" __global__ void half2_narrow(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float a = __uint_as_float(input[i * 3]), b = __uint_as_float(input[i * 3 + 1]);
    output += i * 4;
    output[0] = HALF2_BITS(__floats2half2_rn(a, b));
    output[1] = HALF2_BITS(__float22half2_rn(make_float2(a, b)));
    output[2] = HALF2_BITS(__float2half2_rn(a)); output[3] = HALF2_BITS(__float2half2_rn(b));
}
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 3
extern "C" __global__ void half2_arithmetic(unsigned* output, const unsigned* input, unsigned count) {
    HALF2_INPUTS
    output += i * 6;
    output[0] = HALF2_BITS(__hadd2(a, b)); output[1] = HALF2_BITS(__hsub2(a, b)); output[2] = HALF2_BITS(__hmul2(a, b));
    output[3] = HALF2_BITS(__hadd2_rn(a, b)); output[4] = HALF2_BITS(__hsub2_rn(a, b)); output[5] = HALF2_BITS(__hmul2_rn(a, b));
}
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 4
extern "C" __global__ void half2_fused(unsigned* output, const unsigned* input, unsigned count) {
    HALF2_INPUTS
    output[i * 2] = HALF2_BITS(__hfma2(a, b, c));
    output[i * 2 + 1] = HALF2_BITS(__hadd2_rn(__hmul2_rn(a, b), c));
}
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 5
extern "C" __global__ void half2_saturation(unsigned* output, const unsigned* input, unsigned count) {
    HALF2_INPUTS
    output += i * 5;
    output[0] = HALF2_BITS(__hadd2_sat(a, b)); output[1] = HALF2_BITS(__hsub2_sat(a, b)); output[2] = HALF2_BITS(__hmul2_sat(a, b));
    output[3] = HALF2_BITS(__hfma2_sat(a, b, c)); output[4] = HALF2_BITS(__hfma2_relu(a, b, c));
}
#endif
#define HALF2_DIVIDE_KERNEL(NAME, EXPRESSION) \
extern "C" __global__ void half2_##NAME(unsigned* output, const unsigned* input, unsigned count) { \
    HALF2_INPUTS \
    __half2 value = a; \
    output[i] = HALF2_BITS(EXPRESSION); \
}
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 6
HALF2_DIVIDE_KERNEL(divide, __h2div(a, b))
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 10
HALF2_DIVIDE_KERNEL(divide_operator, a / b)
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 11
HALF2_DIVIDE_KERNEL(divide_assign, value /= b)
#endif
#undef HALF2_DIVIDE_KERNEL
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 7
extern "C" __global__ void half2_operators(unsigned* output, const unsigned* input, unsigned count) {
    HALF2_INPUTS
    output += i * 6;
    output[0] = HALF2_BITS(a + b); output[1] = HALF2_BITS(a - b); output[2] = HALF2_BITS(a * b);
    __half2 value = a; value += b; output[3] = HALF2_BITS(value);
    value = a; value -= b; output[4] = HALF2_BITS(value);
    value = a; value *= b; output[5] = HALF2_BITS(value);
}
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 8
extern "C" __global__ void half2_comparisons(unsigned* output, const unsigned* input, unsigned count) {
    HALF2_INPUTS
    output += i * 12;
    output[0] = HALF2_BITS(__heq2(a, b)); output[1] = HALF2_BITS(__hne2(a, b));
    output[2] = HALF2_BITS(__hlt2(a, b)); output[3] = HALF2_BITS(__hle2(a, b));
    output[4] = HALF2_BITS(__hgt2(a, b)); output[5] = HALF2_BITS(__hge2(a, b));
    output[6] = HALF2_BITS(__hequ2(a, b)); output[7] = HALF2_BITS(__hneu2(a, b));
    output[8] = HALF2_BITS(__hltu2(a, b)); output[9] = HALF2_BITS(__hleu2(a, b));
    output[10] = HALF2_BITS(__hgtu2(a, b)); output[11] = HALF2_BITS(__hgeu2(a, b));
}
#endif
#if !defined(CUDAVM_HALF2_PROFILE) || CUDAVM_HALF2_PROFILE == 9
extern "C" __global__ void half2_predicates(unsigned* output, const unsigned* input, unsigned count) {
    HALF2_INPUTS
    output += i * 12;
    output[0] = __hbeq2(a, b); output[1] = __hbne2(a, b); output[2] = __hblt2(a, b); output[3] = __hble2(a, b);
    output[4] = __hbgt2(a, b); output[5] = __hbge2(a, b); output[6] = __hbequ2(a, b); output[7] = __hbneu2(a, b);
    output[8] = __hbltu2(a, b); output[9] = __hbleu2(a, b); output[10] = __hbgtu2(a, b); output[11] = __hbgeu2(a, b);
}
#endif
#undef HALF2_INPUTS
#undef HALF2_BITS
