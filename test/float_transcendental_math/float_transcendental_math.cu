#define TRANSCENDENTAL(NAME, FUNCTION) \
extern "C" __global__ void float_transcendental_##NAME(unsigned* output, const unsigned* input, unsigned count) { \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    output[i] = __float_as_uint(FUNCTION(__uint_as_float(input[i * 3]))); \
}
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 0
TRANSCENDENTAL(sin, sinf)
#endif
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 1
TRANSCENDENTAL(cos, cosf)
#endif
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 2
TRANSCENDENTAL(exp, expf)
#endif
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 3
TRANSCENDENTAL(exp2, exp2f)
#endif
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 4
TRANSCENDENTAL(exp10, exp10f)
#endif
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 5
TRANSCENDENTAL(log, logf)
#endif
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 6
TRANSCENDENTAL(log2, log2f)
#endif
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 7
TRANSCENDENTAL(log10, log10f)
#endif
#if !defined(CUDAVM_TRANSCENDENTAL_PROFILE) || CUDAVM_TRANSCENDENTAL_PROFILE == 8
TRANSCENDENTAL(tanh, tanhf)
#endif
#undef TRANSCENDENTAL
