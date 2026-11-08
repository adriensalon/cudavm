#define DOUBLE_TRANSCENDENTAL(NAME, FUNCTION) \
extern "C" __global__ void double_transcendental_##NAME(unsigned long long* output, const unsigned long long* input, unsigned count) { \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const double x = __longlong_as_double(static_cast<long long>(input[i * 3])); \
    output[i] = static_cast<unsigned long long>(__double_as_longlong(FUNCTION(x))); \
}
#if !defined(CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE) || CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE == 0
DOUBLE_TRANSCENDENTAL(exp, exp)
#endif
#if !defined(CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE) || CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE == 1
DOUBLE_TRANSCENDENTAL(exp2, exp2)
#endif
#if !defined(CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE) || CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE == 2
DOUBLE_TRANSCENDENTAL(exp10, exp10)
#endif
#if !defined(CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE) || CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE == 3
DOUBLE_TRANSCENDENTAL(log, log)
#endif
#if !defined(CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE) || CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE == 4
DOUBLE_TRANSCENDENTAL(log2, log2)
#endif
#if !defined(CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE) || CUDAVM_DOUBLE_TRANSCENDENTAL_PROFILE == 5
DOUBLE_TRANSCENDENTAL(log10, log10)
#endif
#undef DOUBLE_TRANSCENDENTAL
