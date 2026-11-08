#define NEIGHBOR_BODY(REAL, FROM_BITS, TO_BITS, NEXT, SIGN, INFINITY_BITS) \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const REAL x = FROM_BITS(input[i * 3]), y = FROM_BITS(input[i * 3 + 1]); \
    const REAL first = NEXT(x, y), infinity = FROM_BITS(INFINITY_BITS); \
    output += i * 8; \
    output[0] = TO_BITS(first); output[1] = TO_BITS(NEXT(y, x)); \
    output[2] = TO_BITS(NEXT(x, FROM_BITS(0))); output[3] = TO_BITS(NEXT(x, FROM_BITS(SIGN))); \
    output[4] = TO_BITS(NEXT(x, infinity)); output[5] = TO_BITS(NEXT(x, -infinity)); \
    output[6] = TO_BITS(NEXT(first, x)); \
    const unsigned long long magnitude = static_cast<unsigned long long>(TO_BITS(first)) & (SIGN - 1); \
    const unsigned long long minimum_normal = INFINITY_BITS & (0ull - INFINITY_BITS); \
    output[7] = isnan(first) ? 1u : unsigned(isinf(first) != 0) * 2 \
        | unsigned(!magnitude) * 4 | unsigned(magnitude && magnitude < minimum_normal) * 8 \
        | unsigned(magnitude >= minimum_normal && magnitude < INFINITY_BITS) * 16 | unsigned(signbit(first) != 0) * 32;

// Execution compiles its own precision. The NVRTC-only source check compiles both.
#if !defined(CUDAVM_SPECIAL_PROFILE) || CUDAVM_SPECIAL_PROFILE == 0
extern "C" __global__ void special_float_neighbors(unsigned long long* output, const unsigned long long* input, unsigned count) {
    NEIGHBOR_BODY(float, __uint_as_float, __float_as_uint, nextafterf, 0x80000000ull, 0x7f800000ull)
}
#endif
#if !defined(CUDAVM_SPECIAL_PROFILE) || CUDAVM_SPECIAL_PROFILE == 1
extern "C" __global__ void special_double_neighbors(unsigned long long* output, const unsigned long long* input, unsigned count) {
    NEIGHBOR_BODY(double, __longlong_as_double, __double_as_longlong, nextafter, 0x8000000000000000ull, 0x7ff0000000000000ull)
}
#endif
