#if !defined(CUDAVM_ELEMENTARY_PROFILE) || CUDAVM_ELEMENTARY_PROFILE == 0
extern "C" __global__ void float_elementary_rounding(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = __uint_as_float(input[i * 3]);
    output += i * 6;
    output[0] = __float_as_uint(floorf(value));
    output[1] = __float_as_uint(ceilf(value));
    output[2] = __float_as_uint(truncf(value));
    output[3] = __float_as_uint(roundf(value));
    output[4] = __float_as_uint(rintf(value));
    output[5] = __float_as_uint(nearbyintf(value));
}
#endif
#if !defined(CUDAVM_ELEMENTARY_PROFILE) || CUDAVM_ELEMENTARY_PROFILE == 1
extern "C" __global__ void float_elementary_roots(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = __uint_as_float(input[i * 3]);
    output[i * 2] = __float_as_uint(sqrtf(value));
    output[i * 2 + 1] = __float_as_uint(1.0f / value);
}
#endif
