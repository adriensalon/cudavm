#if !defined(CUDAVM_ELEMENTARY_PROFILE) || CUDAVM_ELEMENTARY_PROFILE == 0
extern "C" __global__ void double_elementary_rounding(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double value = __longlong_as_double(input[i * 3]);
    output += i * 6;
    output[0] = __double_as_longlong(floor(value));
    output[1] = __double_as_longlong(ceil(value));
    output[2] = __double_as_longlong(trunc(value));
    output[3] = __double_as_longlong(round(value));
    output[4] = __double_as_longlong(rint(value));
    output[5] = __double_as_longlong(nearbyint(value));
}
#endif
#if !defined(CUDAVM_ELEMENTARY_PROFILE) || CUDAVM_ELEMENTARY_PROFILE == 1
extern "C" __global__ void double_elementary_sqrt(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double value = __longlong_as_double(input[i * 3]);
    output[i] = __double_as_longlong(sqrt(value));
}
#endif
#if !defined(CUDAVM_ELEMENTARY_PROFILE) || CUDAVM_ELEMENTARY_PROFILE == 2
extern "C" __global__ void double_elementary_reciprocal(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double value = __longlong_as_double(input[i * 3]);
    output[i] = __double_as_longlong(1.0 / value);
}
#endif
