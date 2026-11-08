namespace cudavm {
template<unsigned Dimensions> __device__ float float_special_norm(float a, float b, float c, float d) {
    if constexpr (Dimensions == 2) return hypotf(a, b);
    if constexpr (Dimensions == 3) return norm3df(a, b, c);
    if constexpr (Dimensions == 4) return norm4df(a, b, c, d);
}
template<unsigned Dimensions> __device__ void float_special_execute(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto xy = input[i * 3], zw = input[i * 3 + 1];
    const float a = __uint_as_float(unsigned(xy)), b = __uint_as_float(unsigned(xy >> 32));
    const float c = __uint_as_float(unsigned(zw)), d = __uint_as_float(unsigned(zw >> 32));
    output += i * 4;
    output[0] = __float_as_uint(float_special_norm<Dimensions>(a, b, c, d));
    output[1] = __float_as_uint(Dimensions == 2 ? float_special_norm<Dimensions>(b, a, c, d)
        : Dimensions == 3 ? float_special_norm<Dimensions>(c, a, b, d) : float_special_norm<Dimensions>(d, c, b, a));
    output[2] = __float_as_uint(float_special_norm<Dimensions>(-a, b, -c, d));
    output[3] = __float_as_uint(Dimensions == 2 ? float_special_norm<Dimensions>(-b, -a, c, d)
        : Dimensions == 3 ? float_special_norm<Dimensions>(b, -c, a, d) : float_special_norm<Dimensions>(b, -d, a, c));
}
}
#define FLOAT_SPECIAL_KERNEL(NAME, N) \
extern "C" __global__ void float_special_##NAME(unsigned long long* output, const unsigned long long* input, unsigned count) { \
    cudavm::float_special_execute<N>(output, input, count); \
}
#if !defined(CUDAVM_FLOAT_SPECIAL_PROFILE) || CUDAVM_FLOAT_SPECIAL_PROFILE == 0
FLOAT_SPECIAL_KERNEL(hypot, 2)
#endif
#if !defined(CUDAVM_FLOAT_SPECIAL_PROFILE) || CUDAVM_FLOAT_SPECIAL_PROFILE == 1
FLOAT_SPECIAL_KERNEL(norm3, 3)
#endif
#if !defined(CUDAVM_FLOAT_SPECIAL_PROFILE) || CUDAVM_FLOAT_SPECIAL_PROFILE == 2
FLOAT_SPECIAL_KERNEL(norm4, 4)
#endif
#undef FLOAT_SPECIAL_KERNEL
