namespace cudavm {
template<unsigned Dimensions> __device__ double double_special_norm(double a, double b, double c, double d) {
    if constexpr (Dimensions == 2) return hypot(a, b);
    if constexpr (Dimensions == 3) return norm3d(a, b, c);
    if constexpr (Dimensions == 4) return norm4d(a, b, c, d);
}
template<unsigned Dimensions> __device__ void double_special_execute(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(static_cast<long long>(input[i * 4]));
    const double b = __longlong_as_double(static_cast<long long>(input[i * 4 + 1]));
    const double c = __longlong_as_double(static_cast<long long>(input[i * 4 + 2]));
    const double d = __longlong_as_double(static_cast<long long>(input[i * 4 + 3]));
    output += i * 4;
    output[0] = static_cast<unsigned long long>(__double_as_longlong(double_special_norm<Dimensions>(a, b, c, d)));
    output[1] = static_cast<unsigned long long>(__double_as_longlong(Dimensions == 2 ? double_special_norm<Dimensions>(b, a, c, d)
        : Dimensions == 3 ? double_special_norm<Dimensions>(c, a, b, d) : double_special_norm<Dimensions>(d, c, b, a)));
    output[2] = static_cast<unsigned long long>(__double_as_longlong(double_special_norm<Dimensions>(-a, b, -c, d)));
    output[3] = static_cast<unsigned long long>(__double_as_longlong(Dimensions == 2 ? double_special_norm<Dimensions>(-b, -a, c, d)
        : Dimensions == 3 ? double_special_norm<Dimensions>(b, -c, a, d) : double_special_norm<Dimensions>(b, -d, a, c)));
}
}
#define DOUBLE_SPECIAL_KERNEL(NAME, N) \
extern "C" __global__ void double_special_##NAME(unsigned long long* output, const unsigned long long* input, unsigned count) { \
    cudavm::double_special_execute<N>(output, input, count); \
}
#if !defined(CUDAVM_DOUBLE_SPECIAL_PROFILE) || CUDAVM_DOUBLE_SPECIAL_PROFILE == 0
DOUBLE_SPECIAL_KERNEL(hypot, 2)
#endif
#if !defined(CUDAVM_DOUBLE_SPECIAL_PROFILE) || CUDAVM_DOUBLE_SPECIAL_PROFILE == 1
DOUBLE_SPECIAL_KERNEL(norm3, 3)
#endif
#if !defined(CUDAVM_DOUBLE_SPECIAL_PROFILE) || CUDAVM_DOUBLE_SPECIAL_PROFILE == 2
DOUBLE_SPECIAL_KERNEL(norm4, 4)
#endif
#undef DOUBLE_SPECIAL_KERNEL
