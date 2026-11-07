// Each execution selects one profile; the independent source check parses all.
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 0
extern "C" __global__ void double_arithmetic_add(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 4;
    output[0] = __double_as_longlong(__dadd_rn(a, b));
    output[1] = __double_as_longlong(__dadd_rz(a, b));
    output[2] = __double_as_longlong(__dadd_rd(a, b));
    output[3] = __double_as_longlong(__dadd_ru(a, b));
}
#endif
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 1
extern "C" __global__ void double_arithmetic_subtract(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 4;
    output[0] = __double_as_longlong(__dsub_rn(a, b));
    output[1] = __double_as_longlong(__dsub_rz(a, b));
    output[2] = __double_as_longlong(__dsub_rd(a, b));
    output[3] = __double_as_longlong(__dsub_ru(a, b));
}
#endif
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 2
extern "C" __global__ void double_arithmetic_multiply(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 4;
    output[0] = __double_as_longlong(__dmul_rn(a, b));
    output[1] = __double_as_longlong(__dmul_rz(a, b));
    output[2] = __double_as_longlong(__dmul_rd(a, b));
    output[3] = __double_as_longlong(__dmul_ru(a, b));
}
#endif
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 3
extern "C" __global__ void double_arithmetic_fma_rn(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 1;
    output[0] = __double_as_longlong(__fma_rn(a, b, c));
}
#endif
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 4
extern "C" __global__ void double_arithmetic_fma_rz(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 1;
    output[0] = __double_as_longlong(__fma_rz(a, b, c));
}
#endif
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 5
extern "C" __global__ void double_arithmetic_fma_rd(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 1;
    output[0] = __double_as_longlong(__fma_rd(a, b, c));
}
#endif
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 6
extern "C" __global__ void double_arithmetic_fma_ru(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 1;
    output[0] = __double_as_longlong(__fma_ru(a, b, c));
}
#endif
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 7
extern "C" __global__ void double_arithmetic_separate(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 4;
    output[0] = __double_as_longlong(__dadd_rn(__dmul_rn(a, b), c));
    output[1] = __double_as_longlong(__dadd_rz(__dmul_rz(a, b), c));
    output[2] = __double_as_longlong(__dadd_rd(__dmul_rd(a, b), c));
    output[3] = __double_as_longlong(__dadd_ru(__dmul_ru(a, b), c));
}
#endif
#if !defined(CUDAVM_ARITHMETIC_PROFILE) || CUDAVM_ARITHMETIC_PROFILE == 8
extern "C" __global__ void double_arithmetic_default(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a = __longlong_as_double(input[i * 3]), b = __longlong_as_double(input[i * 3 + 1]), c = __longlong_as_double(input[i * 3 + 2]);
    output += i * 4;
    output[0] = __double_as_longlong(a + b);
    output[1] = __double_as_longlong(a - b);
    output[2] = __double_as_longlong(a * b);
    output[3] = __double_as_longlong(fma(a, b, c));
}
#endif
