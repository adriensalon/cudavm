extern "C" __global__ void float_arithmetic_values(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float a = __uint_as_float(input[i * 3]), b = __uint_as_float(input[i * 3 + 1]), c = __uint_as_float(input[i * 3 + 2]);
    output += i * 20;
#define ARITHMETIC(MODE, INDEX) \
    output[INDEX] = __float_as_uint(__fadd_##MODE(a, b)); \
    output[4 + INDEX] = __float_as_uint(__fsub_##MODE(a, b)); \
    output[8 + INDEX] = __float_as_uint(__fmul_##MODE(a, b)); \
    output[12 + INDEX] = __float_as_uint(__fmaf_##MODE(a, b, c)); \
    output[16 + INDEX] = __float_as_uint(__fadd_##MODE(__fmul_##MODE(a, b), c));
    ARITHMETIC(rn, 0)
    ARITHMETIC(rz, 1)
    ARITHMETIC(rd, 2)
    ARITHMETIC(ru, 3)
#undef ARITHMETIC
}
extern "C" __global__ void float_arithmetic_default(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float a = __uint_as_float(input[i * 3]), b = __uint_as_float(input[i * 3 + 1]), c = __uint_as_float(input[i * 3 + 2]);
    output += i * 4;
    output[0] = __float_as_uint(a + b);
    output[1] = __float_as_uint(a - b);
    output[2] = __float_as_uint(a * b);
    output[3] = __float_as_uint(fmaf(a, b, c));
}
