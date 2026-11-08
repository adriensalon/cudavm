extern "C" __global__ void fast_math_divide(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float a = __uint_as_float(input[i * 3]), b = __uint_as_float(input[i * 3 + 1]);
    output[i * 2] = __float_as_uint(a / b);
    output[i * 2 + 1] = __float_as_uint(__fdividef(a, b));
}
