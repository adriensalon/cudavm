extern "C" __global__ void float_double_widen(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float value = __uint_as_float(static_cast<unsigned>(input[i * 3]));
    output[i] = __double_as_longlong(static_cast<double>(value));
}
extern "C" __global__ void float_double_narrow(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double value = __longlong_as_double(input[i * 3]);
    output += i * 5;
    output[0] = __float_as_uint(__double2float_rn(value));
    output[1] = __float_as_uint(__double2float_rz(value));
    output[2] = __float_as_uint(__double2float_rd(value));
    output[3] = __float_as_uint(__double2float_ru(value));
    output[4] = __float_as_uint(static_cast<float>(value));
}
