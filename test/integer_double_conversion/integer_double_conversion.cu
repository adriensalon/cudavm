extern "C" __global__ void integer_double_to_real(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned long long bits = input[i * 3];
    const unsigned word = static_cast<unsigned>(bits);
    const int signed_word = __builtin_bit_cast(int, word);
    const long long signed_wide = __builtin_bit_cast(long long, bits);
    output += i * 10;
    output[0] = __double_as_longlong(__int2double_rn(signed_word));
    output[1] = __double_as_longlong(__uint2double_rn(word));
#define CONVERT(MODE, INDEX) \
    output[2 + INDEX] = __double_as_longlong(__ll2double_##MODE(signed_wide)); \
    output[6 + INDEX] = __double_as_longlong(__ull2double_##MODE(bits));
    CONVERT(rn, 0)
    CONVERT(rz, 1)
    CONVERT(rd, 2)
    CONVERT(ru, 3)
#undef CONVERT
}
extern "C" __global__ void integer_double_to_integer(unsigned long long* output, const unsigned long long* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double value = __longlong_as_double(input[i * 3]);
    const unsigned domains = static_cast<unsigned>(input[i * 3 + 1]);
    output += i * 16;
#define CONVERT(MODE, INDEX) \
    if (domains & 1) output[INDEX] = static_cast<unsigned long long>(__double2int_##MODE(value)); \
    if (domains & 2) output[4 + INDEX] = __double2uint_##MODE(value); \
    if (domains & 4) output[8 + INDEX] = static_cast<unsigned long long>(__double2ll_##MODE(value)); \
    if (domains & 8) output[12 + INDEX] = __double2ull_##MODE(value);
    CONVERT(rn, 0)
    CONVERT(rz, 1)
    CONVERT(rd, 2)
    CONVERT(ru, 3)
#undef CONVERT
}
