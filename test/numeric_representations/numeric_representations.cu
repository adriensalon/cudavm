extern "C" __global__ void numeric_bits(unsigned* out, const unsigned* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    in += i * 4; out += i * 12;
    // Read and write each representation separately, so inverse conversions
    // cannot hide an incorrect bit pattern by cancelling one another.
    const float* floats = reinterpret_cast<const float*>(in);
    const int* integers = reinterpret_cast<const int*>(in);
    const double real = *reinterpret_cast<const double*>(in + 2);
    out[0] = __float_as_uint(floats[0]);
    out[1] = unsigned(__float_as_int(floats[1]));
    reinterpret_cast<float*>(out)[2] = __uint_as_float(in[0]);
    reinterpret_cast<float*>(out)[3] = __int_as_float(integers[1]);
    *reinterpret_cast<double*>(out + 4) = __longlong_as_double(*reinterpret_cast<const long long*>(in + 2));
    *reinterpret_cast<long long*>(out + 6) = __double_as_longlong(real);
    out[8] = unsigned(__double2loint(real));
    out[9] = unsigned(__double2hiint(real));
    *reinterpret_cast<double*>(out + 10) = __hiloint2double(integers[0], integers[1]);
}
