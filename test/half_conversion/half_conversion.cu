#include <cuda_fp16.h>
extern "C" __global__ void convert_half(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned bits = input[i];
    const float value = __uint_as_float(bits);
    const __half half = __ushort_as_half(static_cast<unsigned short>(bits));
    output[i * 5] = __float_as_uint(__half2float(half));
    output[i * 5 + 1] = __half_as_ushort(__float2half_rn(value));
    output[i * 5 + 2] = __half_as_ushort(__float2half_rz(value));
    output[i * 5 + 3] = __half_as_ushort(__float2half_rd(value));
    output[i * 5 + 4] = __half_as_ushort(__float2half_ru(value));
}
