#include <cuda_bf16.h>
extern "C" __global__ void convert_bfloat16(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned bits = input[i];
    const float value = __uint_as_float(bits);
    const __nv_bfloat16 number = __ushort_as_bfloat16(static_cast<unsigned short>(bits));
    output[i * 5] = __float_as_uint(__bfloat162float(number));
    output[i * 5 + 1] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    output[i * 5 + 2] = __bfloat16_as_ushort(__float2bfloat16_rz(value));
    output[i * 5 + 3] = __bfloat16_as_ushort(__float2bfloat16_rd(value));
    output[i * 5 + 4] = __bfloat16_as_ushort(__float2bfloat16_ru(value));
}
