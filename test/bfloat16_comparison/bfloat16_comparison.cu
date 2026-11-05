#include <cuda_bf16.h>
extern "C" __global__ void compare_bfloat16(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned bits = input[i];
    const __nv_bfloat16 a = __ushort_as_bfloat16(static_cast<unsigned short>(bits));
    const __nv_bfloat16 b = __ushort_as_bfloat16(static_cast<unsigned short>(bits >> 16));
    unsigned* row = output + i * 20;
    row[0] = __heq(a, b); row[1] = __hne(a, b);
    row[2] = __hlt(a, b); row[3] = __hle(a, b);
    row[4] = __hgt(a, b); row[5] = __hge(a, b);
    row[6] = __hequ(a, b); row[7] = __hneu(a, b);
    row[8] = __hltu(a, b); row[9] = __hleu(a, b);
    row[10] = __hgtu(a, b); row[11] = __hgeu(a, b);
    row[12] = __hisnan(a); row[13] = static_cast<unsigned>(__hisinf(a));
    row[14] = __bfloat16_as_ushort(__hmin(a, b));
    row[15] = __bfloat16_as_ushort(__hmax(a, b));
    row[16] = __bfloat16_as_ushort(__hmin_nan(a, b));
    row[17] = __bfloat16_as_ushort(__hmax_nan(a, b));
    row[18] = __bfloat16_as_ushort(__habs(a));
    row[19] = __bfloat16_as_ushort(__hneg(a));
}
