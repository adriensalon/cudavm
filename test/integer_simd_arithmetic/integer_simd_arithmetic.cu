// Unsaturated operations complement integer_packed_arithmetic.
extern "C" __global__ void simd_arithmetic(unsigned* out, const unsigned* in, unsigned count) {
    unsigned block = blockIdx.x + gridDim.x * blockIdx.y;
    unsigned i = block * blockDim.x + threadIdx.x;
    if (i >= count) return;
    in += i * 3; out += i * 20;
    unsigned a = in[0], b = in[1], c = in[2];
    out[0] = __vabs2(a);
    out[1] = __vabs4(a);
    out[2] = __vadd2(a,b);
    out[3] = __vadd4(a,b);
    out[4] = __vsub2(a,b);
    out[5] = __vsub4(a,b);
    out[6] = __vneg2(a);
    out[7] = __vneg4(a);
    out[8] = __vabsdiffs2(a,b);
    out[9] = __vabsdiffs4(a,b);
    out[10] = __vabsdiffu2(a,b);
    out[11] = __vabsdiffu4(a,b);
    out[12] = __vsads2(a,b);
    out[13] = __vsads4(a,b);
    out[14] = __vsadu2(a,b);
    out[15] = __vsadu4(a,b);
    unsigned r16;
    asm("vabsdiff2.s32.s32.s32.add %0,%1,%2,%3;" : "=r"(r16) : "r"(a), "r"(b), "r"(c));
    out[16] = r16;
    unsigned r17;
    asm("vabsdiff4.s32.s32.s32.add %0,%1,%2,%3;" : "=r"(r17) : "r"(a), "r"(b), "r"(c));
    out[17] = r17;
    unsigned r18;
    asm("vabsdiff2.u32.u32.u32.add %0,%1,%2,%3;" : "=r"(r18) : "r"(a), "r"(b), "r"(c));
    out[18] = r18;
    unsigned r19;
    asm("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;" : "=r"(r19) : "r"(a), "r"(b), "r"(c));
    out[19] = r19;
}
