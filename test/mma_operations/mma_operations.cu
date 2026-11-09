// PTX m16n8k8 row/column fragments: each four-lane group owns
// two A rows and one B column; each lane owns two C/D columns.
extern "C" __global__ void mma_f16(float* output, const unsigned short* a, const unsigned short* b,
    const float* initial, unsigned half_stride, unsigned float_stride, unsigned rounds) {
    const unsigned flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned tile = (blockIdx.x * blockDim.x * blockDim.y * blockDim.z + flat) / 32;
    const unsigned group = (flat & 31) / 4, pair = (flat & 3) * 2;
    a += tile * half_stride * 16; b += tile * half_stride * 8;
    initial += tile * float_stride * 16; output += tile * float_stride * 16;
    const unsigned a0 = *reinterpret_cast<const unsigned*>(a + group * half_stride + pair);
    const unsigned a1 = *reinterpret_cast<const unsigned*>(a + (group + 8) * half_stride + pair);
    const unsigned b0 = *reinterpret_cast<const unsigned*>(b + group * half_stride + pair);
    float c0 = initial[group * float_stride + pair], c1 = initial[group * float_stride + pair + 1];
    float c2 = initial[(group + 8) * float_stride + pair], c3 = initial[(group + 8) * float_stride + pair + 1];
    for (unsigned round = 0; round < rounds; ++round)
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(b0));
    output[group * float_stride + pair] = c0; output[group * float_stride + pair + 1] = c1;
    output[(group + 8) * float_stride + pair] = c2; output[(group + 8) * float_stride + pair + 1] = c3;
}
