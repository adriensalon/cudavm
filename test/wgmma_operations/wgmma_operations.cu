#ifndef __CUDA_ARCH_FEAT_SM90_ALL
#error wgmma_requires_architecture_specific_hopper
#endif
extern "C" __global__ void wgmma_f16(float* output, const unsigned short* a, const unsigned short* b,
    const float* initial, unsigned half_stride, unsigned float_stride, unsigned rounds, unsigned accumulate) {
    __shared__ __align__(16) unsigned short shared_b[256];
    const unsigned flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned local = flat % 128, warp = local / 32, group = (local % 32) / 4, pair = (local % 4) * 2;
    const unsigned tile = (blockIdx.x * blockDim.x * blockDim.y * blockDim.z + flat) / 128;
    a += tile * half_stride * 64; b += tile * half_stride * 8;
    initial += tile * float_stride * 64; output += tile * float_stride * 64;
    // No-swizzle K-major B: two contiguous 8-column by 8-K core matrices.
    // Adjacent K cores are 128 bytes apart; an adjacent N group is 256 bytes.
    shared_b[flat] = b[((local % 64) / 8) * half_stride + (local / 64) * 8 + local % 8];
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned long long shared_address;
    asm("cvta.to.shared.u64 %0,%1;" : "=l"(shared_address) : "l"(shared_b + (flat / 128) * 128));
    const unsigned long long descriptor = ((shared_address & 0x3ffffull) >> 4) | (8ull << 16) | (16ull << 32);
    const unsigned row = warp * 16 + group;
    const unsigned a0 = *reinterpret_cast<const unsigned*>(a + row * half_stride + pair);
    const unsigned a1 = *reinterpret_cast<const unsigned*>(a + (row + 8) * half_stride + pair);
    const unsigned a2 = *reinterpret_cast<const unsigned*>(a + row * half_stride + pair + 8);
    const unsigned a3 = *reinterpret_cast<const unsigned*>(a + (row + 8) * half_stride + pair + 8);
    float d0 = initial[row * float_stride + pair], d1 = initial[row * float_stride + pair + 1];
    float d2 = initial[(row + 8) * float_stride + pair], d3 = initial[(row + 8) * float_stride + pair + 1];
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned scale = round || accumulate;
        // One block keeps the register inputs before fence and every output
        // after wait. No ordinary code touches an outstanding accumulator.
        asm volatile("{ .reg .pred use_c; setp.ne.u32 use_c,%9,0; "
            "wgmma.fence.sync.aligned; "
            "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, use_c, 1, 1, 0; "
            "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0; }"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descriptor), "r"(scale) : "memory");
    }
    output[row * float_stride + pair] = d0; output[row * float_stride + pair + 1] = d1;
    output[(row + 8) * float_stride + pair] = d2; output[(row + 8) * float_stride + pair + 1] = d3;
}
