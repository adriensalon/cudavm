extern "C" __global__ void architecture_state(unsigned* output, const unsigned* input, unsigned count, unsigned dynamic_bytes) {
    __shared__ volatile unsigned scratch[64];
    extern __shared__ volatile unsigned char dynamic[];
    const unsigned flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const unsigned i = block * blockDim.x * blockDim.y * blockDim.z + flat;
    if (i >= count) return;
    unsigned sm, sms, warp, warps, allocated, total;
    unsigned long long grid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(sms));
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp));
    asm volatile("mov.u32 %0, %%nwarpid;" : "=r"(warps));
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(grid));
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(allocated));
    asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(total));
    scratch[flat] = input[i] ^ (block * 0x10203u + flat);
    if (dynamic_bytes) dynamic[flat] = static_cast<unsigned char>(input[i]);
    output += i * 16;
    output[0] = sm; output[1] = sms; output[2] = warp; output[3] = warps;
    output[4] = static_cast<unsigned>(grid); output[5] = static_cast<unsigned>(grid >> 32);
    output[6] = allocated; output[7] = total;
    output[8] = scratch[flat] ^ (dynamic_bytes ? unsigned(dynamic[flat]) << 16 : 0u);
}
