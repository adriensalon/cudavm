#ifndef __CUDA_ARCH_FEAT_SM100_ALL
#error tensor_memory_requires_architecture_specific_sm100
#endif
extern "C" __global__ void tensor_memory_exchange(unsigned* output, const unsigned* input,
    unsigned count, unsigned columns, unsigned rounds) {
    __shared__ volatile unsigned tensor_address;
    const unsigned flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned i = blockIdx.x * 128 + flat;
    for (unsigned round = 0; round < rounds; ++round) {
        // Allocation/management is collective within the first complete warp;
        // every CTA thread stays alive through all publication/lifetime barriers.
        if (flat < 32)
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0],%1;"
                :: "l"(&tensor_address), "r"(columns) : "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        // Each of the four CTA warps owns its corresponding 32-row TMEM slab.
        const unsigned row_address = tensor_address + ((flat / 32) * 32 << 16);
        for (unsigned slot = 0; slot < 3; ++slot) {
            const unsigned column = slot == 0 ? 0 : slot == 1 ? columns / 2 : columns - 1;
            const unsigned address = row_address + column;
            const unsigned value = input[i] ^ ((round + 1) * 0x5917ab3du) ^ (column * 0x1020304u);
            asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1}; tcgen05.wait::st.sync.aligned;"
                :: "r"(address), "r"(value) : "memory");
            unsigned loaded;
            // Keep the result inaccessible to ordinary code until the load wait.
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0},[%1]; tcgen05.wait::ld.sync.aligned;"
                : "=r"(loaded) : "r"(address) : "memory");
            if (i < count) output[i * 16 + round * 4 + slot] = loaded;
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
        if (flat < 32)
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"
                :: "r"(unsigned(tensor_address)), "r"(columns) : "memory");
        __syncthreads();
    }
    // No later allocation is permitted after this final release.
    if (flat < 32 && rounds)
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
