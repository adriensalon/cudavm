extern "C" __global__ void cache_access(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned i = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + flat;
    if (i >= count) return;
    output += i * 16; input += i;
    unsigned value;
#define CACHE_LOAD(mode, index) \
    asm volatile("ld.global." #mode ".u32 %0,[%1];" : "=r"(value) : "l"(input) : "memory"); output[index] = value;
    CACHE_LOAD(ca, 0)
    CACHE_LOAD(cg, 1)
    CACHE_LOAD(cs, 2)
    CACHE_LOAD(lu, 3)
    CACHE_LOAD(cv, 4)
#undef CACHE_LOAD
#define CACHE_STORE(mode, index) \
    value = *input ^ (0x1020304u * (index - 4)); \
    asm volatile("st.global." #mode ".u32 [%0],%1;" :: "l"(output + index), "r"(value) : "memory"); \
    output[index + 4] = output[index];
    CACHE_STORE(wb, 5)
    CACHE_STORE(cg, 6)
    CACHE_STORE(cs, 7)
    CACHE_STORE(wt, 8)
#undef CACHE_STORE
    asm volatile("prefetch.global.L1 [%0]; prefetch.global.L2 [%0]; prefetchu.L1 [%0];" :: "l"(input) : "memory");
    output[13] = *input + 0x17u;
}
