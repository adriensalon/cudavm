extern "C" __global__ void permute_bytes(unsigned* out, const unsigned* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned low = in[i * 3], high = in[i * 3 + 1], selector = in[i * 3 + 2], raw;
    out[i * 2] = __byte_perm(low, high, selector);
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(raw) : "r"(low), "r"(high), "r"(selector));
    out[i * 2 + 1] = raw;
}
