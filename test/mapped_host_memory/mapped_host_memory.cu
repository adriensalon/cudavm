extern "C" __global__ void mapped_host_transform(unsigned* out, const unsigned* in, unsigned count, unsigned seed) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (in[i] ^ (seed + i * 17u)) + i;
}
