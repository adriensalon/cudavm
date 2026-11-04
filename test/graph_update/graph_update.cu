extern "C" __global__ void update_write(unsigned* output, unsigned count, unsigned value) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = value + i * 7u;
}
extern "C" __global__ void update_twist(unsigned* output, unsigned count, unsigned value) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = value ^ (i * 13u + 0xa5a55a5au);
}
