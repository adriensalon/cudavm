extern "C" __global__ void lifetime_write(unsigned* output, unsigned value) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 91) output[i] = value + i * 7u;
}
