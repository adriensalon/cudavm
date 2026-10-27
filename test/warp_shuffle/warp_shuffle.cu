extern "C" __global__ void shuffle_modes(unsigned* out, const unsigned* in,
                                        unsigned width, unsigned distance) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned value = in[i];
    out[i * 4] = __shfl_sync(0xffffffffu, value, int(distance), width);
    out[i * 4 + 1] = __shfl_up_sync(0xffffffffu, value, distance, width);
    out[i * 4 + 2] = __shfl_down_sync(0xffffffffu, value, distance, width);
    out[i * 4 + 3] = __shfl_xor_sync(0xffffffffu, value, int(distance), width);
}

extern "C" __global__ void shuffle_scan(unsigned* out, const unsigned* in) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31u, base = threadIdx.x & ~31u;
    unsigned remaining = blockDim.x - base;
    unsigned live = remaining < 32u ? remaining : 32u;
    unsigned value = in[i];
    #pragma unroll 1
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        unsigned other = __shfl_up_sync(0xffffffffu, value, offset);
        if (lane >= offset) value += other;
    }
    out[i * 2] = value;
    out[i * 2 + 1] = __shfl_sync(0xffffffffu, value, int(live - 1));
}
