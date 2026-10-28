extern "C" __global__ void warp_publish(unsigned* out, const unsigned* in,
    unsigned* scratch, unsigned mask, unsigned rounds, unsigned mode) {
    __shared__ volatile unsigned values[96];
    unsigned thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    unsigned i = blockIdx.x * threads + thread, lane = thread & 31u;
    unsigned live = threads - (thread & ~31u); if (live > 32) live = 32;
    mask &= 0xffffffffu >> (32u - live);
    if (mode == 1) mask &= lane & 1u ? 0xaaaaaaaau : 0x55555555u;
    if (!(mask & (1u << lane))) return;
    unsigned peer = (lane + 1u) & 31u;
    while (!(mask & (1u << peer))) peer = (peer + 1u) & 31u;
    peer += thread & ~31u;
    unsigned limit = rounds;
    if (rounds && mode == 1) limit += (lane & 1u) + (thread / 32u) % 3u;
    #pragma unroll 1
    for (unsigned round = 0; round < limit; ++round) {
        unsigned value = in[i] ^ (round * 0x10203u + i);
        values[thread] = value;
        scratch[i] = value + 0x9e3779b9u;
        if (mode == 2 && (lane & 1u)) {
            for (unsigned tick = 0; tick < lane % 3u; ++tick) {
                values[thread] = values[thread] + tick + 1u;
                scratch[i] = scratch[i] + tick + 1u;
            }
            __syncwarp(mask);
        } else {
            unsigned n = mode == 2 ? lane % 3u : 0u;
            unsigned delta = n * (n + 1u) / 2u;
            values[thread] = values[thread] + delta;
            scratch[i] = scratch[i] + delta;
            __syncwarp(mask);
        }
        out[i * 16 + round * 2] = values[peer];
        out[i * 16 + round * 2 + 1] = scratch[blockIdx.x * threads + peer];
        __syncwarp(mask);
    }
}
