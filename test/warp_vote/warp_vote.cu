extern "C" __global__ void masked_vote(unsigned* out, const unsigned* in, unsigned mask) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31u;
    if ((mask & (1u << lane)) == 0) return;
    unsigned value = in[i];
    #pragma unroll
    for (unsigned phase = 0; phase < 4; ++phase) {
        unsigned shift = phase == 3 ? 31 : phase;
        unsigned predicate = value & (1u << shift);
        out[i * 12 + phase * 3] = __ballot_sync(mask, predicate);
        out[i * 12 + phase * 3 + 1] = __all_sync(mask, predicate);
        out[i * 12 + phase * 3 + 2] = __any_sync(mask, predicate);
    }
}

extern "C" __global__ void retiring_vote(unsigned* out, unsigned keep) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31u;
    if (lane >= keep) return;
    #pragma unroll
    for (unsigned phase = 0; phase < 4; ++phase) {
        if (lane < phase) return;
        int predicate = lane % 3u == phase;
        out[i * 12 + phase * 3] = __ballot_sync(0xffffffffu, predicate);
        out[i * 12 + phase * 3 + 1] = __all_sync(0xffffffffu, predicate);
        out[i * 12 + phase * 3 + 2] = __any_sync(0xffffffffu, predicate);
    }
}
