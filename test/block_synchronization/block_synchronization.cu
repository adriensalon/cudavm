extern "C" __global__ void block_rounds(unsigned* out, const unsigned* in,
    unsigned* scratch, unsigned count, unsigned rounds, unsigned pattern) {
    __shared__ unsigned values[96];
    unsigned lane = threadIdx.x, lanes = blockDim.x, i = blockIdx.x * lanes + lane;
    unsigned votes = 0, every = 0, any = 0, shared_sum = 0, global_sum = 0;
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned value = in[i] ^ (round * 17u + i);
        values[lane] = value;
        scratch[i] = value + 0x12345u;
        bool truth = pattern == 0 ? false : pattern == 1 ? true
            : pattern == 2 ? lane == round % lanes : (in[i] + round) % 4 != 0;
        // Negative predicates exercise nonzero truth, including the sign bit.
        int predicate = truth ? (-2147483647 - 1) : 0;
        votes += __syncthreads_count(predicate);
        every += __syncthreads_and(predicate) != 0;
        any += __syncthreads_or(predicate) != 0;
        unsigned peer = (lane + round + 1) % lanes;
        shared_sum += values[peer];
        global_sum += scratch[blockIdx.x * lanes + peer];
        // All readers finish before any lane reuses its storage next round.
        __syncthreads();
    }
    if (i < count) {
        out += i * 5;
        out[0] = votes; out[1] = every; out[2] = any;
        out[3] = shared_sum; out[4] = global_sum;
    }
}
