extern "C" __global__ void replay_step(unsigned* state, unsigned count, unsigned add) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) state[i] = state[i] * 3u + add + i * 7u;
}
