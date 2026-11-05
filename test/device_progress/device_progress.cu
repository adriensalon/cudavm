#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
extern "C" __global__ void progress_rounds(unsigned* output, const unsigned* input,
    unsigned* tickets, unsigned* completed, unsigned count, unsigned seed) {
    __shared__ unsigned values[64];
    unsigned lane = threadIdx.x, lanes = blockDim.x, block = blockIdx.x;
    unsigned id = block * lanes + lane, value = input[id] ^ seed;
    unsigned rounds = 1 + ((block + seed) & 3);
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned steps = (lane + round) % 4;
        for (unsigned step = 0; step < steps; ++step)
            value = (value ^ (round * 13 + step + seed)) * 33 + lane;
        values[lane] = value;
        __syncthreads();
        unsigned peer = lane + 1;
        if (peer == lanes) peer = 0;
        unsigned other = values[peer];
        value += other ^ (seed + round * 17);
        __syncthreads();
    }
    if (id < count) output[id] = value;
    if (lane == 0) tickets[block] = atomicAdd(completed, 1u);
}
