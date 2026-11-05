#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
#include <cuda_awbarrier_primitives.h>

// Exercise the pinned public primitives, including their SM70/75 implementation.
extern "C" __global__ void split_serial(unsigned* out, unsigned count, unsigned rounds) {
    __shared__ __mbarrier_t barrier;
    if (threadIdx.x != 0) return;
    for (unsigned round = 0; round < rounds; ++round) {
        __mbarrier_init(&barrier, count);
        for (unsigned arrival = 0; arrival < count; ++arrival) {
            auto token = __mbarrier_arrive(&barrier);
            unsigned row = (round * count + arrival) * 2;
            out[row] = __mbarrier_token_pending_count(token);
            if (arrival + 1 == count) {
                while (!__mbarrier_test_wait(&barrier, token)) {}
                out[row + 1] = 1;
            } else out[row + 1] = __mbarrier_test_wait(&barrier, token);
        }
        __mbarrier_inval(&barrier);
        // The next init repurposes the same storage after invalidation.
    }
}

extern "C" __global__ void split_publication(unsigned* out, const unsigned* in,
    unsigned* scratch, unsigned rounds, unsigned group_size, unsigned drop) {
    __shared__ __mbarrier_t barriers[2];
    __shared__ unsigned values[128];
    unsigned lane = threadIdx.x, group = lane / group_size;
    unsigned rank = lane % group_size, base = group * group_size;
    unsigned size = blockDim.x - base;
    if (size > group_size) size = group_size;
    unsigned id = blockIdx.x * blockDim.x + lane;
    if (rank == 0) __mbarrier_init(&barriers[group], size);
    __syncthreads();
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned retired = drop ? round : 0;
        if (retired >= size) retired = size - 1;
        unsigned live = size - retired;
        if (rank < live) {
            unsigned value = in[id] ^ (round * 0x10001u + id);
            values[lane] = value;
            scratch[id] = value + 0x12345u;
            auto token = drop && live > 1 && rank + 1 == live
                ? __mbarrier_arrive_and_drop(&barriers[group]) : __mbarrier_arrive(&barriers[group]);
            // Independent work between arrival and wait must not block arrival.
            unsigned work = (in[id] + round) * 33u + rank;
            while (!__mbarrier_test_wait(&barriers[group], token)) {}
            unsigned peer = base + ((rank + round + 1) % live);
            unsigned row = (round * gridDim.x * blockDim.x + id) * 4;
            out[row] = values[peer];
            out[row + 1] = scratch[blockIdx.x * blockDim.x + peer];
            out[row + 2] = work;
            out[row + 3] = __mbarrier_token_pending_count(token);
        }
        // Readers finish before any participant overwrites a published word.
        // Retired participants remain here but never arrive at the mbarrier again.
        __syncthreads();
    }
    if (rank == 0) __mbarrier_inval(&barriers[group]);
}
