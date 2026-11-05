#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
#define _CG_LIMIT_INCLUDED_DEPENDENCIES
#include <cooperative_groups.h>

extern "C" __global__ void cooperative_grid(unsigned* out, const unsigned* in,
    unsigned* scratch, unsigned rounds) {
    auto grid = cooperative_groups::this_grid();
    unsigned rank = grid.thread_rank(), size = grid.size();
    unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned round = 0; round < rounds; ++round) {
        scratch[rank] = in[rank] ^ (round * 0x10001u + rank);
        unsigned work = (in[rank] + round) * 33u + rank;
        if (round & 1) {
            auto token = grid.barrier_arrive();
            work ^= round;
            grid.barrier_wait(static_cast<decltype(token)&&>(token));
        } else grid.sync();
        unsigned peer = (rank + threads + round + 1) % size;
        unsigned row = (round * size + rank) * 8;
        out[row] = scratch[peer]; out[row + 1] = size; out[row + 2] = rank;
        out[row + 3] = grid.block_rank(); out[row + 4] = grid.is_valid();
        out[row + 5] = work; out[row + 6] = grid.num_blocks(); out[row + 7] = peer;
        grid.sync();
    }
}
