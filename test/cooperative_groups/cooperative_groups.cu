#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
// The pinned header offers this mode without libcu++ dependencies.
#define _CG_LIMIT_INCLUDED_DEPENDENCIES
#include <cooperative_groups.h>

extern "C" __global__ void cooperative_block(unsigned* out, const unsigned* in,
    unsigned* scratch, unsigned rounds, unsigned ignored) {
    __shared__ unsigned values[128];
    auto block = cooperative_groups::this_thread_block();
    unsigned rank = block.thread_rank(), size = block.size();
    auto index = block.group_index(), thread = block.thread_index(), dimensions = block.dim_threads();
    unsigned block_rank = index.x + gridDim.x * (index.y + gridDim.y * index.z);
    unsigned id = block_rank * size + rank, total = gridDim.x * gridDim.y * gridDim.z * size;
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned value = in[id] ^ (round * 0x10001u + id);
        values[rank] = value; scratch[id] = value + 0x12345u;
        cooperative_groups::sync(block);
        unsigned peer = (rank + round + 1) % size;
        unsigned row = (round * total + id) * 20;
        out[row] = cooperative_groups::group_size(block);
        out[row + 1] = cooperative_groups::thread_rank(block);
        out[row + 2] = index.x; out[row + 3] = index.y; out[row + 4] = index.z;
        out[row + 5] = thread.x; out[row + 6] = thread.y; out[row + 7] = thread.z;
        out[row + 8] = dimensions.x; out[row + 9] = dimensions.y; out[row + 10] = dimensions.z;
        out[row + 11] = values[peer]; out[row + 12] = scratch[block_rank * size + peer];
        block.sync();
    }
}

template<unsigned Width>
__device__ __forceinline__ void cooperative_tile_body(unsigned* out, const unsigned* in, unsigned* scratch, unsigned rounds) {
    __shared__ unsigned values[128];
    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<Width>(block);
    auto dynamic = cooperative_groups::tiled_partition(block, Width);
    auto parent = cooperative_groups::tiled_partition<32>(block);
    auto nested = cooperative_groups::tiled_partition<4>(parent);
    unsigned rank = tile.thread_rank(), lane = block.thread_rank(), size = block.size();
    unsigned id = blockIdx.x * size + lane, total = gridDim.x * size, base = lane - rank;
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned value = in[id] ^ (round * 0x10001u + id);
        values[lane] = value; scratch[id] = value + 0x12345u;
        tile.sync();
        unsigned peer = base + (rank + round + 1) % Width;
        unsigned prefix = value, reduction = value;
        for (unsigned delta = 1; delta < Width; delta *= 2) {
            unsigned previous = tile.shfl_up(prefix, delta);
            if (rank >= delta) prefix += previous;
        }
        for (unsigned delta = Width / 2; delta; delta /= 2) {
            unsigned next = tile.shfl_down(reduction, delta);
            if (rank + delta < Width) reduction += next;
        }
        unsigned sum = tile.shfl(reduction, 0);
        bool predicate = (in[id] + round) % 3 != 0;
        unsigned row = (round * total + id) * 20;
        out[row] = tile.size(); out[row + 1] = rank;
        out[row + 2] = tile.meta_group_rank(); out[row + 3] = tile.meta_group_size();
        out[row + 4] = values[peer]; out[row + 5] = scratch[blockIdx.x * size + peer];
        out[row + 6] = tile.shfl(value, Width - 1);
        out[row + 7] = tile.shfl_xor(value, Width - 1);
        out[row + 8] = prefix; out[row + 9] = sum;
        out[row + 10] = tile.any(predicate); out[row + 11] = tile.all(predicate);
        out[row + 12] = tile.ballot(predicate);
        out[row + 13] = dynamic.size(); out[row + 14] = dynamic.thread_rank();
        out[row + 15] = nested.meta_group_rank(); out[row + 16] = nested.meta_group_size();
        out[row + 17] = nested.thread_rank(); out[row + 18] = nested.size();
        dynamic.sync();
    }
}
#define COOPERATIVE_TILE(Width) \
extern "C" __global__ void cooperative_tile_##Width(unsigned* out, const unsigned* in, \
    unsigned* scratch, unsigned rounds, unsigned ignored) { \
    cooperative_tile_body<Width>(out, in, scratch, rounds); \
}
COOPERATIVE_TILE(1)
COOPERATIVE_TILE(2)
COOPERATIVE_TILE(4)
COOPERATIVE_TILE(8)
COOPERATIVE_TILE(16)
COOPERATIVE_TILE(32)
