#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
static_assert(__CUDA_ARCH__ >= 900, "cluster execution requires SM90");
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
extern "C" __global__ void cluster_rounds(unsigned* out, const unsigned* in, unsigned* scratch, unsigned rounds) {
    unsigned rank, size, x, y, z, nx, ny, nz;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(size));
    asm("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(x));
    asm("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(y));
    asm("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(z));
    asm("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(nx));
    asm("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(ny));
    asm("mov.u32 %0, %%cluster_nctaid.z;" : "=r"(nz));
    unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    unsigned id = block * blockDim.x + threadIdx.x;
    unsigned total = gridDim.x * gridDim.y * gridDim.z * blockDim.x;
    unsigned peer_rank = (rank + 1) % size;
    unsigned peer_x = blockIdx.x - x + peer_rank % nx;
    unsigned peer_y = blockIdx.y - y + peer_rank / nx % ny;
    unsigned peer_z = blockIdx.z - z + peer_rank / (nx * ny);
    unsigned peer_block = peer_x + gridDim.x * (peer_y + gridDim.y * peer_z);
    unsigned peer = peer_block * blockDim.x + (threadIdx.x + 1) % blockDim.x;
    for (unsigned round = 0; round < rounds; ++round) {
        scratch[id] = in[id] ^ (round * 0x10001u + id);
        cluster_sync();
        unsigned row = (round * total + id) * 10;
        out[row] = rank; out[row + 1] = size;
        out[row + 2] = x; out[row + 3] = y; out[row + 4] = z;
        out[row + 5] = nx; out[row + 6] = ny; out[row + 7] = nz;
        out[row + 8] = scratch[peer]; out[row + 9] = peer;
        cluster_sync();
    }
}
