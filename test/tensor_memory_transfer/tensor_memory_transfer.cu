static_assert(__CUDA_ARCH__ >= 900, "tiled TMA load requires SM90");
// Opaque parameter storage matches the pinned public host CUtensorMap ABI.
struct alignas(128) tensor_descriptor { unsigned long long opaque[16]; };
__device__ __forceinline__ unsigned tensor_shared_address(const void* pointer) {
    unsigned long long address;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(address) : "l"(reinterpret_cast<unsigned long long>(pointer)));
    return unsigned(address);
}
extern "C" __global__ void tensor_load(const __grid_constant__ tensor_descriptor descriptor,
    unsigned* output, unsigned rounds, int origin_x, int origin_y) {
    struct alignas(128) shared_tile { unsigned before[32], values[32], after[32]; };
    __shared__ shared_tile tile;
    __shared__ unsigned long long barrier;
    const unsigned lane = threadIdx.x, bar = tensor_shared_address(&barrier);
    if (lane == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
    for (unsigned round = 0; round < rounds; ++round) {
        for (unsigned i = lane; i < 32; i += blockDim.x) {
            tile.before[i] = 0x9b3d715a; tile.values[i] = 0xfeed1234; tile.after[i] = 0xc6e8294f;
        }
        // Publish barrier initialization and all generic shared writes to TMA.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (lane == 0) {
            const int x = origin_x + int(blockIdx.x) * 4, y = origin_y + int(round);
            const unsigned destination = tensor_shared_address(tile.values);
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];" :: "r"(destination), "l"(&descriptor), "r"(x), "r"(y), "r"(bar) : "memory");
            unsigned long long token;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], 128;"
                : "=l"(token) : "r"(bar) : "memory");
            unsigned ready;
            do {
                asm volatile("{ .reg .pred p; mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2; "
                    "selp.b32 %0, 1, 0, p; }" : "=r"(ready) : "r"(bar), "l"(token) : "memory");
            } while (!ready);
        }
        // The acquiring lane publishes completion to all CTA consumers.
        __syncthreads();
        const unsigned row = (round * gridDim.x + blockIdx.x) * 96;
        for (unsigned i = lane; i < 32; i += blockDim.x) {
            output[row + i] = tile.before[i]; output[row + 32 + i] = tile.values[i]; output[row + 64 + i] = tile.after[i];
        }
        __syncthreads(); // All consumers finish before resetting shared storage.
    }
    if (lane == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
}
