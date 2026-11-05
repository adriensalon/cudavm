#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
template<bool Device>
__device__ __forceinline__ unsigned relaxed_word(const unsigned* pointer) {
    unsigned result;
    if constexpr (Device) asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    else asm volatile("ld.relaxed.cta.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    return result;
}
template<bool Device>
__device__ __forceinline__ void relaxed_word(unsigned* pointer, unsigned value) {
    if constexpr (Device) asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
    else asm volatile("st.relaxed.cta.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
}
template<bool Device>
__device__ __forceinline__ void publication_fence() {
    if constexpr (Device) __threadfence(); else __threadfence_block();
}
template<bool Device, bool Shared>
__device__ __forceinline__ void fence_handoff(unsigned* out, const unsigned* in, unsigned* global_flags,
    unsigned* global_values, unsigned rounds) {
    __shared__ unsigned local_flags[64], local_values[64];
    unsigned lane = threadIdx.x, pair = lane & 31u, id = blockIdx.x * 32 + pair;
    auto* flags = Shared ? local_flags : global_flags + blockIdx.x * 64;
    volatile unsigned* values = Shared ? local_values : global_values + blockIdx.x * 64;
    flags[lane] = 0;
    __syncthreads();
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned generation = round + 1;
        if (lane < 32) {
            unsigned value = in[id] ^ (round * 0x10001u + id);
            values[pair * 2] = value; values[pair * 2 + 1] = value + 0x12345u;
            publication_fence<Device>();
            relaxed_word<Device>(flags + pair, generation);
            while (relaxed_word<Device>(flags + 32 + pair) != generation) {}
            publication_fence<Device>();
        } else {
            while (relaxed_word<Device>(flags + pair) != generation) {}
            publication_fence<Device>();
            unsigned row = (round * gridDim.x * 32 + id) * 3;
            out[row] = values[pair * 2]; out[row + 1] = values[pair * 2 + 1]; out[row + 2] = generation;
            publication_fence<Device>();
            relaxed_word<Device>(flags + 32 + pair, generation);
        }
    }
}
#define FENCE_KERNEL(Name, Device, Shared) \
extern "C" __global__ void fence_##Name(unsigned* out, const unsigned* in, unsigned* flags, unsigned* values, unsigned rounds) { \
    fence_handoff<Device, Shared>(out, in, flags, values, rounds); }
FENCE_KERNEL(cta_global, false, false)
FENCE_KERNEL(cta_shared, false, true)
FENCE_KERNEL(gpu_global, true, false)
