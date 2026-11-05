#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
template<bool Device, bool Acquire>
__device__ __forceinline__ unsigned litmus_load(const unsigned* pointer) {
    unsigned result;
    if constexpr (Device && Acquire) asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    else if constexpr (Device) asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    else if constexpr (Acquire) asm volatile("ld.acquire.cta.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    else asm volatile("ld.relaxed.cta.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    return result;
}
template<bool Device, bool Release>
__device__ __forceinline__ void litmus_store(unsigned* pointer, unsigned value) {
    if constexpr (Device && Release) asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
    else if constexpr (Device) asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
    else if constexpr (Release) asm volatile("st.release.cta.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
    else asm volatile("st.relaxed.cta.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
}
template<bool Device>
__device__ __forceinline__ void litmus_fence() {
    if constexpr (Device) __threadfence(); else __threadfence_block();
}
template<bool Device, bool Shared, bool Fenced>
__device__ __forceinline__ void sample_mp(unsigned* out, const unsigned* in, unsigned* global_flags,
    unsigned* global_values, unsigned trials, unsigned forced) {
    __shared__ unsigned local_flags[32], local_values[32];
    unsigned lane = threadIdx.x, pair = lane & 31u, id = blockIdx.x * 32 + pair;
    auto* flags = Shared ? local_flags : global_flags + blockIdx.x * 32;
    auto* values = Shared ? local_values : global_values + blockIdx.x * 32;
    for (unsigned trial = 0; trial < trials; ++trial) {
        unsigned initial = in[id] + trial * 17u, published = initial ^ 0xa5a55a5au;
        if (lane < 32) {
            litmus_store<Device, false>(values + pair, initial);
            litmus_store<Device, false>(flags + pair, 0);
        }
        __syncthreads();
        if (lane < 32) {
            litmus_store<Device, false>(values + pair, published);
            if constexpr (Fenced) litmus_fence<Device>();
            litmus_store<Device, !Fenced>(flags + pair, 1);
        }
        // This optional control is uniform and supplies a deterministic oracle
        // for the same addresses. The actual litmus mode has no barrier here.
        if (forced) __syncthreads();
        if (lane >= 32) {
            unsigned observed = litmus_load<Device, !Fenced>(flags + pair);
            if constexpr (Fenced) litmus_fence<Device>();
            unsigned value = litmus_load<Device, false>(values + pair);
            unsigned row = (trial * gridDim.x * 32 + id) * 4;
            out[row] = observed; out[row + 1] = value;
        }
        __syncthreads();
    }
}
#define LITMUS_KERNELS(Name, Device, Shared) \
extern "C" __global__ void mp_ordered_##Name(unsigned* out, const unsigned* in, unsigned* flags, unsigned* values, unsigned trials, unsigned forced) { \
    sample_mp<Device, Shared, false>(out, in, flags, values, trials, forced); } \
extern "C" __global__ void mp_fenced_##Name(unsigned* out, const unsigned* in, unsigned* flags, unsigned* values, unsigned trials, unsigned forced) { \
    sample_mp<Device, Shared, true>(out, in, flags, values, trials, forced); }
LITMUS_KERNELS(cta_global, false, false)
LITMUS_KERNELS(cta_shared, false, true)
LITMUS_KERNELS(gpu_global, true, false)
