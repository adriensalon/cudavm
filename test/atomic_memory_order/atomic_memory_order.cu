#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
template<bool Device>
__device__ __forceinline__ unsigned acquire_word(const unsigned* pointer) {
    unsigned result;
    if constexpr (Device) asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    else asm volatile("ld.acquire.cta.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    return result;
}
template<bool Device>
__device__ __forceinline__ void release_word(unsigned* pointer, unsigned value) {
    if constexpr (Device) asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
    else asm volatile("st.release.cta.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
}
template<bool Device>
__device__ __forceinline__ unsigned ordered_add(unsigned* pointer, unsigned value, unsigned mode) {
    unsigned previous = 0;
#define ORDERED_ADD(Mode, Order) case Mode: \
    if constexpr (Device) asm volatile("atom.add." #Order ".gpu.u32 %0, [%1], %2;" : "=r"(previous) : "l"(pointer), "r"(value) : "memory"); \
    else asm volatile("atom.add." #Order ".cta.u32 %0, [%1], %2;" : "=r"(previous) : "l"(pointer), "r"(value) : "memory"); break;
    switch (mode) { ORDERED_ADD(0, relaxed) ORDERED_ADD(1, acquire) ORDERED_ADD(2, release) ORDERED_ADD(3, acq_rel) }
#undef ORDERED_ADD
    return previous;
}
template<bool Device, bool Shared>
__device__ __forceinline__ void scalar_orders(unsigned* out, unsigned* storage, unsigned seed) {
    __shared__ unsigned local;
    unsigned* pointer = Shared ? &local : storage + blockIdx.x;
    release_word<Device>(pointer, seed);
    for (unsigned mode = 0; mode < 4; ++mode) {
        unsigned addend = (mode & 1) ? 0xffffffffu : 0x80000001u;
        out[blockIdx.x * 12 + mode * 2] = ordered_add<Device>(pointer, addend, mode);
        out[blockIdx.x * 12 + mode * 2 + 1] = acquire_word<Device>(pointer);
    }
}
template<bool Device, bool Shared>
__device__ __forceinline__ void publish_orders(unsigned* out, const unsigned* in, unsigned* global_flags,
    unsigned* global_values, unsigned rounds) {
    __shared__ unsigned local_flags[64], local_values[32];
    unsigned lane = threadIdx.x, pair = lane & 31u, id = blockIdx.x * 32 + pair;
    auto* flags = Shared ? local_flags : global_flags + blockIdx.x * 64;
    auto* values = Shared ? local_values : global_values + blockIdx.x * 32;
    flags[lane] = 0;
    __syncthreads();
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned generation = round + 1;
        if (lane < 32) {
            values[pair] = in[id] ^ (round * 0x10001u + id);
            release_word<Device>(flags + pair, generation);
            while (acquire_word<Device>(flags + 32 + pair) != generation) {}
        } else {
            while (acquire_word<Device>(flags + pair) != generation) {}
            unsigned row = (round * gridDim.x * 32 + id) * 2;
            out[row] = values[pair]; out[row + 1] = generation;
            release_word<Device>(flags + 32 + pair, generation);
        }
    }
}
#define ORDER_KERNELS(Name, Device, Shared) \
extern "C" __global__ void orders_##Name(unsigned* out, unsigned* storage, unsigned seed) { scalar_orders<Device, Shared>(out, storage, seed); } \
extern "C" __global__ void publish_##Name(unsigned* out, const unsigned* in, unsigned* flags, unsigned* values, unsigned rounds) { \
    publish_orders<Device, Shared>(out, in, flags, values, rounds); }
ORDER_KERNELS(cta_global, false, false)
ORDER_KERNELS(cta_shared, false, true)
ORDER_KERNELS(gpu_global, true, false)
