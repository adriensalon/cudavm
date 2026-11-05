#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
static_assert(__CUDA_ARCH__ >= 800, "warp reduction profile requires SM80 source");
__device__ inline void reduce_record(unsigned* output, unsigned mask, unsigned value) {
    int signed_value = __builtin_bit_cast(int, value);
    output[0] = __reduce_add_sync(mask, value);
    output[1] = __reduce_min_sync(mask, value);
    output[2] = __reduce_max_sync(mask, value);
    output[3] = unsigned(__reduce_add_sync(mask, signed_value));
    output[4] = unsigned(__reduce_min_sync(mask, signed_value));
    output[5] = unsigned(__reduce_max_sync(mask, signed_value));
    output[6] = __reduce_and_sync(mask, value);
    output[7] = __reduce_or_sync(mask, value);
    output[8] = __reduce_xor_sync(mask, value);
}
extern "C" __global__ void masked_reductions(unsigned* output, const unsigned* input, unsigned mask) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31u;
    if (!(mask & (1u << lane))) return;
    reduce_record(output + id * 9, mask, input[id]);
}
extern "C" __global__ void retiring_reductions(unsigned* output, const unsigned* input, unsigned mask) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31u;
    if (!(mask & (1u << lane))) return;
    #pragma unroll 1
    for (unsigned phase = 0; phase < 4; ++phase) {
        if (lane < phase * 3) return;
        reduce_record(output + id * 36 + phase * 9, mask, input[id] ^ (phase * 0x10010001u));
    }
}
