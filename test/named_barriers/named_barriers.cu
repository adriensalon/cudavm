#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
__device__ inline void named_sync(unsigned id, unsigned count) {
    asm volatile("barrier.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ inline void named_arrive(unsigned id, unsigned count) {
    asm volatile("barrier.arrive %0, %1;" :: "r"(id), "r"(count) : "memory");
}
__device__ inline unsigned named_popc(unsigned id, unsigned count, unsigned truth) {
    unsigned result;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %3, 0; barrier.red.popc.u32 %0, %1, %2, p; }"
        : "=r"(result) : "r"(id), "r"(count), "r"(truth) : "memory");
    return result;
}
__device__ inline unsigned named_and(unsigned id, unsigned count, unsigned truth) {
    unsigned result;
    asm volatile("{ .reg .pred p, q; setp.ne.u32 p, %3, 0; barrier.red.and.pred q, %1, %2, p; selp.u32 %0, 1, 0, q; }"
        : "=r"(result) : "r"(id), "r"(count), "r"(truth) : "memory");
    return result;
}
__device__ inline unsigned named_or(unsigned id, unsigned count, unsigned truth) {
    unsigned result;
    asm volatile("{ .reg .pred p, q; setp.ne.u32 p, %3, 0; barrier.red.or.pred q, %1, %2, p; selp.u32 %0, 1, 0, q; }"
        : "=r"(result) : "r"(id), "r"(count), "r"(truth) : "memory");
    return result;
}
extern "C" __global__ void named_whole(unsigned* out, const unsigned* in, unsigned* scratch,
    unsigned rounds, unsigned id, unsigned pattern) {
    __shared__ unsigned values[128];
    unsigned lane = threadIdx.x, threads = blockDim.x, i = blockIdx.x * threads + lane;
    unsigned shared_sum = 0, global_sum = 0;
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned value = in[i] ^ (round * 17u + i);
        values[lane] = value; scratch[i] = value + 0x12345u;
        asm volatile("barrier.sync %0;" :: "r"(id) : "memory");
        unsigned peer = (lane + round + 1) % threads;
        shared_sum += values[peer]; global_sum += scratch[blockIdx.x * threads + peer];
        asm volatile("barrier.sync %0;" :: "r"(id) : "memory");
    }
    out[i * 5] = shared_sum; out[i * 5 + 1] = global_sum;
    out[i * 5 + 2] = 0; out[i * 5 + 3] = 0; out[i * 5 + 4] = 0;
}
extern "C" __global__ void named_groups(unsigned* out, const unsigned* in, unsigned* scratch,
    unsigned rounds, unsigned ignored, unsigned pattern) {
    __shared__ unsigned values[128];
    unsigned lane = threadIdx.x, i = blockIdx.x * 128 + lane, base = lane & ~63u;
    unsigned id = base == 0 ? 1u : 15u;
    unsigned sums[5] = {};
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned value = in[i] ^ (round * 17u + i);
        values[lane] = value; scratch[i] = value + 0x12345u;
        bool truth = pattern == 0 ? false : pattern == 1 ? true
            : pattern == 2 ? (lane & 63u) == round % 64 : (in[i] + round) % 4 != 0;
        unsigned predicate = truth ? 0x80000000u : 0;
        sums[2] += named_popc(id, 64, predicate);
        sums[3] += named_and(id, 64, predicate);
        sums[4] += named_or(id, 64, predicate);
        unsigned peer = base + ((lane + round + 1) & 63u);
        sums[0] += values[peer]; sums[1] += scratch[blockIdx.x * 128 + peer];
        named_sync(id, 64);
    }
    for (unsigned field = 0; field < 5; ++field) out[i * 5 + field] = sums[field];
}
extern "C" __global__ void named_handoff(unsigned* out, const unsigned* in, unsigned* scratch,
    unsigned rounds, unsigned ignored, unsigned pattern) {
    __shared__ unsigned values[128];
    unsigned lane = threadIdx.x, i = blockIdx.x * 128 + lane;
    unsigned shared_sum = 0, global_sum = 0;
    for (unsigned round = 0; round < rounds; ++round) {
        if (lane < 32) {
            unsigned value = in[i] ^ (round * 17u + i);
            values[lane] = value; scratch[i] = value + 0x12345u;
            named_arrive(1, 64); named_sync(2, 64); named_sync(3, 64);
        } else if (lane < 64) {
            named_sync(2, 64); named_sync(1, 64);
            unsigned peer = (lane + round + 1) & 31u;
            shared_sum += values[peer]; global_sum += scratch[blockIdx.x * 128 + peer];
            named_arrive(3, 64);
        } else {
            unsigned value = in[i] ^ (round * 17u + i);
            values[lane] = value; scratch[i] = value + 0x12345u;
            named_sync(15, 64);
            unsigned peer = 64 + ((lane + round + 1) & 63u);
            shared_sum += values[peer]; global_sum += scratch[blockIdx.x * 128 + peer];
            named_sync(15, 64);
        }
    }
    out[i * 5] = shared_sum; out[i * 5 + 1] = global_sum;
    out[i * 5 + 2] = 0; out[i * 5 + 3] = 0; out[i * 5 + 4] = 0;
}
