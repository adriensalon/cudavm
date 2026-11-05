#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
#define _CUDA_PIPELINE_STATIC_QUALIFIER static __device__ __forceinline__
#include <cuda_pipeline_primitives.h>

template<unsigned Bytes, unsigned Fill>
__device__ __forceinline__ void copy_records(unsigned* out, const unsigned* in,
    unsigned rounds, unsigned count, unsigned seed) {
    __shared__ __align__(16) unsigned records[64 * 16];
    unsigned lane = threadIdx.x, threads = blockDim.x, id = blockIdx.x * threads + lane;
    unsigned total = gridDim.x * threads;
    for (unsigned round = 0; round < rounds; ++round) {
        for (unsigned word = 0; word < 16; ++word) records[lane * 16 + word] = 0xa1b2c3d4u;
        unsigned bytes = Bytes, zfill = Fill;
        unsigned offset = ((id + round) & 1) * bytes;
        if (id < count) __pipeline_memcpy_async(
            reinterpret_cast<char*>(records + lane * 16) + 16 + offset,
            reinterpret_cast<const char*>(in + id * 16) + offset, Bytes, Fill);
        __pipeline_commit();
        unsigned work = (in[id * 16] ^ seed) * 33u + round;
        __pipeline_wait_prior(0);
        __syncthreads();
        unsigned peer = (lane + round + 1) % threads;
        unsigned row = (round * total + id) * 20;
        for (unsigned word = 0; word < 16; ++word) out[row + word] = records[peer * 16 + word];
        out[row + 16] = bytes; out[row + 17] = zfill; out[row + 18] = offset; out[row + 19] = work;
        __syncthreads();
    }
}
#define COPY_KERNEL(Bytes, Fill) \
extern "C" __global__ void async_copy_##Bytes##_##Fill(unsigned* out, const unsigned* in, \
    unsigned rounds, unsigned count, unsigned seed) { copy_records<Bytes, Fill>(out, in, rounds, count, seed); }
#define COPY_BASE(Bytes) COPY_KERNEL(Bytes, 0) COPY_KERNEL(Bytes, 1) COPY_KERNEL(Bytes, 2) COPY_KERNEL(Bytes, 3) COPY_KERNEL(Bytes, 4)
#define COPY_MORE(Bytes) COPY_KERNEL(Bytes, 5) COPY_KERNEL(Bytes, 6) COPY_KERNEL(Bytes, 7) COPY_KERNEL(Bytes, 8)
COPY_BASE(4)
COPY_BASE(8) COPY_MORE(8)
COPY_BASE(16) COPY_MORE(16)
COPY_KERNEL(16, 9) COPY_KERNEL(16, 10) COPY_KERNEL(16, 11) COPY_KERNEL(16, 12)
COPY_KERNEL(16, 13) COPY_KERNEL(16, 14) COPY_KERNEL(16, 15) COPY_KERNEL(16, 16)
