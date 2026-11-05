#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
#include <cuda_pipeline.h>

extern "C" __global__ void pipeline_batches(unsigned* out, const unsigned* in, unsigned rounds) {
    __shared__ unsigned stages[4 * 64];
    unsigned lane = threadIdx.x, threads = blockDim.x, id = blockIdx.x * threads + lane;
    unsigned total = gridDim.x * threads;
    nvcuda::experimental::pipeline pipeline;
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned row = (round * total + id) * 16;
        for (unsigned stage = 0; stage < 4; ++stage) {
            auto* destination = stages + stage * 64 + lane;
            const auto* source = in + ((stage + round) % 4) * total + id;
            nvcuda::experimental::memcpy_async(nvcuda::experimental::segment<1>(destination),
                nvcuda::experimental::segment<1>(source), pipeline);
            out[row + 4 + stage] = unsigned(pipeline.commit());
        }
        // Only inspect stages that the corresponding wait has completed.
        pipeline.wait_prior<3>();
        out[row] = stages[lane];
        pipeline.wait_prior<2>();
        out[row + 1] = stages[64 + lane];
        pipeline.wait_prior<1>();
        out[row + 2] = stages[128 + lane];
        pipeline.wait_prior<0>();
        out[row + 3] = stages[192 + lane];
        __syncthreads();
        unsigned peer = (lane + round + 1) % threads;
        for (unsigned stage = 0; stage < 4; ++stage) out[row + 8 + stage] = stages[stage * 64 + peer];
        __syncthreads();
        // A fifth batch reuses stage zero only after every prior reader retires.
        nvcuda::experimental::memcpy_async(nvcuda::experimental::segment<1>(stages + lane),
            nvcuda::experimental::segment<1>(in + ((round + 3) % 4) * total + id), pipeline);
        pipeline.commit_and_wait();
        out[row + 12] = stages[lane];
        // Empty committed batches remain valid and preserve the local sequence.
        out[row + 13] = unsigned(pipeline.commit());
        pipeline.wait_prior<0>();
        __syncthreads();
    }
}
