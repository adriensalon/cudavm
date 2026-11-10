#include <cuda_runtime.h>
extern __device__ unsigned transform_word(unsigned, unsigned);
__global__ void separate_transform(unsigned* output, const unsigned* input, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = transform_word(input[i], i);
}
extern "C" cudaError_t launch_separate(unsigned* output, const unsigned* input, unsigned count, cudaStream_t stream) {
    separate_transform<<<3, 32, 0, stream>>>(output, input, count);
    return cudaGetLastError();
}
