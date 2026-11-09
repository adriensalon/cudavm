extern "C" __device__ unsigned lto_transform(unsigned value);
extern "C" __global__ void lto_results(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = lto_transform(input[i]);
}
