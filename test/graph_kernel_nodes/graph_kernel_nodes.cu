extern "C" __global__ void kernel_node_write(unsigned* output, unsigned count, unsigned seed) {
    extern __shared__ unsigned values[];
    unsigned lane = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    unsigned block = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    unsigned i = block * threads + lane;
    values[lane] = seed + threadIdx.x * 3u + threadIdx.y * 5u + threadIdx.z * 7u
        + blockIdx.x * 11u + blockIdx.y * 13u + blockIdx.z * 17u
        + blockDim.x * 19u + blockDim.y * 23u + blockDim.z * 29u;
    __syncthreads();
    if (i < count) output[i] = values[threads - 1 - lane];
}
