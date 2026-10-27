extern "C" __global__ void record_builtins(unsigned* out, unsigned count) {
    unsigned local = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    unsigned block = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    unsigned i = block * (blockDim.x * blockDim.y * blockDim.z) + local;
    if (i >= count) return;
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    out += i * 16;
    out[0] = threadIdx.x; out[1] = threadIdx.y; out[2] = threadIdx.z;
    out[3] = blockDim.x; out[4] = blockDim.y; out[5] = blockDim.z;
    out[6] = blockIdx.x; out[7] = blockIdx.y; out[8] = blockIdx.z;
    out[9] = gridDim.x; out[10] = gridDim.y; out[11] = gridDim.z;
    out[12] = lane; out[13] = warpSize; out[14] = local; out[15] = block;
}
