// Independently compiled CUDA 12.4 / SM75 cooperative-kernel witnesses.
extern "C" __global__ void shared_exchange(unsigned* out, const unsigned* in, unsigned count) {
    __shared__ __align__(16) unsigned values[129];
    unsigned lane = threadIdx.x, i = blockIdx.x * blockDim.x + lane;
    unsigned initial = i < count ? in[i] : 0;
    // Keep a real token-controlled loop containing shared stores. Every lane
    // owns its word; the later CTA barrier supplies cross-warp ordering.
    volatile unsigned* changing = values;
    #pragma unroll 1
    for (unsigned step = 0; step < (lane >> 4); ++step) {
        initial = initial * 3 + step;
        changing[lane + 1] = initial;
    }
    values[lane + 1] = initial;
    __syncthreads();
    unsigned peer = blockDim.x - 1 - lane;
    unsigned value = values[peer + 1];
    __syncthreads();
    values[lane + 1] = value * 3 + lane;
    __syncthreads();
    if (i < count) out[i] = values[peer + 1];
}

extern "C" __global__ void shared_transpose(unsigned* out, const unsigned* in, unsigned tiles) {
    __shared__ unsigned tile[8][10];
    unsigned x = threadIdx.x, y = threadIdx.y, base = blockIdx.x * 64;
    tile[y][x + 1] = in[base + y * 8 + x];
    __syncthreads();
    if (blockIdx.x < tiles) out[base + y * 8 + x] = tile[x][y + 1];
}

extern "C" __global__ void shared_reduce(unsigned* out, const unsigned* in, unsigned count) {
    __shared__ unsigned values[128];
    unsigned lane = threadIdx.x, i = blockIdx.x * blockDim.x + lane;
    values[lane] = i < count ? in[i] : 0;
    __syncthreads();
    #pragma unroll 1
    for (unsigned step = blockDim.x / 2; step; step /= 2) {
        if (lane < step) values[lane] += values[lane + step];
        __syncthreads();
    }
    if (lane == 0) out[blockIdx.x] = values[0];
}
