// Independently compiled CUDA 12.4 / SM75 dynamic-storage and atomic witnesses.
extern "C" __global__ void dynamic_reduce(unsigned* out, const unsigned* in, unsigned count) {
    extern __shared__ unsigned values[];
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

extern "C" __global__ void mixed_exchange(unsigned* out, const unsigned* in, unsigned count) {
    __shared__ unsigned fixed[3];
    extern __shared__ __align__(16) unsigned values[];
    unsigned lane = threadIdx.x, i = blockIdx.x * blockDim.x + lane;
    if (lane == 0) fixed[2] = blockIdx.x + 7;
    values[lane] = i < count ? in[i] : 0;
    __syncthreads();
    unsigned value = values[blockDim.x - 1 - lane] + fixed[2];
    if (i < count) out[i] = value;
}

extern "C" __global__ void shared_histogram(unsigned* out, const unsigned* in, unsigned count) {
    extern __shared__ unsigned bins[];
    unsigned lane = threadIdx.x, i = blockIdx.x * blockDim.x + lane;
    if (lane < 4) { bins[lane] = 0; bins[lane + 4] = 0; }
    __syncthreads();
    unsigned value = i < count ? in[i] : 0;
    // Zero contributions preserve partial-input behavior without divergent tokens.
    unsigned old = atomicAdd(bins + (value & 3), unsigned(i < count));
    atomicAdd(bins + 4 + (value & 3), unsigned(i < count));
    out[gridDim.x * 8 + i] = old;
    __syncthreads();
    if (lane < 4) { out[blockIdx.x * 8 + lane] = bins[lane]; out[blockIdx.x * 8 + lane + 4] = bins[lane + 4]; }
}

extern "C" __global__ void global_add(unsigned* out, unsigned* state, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned delta = unsigned(i < count);
    unsigned old = atomicAdd(state + (i & 1), delta);
    out[i] = old;
    atomicAdd(state + 2 + (i & 1), delta);
}

extern "C" __global__ void global_cas(unsigned* out, unsigned* state, unsigned replacement) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = atomicCAS(state, 0u, replacement);
    atomicCAS(state + 1, 0u, replacement);
}

extern "C" __global__ void shared_cas(unsigned* out, unsigned replacement, unsigned unused) {
    extern __shared__ unsigned state[];
    unsigned lane = threadIdx.x;
    if (lane < 2) state[lane] = 0;
    __syncthreads();
    unsigned old = atomicCAS(state, 0u, replacement);
    atomicCAS(state + 1, 0u, replacement);
    out[blockIdx.x * blockDim.x + lane] = old;
    __syncthreads();
    if (lane == 0) out[gridDim.x * blockDim.x + blockIdx.x] = state[1];
}
