extern "C" __global__ void floating_global_add(unsigned* out, float* state, unsigned count, float delta) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = __float_as_uint(atomicAdd(state, delta));
        atomicAdd(state + 1, delta);
    }
}

extern "C" __global__ void floating_global_double_add(unsigned long long* out, double* state,
                                                     unsigned count, double delta) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = __double_as_longlong(atomicAdd(state, delta));
        atomicAdd(state + 1, delta);
    }
}

extern "C" __global__ void floating_shared_add(unsigned* out, unsigned* totals, unsigned count,
                                              float seed, float delta) {
    __shared__ float counters[2];
    if (threadIdx.x == 0) counters[0] = counters[1] = seed;
    __syncthreads();
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = __float_as_uint(atomicAdd(counters, delta));
        atomicAdd(counters + 1, delta);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        totals[blockIdx.x * 2] = __float_as_uint(counters[0]);
        totals[blockIdx.x * 2 + 1] = __float_as_uint(counters[1]);
    }
}

extern "C" __global__ void floating_exchange(unsigned* out, float* state, unsigned replacement) {
    out[0] = __float_as_uint(atomicExch(state, __uint_as_float(replacement)));
}
