static_assert(sizeof(clock_t) == sizeof(long), "NVRTC predefined clock_t ABI");
extern "C" __global__ void timing_samples(unsigned* output, const unsigned* input, unsigned count,
    unsigned sleep, unsigned nanoseconds) {
    const unsigned flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned i = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + flat;
    if (i >= count) return;
    unsigned value = input[i];
    for (unsigned phase = 0; phase < 4; ++phase) {
        if (sleep) __nanosleep(nanoseconds);
        const clock_t narrow = clock();
        const unsigned long long wide = static_cast<unsigned long long>(clock64());
        #pragma unroll 1
        for (unsigned step = 0; step < 19; ++step) value = value * 1664525u + 1013904223u;
        output[i * 16 + phase * 4] = static_cast<unsigned>(narrow);
        output[i * 16 + phase * 4 + 1] = static_cast<unsigned>(wide);
        output[i * 16 + phase * 4 + 2] = static_cast<unsigned>(wide >> 32);
        output[i * 16 + phase * 4 + 3] = value;
    }
}
