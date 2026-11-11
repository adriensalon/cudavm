#ifndef PERF_STAGES
#define PERF_STAGES 8
#endif

template<unsigned stage> __device__ unsigned perf_compile_mix(unsigned value) {
    if constexpr (stage != 0) value = perf_compile_mix<stage - 1>(value);
    return (value * 1664525u + 1013904223u) ^ stage;
}

extern "C" __global__ void perf_compile_0(unsigned* output, unsigned input) {
    output[threadIdx.x] = perf_compile_mix<PERF_STAGES>(input + threadIdx.x);
}

#ifdef PERF_LARGE
#define PERF_COMPILE_KERNEL(index) \
extern "C" __global__ void perf_compile_##index(unsigned* output, unsigned input) { \
    output[threadIdx.x] = perf_compile_mix<PERF_STAGES + index>(input + threadIdx.x + index); \
}
PERF_COMPILE_KERNEL(1)
PERF_COMPILE_KERNEL(2)
PERF_COMPILE_KERNEL(3)
PERF_COMPILE_KERNEL(4)
PERF_COMPILE_KERNEL(5)
PERF_COMPILE_KERNEL(6)
PERF_COMPILE_KERNEL(7)
#endif
