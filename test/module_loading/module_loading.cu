extern "C" { __device__ unsigned module_seed = 13; __constant__ unsigned module_bias = 29; }
extern "C" __global__ void module_alpha(unsigned* out, const unsigned* in, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = in[i] * 3u + module_seed + i;
}
extern "C" __global__ void module_beta(unsigned* out, const unsigned* in, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (in[i] ^ module_bias) + module_seed;
}
