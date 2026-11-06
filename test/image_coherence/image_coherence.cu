extern "C" __global__ void coherence_sample(cudaTextureObject_t object, float4* output) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 32) output[i] = tex2D<float4>(object, float(i % 8) + 0.5f, float(i / 8) + 0.5f);
}
extern "C" __global__ void coherence_store(cudaSurfaceObject_t object, unsigned seed) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 32) return;
    unsigned base = seed * 257 + (i % 8) * 7 + (i / 8) * 13;
    float4 value{float(base), float(base + 31), float(base + 62), float(base + 93)};
    surf2Dwrite(value, object, int(i % 8) * 16, int(i / 8));
}
