extern "C" __global__ void sample_mipmap(cudaTextureObject_t object, float4* output, const float4* samples, unsigned count, unsigned gradients) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    auto sample = samples[i];
    if (gradients) output[i] = tex2DGrad<float4>(object, sample.x, sample.y, {sample.z, 0}, {0, sample.z});
    else output[i] = tex2DLod<float4>(object, sample.x, sample.y, sample.z);
}
