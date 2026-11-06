extern "C" __global__ void filter_texture(cudaTextureObject_t object, float4* output, const float2* coordinates, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = tex2D<float4>(object, coordinates[i].x, coordinates[i].y);
}
