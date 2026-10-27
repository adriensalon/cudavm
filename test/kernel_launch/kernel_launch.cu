extern "C" __global__ void cubin_scalar(unsigned* out, const unsigned* in, unsigned add) {
    out[1] = in[1] + add;
}
extern "C" __global__ void cubin_branch(unsigned* out, const unsigned* in, unsigned threshold) {
    unsigned value = in[0];
    out[0] = value < threshold ? value * 3u : value + 5u;
}
extern "C" __global__ void cubin_indexed(unsigned* out, const unsigned* in, unsigned count) {
    unsigned lane = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    unsigned block = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    unsigned i = block * (blockDim.x * blockDim.y * blockDim.z) + lane;
    if (i < count) out[i] = in[i] + i + gridDim.x * gridDim.y * gridDim.z;
}
extern "C" __global__ void cubin_wide(unsigned* out, unsigned left, unsigned right, unsigned long long add) {
    unsigned long long value = static_cast<unsigned long long>(left) * right + add;
    out[0] = static_cast<unsigned>(value);
    out[1] = static_cast<unsigned>(value >> 32);
}
