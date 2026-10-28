extern "C" __global__ void mixed_width_access(unsigned char* out, const unsigned char* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out += i * 16;
    in += i * 16;
    out[1] = in[0] ^ 0xa5u;
    *reinterpret_cast<unsigned short*>(out + 2) = *reinterpret_cast<const unsigned short*>(in + 2) ^ 0x5aa5u;
    *reinterpret_cast<unsigned*>(out + 4) = *reinterpret_cast<const unsigned*>(in + 4) + 0x1020305u;
    *reinterpret_cast<unsigned long long*>(out + 8) =
        *reinterpret_cast<const unsigned long long*>(in + 8) ^ 0x0123456789abcdefull;
}

extern "C" __global__ void neighbor_byte_access(unsigned char* out, const unsigned char* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = in[i] ^ 0x5au;
}
