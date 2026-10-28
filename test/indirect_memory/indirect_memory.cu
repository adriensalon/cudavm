extern "C" __global__ void follow_pointers(unsigned* out, const unsigned* const* const* roots,
                                           const unsigned* selection, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned* const* table = roots[(i >> 2) & 1u];
    unsigned selector = selection[i];
    const unsigned* data = table[selector & 3u];
    if (!data) return;
    unsigned long long address = reinterpret_cast<unsigned long long>(data);
    out[i * 3] = data[(selector >> 2) & 7u] + i * 0x101u;
    out[i * 3 + 1] = static_cast<unsigned>(address);
    out[i * 3 + 2] = static_cast<unsigned>(address >> 32);
}
