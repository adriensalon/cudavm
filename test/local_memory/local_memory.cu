extern "C" __global__ void local_arrays(unsigned* out, const unsigned* in,
    unsigned count, unsigned rounds) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned seed = in[i];
    // Volatile, dynamically addressed storage prevents scalar promotion.
    __align__(16) volatile unsigned values[19];
    values[0] = 0x13579bdfu ^ i;
    values[18] = 0xeca86420u ^ seed;
    for (unsigned slot = 0; slot < 17; ++slot)
        values[slot + 1] = (seed + slot * 0x10203u) ^ (i * 17u + slot);
    for (unsigned round = 0; round < rounds + i % 4; ++round) {
        unsigned slot = (seed + round * 5u) % 17;
        values[slot + 1] = values[slot + 1] ^ (values[(slot + 7) % 17 + 1] + round);
    }
    unsigned sum = 0, hash = 0;
    for (unsigned slot = 0; slot < 17; ++slot) {
        unsigned value = values[slot + 1];
        sum += value; hash = (hash * 33u) ^ value;
    }
    out += i * 5;
    out[0] = sum; out[1] = hash; out[2] = values[seed % 17 + 1];
    out[3] = values[0]; out[4] = values[18];
}
