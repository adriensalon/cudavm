extern "C" __device__ __noinline__ unsigned frame_leaf(unsigned seed) {
    __align__(16) volatile unsigned frame[3];
    frame[0] = seed; frame[1] = seed ^ 0x89abcdefu; frame[2] = seed + 0x7654321u;
    return (frame[0] * 3u + frame[1]) ^ frame[2];
}
extern "C" __device__ __noinline__ unsigned frame_middle(unsigned seed, unsigned rounds) {
    __align__(16) volatile unsigned frame[5];
    for (unsigned slot = 0; slot < 5; ++slot) frame[slot] = seed + slot * 0x1020304u;
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned slot = (seed + round) % 5;
        frame[slot] ^= frame_leaf(seed + round * 7u) + round;
    }
    unsigned hash = 0;
    for (unsigned slot = 0; slot < 5; ++slot) hash = hash * 33u + frame[slot];
    return hash;
}
extern "C" __global__ void frame_results(unsigned* out, const unsigned* in, unsigned count, unsigned rounds) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned seed = in[i];
    __align__(16) volatile unsigned caller[3];
    caller[0] = seed + 17u; caller[1] = seed ^ 0x13579bdfu; caller[2] = i + 0xeca86420u;
    unsigned first = frame_middle(seed, rounds + i % 3u);
    unsigned second = frame_middle(seed ^ first, rounds + 2u);
    out += i * 5;
    out[0] = first; out[1] = second;
    out[2] = caller[0]; out[3] = caller[1]; out[4] = caller[2];
}
