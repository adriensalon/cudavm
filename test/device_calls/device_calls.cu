using word64 = unsigned long long;
extern "C" __device__ __noinline__ word64 mix_words(word64 a, word64 b, unsigned shift) {
    unsigned n = shift & 63u;
    return ((a << n) | (a >> ((0u - n) & 63u))) + b;
}
extern "C" __device__ __noinline__ unsigned fold_words(unsigned a, unsigned b) {
    unsigned value = a ^ (b * 33u);
    if (value & 0x80000000u) return value - 0x13579bdfu;
    return value + 0x89abcdefu;
}
extern "C" __device__ __noinline__ word64 nested_words(word64 a, word64 b, unsigned rounds) {
    for (unsigned round = 0; round < rounds; ++round) {
        a = mix_words(a, b + round, round * 7u);
        unsigned folded = fold_words(unsigned(a), unsigned(a >> 32));
        a ^= (word64(folded) << 32) | folded;
    }
    return a;
}
extern "C" __global__ void call_results(unsigned* out, const word64* in, unsigned count, unsigned rounds) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    word64 a = in[i * 2], b = in[i * 2 + 1];
    word64 mixed = mix_words(a, b, i);
    unsigned folded = fold_words(unsigned(mixed), unsigned(mixed >> 32));
    word64 nested = nested_words(a, b, rounds + (i & 3u));
    unsigned branch;
    if (i & 1u) branch = fold_words(unsigned(a), unsigned(b));
    else branch = fold_words(unsigned(b >> 32), unsigned(a >> 32));
    out += i * 6;
    out[0] = unsigned(mixed); out[1] = unsigned(mixed >> 32); out[2] = folded;
    out[3] = unsigned(nested); out[4] = unsigned(nested >> 32); out[5] = branch;
}
