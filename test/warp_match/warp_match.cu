template<class T> __device__ __forceinline__ void match_value(unsigned* out, unsigned mask, T value) {
    int predicate = -7;
    out[0] = __match_any_sync(mask, value);
    out[1] = __match_all_sync(mask, value, &predicate);
    out[2] = unsigned(predicate);
}
extern "C" __global__ void masked_match(unsigned* out, const unsigned long long* in, unsigned mask) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31u;
    if (!(mask & (1u << lane))) return;
    unsigned long long value = in[i];
    unsigned low = unsigned(value);
    out += i * 18;
    match_value(out, mask, __builtin_bit_cast(int, low));
    match_value(out + 3, mask, low);
    match_value(out + 6, mask, __builtin_bit_cast(long long, value));
    match_value(out + 9, mask, value);
    match_value(out + 12, mask, __builtin_bit_cast(float, low));
    match_value(out + 15, mask, __builtin_bit_cast(double, value));
}
extern "C" __global__ void retiring_match(unsigned* out, const unsigned long long* in, unsigned mask, unsigned keep) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31u;
    if (lane >= keep || !(mask & (1u << lane))) return;
    unsigned long long value = in[i];
    #pragma unroll 1
    for (unsigned phase = 0; phase < 4; ++phase) {
        if (lane < phase) return;
        match_value(out + i * 24 + phase * 6, mask, unsigned(value));
        match_value(out + i * 24 + phase * 6 + 3, mask, value);
    }
}
