extern "C" __global__ void count_reverse_bits(unsigned* out, const unsigned long long* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned long long wide = in[i];
    unsigned narrow = static_cast<unsigned>(wide);
    unsigned long long reversed = __brevll(wide);
    out += i * 9;
    out[0] = __popc(narrow);
    out[1] = __clz(narrow);
    out[2] = __brev(narrow);
    out[3] = __ffs(static_cast<int>(narrow));
    out[4] = __popcll(wide);
    out[5] = __clzll(wide);
    out[6] = static_cast<unsigned>(reversed);
    out[7] = static_cast<unsigned>(reversed >> 32);
    out[8] = __ffsll(static_cast<long long>(wide));
}
