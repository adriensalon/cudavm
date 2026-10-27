extern "C" __global__ void branch_loop(unsigned* out, const unsigned* in, unsigned count, unsigned seed) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned value = in[i];
    if (value == 0) return;
    unsigned state = value ^ seed;
    unsigned step = 0;
    for (; step < (value & 7u); ++step) {
        if (step == ((value >> 5) & 7u)) break;
        if ((step & 1u) == 0) state = state * 3u + step;
        else state = (state >> 1) ^ (value + step);
        if ((state & 3u) == 0) {
            state += 17u;
            continue;
        }
        state ^= 0x9e3779b9u;
    }
    switch ((value >> 3) & 3u) {
    case 0: state += seed; break;
    case 1: state ^= seed; break;
    case 2: state = (state << 3) | (state >> 29); break;
    default: state = ~state; break;
    }
    out[i * 2] = state;
    out[i * 2 + 1] = step;
}
