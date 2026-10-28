extern "C" __global__ void integer_values(unsigned long long* out,
    const unsigned long long* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned long long a = in[i * 2], b = in[i * 2 + 1];
    unsigned x = unsigned(a), y = unsigned(b);
    int sx = int(x), sy = int(y);
    unsigned divisor = y == 0 || (x == 0x80000000u && y == 0xffffffffu) ? 1 : y;
    int signed_divisor = int(divisor);
    out += i * 12;
    out[0] = x / divisor;
    out[1] = x % divisor;
    out[2] = unsigned(sx / signed_divisor);
    out[3] = unsigned(sx % signed_divisor);
    out[4] = unsigned(__mulhi(sx, sy));
    out[5] = __umulhi(x, y);
    out[6] = static_cast<unsigned long long>(x) * y;
    out[7] = static_cast<long long>(sx) * sy;
    out[8] = a + b;
    out[9] = a - b;
    out[10] = __umul64hi(a, b);
    out[11] = __mul64hi(static_cast<long long>(a), static_cast<long long>(b));
}
