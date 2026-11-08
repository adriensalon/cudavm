namespace cudavm {
__device__ inline unsigned math_norm_encode(unsigned long long significand, int scale) {
    if (!significand) return 0;
    const int top = 63 - __builtin_clzll(significand);
    int exponent = top + scale + 127;
    const unsigned shift = unsigned(exponent <= 0 ? -149 - scale : top - 23);
    const auto retained = significand >> shift;
    const auto discarded = significand & ((1ull << shift) - 1), halfway = 1ull << (shift - 1);
    auto rounded = retained + (discarded > halfway || (discarded == halfway && (retained & 1)));
    if (exponent <= 0) return unsigned(rounded);
    if (rounded == 0x1000000ull) { rounded >>= 1; ++exponent; }
    if (exponent >= 255) return 0x7f800000u;
    return (unsigned(exponent) << 23) | (unsigned(rounded) & 0x7fffffu);
}
struct math_norm_coordinate { unsigned significand; int exponent; };
__device__ inline math_norm_coordinate math_norm_decode(unsigned bits) {
    if (!bits) return {0, -150};
    const unsigned raw_exponent = bits >> 23;
    if (raw_exponent) return {(bits & 0x7fffffu) | 0x800000u, int(raw_exponent) - 127};
    const auto shift = __builtin_clz(bits) - 8;
    return {bits << shift, -126 - shift};
}
__device__ inline unsigned long long math_norm_square(math_norm_coordinate a, int maximum) {
    if (!a.significand) return 0;
    const auto shift = unsigned(2 * (maximum - a.exponent));
    return shift < 64 ? (static_cast<unsigned long long>(a.significand) * a.significand) >> shift : 0;
}
template<unsigned Dimensions> __device__ inline float math_float_norm(float x, float y, float z = 0, float w = 0) {
    const auto a = __builtin_bit_cast(unsigned, x) & 0x7fffffffu, b = __builtin_bit_cast(unsigned, y) & 0x7fffffffu;
    const auto c = Dimensions >= 3 ? __builtin_bit_cast(unsigned, z) & 0x7fffffffu : 0u;
    const auto d = Dimensions >= 4 ? __builtin_bit_cast(unsigned, w) & 0x7fffffffu : 0u;
    if (a == 0x7f800000u || b == 0x7f800000u || c == 0x7f800000u || d == 0x7f800000u) return __builtin_bit_cast(float, 0x7f800000u);
    if (a > 0x7f800000u || b > 0x7f800000u || c > 0x7f800000u || d > 0x7f800000u) return __builtin_bit_cast(float, 0x7fc00000u);
    const auto ax = math_norm_decode(a), bx = math_norm_decode(b), cx = math_norm_decode(c), dx = math_norm_decode(d);
    int maximum = ax.exponent > bx.exponent ? ax.exponent : bx.exponent;
    if (cx.exponent > maximum) maximum = cx.exponent;
    if (dx.exponent > maximum) maximum = dx.exponent;
    if (maximum == -150) return 0.0f;
    const auto sum = math_norm_square(ax, maximum) + math_norm_square(bx, maximum) + math_norm_square(cx, maximum) + math_norm_square(dx, maximum);
    // Up to four 48-bit squares fit in 50 bits. Fourteen extra radicand bits
    // retain seven root guard bits without a general floating-point helper.
    const auto radicand = sum << 14;
    unsigned long long root = 0, remainder = 0;
    #pragma clang loop unroll(disable)
    for (int pair = 31; pair >= 0; --pair) {
        remainder = (remainder << 2) | ((radicand >> (pair * 2)) & 3);
        const auto trial = (root << 2) | 1;
        root <<= 1;
        if (remainder >= trial) { remainder -= trial; ++root; }
    }
    return __builtin_bit_cast(float, math_norm_encode(root, maximum - 30));
}
}
extern "C" __device__ inline float __nv_hypotf(float x, float y) { return cudavm::math_float_norm<2>(x, y); }
extern "C" __device__ inline float __nv_norm3df(float x, float y, float z) { return cudavm::math_float_norm<3>(x, y, z); }
extern "C" __device__ inline float __nv_norm4df(float x, float y, float z, float w) { return cudavm::math_float_norm<4>(x, y, z, w); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_hypot(float x, float y) { return __nv_hypotf(x, y); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_norm3d(float x, float y, float z) { return __nv_norm3df(x, y, z); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_norm4d(float x, float y, float z, float w) { return __nv_norm4df(x, y, z, w); }

namespace cudavm {
struct math_double_norm_coordinate { unsigned long long significand; int exponent; };
__device__ inline math_double_norm_coordinate math_double_norm_decode(unsigned long long bits) {
    if (!bits) return {0, -1075};
    const auto raw_exponent = unsigned(bits >> 52);
    if (raw_exponent) return {(bits & 0xfffffffffffffull) | 0x10000000000000ull, int(raw_exponent) - 1023};
    const auto shift = __builtin_clzll(bits) - 11;
    return {bits << shift, -1022 - shift};
}
__device__ inline math_wide math_norm_wide_shift(math_wide value, unsigned shift) {
    if (!shift) return value;
    if (shift < 64) return {(value.low >> shift) | (value.high << (64 - shift)), value.high >> shift};
    return shift < 128 ? math_wide{value.high >> (shift - 64), 0} : math_wide{0, 0};
}
__device__ inline math_wide math_double_norm_square(math_double_norm_coordinate a, int maximum) {
    if (!a.significand) return {0, 0};
    return math_norm_wide_shift(math_wide_product(a.significand, a.significand), unsigned(2 * (maximum - a.exponent)));
}
template<unsigned Dimensions> __device__ inline double math_double_norm(double x, double y, double z = 0, double w = 0) {
    const auto a = math_double_word(x) & 0x7fffffffffffffffull, b = math_double_word(y) & 0x7fffffffffffffffull;
    const auto c = Dimensions >= 3 ? math_double_word(z) & 0x7fffffffffffffffull : 0ull;
    const auto d = Dimensions >= 4 ? math_double_word(w) & 0x7fffffffffffffffull : 0ull;
    if (a == 0x7ff0000000000000ull || b == 0x7ff0000000000000ull || c == 0x7ff0000000000000ull || d == 0x7ff0000000000000ull) return math_double_bits(0x7ff0000000000000ull);
    if (a > 0x7ff0000000000000ull || b > 0x7ff0000000000000ull || c > 0x7ff0000000000000ull || d > 0x7ff0000000000000ull) return math_double_bits(0x7ff8000000000000ull);
    const auto ax = math_double_norm_decode(a), bx = math_double_norm_decode(b), cx = math_double_norm_decode(c), dx = math_double_norm_decode(d);
    int maximum = ax.exponent > bx.exponent ? ax.exponent : bx.exponent;
    if (cx.exponent > maximum) maximum = cx.exponent;
    if (dx.exponent > maximum) maximum = dx.exponent;
    if (maximum == -1075) return 0.0;
    const auto sum = math_wide_add(math_wide_add(math_double_norm_square(ax, maximum), math_double_norm_square(bx, maximum)),
        math_wide_add(math_double_norm_square(cx, maximum), math_double_norm_square(dx, maximum)));
    // Four 106-bit squares and fourteen guard bits fit in a 122-bit radicand.
    const math_wide radicand{sum.low << 14, (sum.high << 14) | (sum.low >> 50)};
    unsigned long long root = 0, remainder = 0;
    #pragma clang loop unroll(disable)
    for (int pair = 63; pair >= 0; --pair) {
        const auto word = pair >= 32 ? radicand.high : radicand.low;
        remainder = (remainder << 2) | ((word >> ((pair & 31) * 2)) & 3);
        const auto trial = (root << 2) | 1;
        root <<= 1;
        if (remainder >= trial) { remainder -= trial; ++root; }
    }
    return math_wide_double({root, 0}, maximum - 59);
}
}
extern "C" __device__ inline double __nv_hypot(double x, double y) { return cudavm::math_double_norm<2>(x, y); }
extern "C" __device__ inline double __nv_norm3d(double x, double y, double z) { return cudavm::math_double_norm<3>(x, y, z); }
extern "C" __device__ inline double __nv_norm4d(double x, double y, double z, double w) { return cudavm::math_double_norm<4>(x, y, z, w); }
