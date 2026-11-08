// Q63 series keep the bounded reductions independent of general binary64
// arithmetic helpers. Wide integer products retain ten fractional guard bits.
namespace cudavm {
struct math_wide { unsigned long long low, high; };
__device__ inline double math_double_bits(unsigned long long bits) { return __builtin_bit_cast(double, bits); }
__device__ inline unsigned long long math_double_word(double x) { return __builtin_bit_cast(unsigned long long, x); }
__device__ inline unsigned long long math_high_product(unsigned long long a, unsigned long long b) {
    return join(cvMulHi64(split(a), split(b)));
}
__device__ inline math_wide math_wide_product(unsigned long long a, unsigned long long b) {
    return {a * b, math_high_product(a, b)};
}
__device__ inline math_wide math_wide_add(math_wide a, math_wide b) {
    const auto low = a.low + b.low;
    return {low, a.high + b.high + (low < a.low)};
}
__device__ inline math_wide math_wide_subtract(math_wide a, math_wide b) {
    return {a.low - b.low, a.high - b.high - (a.low < b.low)};
}
__device__ inline bool math_wide_less(math_wide a, math_wide b) {
    return a.high < b.high || (a.high == b.high && a.low < b.low);
}
__device__ inline math_wide math_wide_half(math_wide a) { return {(a.low >> 1) | (a.high << 63), a.high >> 1}; }
__device__ inline math_wide math_wide_fraction(math_wide a, unsigned long long fraction) {
    return math_wide_add(math_wide_product(a.high, fraction), {math_high_product(a.low, fraction), 0});
}
__device__ inline unsigned long long math_q63_product(unsigned long long a, unsigned long long b) {
    return (math_high_product(a, b) << 1) | ((a * b) >> 63);
}
// Numerator/denominator is in [0,2); generate fractional bits with bounded
// remainders instead of selecting a target's integer division implementation.
__device__ inline unsigned long long math_q63_quotient(unsigned long long numerator, unsigned long long denominator) {
    unsigned long long result = numerator >= denominator;
    if (result) numerator -= denominator;
    #pragma clang loop unroll(disable)
    for (unsigned bit = 0; bit != 63; ++bit) {
        const auto carry = numerator >> 63;
        numerator <<= 1;
        const bool take = carry || numerator >= denominator;
        if (take) numerator -= denominator;
        result = (result << 1) | take;
    }
    return result;
}
__device__ inline unsigned math_wide_quotient(math_wide numerator, unsigned long long denominator) {
    unsigned low = 0, high = 8192;
    #pragma clang loop unroll(disable)
    while (high - low > 1) {
        const unsigned middle = (low + high) / 2;
        if (math_wide_less(numerator, math_wide_product(middle, denominator))) high = middle;
        else low = middle;
    }
    return low;
}
__device__ inline unsigned long long math_wide_round(math_wide a, unsigned shift) {
    if (!shift) return a.low;
    if (shift < 64) {
        const auto retained = (a.low >> shift) | (a.high << (64 - shift));
        const auto discarded = a.low & ((1ull << shift) - 1), half = 1ull << (shift - 1);
        return retained + (discarded > half || (discarded == half && (retained & 1)));
    }
    if (shift == 64) return a.high + (a.low > 0x8000000000000000ull || (a.low == 0x8000000000000000ull && (a.high & 1)));
    if (shift >= 128) return shift == 128 && (a.high > 0x8000000000000000ull || (a.high == 0x8000000000000000ull && a.low));
    shift -= 64;
    const auto retained = a.high >> shift, discarded = a.high & ((1ull << shift) - 1), half = 1ull << (shift - 1);
    return retained + (discarded > half || (discarded == half && (a.low || (retained & 1))));
}
__device__ inline double math_wide_double(math_wide magnitude, int scale, bool negative = false) {
    const auto sign = static_cast<unsigned long long>(negative) << 63;
    if (!(magnitude.low | magnitude.high)) return math_double_bits(sign);
    const int top = magnitude.high ? 127 - __builtin_clzll(magnitude.high) : 63 - __builtin_clzll(magnitude.low);
    int exponent = top + scale + 1023;
    if (exponent <= 0) return math_double_bits(sign | math_wide_round(magnitude, unsigned(-1074 - scale)));
    auto significand = top > 52 ? math_wide_round(magnitude, unsigned(top - 52)) : magnitude.low << (52 - top);
    if (significand == 0x20000000000000ull) { significand >>= 1; ++exponent; }
    if (exponent >= 2047) return math_double_bits(sign | 0x7ff0000000000000ull);
    return math_double_bits(sign | (static_cast<unsigned long long>(exponent) << 52) | (significand & 0xfffffffffffffull));
}
constexpr unsigned long long math_ln2_q63 = 0x58b90bfbe8e7bcd6ull;
constexpr unsigned long long math_ln2_q62 = 0x2c5c85fdf473de6bull;
constexpr unsigned long long math_ln10_q62 = 0x935d8dddaaa8ac17ull;
__device__ inline double math_double_exponential(double x, unsigned base) {
    const auto bits = math_double_word(x), magnitude = bits & 0x7fffffffffffffffull;
    const bool negative = bits >> 63;
    if (magnitude > 0x7ff0000000000000ull) return math_double_bits(bits | 0x8000000000000ull);
    if (magnitude > 0x40a0000000000000ull) return math_double_bits(negative ? 0 : 0x7ff0000000000000ull);
    if (magnitude < 0x3c30000000000000ull) return 1.0; // |x| < 2^-60.
    const auto significand = (magnitude & 0xfffffffffffffull) | 0x10000000000000ull;
    const int shift = int(magnitude >> 52) - 1012;
    math_wide argument = shift < 0 ? math_wide{significand >> -shift, 0}
        : math_wide{significand << shift, shift ? significand >> (64 - shift) : 0};
    if (base == 1) argument = math_wide_fraction(argument, 0xb17217f7d1cf79acull);
    if (base == 2) argument = math_wide_add(math_wide_add(argument, argument), math_wide_fraction(argument, 0x4d763776aaa2b05cull));
    unsigned n = math_wide_quotient(argument, math_ln2_q63);
    auto remainder = math_wide_subtract(argument, math_wide_product(n, math_ln2_q63)).low;
    bool subtract = false;
    if (remainder >= math_ln2_q63 / 2) { remainder = math_ln2_q63 - remainder; ++n; subtract = true; }
    subtract ^= negative;
    const int exponent = negative ? -int(n) : int(n);
    if (exponent > 1024) return math_double_bits(0x7ff0000000000000ull);
    if (exponent < -1075) return 0.0;
    unsigned long long term = 0x8000000000000000ull, sum = term;
    #pragma clang loop unroll(disable)
    for (unsigned degree = 1; degree <= 20; ++degree) {
        term = math_q63_product(term, remainder) / degree;
        if (!term) break;
        if (subtract && (degree & 1)) sum -= term;
        else sum += term;
    }
    return math_wide_double({sum, 0}, exponent - 63);
}
__device__ inline double math_double_logarithm(double x, unsigned base) {
    const auto bits = math_double_word(x), magnitude = bits & 0x7fffffffffffffffull;
    if (magnitude > 0x7ff0000000000000ull) return math_double_bits(bits | 0x8000000000000ull);
    if (!magnitude) return math_double_bits(0xfff0000000000000ull);
    if (bits >> 63) return math_double_bits(0x7ff8000000000000ull);
    if (magnitude == 0x7ff0000000000000ull) return x;
    int exponent = int(bits >> 52) - 1023;
    auto significand = (bits & 0xfffffffffffffull) | 0x10000000000000ull;
    if (!(bits >> 52)) {
        const int top = 63 - __builtin_clzll(bits);
        significand = bits << (52 - top); exponent = top - 1074;
    }
    // The mantissa stays in [0.75,1.5). Retain its exact dyadic delta near one
    // and multiply by a relative correction instead of subtracting logarithms.
    const bool below_one = significand >= 0x18000000000000ull;
    if (below_one) ++exponent;
    const auto delta = below_one ? 0x20000000000000ull - significand : significand - 0x10000000000000ull;
    const int delta_scale = below_one ? -53 : -52;
    const auto delta_q62 = delta << (62 + delta_scale);
    const auto denominator = below_one ? 0x8000000000000000ull - delta_q62 : 0x8000000000000000ull + delta_q62;
    const auto t = math_q63_quotient(delta_q62, denominator), square = math_q63_product(t, t);
    unsigned long long term = 0x8000000000000000ull, sum = term;
    #pragma clang loop unroll(disable)
    for (unsigned degree = 3; degree <= 49; degree += 2) {
        term = math_q63_product(term, square);
        if (!term) break;
        sum += term / degree;
    }
    auto ratio = math_q63_product(math_q63_quotient(0x8000000000000000ull, denominator), sum);
    if (!exponent) {
        if (base) ratio = math_q63_quotient(ratio >> 1, base == 1 ? math_ln2_q62 : math_ln10_q62);
        return math_wide_double(math_wide_product(delta, ratio), delta_scale - 63, below_one);
    }
    auto logarithm = math_wide_product(unsigned(exponent < 0 ? -exponent : exponent), math_ln2_q63);
    const auto correction = math_wide_product(delta, ratio);
    const auto correction_q63 = (correction.low >> -delta_scale) | (correction.high << (64 + delta_scale));
    logarithm = (exponent < 0) == below_one ? math_wide_add(logarithm, {correction_q63, 0})
        : math_wide_subtract(logarithm, {correction_q63, 0});
    if (base) {
        const auto numerator = base == 1 ? logarithm : math_wide_half(logarithm);
        const auto divisor = base == 1 ? math_ln2_q63 : math_ln10_q62;
        const auto integral = math_wide_quotient(numerator, divisor);
        const auto residual = math_wide_subtract(numerator, math_wide_product(integral, divisor));
        const auto fraction = math_q63_quotient(residual.low, divisor);
        logarithm = {fraction | (static_cast<unsigned long long>(integral) << 63), integral >> 1};
    }
    return math_wide_double(logarithm, -63, exponent < 0);
}
}
extern "C" __device__ inline double __nv_exp(double x) { return cudavm::math_double_exponential(x, 0); }
extern "C" __device__ inline double __nv_exp2(double x) { return cudavm::math_double_exponential(x, 1); }
extern "C" __device__ inline double __nv_exp10(double x) { return cudavm::math_double_exponential(x, 2); }
extern "C" __device__ inline double __nv_log(double x) { return cudavm::math_double_logarithm(x, 0); }
extern "C" __device__ inline double __nv_log2(double x) { return cudavm::math_double_logarithm(x, 1); }
extern "C" __device__ inline double __nv_log10(double x) { return cudavm::math_double_logarithm(x, 2); }
