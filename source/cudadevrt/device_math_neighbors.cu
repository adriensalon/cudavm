namespace cudavm {
template<class Word, unsigned Fraction, unsigned Exponent>
__device__ inline Word math_nextafter_word(Word a, Word b) {
    constexpr Word sign = Word(1) << (Fraction + Exponent);
    constexpr Word infinity = ((Word(1) << Exponent) - 1) << Fraction;
    constexpr Word nan = infinity | (Word(1) << (Fraction - 1));
    const Word ax = a & (sign - 1), bx = b & (sign - 1);
    if (ax > infinity || bx > infinity) return nan;
    if (a == b || !(ax | bx)) return b;
    if (!ax) return (b & sign) | 1;
    const bool larger = (a ^ b) & sign ? (a & sign) != 0 : a & sign ? ax > bx : ax < bx;
    const bool increment = larger == ((a & sign) == 0);
    return increment ? a + 1 : a - 1;
}
__device__ inline float math_float_nextafter(float x, float y) {
    return __builtin_bit_cast(float, math_nextafter_word<unsigned, 23, 8>(
        __builtin_bit_cast(unsigned, x), __builtin_bit_cast(unsigned, y)));
}
__device__ inline double math_double_nextafter(double x, double y) {
    return __builtin_bit_cast(double, math_nextafter_word<unsigned long long, 52, 11>(
        __builtin_bit_cast(unsigned long long, x), __builtin_bit_cast(unsigned long long, y)));
}
}
extern "C" __device__ inline float __nv_nextafterf(float x, float y) { return cudavm::math_float_nextafter(x, y); }
extern "C" __device__ inline double __nv_nextafter(double x, double y) { return cudavm::math_double_nextafter(x, y); }
