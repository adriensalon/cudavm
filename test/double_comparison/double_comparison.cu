template<class T> struct comparison_returns_double { static constexpr bool value = false; };
template<> struct comparison_returns_double<double> { static constexpr bool value = true; };
static_assert(comparison_returns_double<decltype(min(1.0, 1.0f))>::value, "mixed min must return double");
static_assert(comparison_returns_double<decltype(min(1.0f, 1.0))>::value, "mixed min must return double");
static_assert(comparison_returns_double<decltype(max(1.0, 1.0f))>::value, "mixed max must return double");
static_assert(comparison_returns_double<decltype(max(1.0f, 1.0))>::value, "mixed max must return double");

extern "C" __global__ void compare_double(unsigned* out, const unsigned* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    in += i * 6; out += i * 44;
    double a = *reinterpret_cast<const double*>(in), b = *reinterpret_cast<const double*>(in + 2);
    float c = *reinterpret_cast<const float*>(in + 4);
    out[0] = a == b; out[1] = a != b;
    out[2] = a < b; out[3] = a <= b; out[4] = a > b; out[5] = a >= b;
    out[6] = !(a >= b); out[7] = !(a > b); out[8] = !(a <= b); out[9] = !(a < b);
    out[10] = isnan(a) != 0; out[11] = isinf(a) != 0;
    out[12] = isfinite(a) != 0; out[13] = signbit(a) != 0;
    auto values = reinterpret_cast<unsigned long long*>(out + 14);
    values[0] = __double_as_longlong(fmin(a, b)); values[1] = __double_as_longlong(fmax(a, b));
    values[2] = __double_as_longlong(fabs(a)); values[3] = __double_as_longlong(copysign(a, b));
    double minimum, maximum, absolute, copy_sign;
    asm("min.f64 %0,%1,%2;" : "=d"(minimum) : "d"(a), "d"(b));
    asm("max.f64 %0,%1,%2;" : "=d"(maximum) : "d"(a), "d"(b));
    values[4] = __double_as_longlong(minimum); values[5] = __double_as_longlong(maximum);
    values[6] = __double_as_longlong(min(a, b)); values[7] = __double_as_longlong(max(a, b));
    unsigned normal, subnormal;
    asm("{ .reg .pred p; testp.normal.f64 p,%1; selp.u32 %0,1,0,p; }" : "=r"(normal) : "d"(a));
    asm("{ .reg .pred p; testp.subnormal.f64 p,%1; selp.u32 %0,1,0,p; }" : "=r"(subnormal) : "d"(a));
    out[30] = normal; out[31] = subnormal;
    auto mixed = reinterpret_cast<unsigned long long*>(out + 32);
    mixed[0] = __double_as_longlong(min(a, c)); mixed[1] = __double_as_longlong(min(c, a));
    mixed[2] = __double_as_longlong(max(a, c)); mixed[3] = __double_as_longlong(max(c, a));
    asm("abs.f64 %0,%1;" : "=d"(absolute) : "d"(a));
    asm("copysign.f64 %0,%1,%2;" : "=d"(copy_sign) : "d"(b), "d"(a));
    auto raw = reinterpret_cast<unsigned long long*>(out + 40);
    raw[0] = __double_as_longlong(absolute); raw[1] = __double_as_longlong(copy_sign);
}
