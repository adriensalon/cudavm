template<class T> struct comparison_returns_int { static constexpr bool value = false; };
template<> struct comparison_returns_int<int> { static constexpr bool value = true; };
static_assert(comparison_returns_int<decltype(min(1, 2))>::value, "float min must preserve integer overload resolution");
static_assert(comparison_returns_int<decltype(max(1, 2))>::value, "float max must preserve integer overload resolution");

extern "C" __global__ void compare_float(unsigned* out, const float* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float a = in[i * 2], b = in[i * 2 + 1];
    out += i * 24;
    out[0] = a == b; out[1] = a != b;
    out[2] = a < b; out[3] = a <= b; out[4] = a > b; out[5] = a >= b;
    out[6] = !(a >= b); out[7] = !(a > b); out[8] = !(a <= b); out[9] = !(a < b);
    out[10] = isnan(a) != 0; out[11] = isinf(a) != 0;
    out[12] = isfinite(a) != 0; out[13] = signbit(a) != 0;
    out[14] = __float_as_uint(fminf(a, b)); out[15] = __float_as_uint(fmaxf(a, b));
    out[16] = __float_as_uint(fabsf(a)); out[17] = __float_as_uint(copysignf(a, b));
    float minimum, maximum;
    asm("min.f32 %0,%1,%2;" : "=f"(minimum) : "f"(a), "f"(b));
    asm("max.f32 %0,%1,%2;" : "=f"(maximum) : "f"(a), "f"(b));
    out[18] = __float_as_uint(minimum); out[19] = __float_as_uint(maximum);
    unsigned normal, subnormal;
    asm("{ .reg .pred p; testp.normal.f32 p,%1; selp.u32 %0,1,0,p; }" : "=r"(normal) : "f"(a));
    asm("{ .reg .pred p; testp.subnormal.f32 p,%1; selp.u32 %0,1,0,p; }" : "=r"(subnormal) : "f"(a));
    out[20] = normal; out[21] = subnormal;
    out[22] = __float_as_uint(min(a, b)); out[23] = __float_as_uint(max(a, b));
}
