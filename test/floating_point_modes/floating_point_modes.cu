#define MODE_INPUTS \
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= count) return; \
    const float a = __uint_as_float(input[i * 3]), b = __uint_as_float(input[i * 3 + 1]), c = __uint_as_float(input[i * 3 + 2]); \
    output += i * 8;
#define MODE_BINARY(OP, ROUND, SLOT) do { \
    float gradual, flush; \
    asm volatile(OP "." ROUND ".f32 %0, %1, %2;" : "=f"(gradual) : "f"(a), "f"(b)); \
    asm volatile(OP "." ROUND ".ftz.f32 %0, %1, %2;" : "=f"(flush) : "f"(a), "f"(b)); \
    output[SLOT] = __float_as_uint(gradual); output[4 + SLOT] = __float_as_uint(flush); \
} while (false)
#define MODE_ROUNDINGS(MACRO, OP) \
    MACRO(OP, "rn", 0); MACRO(OP, "rz", 1); MACRO(OP, "rm", 2); MACRO(OP, "rp", 3);
extern "C" __global__ void mode_add(unsigned* output, const unsigned* input, unsigned count) {
    MODE_INPUTS
    MODE_ROUNDINGS(MODE_BINARY, "add")
}
extern "C" __global__ void mode_multiply(unsigned* output, const unsigned* input, unsigned count) {
    MODE_INPUTS
    MODE_ROUNDINGS(MODE_BINARY, "mul")
}
#define MODE_FUSED(UNUSED, ROUND, SLOT) do { \
    float gradual, flush; \
    asm volatile("fma." ROUND ".f32 %0, %1, %2, %3;" : "=f"(gradual) : "f"(a), "f"(b), "f"(c)); \
    asm volatile("fma." ROUND ".ftz.f32 %0, %1, %2, %3;" : "=f"(flush) : "f"(a), "f"(b), "f"(c)); \
    output[SLOT] = __float_as_uint(gradual); output[4 + SLOT] = __float_as_uint(flush); \
} while (false)
extern "C" __global__ void mode_fused(unsigned* output, const unsigned* input, unsigned count) {
    MODE_INPUTS
    MODE_ROUNDINGS(MODE_FUSED, "")
}
#define MODE_SEPARATE(UNUSED, ROUND, SLOT) do { \
    float gradual, flush; \
    asm volatile("{ .reg .f32 p; mul." ROUND ".f32 p, %1, %2; add." ROUND ".f32 %0, p, %3; }" \
        : "=f"(gradual) : "f"(a), "f"(b), "f"(c)); \
    asm volatile("{ .reg .f32 p; mul." ROUND ".ftz.f32 p, %1, %2; add." ROUND ".ftz.f32 %0, p, %3; }" \
        : "=f"(flush) : "f"(a), "f"(b), "f"(c)); \
    output[SLOT] = __float_as_uint(gradual); output[4 + SLOT] = __float_as_uint(flush); \
} while (false)
extern "C" __global__ void mode_separate(unsigned* output, const unsigned* input, unsigned count) {
    MODE_INPUTS
    MODE_ROUNDINGS(MODE_SEPARATE, "")
}
