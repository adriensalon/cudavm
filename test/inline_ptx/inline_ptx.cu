#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750
#error inline_ptx_fixture_requires_sm75
#endif
extern "C" __global__ void inline_results(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = input[i];
    unsigned* row = output + i * 13;
    unsigned added;
    asm("add.u32 %0, %1, %2;" : "=r"(added) : "r"(value), "n"(17));
    row[0] = added;
    unsigned modified = value;
    asm("xor.b32 %0, %0, 0x193759bd;" : "+r"(modified));
    row[1] = modified;
    unsigned low, high;
    asm("{ add.cc.u32 %0, %2, 0xfffffff0; addc.u32 %1, 0, 0; }"
        : "=&r"(low), "=r"(high) : "r"(value));
    row[2] = low; row[3] = high;
    const unsigned rounds = value & 7;
    unsigned accumulated = value;
    asm("{ .reg .u32 temporary; .reg .pred condition; mov.u32 temporary, %1; "
        "again: setp.eq.u32 condition, temporary, 0; @condition bra done; "
        "add.u32 %0, %0, 11; sub.u32 temporary, temporary, 1; bra again; done: }"
        : "+r"(accumulated) : "r"(rounds));
    row[4] = accumulated;
    const unsigned long long wide = (static_cast<unsigned long long>(value) << 32) | (value ^ 0xabcdef01u);
    unsigned long long wide_result;
    asm("add.u64 %0, %1, %2;" : "=l"(wide_result) : "l"(wide), "l"(0x1234567887654321ull));
    row[5] = unsigned(wide_result); row[6] = unsigned(wide_result >> 32);
    const unsigned short halfword = static_cast<unsigned short>(value);
    unsigned short halfword_result;
    asm("xor.b16 %0, %1, 0x5a39;" : "=h"(halfword_result) : "h"(halfword));
    row[7] = halfword_result;
    const float finite = __uint_as_float((value & 0x807fffffu) | 0x3f000000u);
    float negative;
    asm("neg.f32 %0, %1;" : "=f"(negative) : "f"(finite));
    row[8] = __float_as_uint(negative);
    unsigned tied = value;
    asm("add.u32 %0, %0, %1;" : "+r"(tied) : "r"(i));
    row[9] = tied;
    const unsigned stored = value ^ 0x76543210u;
    row[10] = 0;
    asm volatile("st.global.u32 [%0], %1;" :: "l"(row + 10), "r"(stored) : "memory");
    row[11] = row[10] + 1;
    unsigned inverted;
    asm("{ .reg .u32 temporary; mov.u32 temporary, %1; not.b32 %0, temporary; }" : "=r"(inverted) : "r"(value));
    row[12] = inverted;
}
