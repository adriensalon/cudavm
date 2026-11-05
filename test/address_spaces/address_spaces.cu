#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
using address_word = unsigned long long;
extern "C" __constant__ unsigned address_constants[9] = {
    0x13579bdfu, 0, 1, 0x80000000u, 0xffffffffu, 0x12345678u, 0x89abcdefu, 17, 0xeca86420u};
__device__ __forceinline__ unsigned address_mask(const void* pointer) {
    return __isGlobal(pointer) | (__isShared(pointer) << 1) | (__isLocal(pointer) << 2) | (__isConstant(pointer) << 3);
}
__device__ __forceinline__ void address_group(unsigned* out, const unsigned* pointer,
    address_word first, address_word second, const void* restored, const void* next) {
    out[0] = address_mask(pointer);
    out[1] = restored == pointer;
    out[2] = *static_cast<const unsigned*>(restored);
    out[3] = *static_cast<const unsigned*>(next);
    out[4] = unsigned(second - first);
    out[5] = next == pointer + 1;
}
extern "C" __global__ void address_results(unsigned* out, const unsigned* in, unsigned count, unsigned selector) {
    unsigned lane = threadIdx.x, block = blockIdx.x, i = block * blockDim.x + lane;
    if (i >= count) return;
    __shared__ unsigned shared_words[256];
    unsigned local_words[4];
    const unsigned *global = in + i * 2, *constant = address_constants + i % 6 + 1;
    unsigned a = global[0], b = global[1];
    unsigned* shared = shared_words + lane * 4 + 1;
    unsigned* local = local_words + 1;
    shared[-1] = 0x13579bdfu ^ i; shared[0] = a ^ 0xaaaaaaaau; shared[1] = b + block + 7; shared[2] = 0xeca86420u ^ a;
    local[-1] = 0x89abcdefu ^ b; local[0] = a + i * 17; local[1] = b ^ (i * 0x10203u); local[2] = 0x76543210u ^ i;
    out += i * 36;
    address_word g = __cvta_generic_to_global(global), s = __cvta_generic_to_shared(shared);
    address_word l = __cvta_generic_to_local(local), c = __cvta_generic_to_constant(constant);
    const void *gr = __cvta_global_to_generic(g), *sr = __cvta_shared_to_generic(s);
    const void *lr = __cvta_local_to_generic(l), *cr = __cvta_constant_to_generic(c);
    address_group(out, global, g, __cvta_generic_to_global(global + 1), gr, __cvta_global_to_generic(g + 4));
    address_group(out + 6, shared, s, __cvta_generic_to_shared(shared + 1), sr, __cvta_shared_to_generic(s + 4));
    address_group(out + 12, local, l, __cvta_generic_to_local(local + 1), lr, __cvta_local_to_generic(l + 4));
    address_group(out + 18, constant, c, __cvta_generic_to_constant(constant + 1), cr, __cvta_constant_to_generic(c + 4));
    out[24] = unsigned(g); out[25] = unsigned(g >> 32);
    address_word constant_bits = reinterpret_cast<address_word>(constant);
    out[26] = unsigned(constant_bits); out[27] = unsigned(constant_bits >> 32);
    unsigned which = (i + selector) & 3;
    const unsigned* selected = which == 0 ? global : which == 1 ? shared : which == 2 ? local : constant;
    const void* roundtrip = which == 0 ? gr : which == 1 ? sr : which == 2 ? lr : cr;
    out[28] = address_mask(selected);
    out[29] = selected[0];
    out[30] = selected[1];
    out[31] = roundtrip == selected;
    out[32] = shared[-1]; out[33] = shared[2]; out[34] = local[-1]; out[35] = local[2];
}
