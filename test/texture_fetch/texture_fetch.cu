template<class T> __device__ T fetch_texel(cudaTextureObject_t object, unsigned index, unsigned width, bool linear) {
    return linear ? tex1Dfetch<T>(object, int(index)) : tex2D<T>(object, float(index % width) + 0.5f, float(index / width) + 0.5f);
}
#define FETCH_KERNEL(name, scalar, pair, quad, result) \
extern "C" __global__ void name(cudaTextureObject_t object, result* output, unsigned width, unsigned count, unsigned channels, unsigned linear) { \
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= count) return; \
    unsigned index = (i * 37 + 11) % count; result value{}; \
    if (channels == 1) value.x = fetch_texel<scalar>(object, index, width, linear); \
    else if (channels == 2) { auto v = fetch_texel<pair>(object, index, width, linear); value.x = v.x; value.y = v.y; } \
    else { auto v = fetch_texel<quad>(object, index, width, linear); value.x = v.x; value.y = v.y; value.z = v.z; value.w = v.w; } \
    output[i] = value; \
}
FETCH_KERNEL(fetch_s8, signed char, char2, char4, int4)
FETCH_KERNEL(fetch_u8, unsigned char, uchar2, uchar4, uint4)
FETCH_KERNEL(fetch_s16, short, short2, short4, int4)
FETCH_KERNEL(fetch_u16, unsigned short, ushort2, ushort4, uint4)
FETCH_KERNEL(fetch_s32, int, int2, int4, int4)
FETCH_KERNEL(fetch_u32, unsigned, uint2, uint4, uint4)
FETCH_KERNEL(fetch_float, float, float2, float4, float4)
#undef FETCH_KERNEL
