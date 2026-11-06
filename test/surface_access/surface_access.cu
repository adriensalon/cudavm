__device__ uint4 surface_pack(unsigned char v) { return {v, 0, 0, 0}; }
__device__ uint4 surface_pack(uchar2 v) { return {unsigned(v.x) | (unsigned(v.y) << 8), 0, 0, 0}; }
__device__ uint4 surface_pack(uchar4 v) { return {unsigned(v.x) | (unsigned(v.y) << 8) | (unsigned(v.z) << 16) | (unsigned(v.w) << 24), 0, 0, 0}; }
__device__ uint4 surface_pack(unsigned short v) { return {v, 0, 0, 0}; }
__device__ uint4 surface_pack(ushort2 v) { return {unsigned(v.x) | (unsigned(v.y) << 16), 0, 0, 0}; }
__device__ uint4 surface_pack(ushort4 v) { return {unsigned(v.x) | (unsigned(v.y) << 16), unsigned(v.z) | (unsigned(v.w) << 16), 0, 0}; }
__device__ uint4 surface_pack(unsigned v) { return {v, 0, 0, 0}; }
__device__ uint4 surface_pack(uint2 v) { return {v.x, v.y, 0, 0}; }
__device__ uint4 surface_pack(uint4 v) { return v; }
__device__ uint4 surface_pack(unsigned long long v) { return {unsigned(v), unsigned(v >> 32), 0, 0}; }
__device__ uint4 surface_pack(ulonglong2 v) { return {unsigned(v.x), unsigned(v.x >> 32), unsigned(v.y), unsigned(v.y >> 32)}; }
template<class T> __device__ void access_surface(cudaSurfaceObject_t object, const unsigned* input, uint4* output, unsigned phase) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr unsigned width = 11, height = 3, count = width * height;
    if (phase == 2) {
        if (!i) {
            auto first = *reinterpret_cast<const T*>(input + count * 4);
            auto last = *reinterpret_cast<const T*>(input + (count + 1) * 4);
            surf2Dwrite(last, object, -int(sizeof(T)), -1, cudaBoundaryModeZero);
            surf2Dwrite(first, object, -int(sizeof(T)), -1, cudaBoundaryModeClamp);
            surf2Dwrite(last, object, int(width * sizeof(T)), height, cudaBoundaryModeClamp);
        }
    } else if (!phase) {
        if (i < count) surf2Dwrite(*reinterpret_cast<const T*>(input + i * 4), object, int((i % width) * sizeof(T)), int(i / width));
    } else {
        if (i < count) output[i] = surface_pack(surf2Dread<T>(object, int((i % width) * sizeof(T)), int(i / width)));
        if (!i) {
            output[count] = surface_pack(surf2Dread<T>(object, -int(sizeof(T)), -1, cudaBoundaryModeZero));
            output[count + 1] = surface_pack(surf2Dread<T>(object, -int(sizeof(T)), -1, cudaBoundaryModeClamp));
            output[count + 2] = surface_pack(surf2Dread<T>(object, int(width * sizeof(T)), height, cudaBoundaryModeZero));
            output[count + 3] = surface_pack(surf2Dread<T>(object, int(width * sizeof(T)), height, cudaBoundaryModeClamp));
        }
    }
}
#define SURFACE_KERNEL(name, type) \
extern "C" __global__ void name(cudaSurfaceObject_t object, const unsigned* input, uint4* output, unsigned phase) { access_surface<type>(object, input, output, phase); }
SURFACE_KERNEL(access_8_1, unsigned char)
SURFACE_KERNEL(access_8_2, uchar2)
SURFACE_KERNEL(access_8_4, uchar4)
SURFACE_KERNEL(access_16_1, unsigned short)
SURFACE_KERNEL(access_16_2, ushort2)
SURFACE_KERNEL(access_16_4, ushort4)
SURFACE_KERNEL(access_32_1, unsigned)
SURFACE_KERNEL(access_32_2, uint2)
SURFACE_KERNEL(access_32_4, uint4)
SURFACE_KERNEL(access_64_1, unsigned long long)
SURFACE_KERNEL(access_64_2, ulonglong2)
#undef SURFACE_KERNEL
