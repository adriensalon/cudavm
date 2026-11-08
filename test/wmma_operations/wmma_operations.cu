#include <mma.h>
namespace wmma = nvcuda::wmma;
template<class ALayout, class BLayout> __device__ __forceinline__ void wmma_product(float* output, const __half* a, const __half* b,
    const float* initial, unsigned half_stride, unsigned float_stride, unsigned control, unsigned rounds) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, ALayout> left;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, BLayout> right;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    const auto layout = control & 1 ? wmma::mem_col_major : wmma::mem_row_major;
    wmma::load_matrix_sync(left, a, half_stride); wmma::load_matrix_sync(right, b, half_stride);
    if (control & 2) wmma::fill_fragment(accumulator, 2.0f);
    else wmma::load_matrix_sync(accumulator, initial, float_stride, layout);
    // Apply the same elementwise operation in every participating lane.
    for (unsigned i = 0; i < accumulator.num_elements; ++i) accumulator.x[i] = -accumulator.x[i];
    for (unsigned round = 0; round < rounds; ++round)
        wmma::mma_sync(accumulator, left, right, accumulator, bool(control & 4));
    wmma::store_matrix_sync(output, accumulator, float_stride, layout);
}
template<class ALayout, class BLayout> __device__ __forceinline__ void wmma_tile(float* output, const __half* a, const __half* b, const float* initial,
    unsigned half_stride, unsigned float_stride, unsigned control, unsigned rounds) {
    const unsigned flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned tile = (blockIdx.x * blockDim.x * blockDim.y * blockDim.z + flat) / 32;
    output += tile * float_stride * 16; initial += tile * float_stride * 16;
    a += tile * half_stride * 16; b += tile * half_stride * 16;
    wmma_product<ALayout, BLayout>(output, a, b, initial, half_stride, float_stride, control, rounds);
}
#define WMMA_TILE_KERNEL(name, a_layout, b_layout) \
extern "C" __global__ void name(float* output, const __half* a, const __half* b, const float* initial, \
    unsigned half_stride, unsigned float_stride, unsigned control, unsigned rounds) { \
    wmma_tile<wmma::a_layout, wmma::b_layout>(output, a, b, initial, half_stride, float_stride, control, rounds); }
WMMA_TILE_KERNEL(wmma_rr, row_major, row_major)
WMMA_TILE_KERNEL(wmma_rc, row_major, col_major)
WMMA_TILE_KERNEL(wmma_cr, col_major, row_major)
WMMA_TILE_KERNEL(wmma_cc, col_major, col_major)
#undef WMMA_TILE_KERNEL
extern "C" __global__ void wmma_fill(float* output, unsigned bits, unsigned saturate) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> value;
    __half_raw zero, two; zero.x = 0; two.x = 0x4000;
    wmma::fill_fragment(a, __half(zero)); wmma::fill_fragment(b, __half(two));
    wmma::fill_fragment(value, __uint_as_float(bits));
    wmma::mma_sync(value, a, b, value, bool(saturate));
    wmma::store_matrix_sync(output, value, 20, wmma::mem_row_major);
}
