#include <cuda_tile.h>
namespace tiles = cuda::tiles;
static_assert(__cplusplus == 202002L, "tile header profile requires C++20");
static_assert(sizeof(ptrdiff_t) == sizeof(void*) && ptrdiff_t(-1) < 0, "NVRTC predefined signed pointer difference type");

template<class Mapping> __device__ void tile_span_results(unsigned* output, const unsigned* input, const Mapping& mapping,
    unsigned flat, unsigned threads, unsigned count) {
    using extents = typename Mapping::extents_type;
    using layout = typename Mapping::layout_type;
    tiles::tensor_span<const unsigned, extents, layout> span(input, mapping);
    const auto rows = span.extent(0), columns = span.extent(1);
    const auto copy = span.mapping();
    if (!flat) {
        output[0] = span.rank(); output[1] = span.rank_dynamic();
        output[2] = rows; output[3] = columns;
        output[4] = span.mapping().stride(0); output[5] = span.mapping().stride(1);
        output[6] = span.static_extent(0); output[7] = span.static_extent(1);
        output[8] = Mapping::static_stride(0); output[9] = Mapping::static_stride(1);
        output[10] = tiles::layout_mapping_equal(copy, span.mapping());
        output[11] = tiles::extents_equal(mapping.extents(), span.extents());
        output[12] = sizeof(ptrdiff_t); output[13] = ptrdiff_t(-1) < 0;
        output[14] = span.data_handle() == input; output[15] = 0x5917ab3d;
    }
    for (unsigned cell = flat; cell < rows * columns; cell += threads) if (cell < count) {
        const unsigned row = cell / columns, column = cell % columns;
        const auto offset = row * span.mapping().stride(0) + column * span.mapping().stride(1);
        output[16 + row * (columns + 3) + column] = span.data_handle()[offset] ^ (row * 0x10203u + column * 0x105u);
    }
}
extern "C" __global__ void tile_header_views(unsigned* output, const unsigned* input, unsigned rows,
    unsigned columns, unsigned granularity, unsigned mode, unsigned count) {
    const unsigned flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    output += blockIdx.x * 256; input += blockIdx.x * 256;
    using extents = tiles::extents<unsigned, tiles::dynamic_extent, tiles::dynamic_extent>;
    extents dimensions(rows, columns);
    if (mode == 0) tile_span_results(output, input, tiles::layout_right::mapping<extents>(dimensions), flat, threads, count);
    else if (mode == 1) tile_span_results(output, input, tiles::layout_left::mapping<extents>(dimensions), flat, threads, count);
    else if (mode == 2) tile_span_results(output, input, tiles::layout_right_padded<tiles::dynamic_extent>::mapping<extents>(dimensions, granularity), flat, threads, count);
    else if (mode == 3) tile_span_results(output, input, tiles::layout_left_padded<tiles::dynamic_extent>::mapping<extents>(dimensions, granularity), flat, threads, count);
    else if (mode == 4) {
        using strides = tiles::strides<unsigned, tiles::dynamic_stride, tiles::dynamic_stride>;
        strides steps(columns * 2 + 3, 2);
        tile_span_results(output, input, tiles::layout_strided<strides>::mapping<extents>(dimensions, steps), flat, threads, count);
    } else {
        using fixed = tiles::extents<unsigned, 3, 5>;
        tile_span_results(output, input, tiles::layout_right_padded<4>::mapping<fixed>(fixed{}, 4), flat, threads, count);
    }
}
