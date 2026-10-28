#pragma once
#include <cuda.h>
#include <array>
#include <cstddef>

namespace cudavm {
enum class array_storage_encoding { packed, block_compressed, semi_planar, planar };

// Logical channel descriptions and byte-copy blocks are separate: e.g. BC6H
// has three half channels per texel, but copies sixteen-byte 4x4 blocks.
struct array_format_layout {
    unsigned component_bits = 0, channels = 0;
    unsigned block_width = 1, block_height = 1, block_bytes = 0;
    unsigned planes = 1, subsample_x = 1, subsample_y = 1;
    array_storage_encoding encoding = array_storage_encoding::packed;
    bool valid = false;
};
inline array_format_layout array_format_properties(CUarray_format format) {
    using encoding = array_storage_encoding;
    switch (format) {
    case CU_AD_FORMAT_UNSIGNED_INT8: case CU_AD_FORMAT_SIGNED_INT8: return {8, 0, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_UNSIGNED_INT16: case CU_AD_FORMAT_SIGNED_INT16: case CU_AD_FORMAT_HALF: return {16, 0, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_UNSIGNED_INT32: case CU_AD_FORMAT_SIGNED_INT32: case CU_AD_FORMAT_FLOAT: return {32, 0, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_UNORM_INT8X1:     case CU_AD_FORMAT_SNORM_INT8X1: return {8, 1, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_UNORM_INT8X2:     case CU_AD_FORMAT_SNORM_INT8X2: return {8, 2, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_UNORM_INT8X4:     case CU_AD_FORMAT_SNORM_INT8X4: return {8, 4, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_UNORM_INT16X1:     case CU_AD_FORMAT_SNORM_INT16X1: return {16, 1, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_UNORM_INT16X2:     case CU_AD_FORMAT_SNORM_INT16X2: return {16, 2, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_UNORM_INT16X4:     case CU_AD_FORMAT_SNORM_INT16X4: return {16, 4, 1, 1, 0, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_BC1_UNORM: return {8, 4, 4, 4, 8, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC1_UNORM_SRGB: return {8, 4, 4, 4, 8, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC2_UNORM: return {8, 4, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC2_UNORM_SRGB: return {8, 4, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC3_UNORM: return {8, 4, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC3_UNORM_SRGB: return {8, 4, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC4_UNORM: return {8, 1, 4, 4, 8, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC4_SNORM: return {8, 1, 4, 4, 8, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC5_UNORM: return {8, 2, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC5_SNORM: return {8, 2, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC6H_UF16: return {16, 3, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC6H_SF16: return {16, 3, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC7_UNORM: return {8, 4, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_BC7_UNORM_SRGB: return {8, 4, 4, 4, 16, 1, 1, 1, encoding::block_compressed, true};
    case CU_AD_FORMAT_UNORM_INT_101010_2: case CU_AD_FORMAT_Y410: return {10, 4, 1, 1, 4, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_YUY2: case CU_AD_FORMAT_UINT8_PACKED_422: return {8, 4, 1, 1, 4, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_Y210: case CU_AD_FORMAT_Y216: return {16, 4, 1, 1, 8, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_AYUV: case CU_AD_FORMAT_UINT8_PACKED_444: return {8, 4, 1, 1, 4, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_Y416: return {16, 4, 1, 1, 8, 1, 1, 1, encoding::packed, true};
    case CU_AD_FORMAT_NV12: case CU_AD_FORMAT_UINT8_SEMIPLANAR_420: return {8, 3, 1, 1, 0, 2, 2, 2, encoding::semi_planar, true};
    case CU_AD_FORMAT_P010: case CU_AD_FORMAT_P016: case CU_AD_FORMAT_UINT16_SEMIPLANAR_420: return {16, 3, 1, 1, 0, 2, 2, 2, encoding::semi_planar, true};
    case CU_AD_FORMAT_NV16: case CU_AD_FORMAT_UINT8_SEMIPLANAR_422: return {8, 3, 1, 1, 0, 2, 2, 1, encoding::semi_planar, true};
    case CU_AD_FORMAT_P210: case CU_AD_FORMAT_P216: case CU_AD_FORMAT_UINT16_SEMIPLANAR_422: return {16, 3, 1, 1, 0, 2, 2, 1, encoding::semi_planar, true};
    case CU_AD_FORMAT_YUV444_8bit_SemiPlanar: case CU_AD_FORMAT_UINT8_SEMIPLANAR_444: return {8, 3, 1, 1, 0, 2, 1, 1, encoding::semi_planar, true};
    case CU_AD_FORMAT_YUV444_16bit_SemiPlanar: case CU_AD_FORMAT_UINT16_SEMIPLANAR_444: return {16, 3, 1, 1, 0, 2, 1, 1, encoding::semi_planar, true};
    case CU_AD_FORMAT_UINT8_PLANAR_420: return {8, 3, 1, 1, 0, 3, 2, 2, encoding::planar, true};
    case CU_AD_FORMAT_UINT16_PLANAR_420: return {16, 3, 1, 1, 0, 3, 2, 2, encoding::planar, true};
    case CU_AD_FORMAT_UINT8_PLANAR_422: return {8, 3, 1, 1, 0, 3, 2, 1, encoding::planar, true};
    case CU_AD_FORMAT_UINT16_PLANAR_422: return {16, 3, 1, 1, 0, 3, 2, 1, encoding::planar, true};
    case CU_AD_FORMAT_Y444_PLANAR8: case CU_AD_FORMAT_UINT8_PLANAR_444: return {8, 3, 1, 1, 0, 3, 1, 1, encoding::planar, true};
    case CU_AD_FORMAT_Y444_PLANAR10: case CU_AD_FORMAT_UINT16_PLANAR_444: return {16, 3, 1, 1, 0, 3, 1, 1, encoding::planar, true};
    default: return {};
    }
}
inline std::array<int, 4> array_channel_bits(const array_format_layout& format, unsigned channels) {
    if (format.component_bits == 10 && format.block_bytes == 4) return {10, 10, 10, 2};
    std::array<int, 4> result{};
    for (unsigned i = 0; i < channels && i < result.size(); ++i) result[i] = int(format.component_bits);
    return result;
}
inline size_t array_block_extent(size_t extent, unsigned block) {
    return extent / block + (extent % block != 0);
}
}
