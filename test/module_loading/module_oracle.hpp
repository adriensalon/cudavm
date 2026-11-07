#pragma once
#include <cstdint>
namespace cudavm {
inline uint32_t module_boundary_expected(bool beta, uint32_t input, uint32_t seed, uint32_t index, uint32_t bias = 29) {
    return beta ? (input ^ bias) + seed : input * 3u + seed + index;
}
}
