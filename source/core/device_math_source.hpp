#pragma once
#include <bit>
#include <cstdint>
#include <cmath>

// Compile the device scalar source on the host for numerical oracle checks.
#define CV_MATH_FN inline
#define CV_BITS(x) std::bit_cast<uint32_t>(x)
#define CV_FLOAT(x) std::bit_cast<float>(uint32_t(x))
#define CV_MATH_PRECISE
#define CV_MATH_FMA(x,y,z) std::fma(x,y,z)
#define CV_MATH_CLZ(x) std::countl_zero(uint32_t(x))
#define CV_MATH_CONTRACT
#include "../cudadevrt/device_math_scalar.cu"
#undef CV_MATH_FN
#undef CV_MATH_PRECISE
#undef CV_MATH_FMA
#undef CV_MATH_CLZ
#undef CV_MATH_CONTRACT
#undef CV_BITS
#undef CV_FLOAT
