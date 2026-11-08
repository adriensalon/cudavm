#pragma once
namespace cudavm {
struct device_math_operation {
    const char* name;
    const char* intrinsic;
    const char* shader_helper;
    unsigned arguments;
    const char* llvm_intrinsic = nullptr;
    const char* public_name = nullptr;
};
inline constexpr device_math_operation device_math_operations[] = {
    {"sqrtf", "llvm.nvvm.sqrt.rn.f", "cvMathSqrt", 1},
    {"fmaxf", "llvm.nvvm.fmax.f", "cvMathMax", 2},
    {"sinf", nullptr, "cvMathSin", 1},
    {"cosf", nullptr, "cvMathCos", 1},
    {"exp2f", nullptr, "math_exp2", 1},
    {"expf", nullptr, "math_exp", 1},
    {"exp10f", nullptr, "math_exp10", 1},
    {"log2f", nullptr, "math_log2", 1},
    {"logf", nullptr, "math_log", 1},
    {"log10f", nullptr, "math_log10", 1},
    {"tanhf", nullptr, "math_tanh", 1},
    {"fast_fdividef", nullptr, "math_fast_divide", 2, nullptr, "__fdividef"},
    {"hypotf", nullptr, "math_hypot", 2},
    {"norm3df", nullptr, "math_norm3d", 3},
    {"norm4df", nullptr, "math_norm4d", 4},
    {"nextafterf", nullptr, "math_float_nextafter", 2},
    {"fabsf", nullptr, "cvMathAbs", 1, "llvm.fabs.f32"},
    {"fminf", nullptr, "cvMathMin", 2, "llvm.minnum.f32"},
    {"copysignf", nullptr, "cvMathCopySign", 2, "llvm.copysign.f32"},
    {"floorf", nullptr, "cvMathFloor", 1, "llvm.floor.f32"},
    {"ceilf", nullptr, "cvMathCeil", 1, "llvm.ceil.f32"},
    {"truncf", nullptr, "cvMathTrunc", 1, "llvm.trunc.f32"},
    {"roundf", nullptr, "cvMathRound", 1, "llvm.round.f32"},
    {"rintf", nullptr, "cvMathRint", 1, "llvm.rint.f32"},
    {"nearbyintf", nullptr, "cvMathRint", 1, "llvm.nearbyint.f32"},
};
inline constexpr device_math_operation device_double_math_operations[] = {
    {"hypot", nullptr, "math_double_hypot", 2},
    {"norm3d", nullptr, "math_double_norm3d", 3},
    {"norm4d", nullptr, "math_double_norm4d", 4},
    {"nextafter", nullptr, "math_double_nextafter", 2},
};
}
