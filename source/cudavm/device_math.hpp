#pragma once
namespace cudavm {
struct device_math_operation {
    const char* name;
    const char* intrinsic;
    const char* shader_helper;
    unsigned arguments;
    const char* llvm_intrinsic = nullptr;
};
inline constexpr device_math_operation device_math_operations[] = {
    {"sqrtf", "llvm.nvvm.sqrt.rn.f", "cvMathSqrt", 1},
    {"fmaxf", "llvm.nvvm.fmax.f", "cvMathMax", 2},
    {"sinf", nullptr, "cvMathSin", 1},
    {"cosf", nullptr, "cvMathCos", 1},
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
}
