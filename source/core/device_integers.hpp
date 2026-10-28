#pragma once
namespace cudavm {
inline constexpr const char* device_dot_intrinsics[] = {
    "llvm.nvvm.idp4a.s.s", "llvm.nvvm.idp4a.s.u", "llvm.nvvm.idp4a.u.s", "llvm.nvvm.idp4a.u.u",
    "llvm.nvvm.idp2a.s.s", "llvm.nvvm.idp2a.s.u", "llvm.nvvm.idp2a.u.s", "llvm.nvvm.idp2a.u.u",
};
struct device_integer_operation {
    const char* name;
    const char* intrinsic;
    const char* helper;
    unsigned bits, arguments;
};
inline constexpr device_integer_operation device_integer_operations[] = {
    {"mul24", "llvm.nvvm.mul24.i", "cvSMul24", 32, 2},
    {"umul24", "llvm.nvvm.mul24.ui", "cvUMul24", 32, 2},
    {"mulhi", "llvm.nvvm.mulhi.i", "cvSMulHi32", 32, 2},
    {"umulhi", "llvm.nvvm.mulhi.ui", "cvMulHi", 32, 2},
    {"mul64hi", "llvm.nvvm.mulhi.ll", "cvSMulHi64", 64, 2},
    {"umul64hi", "llvm.nvvm.mulhi.ull", "cvMulHi64", 64, 2},
    {"sad", "llvm.nvvm.sad.i", "cvSad32", 32, 3},
    {"usad", "llvm.nvvm.sad.ui", "cvUSad32", 32, 3},
};
struct device_dot_operation {
    const char* symbol;
    const char* name;
    const char* intrinsic;
    bool signed_value;
    int high; // -1 for four bytes, otherwise selects the dp2a byte pair.
};
inline constexpr device_dot_operation device_dot_operations[] = {
    {"__nv_idp4a_s_s_impl", "dp4a", "llvm.nvvm.idp4a.s.s", true, -1},
    {"__nv_idp4a_u_u_impl", "dp4a", "llvm.nvvm.idp4a.u.u", false, -1},
    {"__nv_idp2a_lo_s_s_impl", "dp2a_lo", "llvm.nvvm.idp2a.s.s", true, 0},
    {"__nv_idp2a_lo_u_u_impl", "dp2a_lo", "llvm.nvvm.idp2a.u.u", false, 0},
    {"__nv_idp2a_hi_s_s_impl", "dp2a_hi", "llvm.nvvm.idp2a.s.s", true, 1},
    {"__nv_idp2a_hi_u_u_impl", "dp2a_hi", "llvm.nvvm.idp2a.u.u", false, 1},
};
}
