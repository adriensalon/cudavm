#pragma once
namespace cudavm {
// Public CUDA/libdevice names and their exact NVVM identities. Modes are
// nearest-even (0), toward zero (1), toward -infinity (2), toward +infinity (3).
struct device_conversion {
    const char* name;
    const char* intrinsic;
    bool to_real, signed_value;
    unsigned mode;
    unsigned integer_bits = 64;
    unsigned real_bits = 32;
};
inline constexpr device_conversion device_conversions[] = {
    {"int2float_rn", "llvm.nvvm.i2f.rn", true, true, 0, 32},
    {"int2float_rz", "llvm.nvvm.i2f.rz", true, true, 1, 32},
    {"int2float_rd", "llvm.nvvm.i2f.rm", true, true, 2, 32},
    {"int2float_ru", "llvm.nvvm.i2f.rp", true, true, 3, 32},
    {"uint2float_rn", "llvm.nvvm.ui2f.rn", true, false, 0, 32},
    {"uint2float_rz", "llvm.nvvm.ui2f.rz", true, false, 1, 32},
    {"uint2float_rd", "llvm.nvvm.ui2f.rm", true, false, 2, 32},
    {"uint2float_ru", "llvm.nvvm.ui2f.rp", true, false, 3, 32},
    {"float2int_rn", "llvm.nvvm.f2i.rn", false, true, 0, 32},
    {"float2int_rz", "llvm.nvvm.f2i.rz", false, true, 1, 32},
    {"float2int_rd", "llvm.nvvm.f2i.rm", false, true, 2, 32},
    {"float2int_ru", "llvm.nvvm.f2i.rp", false, true, 3, 32},
    {"float2uint_rn", "llvm.nvvm.f2ui.rn", false, false, 0, 32},
    {"float2uint_rz", "llvm.nvvm.f2ui.rz", false, false, 1, 32},
    {"float2uint_rd", "llvm.nvvm.f2ui.rm", false, false, 2, 32},
    {"float2uint_ru", "llvm.nvvm.f2ui.rp", false, false, 3, 32},
    {"ll2float_rn", "llvm.nvvm.ll2f.rn", true, true, 0},
    {"ll2float_rz", "llvm.nvvm.ll2f.rz", true, true, 1},
    {"ll2float_rd", "llvm.nvvm.ll2f.rm", true, true, 2},
    {"ll2float_ru", "llvm.nvvm.ll2f.rp", true, true, 3},
    {"ull2float_rn", "llvm.nvvm.ull2f.rn", true, false, 0},
    {"ull2float_rz", "llvm.nvvm.ull2f.rz", true, false, 1},
    {"ull2float_rd", "llvm.nvvm.ull2f.rm", true, false, 2},
    {"ull2float_ru", "llvm.nvvm.ull2f.rp", true, false, 3},
    {"float2ll_rn", "llvm.nvvm.f2ll.rn", false, true, 0},
    {"float2ll_rz", "llvm.nvvm.f2ll.rz", false, true, 1},
    {"float2ll_rd", "llvm.nvvm.f2ll.rm", false, true, 2},
    {"float2ll_ru", "llvm.nvvm.f2ll.rp", false, true, 3},
    {"float2ull_rn", "llvm.nvvm.f2ull.rn", false, false, 0},
    {"float2ull_rz", "llvm.nvvm.f2ull.rz", false, false, 1},
    {"float2ull_rd", "llvm.nvvm.f2ull.rm", false, false, 2},
    {"float2ull_ru", "llvm.nvvm.f2ull.rp", false, false, 3},
    {"int2double_rn", "llvm.nvvm.i2d.rn", true, true, 0, 32, 64},
    {"uint2double_rn", "llvm.nvvm.ui2d.rn", true, false, 0, 32, 64},
    {"ll2double_rn", "llvm.nvvm.ll2d.rn", true, true, 0, 64, 64},
    {"ll2double_rz", "llvm.nvvm.ll2d.rz", true, true, 1, 64, 64},
    {"ll2double_rd", "llvm.nvvm.ll2d.rm", true, true, 2, 64, 64},
    {"ll2double_ru", "llvm.nvvm.ll2d.rp", true, true, 3, 64, 64},
    {"ull2double_rn", "llvm.nvvm.ull2d.rn", true, false, 0, 64, 64},
    {"ull2double_rz", "llvm.nvvm.ull2d.rz", true, false, 1, 64, 64},
    {"ull2double_rd", "llvm.nvvm.ull2d.rm", true, false, 2, 64, 64},
    {"ull2double_ru", "llvm.nvvm.ull2d.rp", true, false, 3, 64, 64},
    {"double2int_rn", "llvm.nvvm.d2i.rn", false, true, 0, 32, 64},
    {"double2int_rz", "llvm.nvvm.d2i.rz", false, true, 1, 32, 64},
    {"double2int_rd", "llvm.nvvm.d2i.rm", false, true, 2, 32, 64},
    {"double2int_ru", "llvm.nvvm.d2i.rp", false, true, 3, 32, 64},
    {"double2uint_rn", "llvm.nvvm.d2ui.rn", false, false, 0, 32, 64},
    {"double2uint_rz", "llvm.nvvm.d2ui.rz", false, false, 1, 32, 64},
    {"double2uint_rd", "llvm.nvvm.d2ui.rm", false, false, 2, 32, 64},
    {"double2uint_ru", "llvm.nvvm.d2ui.rp", false, false, 3, 32, 64},
    {"double2ll_rn", "llvm.nvvm.d2ll.rn", false, true, 0, 64, 64},
    {"double2ll_rz", "llvm.nvvm.d2ll.rz", false, true, 1, 64, 64},
    {"double2ll_rd", "llvm.nvvm.d2ll.rm", false, true, 2, 64, 64},
    {"double2ll_ru", "llvm.nvvm.d2ll.rp", false, true, 3, 64, 64},
    {"double2ull_rn", "llvm.nvvm.d2ull.rn", false, false, 0, 64, 64},
    {"double2ull_rz", "llvm.nvvm.d2ull.rz", false, false, 1, 64, 64},
    {"double2ull_rd", "llvm.nvvm.d2ull.rm", false, false, 2, 64, 64},
    {"double2ull_ru", "llvm.nvvm.d2ull.rp", false, false, 3, 64, 64},
};
struct double_conversion {
    const char* name;
    const char* intrinsic;
    unsigned mode;
};
inline constexpr double_conversion double_conversions[] = {
    {"double2float_rn", "llvm.nvvm.d2f.rn", 0},
    {"double2float_rz", "llvm.nvvm.d2f.rz", 1},
    {"double2float_rd", "llvm.nvvm.d2f.rm", 2},
    {"double2float_ru", "llvm.nvvm.d2f.rp", 3},
};
}
