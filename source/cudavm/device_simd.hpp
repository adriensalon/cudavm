#pragma once
namespace cudavm {
// Public SIMD arithmetic/comparison profile. The operation index is shared with cvSimd;
// asm is the exact unmasked, same-signedness Clang CUDA wrapper form.
struct device_simd_operation {
    const char* name;
    const char* assembly;
    unsigned bits, operation, arguments;
};
inline constexpr device_simd_operation device_simd_operations[] = {
#define CV_SIMD_PAIR(name, instruction, type, modifier, op, args) \
    {name "2", instruction "2." type "." type "." type modifier " $0,$1,$2,$3;", 16, op, args}, \
    {name "4", instruction "4." type "." type "." type modifier " $0,$1,$2,$3;", 8, op, args}
    CV_SIMD_PAIR("vabs", "vabsdiff", "s32", "", 0, 1),
    CV_SIMD_PAIR("vabsss", "vabsdiff", "s32", ".sat", 1, 1),
    CV_SIMD_PAIR("vadd", "vadd", "u32", "", 2, 2),
    CV_SIMD_PAIR("vaddss", "vadd", "s32", ".sat", 3, 2),
    CV_SIMD_PAIR("vaddus", "vadd", "u32", ".sat", 4, 2),
    CV_SIMD_PAIR("vavgs", "vavrg", "s32", "", 5, 2),
    CV_SIMD_PAIR("vavgu", "vavrg", "u32", "", 6, 2),
    {"vhaddu2", nullptr, 16, 7, 2}, {"vhaddu4", nullptr, 8, 7, 2},
    CV_SIMD_PAIR("vsub", "vsub", "u32", "", 8, 2),
    CV_SIMD_PAIR("vsubss", "vsub", "s32", ".sat", 9, 2),
    CV_SIMD_PAIR("vsubus", "vsub", "u32", ".sat", 10, 2),
    CV_SIMD_PAIR("vneg", "vsub", "u32", "", 11, 1),
    CV_SIMD_PAIR("vnegss", "vsub", "s32", ".sat", 12, 1),
    CV_SIMD_PAIR("vabsdiffs", "vabsdiff", "s32", "", 13, 2),
    CV_SIMD_PAIR("vabsdiffu", "vabsdiff", "u32", "", 14, 2),
    CV_SIMD_PAIR("vsads", "vabsdiff", "s32", ".add", 15, 2),
    CV_SIMD_PAIR("vsadu", "vabsdiff", "u32", ".add", 16, 2),
    CV_SIMD_PAIR("vmaxs", "vmax", "s32", "", 17, 2),
    CV_SIMD_PAIR("vmaxu", "vmax", "u32", "", 18, 2),
    CV_SIMD_PAIR("vmins", "vmin", "s32", "", 19, 2),
    CV_SIMD_PAIR("vminu", "vmin", "u32", "", 20, 2),
#undef CV_SIMD_PAIR
#define CV_SIMD_SET(name, type, predicate, op) \
    {name "2", "vset2." type "." type "." predicate " $0,$1,$2,$3;", 16, op, 2}, \
    {name "4", "vset4." type "." type "." predicate " $0,$1,$2,$3;", 8, op, 2}
    CV_SIMD_SET("vseteq", "u32", "eq", 21),
    CV_SIMD_SET("vsetges", "s32", "ge", 22),
    CV_SIMD_SET("vsetgeu", "u32", "ge", 23),
    CV_SIMD_SET("vsetgts", "s32", "gt", 24),
    CV_SIMD_SET("vsetgtu", "u32", "gt", 25),
    CV_SIMD_SET("vsetles", "s32", "le", 26),
    CV_SIMD_SET("vsetleu", "u32", "le", 27),
    CV_SIMD_SET("vsetlts", "s32", "lt", 28),
    CV_SIMD_SET("vsetltu", "u32", "lt", 29),
    CV_SIMD_SET("vsetne", "u32", "ne", 30),
#undef CV_SIMD_SET
// Mask comparisons expand the corresponding vset boolean lanes to all bits.
#define CV_SIMD_CMP(name, op) {name "2", nullptr, 16, op, 2}, {name "4", nullptr, 8, op, 2}
    CV_SIMD_CMP("vcmpeq", 31), CV_SIMD_CMP("vcmpges", 32),
    CV_SIMD_CMP("vcmpgeu", 33), CV_SIMD_CMP("vcmpgts", 34),
    CV_SIMD_CMP("vcmpgtu", 35), CV_SIMD_CMP("vcmples", 36),
    CV_SIMD_CMP("vcmpleu", 37), CV_SIMD_CMP("vcmplts", 38),
    CV_SIMD_CMP("vcmpltu", 39), CV_SIMD_CMP("vcmpne", 40),
#undef CV_SIMD_CMP
};
}
