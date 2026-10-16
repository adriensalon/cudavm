#pragma once
#include <string>
namespace cudavm {
// CUDA DPX public functions also have fallbacks on pre-SM90 architectures.
// Keep the checked wrapper signatures and semantics shared by AOT and NVRTC.
struct device_dpx_operation {
    const char* name;
    unsigned bits, arguments;
    bool signed_value, maximum, add, relu, predicate;
};
inline constexpr device_dpx_operation device_dpx_operations[] = {
#define CV_DPX_PAIR(name, args, sign, maximum, add, relu, pred) \
    {name "32" relu, 32, args, sign, maximum, add, sizeof(relu)>1, pred}, \
    {name "16x2" relu, 16, args, sign, maximum, add, sizeof(relu)>1, pred}
    CV_DPX_PAIR("vimax_s", 2, true, true, false, "_relu", false),
    CV_DPX_PAIR("vimin_s", 2, true, false, false, "_relu", false),
    CV_DPX_PAIR("vimax3_s", 3, true, true, false, "", false),
    CV_DPX_PAIR("vimax3_u", 3, false, true, false, "", false),
    CV_DPX_PAIR("vimin3_s", 3, true, false, false, "", false),
    CV_DPX_PAIR("vimin3_u", 3, false, false, false, "", false),
    CV_DPX_PAIR("vimax3_s", 3, true, true, false, "_relu", false),
    CV_DPX_PAIR("vimin3_s", 3, true, false, false, "_relu", false),
    CV_DPX_PAIR("viaddmax_s", 3, true, true, true, "", false),
    CV_DPX_PAIR("viaddmax_u", 3, false, true, true, "", false),
    CV_DPX_PAIR("viaddmin_s", 3, true, false, true, "", false),
    CV_DPX_PAIR("viaddmin_u", 3, false, false, true, "", false),
    CV_DPX_PAIR("viaddmax_s", 3, true, true, true, "_relu", false),
    CV_DPX_PAIR("viaddmin_s", 3, true, false, true, "_relu", false),
    CV_DPX_PAIR("vibmax_s", 2, true, true, false, "", true),
    CV_DPX_PAIR("vibmax_u", 2, false, true, false, "", true),
    CV_DPX_PAIR("vibmin_s", 2, true, false, false, "", true),
    CV_DPX_PAIR("vibmin_u", 2, false, false, false, "", true),
#undef CV_DPX_PAIR
};
inline std::string device_dpx_header() {
    std::string result;
    for(auto operation:device_dpx_operations) {
        std::string word=operation.bits==32 && operation.signed_value?"int":"unsigned int";
        std::string arguments=word+" a,"+word+" b", call="a,b";
        if(operation.arguments==3){arguments+=","+word+" c";call+=",c";}
        if(operation.predicate) {
            arguments+=",bool* hi";call+=",hi";
            if(operation.bits==16){arguments+=",bool* lo";call+=",lo";}
        }
        auto symbol=std::string("__cudavm_rtc_dpx_")+operation.name;
        result+="extern \"C\" __device__ "+word+" "+symbol+"("+arguments+");\n";
        result+="__device__ inline "+word+" __"+operation.name+"("+arguments+"){return "+symbol+"("+call+");}\n";
    }
    return result;
}
}
