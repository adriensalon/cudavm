#pragma once
#include <string>
#include <string_view>
#include "device_sources.hpp"

namespace cudavm {
// NVRTC uses named declarations rather than the consumed Clang archive's
// target-builtin enum. Match-all additionally receives predicate storage;
// the frontend converts its result and store to the native NVVM aggregate.
inline std::string device_warp_header() {
    std::string header = std::string(device_source("device_warp.cu"));
    for(auto mode : {"idx", "up", "down", "bfly"}) {
        std::string name=std::string(mode)=="idx"?"__shfl_sync":std::string(mode)=="bfly"?"__shfl_xor_sync":std::string("__shfl_")+mode+"_sync";
        for(auto type : {"int","unsigned","long","unsigned long","long long","unsigned long long","float","double"}) {
            header += render_device_source("device_shuffle.cu.in", {
                {"type", type}, {"name", name}, {"mode", mode},
                {"offset_type", std::string_view(mode)=="up" || std::string_view(mode)=="down" ? "unsigned" : "int"},
                {"clamp", std::string_view(mode)=="up" ? "0u" : "31u"}});
        }
    }
    for(auto type : {"int","unsigned","long","unsigned long","long long","unsigned long long","float","double"}) {
        for(auto mode : {"any","all"}) {
            bool all=std::string_view(mode)=="all";
            header += render_device_source("device_match.cu.in", {
                {"type", type}, {"mode", mode},
                {"predicate_argument", all ? ",int* pred" : ""}, {"predicate_call", all ? ",pred" : ""}});
        }
    }
    header += "#if __CUDA_ARCH__ >= 800\n";
    for(auto mode : {"add","min","max","umin","umax","and","or","xor"}) {
        std::string suffix=mode;
        bool signed_value=suffix=="min" || suffix=="max";
        auto type=signed_value?"int":"unsigned";
        auto name=suffix=="umin"?"min":suffix=="umax"?"max":mode;
        header += render_device_source("device_reduce.cu.in", {
            {"type", type}, {"suffix", suffix}, {"name", name}});
    }
    header += device_source("device_reduce.cu");
    header += "#endif\n";
    return header;
}
}
