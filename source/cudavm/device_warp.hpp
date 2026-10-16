#pragma once
#include <string>
#include <string_view>

namespace cudavm {
// NVRTC uses named declarations rather than the consumed Clang archive's
// target-builtin enum. Match-all additionally receives predicate storage;
// the frontend converts its result and store to the native NVVM aggregate.
inline std::string device_warp_header() {
    std::string header = R"warp(
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_shfl_idx(unsigned,unsigned,unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_shfl_up(unsigned,unsigned,unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_shfl_down(unsigned,unsigned,unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_shfl_bfly(unsigned,unsigned,unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) bool __cudavm_rtc_vote_all(unsigned,bool);
extern "C" __device__ __attribute__((convergent)) bool __cudavm_rtc_vote_any(unsigned,bool);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_vote_ballot(unsigned,bool);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_activemask();
extern "C" __device__ __attribute__((convergent)) void __cudavm_rtc_syncwarp(unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_match_any32(unsigned,unsigned);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_match_any64(unsigned,unsigned long long);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_match_all32(unsigned,unsigned,int*);
extern "C" __device__ __attribute__((convergent)) unsigned __cudavm_rtc_match_all64(unsigned,unsigned long long,int*);
__device__ inline void __syncwarp(unsigned mask=0xffffffffu) {__cudavm_rtc_syncwarp(mask);}
__device__ inline int __all_sync(unsigned mask,int predicate) {return __cudavm_rtc_vote_all(mask,predicate!=0);}
__device__ inline int __any_sync(unsigned mask,int predicate) {return __cudavm_rtc_vote_any(mask,predicate!=0);}
__device__ inline unsigned __ballot_sync(unsigned mask,int predicate) {return __cudavm_rtc_vote_ballot(mask,predicate!=0);}
__device__ inline unsigned __activemask() {return __cudavm_rtc_activemask();}
)warp";
    for(auto mode : {"idx", "up", "down", "bfly"}) {
        std::string name=std::string(mode)=="idx"?"__shfl_sync":std::string(mode)=="bfly"?"__shfl_xor_sync":std::string("__shfl_")+mode+"_sync";
        for(auto type : {"int","unsigned","long","unsigned long","long long","unsigned long long","float","double"}) {
            header += "__device__ inline "+std::string(type)+" "+name+"(unsigned mask,"+type+" value,"+
                (std::string(mode)=="up" || std::string(mode)=="down"?"unsigned":"int")+" offset,int width=32) {\n";
            header += "union {"+std::string(type)+" value;unsigned words[sizeof("+type+")/4];} bits;bits.value=value;\n";
            header += "for(unsigned i=0;i<sizeof(value)/4;++i) bits.words[i]=__cudavm_rtc_shfl_"+std::string(mode)+
                "(mask,bits.words[i],unsigned(offset),((32-unsigned(width))<<8)|"+(std::string(mode)=="up"?"0u":"31u")+");\nreturn bits.value;\n}\n";
        }
    }
    for(auto type : {"int","unsigned","long","unsigned long","long long","unsigned long long","float","double"}) {
        for(auto mode : {"any","all"}) {
            bool all=std::string_view(mode)=="all";
            header += "__device__ inline unsigned __match_"+std::string(mode)+"_sync(unsigned mask,"+type+" value"+(all?",int* pred":"")+") {\n";
            header += "union {"+std::string(type)+" value;unsigned word;unsigned long long wide;} bits;bits.value=value;\n";
            header += "if(sizeof(value)==4)return __cudavm_rtc_match_"+std::string(mode)+"32(mask,bits.word"+(all?",pred":"")+");\n";
            header += "else return __cudavm_rtc_match_"+std::string(mode)+"64(mask,bits.wide"+(all?",pred":"")+");\n}\n";
        }
    }
    header += "#if __CUDA_ARCH__ >= 800\n";
    for(auto mode : {"add","min","max","umin","umax","and","or","xor"}) {
        std::string suffix=mode;
        bool signed_value=suffix=="min" || suffix=="max";
        auto type=signed_value?"int":"unsigned";
        auto name=suffix=="umin"?"min":suffix=="umax"?"max":mode;
        header += "extern \"C\" __device__ __attribute__((convergent)) "+std::string(type)+" __cudavm_rtc_reduce_"+suffix+"("+type+",unsigned);\n";
        header += "__device__ inline "+std::string(type)+" __reduce_"+name+"_sync(unsigned mask,"+type+" value) {return __cudavm_rtc_reduce_"+suffix+"(value,mask);}\n";
    }
    header += "__device__ inline int __reduce_add_sync(unsigned mask,int value) {return int(__cudavm_rtc_reduce_add(unsigned(value),mask));}\n#endif\n";
    return header;
}
}
