#pragma once
#include <string>
namespace cudavm {
inline std::string device_half_header() { return R"cudavm_half(
namespace cudavm {
__device__ static inline unsigned long long half_shift_jam(unsigned long long a,unsigned shift) {
    return shift==0?a:shift<64?(a>>shift)|((a<<(64-shift))!=0):a!=0;
}
}
// Round magnitude * 2^scale directly to binary16, without an intermediate
// floating-point rounding. Modes match device_conversions.hpp.
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f16_round(unsigned long long magnitude,int scale,unsigned sign,unsigned mode) {
    if(!magnitude)return sign;
    int top=63-__builtin_clzll(magnitude),exponent=top+scale;
    int shift=(exponent<-14?-24:exponent-10)-scale;
    unsigned long long retained=magnitude,remainder=0,midpoint=0;
    if(shift>0) {
        retained=shift<64?magnitude>>shift:0;
        remainder=shift<64?magnitude&((1ull<<shift)-1):magnitude;
        midpoint=shift<=64?1ull<<(shift-1):0;
    } else retained=magnitude<<unsigned(-shift);
    bool increment=mode==0?(shift>0 && shift<=64 && (remainder>midpoint || (remainder==midpoint && (retained&1)))):
        mode==1?false:mode==2?(sign!=0 && remainder!=0):(sign==0 && remainder!=0);
    retained+=increment;
    if(exponent<-14)return sign|unsigned(retained);
    if(retained>=2048){retained>>=1;++exponent;}
    if(exponent>15)return sign|((mode==1 || (mode==2 && !sign) || (mode==3 && sign))?0x7bffu:0x7c00u);
    return sign|unsigned((exponent+15)<<10)|(unsigned(retained)&1023u);
}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f16_to_f32(unsigned a) {
    unsigned sign=(a&0x8000u)<<16,exponent=(a>>10)&31u,mantissa=a&1023u;
    if(exponent==31)return mantissa?0x7fffffffu:sign|0x7f800000u;
    if(!exponent) {
        if(!mantissa)return sign;
        unsigned shift=__builtin_clz(mantissa)-21u;
        return sign|((113u-shift)<<23)|((mantissa<<shift&1023u)<<13);
    }
    return sign|((exponent+112u)<<23)|(mantissa<<13);
}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f32_to_f16(unsigned a,unsigned mode) {
    unsigned sign=(a>>16)&0x8000u,exponent=(a>>23)&255u,mantissa=a&0x7fffffu;
    if(exponent==255)return mantissa?0x7fffu:sign|0x7c00u;
    // FP32 needs only 24 significand bits. Keep this common conversion in
    // native word arithmetic; the wider rounder remains for FP64 and FMA.
    if(exponent<102)return sign|unsigned((a&0x7fffffffu)!=0 && (mode==2?sign!=0:mode==3?sign==0:false));
    auto overflow=[&](){return sign|((mode==1 || (mode==2 && !sign) || (mode==3 && sign))?0x7bffu:0x7c00u);};
    if(exponent>142)return overflow();
    unsigned shift=exponent>=113?13:126-exponent,magnitude=mantissa|0x800000u;
    unsigned retained=magnitude>>shift,remainder=magnitude&((1u<<shift)-1),midpoint=1u<<(shift-1);
    retained+=mode==0?(remainder>midpoint || (remainder==midpoint && (retained&1))):mode==1?false:mode==2?(sign && remainder):(!sign && remainder);
    if(exponent<113)return sign|retained;
    if(retained>=2048){retained>>=1;++exponent;}
    return exponent>142?overflow():sign|((exponent-112)<<10)|(retained&1023u);
}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f64_to_f16(unsigned long long a,unsigned mode) {
    unsigned sign=unsigned(a>>48)&0x8000u,exponent=unsigned(a>>52)&2047u;
    unsigned long long mantissa=a&0xfffffffffffffull;
    if(exponent==2047)return mantissa?0x7fffu:sign|0x7c00u;
    return __cudavm_native_word_f16_round(exponent?mantissa|0x10000000000000ull:mantissa,exponent?int(exponent)-1075:-1074,sign,mode);
}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f16_to_integer(unsigned h,unsigned mode,unsigned bits,unsigned unsign) {
    unsigned sign=h&0x8000u,exponent=(h>>10)&31u,mantissa=h&1023u;
    unsigned long long maximum=unsign?(bits==64?~0ull:(1ull<<bits)-1):(1ull<<(bits-1))-1;
    if(exponent==31) {
        if(mantissa)return bits==64?0x8000000000000000ull:0;
        return sign?(unsign?0:0ull-maximum-1):maximum;
    }
    int scale=int(exponent?exponent:1)-25;
    unsigned magnitude=mantissa|(exponent?1024u:0u),rounded=magnitude;
    if(scale<0) {
        unsigned shift=unsigned(-scale),discarded=magnitude&((1u<<shift)-1),midpoint=1u<<(shift-1);
        rounded=magnitude>>shift;
        rounded+=mode==0?(discarded>midpoint || (discarded==midpoint && (rounded&1))):mode==1?false:mode==2?(sign && discarded):(!sign && discarded);
    } else rounded<<=unsigned(scale);
    if(unsign && sign)return 0;
    unsigned long long limit=maximum+(!unsign && sign?1u:0u),value=rounded>limit?limit:rounded;
    return sign?0ull-value:value;
}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f16_add(unsigned a,unsigned b) {
    float x=__builtin_bit_cast(float,__cudavm_native_word_f16_to_f32(a));
    float y=__builtin_bit_cast(float,__cudavm_native_word_f16_to_f32(b));
    return __cudavm_native_word_f32_to_f16(__builtin_bit_cast(unsigned,x+y),0);
}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f16_mul(unsigned a,unsigned b) {
    float x=__builtin_bit_cast(float,__cudavm_native_word_f16_to_f32(a));
    float y=__builtin_bit_cast(float,__cudavm_native_word_f16_to_f32(b));
    return __cudavm_native_word_f32_to_f16(__builtin_bit_cast(unsigned,x*y),0);
}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f16_minmax(unsigned a,unsigned b,unsigned maximum,unsigned propagate) {
    unsigned ax=a&0x7fffu,bx=b&0x7fffu;
    bool an=ax>0x7c00u,bn=bx>0x7c00u;
    if(an || bn)return propagate || (an && bn)?0x7fffu:an?b:a;
    if(!(ax|bx))return maximum?a&b:a|b;
    unsigned ao=a^((a&0x8000u)?0xffffu:0x8000u),bo=b^((b&0x8000u)?0xffffu:0x8000u);
    return (maximum?ao>bo:ao<bo)?a:b;
}
// Divide exact binary16 significands, retaining a sticky remainder before the
// single binary16 rounding. Widening division through FP32 can double round.
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f16_div(unsigned a,unsigned b) {
    unsigned ax=a&0x7fffu,bx=b&0x7fffu,sign=(a^b)&0x8000u;
    if(ax>0x7c00u || bx>0x7c00u || (!ax && !bx) || (ax==0x7c00u && bx==0x7c00u))return 0x7fffu;
    if(ax==0x7c00u || !bx)return sign|0x7c00u;
    if(!ax || bx==0x7c00u)return sign;
    unsigned ae=ax>>10,be=bx>>10,m=(ax&1023u)|(ae?1024u:0u),n=(bx&1023u)|(be?1024u:0u);
    unsigned long long numerator=static_cast<unsigned long long>(m)<<32;
    unsigned long long quotient=numerator/n;
    quotient|=numerator%n!=0;
    return __cudavm_native_word_f16_round(quotient,int(ae?ae:1)-int(be?be:1)-32,sign,0);
}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f16_fma(unsigned a,unsigned b,unsigned c) {
    unsigned ax=a&0x7fffu,bx=b&0x7fffu,cx=c&0x7fffu,sign=(a^b)&0x8000u,cs=c&0x8000u;
    if(ax>0x7c00u || bx>0x7c00u || cx>0x7c00u || (ax==0x7c00u && !bx) || (bx==0x7c00u && !ax))return 0x7fffu;
    if(ax==0x7c00u || bx==0x7c00u)return cx==0x7c00u && sign!=cs?0x7fffu:sign|0x7c00u;
    if(cx==0x7c00u)return c;
    unsigned ae=ax>>10,be=bx>>10,ce=cx>>10;
    unsigned long long product=((ax&1023u)|(ae?1024u:0u))*((bx&1023u)|(be?1024u:0u));
    unsigned long long addend=(cx&1023u)|(ce?1024u:0u);
    int pe=int(ae?ae:1)+int(be?be:1)-50,qe=int(ce?ce:1)-25;
    if(!product && !addend)return sign==cs?sign:0u;
    if(!product)return c;
    int top=pe+22>qe+11?pe+22:qe+11,base=top-60;
    product=pe>=base?product<<unsigned(pe-base):cudavm::half_shift_jam(product,unsigned(base-pe));
    addend=qe>=base?addend<<unsigned(qe-base):cudavm::half_shift_jam(addend,unsigned(base-qe));
    unsigned long long magnitude;
    if(sign==cs)magnitude=product+addend;
    else if(product>=addend)magnitude=product-addend;
    else {magnitude=addend-product;sign=cs;}
    return magnitude?__cudavm_native_word_f16_round(magnitude,base,sign,0):0u;
}
// DXIL's multiply-add does not promise a single binary32 rounding. Keep the
// exact product and a jammed addend until the final rounding, including FTZ
// sensitive inputs. Other shader inputs use this same strict FMA contract.
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f32_fma(unsigned a,unsigned b,unsigned c) {
    unsigned ax=a&0x7fffffffu,bx=b&0x7fffffffu,cx=c&0x7fffffffu,sign=(a^b)&0x80000000u,cs=c&0x80000000u;
    if(ax>0x7f800000u || bx>0x7f800000u || cx>0x7f800000u || (ax==0x7f800000u && !bx) || (bx==0x7f800000u && !ax))return 0x7fc00000u;
    if(ax==0x7f800000u || bx==0x7f800000u)return cx==0x7f800000u && sign!=cs?0x7fc00000u:sign|0x7f800000u;
    if(cx==0x7f800000u)return c;
    unsigned ae=ax>>23,be=bx>>23,ce=cx>>23;
    unsigned long long product=static_cast<unsigned long long>((ax&0x7fffffu)|(ae?0x800000u:0u))*((bx&0x7fffffu)|(be?0x800000u:0u));
    unsigned long long addend=(cx&0x7fffffu)|(ce?0x800000u:0u);
    int pe=int(ae?ae:1)+int(be?be:1)-300,qe=int(ce?ce:1)-150;
    if(!product && !addend)return sign==cs?sign:0u;
    if(!product)return c;
    int top=pe+48>qe+24?pe+48:qe+24,base=top-60;
    product=pe>=base?product<<unsigned(pe-base):cudavm::half_shift_jam(product,unsigned(base-pe));
    addend=qe>=base?addend<<unsigned(qe-base):cudavm::half_shift_jam(addend,unsigned(base-qe));
    unsigned long long magnitude;
    if(sign==cs)magnitude=product+addend;
    else if(product>=addend)magnitude=product-addend;
    else {magnitude=addend-product;sign=cs;}
    if(!magnitude)return 0;
    int exponent=63-__builtin_clzll(magnitude)+base;
    int shift=(exponent<-126?-149:exponent-23)-base;
    unsigned long long q=magnitude,rem=0,mid=0;
    if(shift>0){q=shift<64?magnitude>>shift:0;rem=shift<64?magnitude&((1ull<<shift)-1):magnitude;mid=shift<=64?1ull<<(shift-1):0;}
    else q=magnitude<<unsigned(-shift);
    q+=shift>0 && shift<=64 && (rem>mid || (rem==mid && (q&1)));
    if(exponent<-126)return sign|unsigned(q);
    if(q>=0x1000000ull){q>>=1;++exponent;}
    return exponent>127?sign|0x7f800000u:sign|unsigned(exponent+127)<<23|(unsigned(q)&0x7fffffu);
}
)cudavm_half"; }
}
