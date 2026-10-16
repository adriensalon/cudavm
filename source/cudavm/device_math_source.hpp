#pragma once
#include <bit>
#include <cstdint>
#include <string>
#include "device_numeric_source.hpp"
#include "device_half_source.hpp"

namespace cudavm {
// One scalar implementation is compiled as C++, as CUDA device code in the
// frontend's virtual filesystem before native LLVM target lowering.
#define CV_MATH_FN inline
#define CV_BITS(x) std::bit_cast<uint32_t>(x)
#define CV_FLOAT(x) std::bit_cast<float>(uint32_t(x))
#define CV_MATH_PRECISE
#define CV_MATH_CONTRACT
#define CV_MATH(...) using uint = uint32_t; __VA_ARGS__ inline constexpr char math_source[] = #__VA_ARGS__;
CV_MATH(
CV_MATH_FN uint cvFloatCompare(uint a,uint b,uint mask) {
    uint ax=a&0x7fffffffu,bx=b&0x7fffffffu;
    if(ax>0x7f800000u || bx>0x7f800000u) return (mask&8u)!=0u?1u:0u;
    uint relation=1u;
    if(a!=b && (ax|bx)!=0u) {
        uint oa=(a&0x80000000u)!=0u?~a:a|0x80000000u;
        uint ob=(b&0x80000000u)!=0u?~b:b|0x80000000u;
        relation=oa<ob?4u:2u;
    }
    return (mask&relation)!=0u?1u:0u;
}
CV_MATH_FN float cvFloatDivide(float a,float b) {
    uint x=CV_BITS(a),y=CV_BITS(b),ax=x&0x7fffffffu,ay=y&0x7fffffffu,sign=(x^y)&0x80000000u;
    if(ax>0x7f800000u) return CV_FLOAT(x|0x00400000u);
    if(ay>0x7f800000u) return CV_FLOAT(y|0x00400000u);
    if(ax==0x7f800000u) return CV_FLOAT(ay==0x7f800000u?0x7fc00000u:sign|0x7f800000u);
    if(ay==0x7f800000u) return CV_FLOAT(sign);
    if(ay==0u) return CV_FLOAT(ax==0u?0x7fc00000u:sign|0x7f800000u);
    if(ax==0u) return CV_FLOAT(sign);
    int ex=int(ax>>23u)-127,ey=int(ay>>23u)-127;
    uint m=(ax&0x007fffffu)|0x00800000u,n=(ay&0x007fffffu)|0x00800000u;
    if((ax>>23u)==0u) {m=ax;ex=-126;while(m<0x00800000u){m<<=1u;--ex;}}
    if((ay>>23u)==0u) {n=ay;ey=-126;while(n<0x00800000u){n<<=1u;--ey;}}
    int exponent=ex-ey;
    if(m<n) {m<<=1u;--exponent;}
    if(exponent>127) return CV_FLOAT(sign|0x7f800000u);
    if(exponent< -150) return CV_FLOAT(sign);
    if(exponent== -150) return CV_FLOAT(sign|(m>n?1u:0u));
    uint digits=exponent>=-126?24u:uint(exponent+150);
    uint quotient=1u<<(digits-1u),remainder=m-n;
    for(uint i=digits-1u;i>0u;) {
        --i;remainder<<=1u;
        if(remainder>=n) {remainder-=n;quotient|=1u<<i;}
    }
    uint twice=remainder<<1u;
    if(twice>n || (twice==n && (quotient&1u)!=0u)) ++quotient;
    if(exponent< -126) return CV_FLOAT(sign|quotient);
    if(quotient==0x01000000u) {quotient>>=1u;++exponent;}
    if(exponent>127) return CV_FLOAT(sign|0x7f800000u);
    return CV_FLOAT(sign|(uint(exponent+127)<<23u)|(quotient&0x007fffffu));
}
CV_MATH_FN float cvMathMax(float a, float b) {
    uint x=CV_BITS(a),y=CV_BITS(b),ax=x&0x7fffffffu,ay=y&0x7fffffffu;
    if(ax>0x7f800000u) return ay>0x7f800000u?CV_FLOAT(x|0x00400000u):b;
    if(ay>0x7f800000u) return a;
    if((ax|ay)==0u) return CV_FLOAT(x&y);
    uint ox=(x&0x80000000u)!=0u?~x:x|0x80000000u;
    uint oy=(y&0x80000000u)!=0u?~y:y|0x80000000u;
    return ox>oy?a:b;
}
CV_MATH_FN float cvMathAbs(float x) {return CV_FLOAT(CV_BITS(x)&0x7fffffffu);}
CV_MATH_FN float cvMathCopySign(float x,float y) {return CV_FLOAT((CV_BITS(x)&0x7fffffffu)|(CV_BITS(y)&0x80000000u));}
CV_MATH_FN float cvMathMin(float a,float b) {
    uint x=CV_BITS(a),y=CV_BITS(b),ax=x&0x7fffffffu,ay=y&0x7fffffffu;
    if(ax>0x7f800000u) return ay>0x7f800000u?CV_FLOAT(x|0x00400000u):b;
    if(ay>0x7f800000u) return a;
    if((ax|ay)==0u) return CV_FLOAT(x|y);
    uint ox=(x&0x80000000u)!=0u?~x:x|0x80000000u;
    uint oy=(y&0x80000000u)!=0u?~y:y|0x80000000u;
    return ox<oy?a:b;
}
// Integral rounding uses only binary32 bits, including subnormal inputs and
// signed-zero results. Modes: truncate, floor, ceil, ties-away, ties-even.
CV_MATH_FN float cvMathIntegral(float x,uint mode) {
    uint bits=CV_BITS(x),sign=bits&0x80000000u,m=bits&0x7fffffffu;
    if(m>0x7f800000u) return CV_FLOAT(bits|0x00400000u);
    if(m==0u || m>=0x4b000000u) return x;
    if(m<0x3f800000u) {
        uint up=mode==1u?sign!=0u?1u:0u:mode==2u?sign==0u?1u:0u:
            mode==3u?m>=0x3f000000u?1u:0u:mode==4u?m>0x3f000000u?1u:0u:0u;
        return CV_FLOAT(sign|(up!=0u?0x3f800000u:0u));
    }
    uint shift=150u-(m>>23u),unit=1u<<shift,mask=unit-1u,lost=m&mask,base=m&~mask;
    if(lost==0u) return x;
    uint up=mode==1u?sign!=0u?1u:0u:mode==2u?sign==0u?1u:0u:
        mode==3u?lost>=unit/2u?1u:0u:mode==4u?(lost>unit/2u || (lost==unit/2u && ((base>>shift)&1u)!=0u))?1u:0u:0u;
    return CV_FLOAT(sign|(base+(up!=0u?unit:0u)));
}
CV_MATH_FN float cvMathFloor(float x) {return cvMathIntegral(x,1u);}
CV_MATH_FN float cvMathCeil(float x) {return cvMathIntegral(x,2u);}
CV_MATH_FN float cvMathTrunc(float x) {return cvMathIntegral(x,0u);}
CV_MATH_FN float cvMathRound(float x) {return cvMathIntegral(x,3u);}
CV_MATH_FN float cvMathRint(float x) {return cvMathIntegral(x,4u);}
CV_MATH_FN float cvMathSqrt(float x) {
    uint bits=CV_BITS(x),m=bits&0x7fffffffu;
    if(m==0u) return x;
    if(m>0x7f800000u) return CV_FLOAT(bits|0x00400000u);
    if((bits&0x80000000u)!=0u) return CV_FLOAT(0x7fc00000u);
    if(m==0x7f800000u) return x;
    int exponent=int(m>>23u)-127;
    m=(m&0x007fffffu)|0x00800000u;
    if((bits&0x7f800000u)==0u) {
        exponent=-126;m=bits;
        while(m<0x00800000u) {m<<=1u;--exponent;}
    }
    // Restoring integer square root of the exact 48-bit scaled significand.
    uint shift=23u+(uint(exponent)&1u),lo=m<<shift,hi=m>>(32u-shift);
    uint root=0u,remainder=0u;
    for(uint i=24u;i>0u;) {
        --i;uint word=i>=16u?hi:lo;
        remainder=(remainder<<2u)|((word>>((i&15u)*2u))&3u);
        uint trial=(root<<2u)|1u;root<<=1u;
        if(remainder>=trial) {remainder-=trial;++root;}
    }
    // The exact midpoint square has a fractional .25, so no integer tie exists.
    if(remainder>root) ++root;
    int root_exponent=exponent>=0?exponent/2:-((-exponent+1)/2);
    return CV_FLOAT((uint(root_exponent+127)<<23u)|(root&0x007fffffu));
}
CV_MATH_FN uint cvMathTwoOverPi(uint i) {
    // floor((2/pi) * 2^192), least significant word first.
    if(i==0u) return 0x3c439041u;
    if(i==1u) return 0xdb629599u;
    if(i==2u) return 0xf534ddc0u;
    if(i==3u) return 0xfc2757d1u;
    if(i==4u) return 0x4e441529u;
    return 0xa2f9836eu;
}
CV_MATH_FN float cvMathScaled(uint m,int scale) {
    if(m==0u) return 0.0f;
    while(m<0x00800000u) {m<<=1u;--scale;}
    int exponent=scale+150;
    if(exponent>0) return CV_FLOAT((uint(exponent)<<23u)|(m&0x007fffffu));
    uint shift=uint(1-exponent);
    return shift<24u?CV_FLOAT(m>>shift):0.0f;
}
CV_MATH_FN float cvMathTrig(float x,uint cosine) {
    CV_MATH_CONTRACT
    uint bits=CV_BITS(x),absolute=bits&0x7fffffffu;
    if(absolute>=0x7f800000u) return CV_FLOAT(bits|0x00400000u);
    if(absolute<0x39800000u) return cosine!=0u?1.0f:x;
    CV_MATH_PRECISE float a=CV_FLOAT(absolute),r=0.0f;
    uint quadrant=0u;
    if(a<32.0f) {
        // Each high/middle product is exact in this bounded fast path.
        quadrant=uint(a*0.63661977236758134308f+0.5f);
        CV_MATH_PRECISE float n=float(quadrant);
        r=((a-n*1.5703125f)-n*0.0004837512969970703125f)-n*7.54978941586159635335e-8f;
    } else {
        // Payne-Hanek reduction for every finite binary32 exponent. All integer
        // products use 16-bit limbs; no FP64 or native 64-bit integer is needed.
        uint m=(absolute&0x007fffffu)|0x00800000u;
        uint product[7];uint carry=0u;
        for(uint i=0u;i<6u;++i) {
            uint word=cvMathTwoOverPi(i),p0=(m&65535u)*(word&65535u);
            uint p1=(m>>16u)*(word&65535u)+(p0>>16u);
            uint p2=(m&65535u)*(word>>16u)+(p1&65535u);
            uint lo=(p2<<16u)|(p0&65535u);
            uint hi=(m>>16u)*(word>>16u)+(p1>>16u)+(p2>>16u);
            uint sum=lo+carry;product[i]=sum;carry=hi+(sum<lo?1u:0u);
        }
        product[6]=carry;
        uint binary_point=342u-(absolute>>23u),word=binary_point>>5u,offset=binary_point&31u;
        quadrant=product[word]>>offset;
        if(offset!=0u && word<6u) quadrant|=product[word+1u]<<(32u-offset);
        quadrant&=3u;
        uint above_half=(product[(binary_point-1u)>>5u]>>((binary_point-1u)&31u))&1u;
        if(above_half!=0u) {
            ++quadrant;carry=1u;
            for(uint i=0u;i<=word;++i) {uint v=~product[i]+carry;carry=carry!=0u && v==0u?1u:0u;product[i]=v;}
        }
        product[word]&=offset==0u?0u:((1u<<offset)-1u);
        int lead=int(binary_point)-1;
        while(lead>=0 && ((product[uint(lead)>>5u]>>(uint(lead)&31u))&1u)==0u) --lead;
        uint head=0u,tail=0u;
        for(uint i=0u;i<48u;++i) {
            int position=lead-int(i);uint digit=0u;
            if(position>=0) digit=(product[uint(position)>>5u]>>(uint(position)&31u))&1u;
            if(i<24u) head=(head<<1u)|digit;else tail=(tail<<1u)|digit;
        }
        CV_MATH_PRECISE float f=cvMathScaled(head,lead-int(binary_point)-23);
        CV_MATH_PRECISE float lower=cvMathScaled(tail,lead-int(binary_point)-47);
        CV_MATH_PRECISE float high=CV_FLOAT(CV_BITS(f)&0xfffff000u),low=f-high;
        CV_MATH_PRECISE float first=high*1.5703125f;
        CV_MATH_PRECISE float rest=low*1.5703125f+high*0.0004837512969970703125f;
        rest=rest+low*0.0004837512969970703125f;
        rest=rest+f*7.54978941586159635335e-8f;
        rest=rest+lower*1.57079632679489661923f;
        r=first+rest;
        if(above_half!=0u) r=-r;
    }
    quadrant=(quadrant+cosine)&3u;
    CV_MATH_PRECISE float z=r*r,result;
    if((quadrant&1u)==0u) {
        CV_MATH_PRECISE float p=2.7557319223985893e-6f+z*(-2.505210838544172e-8f);
        p=-0.0001984126984126984f+z*p;
        p=0.008333333333333333f+z*p;
        p=-0.16666666666666667f+z*p;
        result=r+r*z*p;
    } else {
        CV_MATH_PRECISE float p=-2.755731922398589e-7f+z*2.08767569878681e-9f;
        p=2.48015873015873e-5f+z*p;
        p=-0.001388888888888889f+z*p;
        p=0.041666666666666664f+z*p;
        result=(1.0f-z*0.5f)+(z*z)*p;
    }
    uint sign=((quadrant&2u)<<30u)^(cosine==0u?(bits&0x80000000u):0u);
    return CV_FLOAT(CV_BITS(result)^sign);
}
CV_MATH_FN float cvMathSin(float x) {return cvMathTrig(x,0u);}
CV_MATH_FN float cvMathCos(float x) {return cvMathTrig(x,1u);}
)
#undef CV_MATH
#undef CV_MATH_FN
#undef CV_MATH_PRECISE
#undef CV_MATH_CONTRACT
#undef CV_BITS
#undef CV_FLOAT

inline std::string device_math_header() {
    return native_numeric_header()+device_half_header()+std::string("#pragma clang system_header\n"
        "#define CV_MATH_FN __device__ inline\n#define CV_MATH_PRECISE\n"
        "#define CV_MATH_CONTRACT _Pragma(\"clang fp contract(off)\")\n"
        "#define CV_BITS(x) __builtin_bit_cast(unsigned, (x))\n"
        "#define CV_FLOAT(x) __builtin_bit_cast(float, unsigned(x))\n"
        "namespace cudavm { typedef unsigned uint;\n")+math_source+
        "\n}\nextern \"C\" __device__ inline float __nv_sinf(float x) {return cudavm::cvMathSin(x);}\n"
        "extern \"C\" __device__ inline float __nv_cosf(float x) {return cudavm::cvMathCos(x);}\n"
        "extern \"C\" __device__ inline float __nv_fabsf(float x) {return cudavm::cvMathAbs(x);}\n"
        "extern \"C\" __device__ inline float __nv_fminf(float x,float y) {return cudavm::cvMathMin(x,y);}\n"
        "extern \"C\" __device__ inline float __nv_copysignf(float x,float y) {return cudavm::cvMathCopySign(x,y);}\n"
        "extern \"C\" __device__ inline float __nv_floorf(float x) {return cudavm::cvMathFloor(x);}\n"
        "extern \"C\" __device__ inline float __nv_ceilf(float x) {return cudavm::cvMathCeil(x);}\n"
        "extern \"C\" __device__ inline float __nv_truncf(float x) {return cudavm::cvMathTrunc(x);}\n"
        "extern \"C\" __device__ inline float __nv_roundf(float x) {return cudavm::cvMathRound(x);}\n"
        "extern \"C\" __device__ inline float __nv_rintf(float x) {return cudavm::cvMathRint(x);}\n"
        "extern \"C\" __device__ inline float __nv_nearbyintf(float x) {return cudavm::cvMathRint(x);}\n"
        // Retain exact scalar helpers in portable IR. Shader lowering can
        // select them on its private clone while NVPTX retains CUDA operations.
        "extern \"C\" __device__ __attribute__((used)) inline float __cudavm_native_sqrt(float x) {return cudavm::cvMathSqrt(x);}\n"
        "extern \"C\" __device__ __attribute__((used)) inline float __cudavm_native_max(float x,float y) {return cudavm::cvMathMax(x,y);}\n"
        "extern \"C\" __device__ __attribute__((used)) inline float __cudavm_native_divide(float x,float y) {return cudavm::cvFloatDivide(x,y);}\n"
        "extern \"C\" __device__ __attribute__((used)) inline float __cudavm_native_integral(float x,unsigned mode) {return cudavm::cvMathIntegral(x,mode);}\n"
        "extern \"C\" __device__ __attribute__((used)) inline unsigned __cudavm_native_compare(unsigned x,unsigned y,unsigned predicate) {return cudavm::cvFloatCompare(x,y,predicate);}\n"
        "#undef CV_MATH_FN\n#undef CV_MATH_PRECISE\n#undef CV_MATH_CONTRACT\n#undef CV_BITS\n#undef CV_FLOAT\n";
}
}
