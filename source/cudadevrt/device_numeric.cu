#define CV_FN __device__ static inline
namespace cudavm {
typedef unsigned uint;
struct cv64 {uint lo,hi;};
CV_FN cv64 cvMake(uint lo, uint hi) { cv64 r; r.lo=lo; r.hi=hi; return r; }
// Packed byte/halfword arithmetic uses bounded signed intermediates. No carry
// crosses a lane, and even the largest difference or sum fits a signed word.
CV_FN uint cvSimd(uint a,uint b,uint bits,uint operation) {
    uint mask=(1u<<bits)-1u,sign=1u<<(bits-1u),result=0u;
    int limit=int(sign)-1;
    for(uint shift=0u;shift<32u;shift+=bits) {
        uint x=(a>>shift)&mask,y=(b>>shift)&mask,z=0u;
        int sx=int(x^sign)-int(sign),sy=int(y^sign)-int(sign),value=0;
        if(operation==0u || operation==1u)value=sx<0?-sx:sx;
        else if(operation==2u)z=x+y;
        else if(operation==3u)value=sx+sy;
        else if(operation==4u)z=x+y>mask?mask:x+y;
        else if(operation==5u){value=sx+sy;value=value<0?-((-value+1)/2):(value+1)/2;}
        else if(operation==6u)z=(x+y+1u)/2u;
        else if(operation==7u)z=(x+y)/2u;
        else if(operation==8u)z=x-y;
        else if(operation==9u)value=sx-sy;
        else if(operation==10u)z=x<y?0u:x-y;
        else if(operation==11u)z=0u-x;
        else if(operation==12u)value=-sx;
        else if(operation==13u || operation==15u){value=sx-sy;value=value<0?-value:value;}
        else if(operation==14u || operation==16u)z=x<y?y-x:x-y;
        else if(operation==17u)z=uint(sx>sy?sx:sy);
        else if(operation==18u)z=x>y?x:y;
        else if(operation==19u)z=uint(sx<sy?sx:sy);
        else if(operation==20u)z=x<y?x:y;
        else if(operation>=21u) {
            uint predicate=operation>=31u?operation-31u:operation-21u;
            if(predicate==0u)z=x==y?1u:0u;
            else if(predicate==1u)z=sx>=sy?1u:0u;
            else if(predicate==2u)z=x>=y?1u:0u;
            else if(predicate==3u)z=sx>sy?1u:0u;
            else if(predicate==4u)z=x>y?1u:0u;
            else if(predicate==5u)z=sx<=sy?1u:0u;
            else if(predicate==6u)z=x<=y?1u:0u;
            else if(predicate==7u)z=sx<sy?1u:0u;
            else if(predicate==8u)z=x<y?1u:0u;
            else if(predicate==9u)z=x!=y?1u:0u;
            if(operation>=31u)z=z!=0u?mask:0u;
        }
        if(operation==1u || operation==3u || operation==9u || operation==12u)
            value=value>limit?limit:value< -int(sign)?-int(sign):value;
        if(operation==0u || operation==1u || operation==3u || operation==5u || operation==9u || operation==12u || operation==13u || operation==15u)z=uint(value);
        if(operation==15u || operation==16u)result+=z;
        else result=result|((z&mask)<<shift);
    }
    return result;
}
// Sign extension and all accumulation stay unsigned so overflow wraps in every
// shader dialect and the host helper, including negative packed products.
CV_FN uint cvPackedLane(uint x,uint shift,uint bits,uint signed_value) {
    uint mask=(1u<<bits)-1u,sign=signed_value!=0u?1u<<(bits-1u):0u;
    return (((x>>shift)&mask)^sign)-sign;
}
CV_FN uint cvUMul24(uint a,uint b) {return (a&0xffffffu)*(b&0xffffffu);}
CV_FN uint cvSMul24(uint a,uint b) {return cvPackedLane(a,0u,24u,1u)*cvPackedLane(b,0u,24u,1u);}
CV_FN uint cvDot4(uint a,uint b,uint c,uint signed_a,uint signed_b) {
    for(uint lane=0u;lane<4u;++lane)c+=cvPackedLane(a,lane*8u,8u,signed_a)*cvPackedLane(b,lane*8u,8u,signed_b);
    return c;
}
CV_FN uint cvDot2(uint a,uint b,uint c,uint high,uint signed_a,uint signed_b) {
    for(uint lane=0u;lane<2u;++lane)c+=cvPackedLane(a,lane*16u,16u,signed_a)*cvPackedLane(b,(lane+high*2u)*8u,8u,signed_b);
    return c;
}
CV_FN uint cvPrmt32(uint low, uint high, uint selector) {
    uint result=0u;
    for(uint i=0u;i<4u;++i) {
        uint nibble=(selector>>(i*4u))&15u;
        uint word=(nibble&4u)!=0u?high:low;
        uint byte=(word>>((nibble&3u)*8u))&255u;
        if((nibble&8u)!=0u)byte=(byte&128u)!=0u?255u:0u;
        result=result|(byte<<(i*8u));
    }
    return result;
}
// LLVM funnel operands describe high:low. Handle zero before the complementary
// shift so no C++, OpenCL, GLSL or HLSL expression shifts by a full word width.
CV_FN uint cvFshl32(uint high, uint low, uint shift) {
    shift=shift&31u;
    if(shift==0u)return high;
    return (high<<shift)|(low>>(32u-shift));
}
CV_FN uint cvFshr32(uint high, uint low, uint shift) {
    shift=shift&31u;
    if(shift==0u)return low;
    return (low>>shift)|(high<<(32u-shift));
}
CV_FN uint cvPopc32(uint x) {
    x=x-((x>>1u)&0x55555555u);
    x=(x&0x33333333u)+((x>>2u)&0x33333333u);
    x=(x+(x>>4u))&0x0f0f0f0fu;
    return (x*0x01010101u)>>24u;
}
CV_FN uint cvClz32(uint x) {
    if(x==0u)return 32u;
    uint n=0u;
    if((x&0xffff0000u)==0u){n+=16u;x=x<<16u;}
    if((x&0xff000000u)==0u){n+=8u;x=x<<8u;}
    if((x&0xf0000000u)==0u){n+=4u;x=x<<4u;}
    if((x&0xc0000000u)==0u){n+=2u;x=x<<2u;}
    return n+((x&0x80000000u)==0u?1u:0u);
}
CV_FN uint cvBrev32(uint x) {
    x=((x&0x55555555u)<<1u)|((x>>1u)&0x55555555u);
    x=((x&0x33333333u)<<2u)|((x>>2u)&0x33333333u);
    x=((x&0x0f0f0f0fu)<<4u)|((x>>4u)&0x0f0f0f0fu);
    x=((x&0x00ff00ffu)<<8u)|((x>>8u)&0x00ff00ffu);
    return (x<<16u)|(x>>16u);
}
CV_FN uint cvCtz32(uint x) {return cvClz32(cvBrev32(x));}
CV_FN uint cvPopc64(cv64 x) {return cvPopc32(x.lo)+cvPopc32(x.hi);}
CV_FN uint cvClz64(cv64 x) {return x.hi!=0u?cvClz32(x.hi):32u+cvClz32(x.lo);}
CV_FN uint cvCtz64(cv64 x) {return x.lo!=0u?cvCtz32(x.lo):32u+cvCtz32(x.hi);}
CV_FN cv64 cvBrev64(cv64 x) {return cvMake(cvBrev32(x.hi),cvBrev32(x.lo));}
CV_FN cv64 cvAdd(cv64 a, cv64 b) { uint lo=a.lo+b.lo; return cvMake(lo,a.hi+b.hi+(lo<a.lo?1u:0u)); }
CV_FN cv64 cvSub(cv64 a, cv64 b) { return cvMake(a.lo-b.lo,a.hi-b.hi-(a.lo<b.lo?1u:0u)); }
CV_FN cv64 cvNeg(cv64 a) { return cvSub(cvMake(0u,0u),a); }
CV_FN cv64 cvAnd(cv64 a, cv64 b) { return cvMake(a.lo&b.lo,a.hi&b.hi); }
CV_FN cv64 cvOr(cv64 a, cv64 b) { return cvMake(a.lo|b.lo,a.hi|b.hi); }
CV_FN cv64 cvXor(cv64 a, cv64 b) { return cvMake(a.lo^b.lo,a.hi^b.hi); }
CV_FN uint cvEq(cv64 a, cv64 b) { return a.lo==b.lo && a.hi==b.hi?1u:0u; }
CV_FN uint cvLt(cv64 a, cv64 b) { return a.hi<b.hi || (a.hi==b.hi && a.lo<b.lo)?1u:0u; }
CV_FN uint cvSLt(cv64 a, cv64 b) { uint ah=a.hi^0x80000000u,bh=b.hi^0x80000000u; return ah<bh || (ah==bh && a.lo<b.lo)?1u:0u; }
CV_FN uint cvDoubleIsNaN(cv64 a) {
    return (a.hi&0x7ff00000u)==0x7ff00000u && ((a.hi&0xfffffu)!=0u || a.lo!=0u)?1u:0u;
}
// Predicate masks select equal (1), greater (2), less (4), unordered (8).
// Compare the full IEEE encoding with integer operations, preserving subnormals
// and treating the two signed zero encodings as numerically equal.
CV_FN uint cvDoubleCompare(cv64 a, cv64 b, uint predicate) {
    uint relation=1u;
    if(cvDoubleIsNaN(a)!=0u || cvDoubleIsNaN(b)!=0u) relation=8u;
    else if(cvEq(a,b)==0u && (((a.hi|b.hi)&0x7fffffffu)!=0u || (a.lo|b.lo)!=0u)) {
        uint sa=a.hi>>31u,sb=b.hi>>31u;
        uint less=sa!=sb?sa:(sa!=0u?cvLt(b,a):cvLt(a,b));
        relation=less!=0u?4u:2u;
    }
    return (predicate&relation)!=0u?1u:0u;
}
CV_FN cv64 double_minmax(cv64 a,cv64 b,uint maximum) {
    if(cvDoubleIsNaN(a)!=0u) return cvDoubleIsNaN(b)!=0u?cvMake(a.lo,a.hi|0x00080000u):b;
    if(cvDoubleIsNaN(b)!=0u) return a;
    if((((a.hi|b.hi)&0x7fffffffu)|(a.lo|b.lo))==0u)
        return cvMake(0u,maximum!=0u?a.hi&b.hi:a.hi|b.hi);
    return cvDoubleCompare(a,b,maximum!=0u?2u:4u)!=0u?a:b;
}
CV_FN uint cvMulHi(uint a, uint b) {
    uint a0=a&65535u,a1=a>>16u,b0=b&65535u,b1=b>>16u;
    uint t=a0*b0;
    t=a1*b0+(t>>16u);
    uint w1=t&65535u,w2=t>>16u;
    w1=a0*b1+w1;
    return a1*b1+w2+(w1>>16u);
}
CV_FN cv64 cvMul(cv64 a, cv64 b) { return cvMake(a.lo*b.lo,cvMulHi(a.lo,b.lo)+a.hi*b.lo+a.lo*b.hi); }
CV_FN uint cvSMulHi32(uint a, uint b) {
    return cvMulHi(a,b)-((a>>31u)!=0u?b:0u)-((b>>31u)!=0u?a:0u);
}
CV_FN cv64 cvMulHi64(cv64 a, cv64 b) {
    // Accumulate the three contributions to product word 1 separately.
    // Its carry can be 2; a single carry bit would lose extreme products.
    uint middle=cvMulHi(a.lo,b.lo),next=middle+a.lo*b.hi;
    uint carry=next<middle?1u:0u;
    middle=next+a.hi*b.lo;carry+=middle<next?1u:0u;
    cv64 high=cvMake(a.hi*b.hi,cvMulHi(a.hi,b.hi));
    high=cvAdd(high,cvMake(cvMulHi(a.lo,b.hi),0u));
    high=cvAdd(high,cvMake(cvMulHi(a.hi,b.lo),0u));
    return cvAdd(high,cvMake(carry,0u));
}
CV_FN cv64 cvSMulHi64(cv64 a, cv64 b) {
    cv64 high=cvMulHi64(a,b);
    if((a.hi>>31u)!=0u)high=cvSub(high,b);
    if((b.hi>>31u)!=0u)high=cvSub(high,a);
    return high;
}
CV_FN uint cvUSad32(uint a, uint b, uint c) {return (a<b?b-a:a-b)+c;}
CV_FN uint cvSad32(uint a, uint b, uint c) {
    // Bias only the comparison. Subtraction and accumulation stay unsigned,
    // preserving |INT_MIN-INT_MAX| and modulo-2^32 accumulation.
    return ((a^0x80000000u)<(b^0x80000000u)?b-a:a-b)+c;
}
CV_FN cv64 cvShl(cv64 a, uint n) {
    n=n&63u;
    if(n==0u) return a;
    if(n>=32u) return cvMake(0u,a.lo<<(n-32u));
    return cvMake(a.lo<<n,(a.hi<<n)|(a.lo>>(32u-n)));
}
CV_FN cv64 cvShr(cv64 a, uint n) {
    n=n&63u;
    if(n==0u) return a;
    if(n>=32u) return cvMake(a.hi>>(n-32u),0u);
    return cvMake((a.lo>>n)|(a.hi<<(32u-n)),a.hi>>n);
}
CV_FN cv64 cvAShr(cv64 a, uint n) {
    n=n&63u;
    if(n==0u) return a;
    uint sign=(a.hi&0x80000000u)!=0u?0xffffffffu:0u;
    if(n>=32u) return cvMake(uint(int(a.hi)>>(n-32u)),sign);
    return cvMake((a.lo>>n)|(a.hi<<(32u-n)),uint(int(a.hi)>>n));
}
CV_FN cv64 cvUDivRem(cv64 a, cv64 b, uint remainder) {
    cv64 q=cvMake(0u,0u),r=cvMake(0u,0u);
    for(uint i=64u;i>0u;--i) {
        uint bit=i-1u,carry=r.hi>>31u;
        r=cvShl(r,1u); r.lo=r.lo|(cvShr(a,bit).lo&1u);
        if(carry!=0u || cvLt(r,b)==0u) {
            r=cvSub(r,b);
            if(bit>=32u) q.hi=q.hi|(1u<<(bit-32u)); else q.lo=q.lo|(1u<<bit);
        }
    }
    if(remainder!=0u) return r;
    return q;
}
CV_FN cv64 cvSDivRem(cv64 a, cv64 b, uint remainder) {
    uint sa=a.hi>>31u,sb=b.hi>>31u;
    if(sa!=0u) a=cvNeg(a);
    if(sb!=0u) b=cvNeg(b);
    cv64 r=cvUDivRem(a,b,remainder);
    if((remainder!=0u?sa:sa^sb)!=0u) r=cvNeg(r);
    return r;
}
// Construct binary32 bits directly so the two integer words cannot introduce
// double rounding. Modes are nearest-even, zero, down and up (0 through 3).
CV_FN uint cvMagnitudeToFloatBits(cv64 a, uint sign, uint mode) {
    if((a.lo|a.hi)==0u) return 0u;
    uint exponent=a.hi!=0u?32u:0u,word=a.hi!=0u?a.hi:a.lo;
    while(word>1u) {word=word>>1u;++exponent;}
    uint significand=0u;
    if(exponent<=23u) significand=a.lo<<(23u-exponent);
    else {
        uint shift=exponent-23u;
        significand=cvShr(a,shift).lo;
        cv64 discarded=cvSub(a,cvShl(cvMake(significand,0u),shift));
        cv64 halfway=cvShl(cvMake(1u,0u),shift-1u);
        uint increment=0u;
        if(mode==0u) increment=cvLt(halfway,discarded)!=0u || (cvEq(halfway,discarded)!=0u && (significand&1u)!=0u)?1u:0u;
        else if((discarded.lo|discarded.hi)!=0u) increment=(mode==2u && sign!=0u) || (mode==3u && sign==0u)?1u:0u;
        if(increment!=0u) {
            ++significand;
            if(significand==0x1000000u) {significand=significand>>1u;++exponent;}
        }
    }
    return sign|((exponent+127u)<<23u)|(significand&0x7fffffu);
}
CV_FN uint cvU64ToFloatBitsMode(cv64 a, uint mode) { return cvMagnitudeToFloatBits(a,0u,mode); }
CV_FN uint cvI64ToFloatBitsMode(cv64 a, uint mode) {
    uint sign=a.hi&0x80000000u;
    if(sign!=0u) a=cvNeg(a);
    return cvMagnitudeToFloatBits(a,sign,mode);
}
CV_FN uint cvU64ToFloatBits(cv64 a) { return cvU64ToFloatBitsMode(a,0u); }
CV_FN uint cvI64ToFloatBits(cv64 a) { return cvI64ToFloatBitsMode(a,0u); }
// Decode the exponent
// and significand without shader floating-point arithmetic, preserving tiny
// inputs and negative zero even on devices that flush float subnormals.
// NaN, infinity and rounded results outside the return type are undefined.
CV_FN cv64 cvFloatBitsTo64Mode(uint bits, uint mode) {
    uint exponent=(bits>>23u)&255u,sign=bits&0x80000000u;
    if(exponent>190u) return cvMake(0u,0u);
    cv64 magnitude=cvMake(0u,0u);
    uint fractional=0u,increment=0u;
    if(exponent<127u) {
        fractional=(bits&0x7fffffffu)!=0u?1u:0u;
        if(mode==0u && exponent==126u && (bits&0x7fffffu)!=0u) increment=1u;
    } else {
        uint significand=(bits&0x7fffffu)|0x800000u;
        if(exponent>=150u) magnitude=cvShl(cvMake(significand,0u),exponent-150u);
        else {
            uint shift=150u-exponent,remainder=significand&((1u<<shift)-1u),halfway=1u<<(shift-1u);
            magnitude=cvMake(significand>>shift,0u);
            fractional=remainder!=0u?1u:0u;
            if(mode==0u) increment=remainder>halfway || (remainder==halfway && (magnitude.lo&1u)!=0u)?1u:0u;
        }
    }
    if(mode!=0u && fractional!=0u) increment=(mode==2u && sign!=0u) || (mode==3u && sign==0u)?1u:0u;
    if(increment!=0u) magnitude=cvAdd(magnitude,cvMake(1u,0u));
    if(sign!=0u) magnitude=cvNeg(magnitude);
    return magnitude;
}
CV_FN cv64 cvFloatBitsTo64(uint bits) { return cvFloatBitsTo64Mode(bits,1u); }
// Binary64 travels as two integer words. Widening binary32 is exact, including
// its subnormals; numerical conversion quiets NaNs, while transport is bitwise.
CV_FN cv64 cvFloatBitsToDoubleBits(uint bits) {
    uint sign=bits&0x80000000u,exponent=(bits>>23u)&255u,fraction=bits&0x7fffffu;
    if(exponent==255u) {
        cv64 payload=cvShl(cvMake(fraction,0u),29u);
        if(fraction!=0u) payload.hi=payload.hi|0x80000u;
        return cvMake(payload.lo,sign|0x7ff00000u|payload.hi);
    }
    if(exponent==0u) {
        if(fraction==0u) return cvMake(0u,sign);
        exponent=897u;
        while((fraction&0x800000u)==0u) {fraction=fraction<<1u;--exponent;}
        fraction=fraction&0x7fffffu;
    } else exponent=exponent+896u;
    cv64 payload=cvShl(cvMake(fraction,0u),29u);
    return cvMake(payload.lo,sign|(exponent<<20u)|payload.hi);
}
CV_FN uint cvDoubleOverflowBits(uint sign,uint mode) {
    uint infinite=mode==0u || (mode==2u && sign!=0u) || (mode==3u && sign==0u)?1u:0u;
    return sign|(infinite!=0u?0x7f800000u:0x7f7fffffu);
}
// Retain 24 bits for normals, or fewer for subnormals, and round once. Large
// shifts and overflow are handled separately so no shift wraps modulo 64.
CV_FN uint cvDoubleBitsToFloatBitsMode(cv64 bits,uint mode) {
    uint sign=bits.hi&0x80000000u,exponent=(bits.hi>>20u)&2047u;
    cv64 fraction=cvMake(bits.lo,bits.hi&0xfffffu);
    if(exponent==2047u) {
        if((fraction.lo|fraction.hi)==0u) return sign|0x7f800000u;
        return sign|0x7fc00000u|cvShr(fraction,29u).lo;
    }
    if(exponent>1150u) return cvDoubleOverflowBits(sign,mode);
    if(exponent<873u) {
        uint nonzero=(exponent|fraction.lo|fraction.hi)!=0u?1u:0u;
        uint away=(mode==2u && sign!=0u) || (mode==3u && sign==0u)?1u:0u;
        return sign|(nonzero&away);
    }
    cv64 significand=cvMake(fraction.lo,fraction.hi|0x100000u);
    uint shift=exponent<897u?926u-exponent:29u;
    uint retained=cvShr(significand,shift).lo;
    cv64 discarded=cvSub(significand,cvShl(cvMake(retained,0u),shift));
    cv64 halfway=cvShl(cvMake(1u,0u),shift-1u);
    uint increment=0u;
    if(mode==0u) increment=cvLt(halfway,discarded)!=0u || (cvEq(halfway,discarded)!=0u && (retained&1u)!=0u)?1u:0u;
    else if((discarded.lo|discarded.hi)!=0u) increment=(mode==2u && sign!=0u) || (mode==3u && sign==0u)?1u:0u;
    retained=retained+increment;
    if(exponent<897u) return sign|retained;
    uint result_exponent=exponent-896u;
    if(retained==0x1000000u) {retained=retained>>1u;++result_exponent;}
    if(result_exponent==255u) return cvDoubleOverflowBits(sign,mode);
    return sign|(result_exponent<<23u)|(retained&0x7fffffu);
}
// Retain all 53 binary64 significand bits; at most eleven integer bits need
// rounding. Rounding the two words independently would lose ties and carries.
CV_FN cv64 cvMagnitudeToDoubleBits(cv64 a,uint sign,uint mode) {
    if((a.lo|a.hi)==0u) return cvMake(0u,0u);
    uint exponent=a.hi!=0u?32u:0u,word=a.hi!=0u?a.hi:a.lo;
    while(word>1u) {word=word>>1u;++exponent;}
    cv64 significand=cvMake(0u,0u);
    if(exponent<=52u) significand=cvShl(a,52u-exponent);
    else {
        uint shift=exponent-52u,discarded=a.lo&((1u<<shift)-1u),halfway=1u<<(shift-1u);
        significand=cvShr(a,shift);
        uint increment=0u;
        if(mode==0u) increment=discarded>halfway || (discarded==halfway && (significand.lo&1u)!=0u)?1u:0u;
        else if(discarded!=0u) increment=(mode==2u && sign!=0u) || (mode==3u && sign==0u)?1u:0u;
        if(increment!=0u) {
            significand=cvAdd(significand,cvMake(1u,0u));
            if(significand.hi==0x200000u) {significand=cvShr(significand,1u);++exponent;}
        }
    }
    return cvMake(significand.lo,sign|((exponent+1023u)<<20u)|(significand.hi&0xfffffu));
}
CV_FN cv64 cvU64ToDoubleBitsMode(cv64 a,uint mode) { return cvMagnitudeToDoubleBits(a,0u,mode); }
CV_FN cv64 cvI64ToDoubleBitsMode(cv64 a,uint mode) {
    uint sign=a.hi&0x80000000u;
    if(sign!=0u) a=cvNeg(a);
    return cvMagnitudeToDoubleBits(a,sign,mode);
}
// The input must be finite and its rounded result representable by the return
// type. Decode the full binary64 significand before applying sign and rounding.
CV_FN cv64 cvDoubleBitsTo64Mode(cv64 bits,uint mode) {
    uint exponent=(bits.hi>>20u)&2047u,sign=bits.hi&0x80000000u;
    if(exponent>1086u) return cvMake(0u,0u);
    cv64 magnitude=cvMake(0u,0u);
    uint fractional=0u,increment=0u;
    if(exponent<1023u) {
        fractional=(bits.lo|(bits.hi&0x7fffffffu))!=0u?1u:0u;
        if(mode==0u && exponent==1022u && (bits.lo|(bits.hi&0xfffffu))!=0u) increment=1u;
    } else {
        cv64 significand=cvMake(bits.lo,(bits.hi&0xfffffu)|0x100000u);
        if(exponent>=1075u) magnitude=cvShl(significand,exponent-1075u);
        else {
            uint shift=1075u-exponent;
            magnitude=cvShr(significand,shift);
            cv64 discarded=cvSub(significand,cvShl(magnitude,shift));
            cv64 halfway=cvShl(cvMake(1u,0u),shift-1u);
            fractional=(discarded.lo|discarded.hi)!=0u?1u:0u;
            if(mode==0u) increment=cvLt(halfway,discarded)!=0u || (cvEq(halfway,discarded)!=0u && (magnitude.lo&1u)!=0u)?1u:0u;
        }
    }
    if(mode!=0u && fractional!=0u) increment=(mode==2u && sign!=0u) || (mode==3u && sign==0u)?1u:0u;
    if(increment!=0u) magnitude=cvAdd(magnitude,cvMake(1u,0u));
    if(sign!=0u) magnitude=cvNeg(magnitude);
    return magnitude;
}
CV_FN cv64 cvDoubleShiftJam(cv64 a,uint shift) {
    if(shift==0u) return a;
    if(shift>=64u) return cvMake((a.lo|a.hi)!=0u?1u:0u,0u);
    cv64 result=cvShr(a,shift),discarded=cvSub(a,cvShl(result,shift));
    result.lo=result.lo|((discarded.lo|discarded.hi)!=0u?1u:0u);
    return result;
}
// These two helpers accept a nonzero finite magnitude. Subnormal operands are
// normalized exactly; their biased exponent can be negative.
CV_FN cv64 cvDoubleSignificand(cv64 a) {
    cv64 significand=cvMake(a.lo,a.hi&0xfffffu);
    if((a.hi&0x7ff00000u)!=0u) significand.hi=significand.hi|0x100000u;
    else while((significand.hi&0x100000u)==0u) significand=cvShl(significand,1u);
    return significand;
}
CV_FN int cvDoubleExponent(cv64 a) {
    int exponent=int((a.hi>>20u)&2047u);
    if(exponent!=0) return exponent;
    cv64 significand=cvMake(a.lo,a.hi&0xfffffu);exponent=1;
    while((significand.hi&0x100000u)==0u) {significand=cvShl(significand,1u);--exponent;}
    return exponent;
}
// The significand carries three guard/round/sticky bits. Underflow shifts at
// the destination precision before the single nearest-even rounding decision.
CV_FN cv64 cvDoubleRound(cv64 significand,int exponent,uint sign,uint mode=0u) {
    if((significand.lo|significand.hi)==0u) return cvMake(0u,sign);
    if(significand.hi>=0x1000000u) {significand=cvDoubleShiftJam(significand,1u);++exponent;}
    while(significand.hi<0x800000u && exponent>1) {significand=cvShl(significand,1u);--exponent;}
    if(exponent<1) {significand=cvDoubleShiftJam(significand,uint(1-exponent));exponent=1;}
    uint discarded=significand.lo&7u;
    significand=cvShr(significand,3u);
    uint increment=mode==0u?(discarded>4u || (discarded==4u && (significand.lo&1u)!=0u)):
        discarded!=0u && ((mode==2u && sign!=0u) || (mode==3u && sign==0u));
    if(increment!=0u) significand=cvAdd(significand,cvMake(1u,0u));
    if(significand.hi>=0x200000u) {significand=cvShr(significand,1u);++exponent;}
    if(exponent>=2047) return mode==0u || (mode==2u && sign!=0u) || (mode==3u && sign==0u)?
        cvMake(0u,sign|0x7ff00000u):cvMake(0xffffffffu,sign|0x7fefffffu);
    if(significand.hi<0x100000u) exponent=0;
    return cvMake(significand.lo,sign|(uint(exponent)<<20u)|(significand.hi&0xfffffu));
}
CV_FN cv64 cvDoubleAdd(cv64 a,cv64 b,uint mode=0u) {
    // Imported binary64 instructions retain an incoming NaN's payload while
    // quieting signaling values, including DADD used for abs.f64 by ptxas.
    if(cvDoubleIsNaN(a)!=0u) return cvMake(a.lo,a.hi|0x00080000u);
    if(cvDoubleIsNaN(b)!=0u) return cvMake(b.lo,b.hi|0x00080000u);
    cv64 ax=cvMake(a.lo,a.hi&0x7fffffffu),bx=cvMake(b.lo,b.hi&0x7fffffffu);
    uint sa=a.hi&0x80000000u,sb=b.hi&0x80000000u;
    if(ax.hi==0x7ff00000u || bx.hi==0x7ff00000u) {
        if(ax.hi==0x7ff00000u && bx.hi==0x7ff00000u && sa!=sb) return cvMake(0u,0x7ff80000u);
        if(ax.hi==0x7ff00000u) return a;
        return b;
    }
    if((ax.lo|ax.hi)==0u && (bx.lo|bx.hi)==0u) return cvMake(0u,sa==sb?sa:mode==2u?0x80000000u:0u);
    if((ax.lo|ax.hi)==0u) return b;
    if((bx.lo|bx.hi)==0u) return a;
    if(cvLt(ax,bx)!=0u) {cv64 temporary=a;a=b;b=temporary;uint sign=sa;sa=sb;sb=sign;}
    int exponent=cvDoubleExponent(a),other_exponent=cvDoubleExponent(b);
    cv64 left=cvShl(cvDoubleSignificand(a),3u);
    cv64 right=cvDoubleShiftJam(cvShl(cvDoubleSignificand(b),3u),uint(exponent-other_exponent));
    cv64 result;
    if(sa==sb) result=cvAdd(left,right);else result=cvSub(left,right);
    if((result.lo|result.hi)==0u) return cvMake(0u,mode==2u?0x80000000u:0u);
    return cvDoubleRound(result,exponent,sa,mode);
}
CV_FN cv64 cvDoubleSubtract(cv64 a,cv64 b) {b.hi=b.hi^0x80000000u;return cvDoubleAdd(a,b);}
CV_FN cv64 cvDoubleMultiply(cv64 a,cv64 b,uint mode=0u) {
    if(cvDoubleIsNaN(a)!=0u || cvDoubleIsNaN(b)!=0u) return cvMake(0u,0x7ff80000u);
    cv64 ax=cvMake(a.lo,a.hi&0x7fffffffu),bx=cvMake(b.lo,b.hi&0x7fffffffu);
    uint sign=(a.hi^b.hi)&0x80000000u;
    if(ax.hi==0x7ff00000u || bx.hi==0x7ff00000u) {
        if((ax.lo|ax.hi)==0u || (bx.lo|bx.hi)==0u) return cvMake(0u,0x7ff80000u);
        return cvMake(0u,sign|0x7ff00000u);
    }
    if((ax.lo|ax.hi)==0u || (bx.lo|bx.hi)==0u) return cvMake(0u,sign);
    int exponent=cvDoubleExponent(a)+cvDoubleExponent(b)-1023;
    cv64 x=cvDoubleSignificand(a),y=cvDoubleSignificand(b);
    // Preserve the full 106-bit product as two cv64 words. Carries between
    // partial products must reach the upper word before any bits are discarded.
    cv64 low=cvMul(cvMake(x.lo,0u),cvMake(y.lo,0u));
    cv64 high=cvMul(cvMake(x.hi,0u),cvMake(y.hi,0u));
    cv64 cross=cvMul(cvMake(x.lo,0u),cvMake(y.hi,0u)),sum=cvAdd(low,cvMake(0u,cross.lo));
    high=cvAdd(high,cvMake(cross.hi,0u));high=cvAdd(high,cvMake(cvLt(sum,low),0u));low=sum;
    cross=cvMul(cvMake(x.hi,0u),cvMake(y.lo,0u));sum=cvAdd(low,cvMake(0u,cross.lo));
    high=cvAdd(high,cvMake(cross.hi,0u));high=cvAdd(high,cvMake(cvLt(sum,low),0u));low=sum;
    uint shift=49u;
    if((high.hi&0x200u)!=0u) {shift=50u;++exponent;}
    cv64 significand=cvOr(cvShr(low,shift),cvShl(high,64u-shift));
    cv64 discarded=cvSub(low,cvShl(cvShr(low,shift),shift));
    significand.lo=significand.lo|((discarded.lo|discarded.hi)!=0u?1u:0u);
    return cvDoubleRound(significand,exponent,sign,mode);
}
// Exact binary64 FMA keeps the full product and addend until one rounding.
// Four little-endian words leave room for carry and three final rounding bits.
struct double_wide_words { uint words[4]; };
CV_FN double_wide_words double_wide_from(cv64 value) { double_wide_words r={{value.lo,value.hi,0u,0u}};return r; }
CV_FN uint double_wide_bit(double_wide_words a,uint bit) {return bit<128u?(a.words[bit/32u]>>(bit%32u))&1u:0u;}
CV_FN double_wide_words double_wide_shift(double_wide_words a,int shift) {
    double_wide_words r={{0u,0u,0u,0u}};
    for(uint bit=0u;bit<128u;++bit) {int source=int(bit)-shift;if(source>=0 && source<128)r.words[bit/32u]|=double_wide_bit(a,uint(source))<<(bit%32u);}
    if(shift<0) {uint discarded=0u;for(uint bit=0u;bit<128u;++bit)if(bit<uint(-shift))discarded|=double_wide_bit(a,bit);r.words[0]|=discarded;}
    return r;
}
CV_FN int double_wide_top(double_wide_words a) {for(int bit=127;bit>=0;--bit)if(double_wide_bit(a,uint(bit))!=0u)return bit;return -1;}
CV_FN uint double_wide_less(double_wide_words a,double_wide_words b) {
    for(int i=3;i>=0;--i)if(a.words[i]!=b.words[i])return a.words[i]<b.words[i]?1u:0u;return 0u;
}
CV_FN double_wide_words double_wide_sum(double_wide_words a,double_wide_words b,uint subtract) {
    double_wide_words r;uint carry=subtract;
    for(uint i=0u;i<4u;++i) {uint y=subtract!=0u?~b.words[i]:b.words[i],sum=a.words[i]+y;
        uint next=sum+carry;carry=(sum<a.words[i] || next<sum)?1u:0u;r.words[i]=next;}
    return r;
}
CV_FN cv64 double_wide_round(double_wide_words value,int exponent,uint sign,uint mode) {
    int top=double_wide_top(value);if(top<0)return cvMake(0u,sign);
    value=double_wide_shift(value,55-top);
    return cvDoubleRound(cvMake(value.words[0],value.words[1]),exponent+top-120+1023,sign,mode);
}
CV_FN cv64 double_fma(cv64 a,cv64 b,cv64 c,uint mode) {
    if(cvDoubleIsNaN(a)!=0u || cvDoubleIsNaN(b)!=0u || cvDoubleIsNaN(c)!=0u)return cvMake(0u,0x7ff80000u);
    uint sign=(a.hi^b.hi)&0x80000000u,cs=c.hi&0x80000000u;
    cv64 ax=cvMake(a.lo,a.hi&0x7fffffffu),bx=cvMake(b.lo,b.hi&0x7fffffffu),cx=cvMake(c.lo,c.hi&0x7fffffffu);
    uint az=(ax.lo|ax.hi)==0u,bz=(bx.lo|bx.hi)==0u,cz=(cx.lo|cx.hi)==0u;
    if(ax.hi==0x7ff00000u || bx.hi==0x7ff00000u) {
        if(az!=0u || bz!=0u || (cx.hi==0x7ff00000u && cs!=sign))return cvMake(0u,0x7ff80000u);
        return cvMake(0u,sign|0x7ff00000u);
    }
    if(cx.hi==0x7ff00000u)return c;
    if(az!=0u || bz!=0u)return cvDoubleAdd(cvMake(0u,sign),c,mode);
    cv64 x=cvDoubleSignificand(a),y=cvDoubleSignificand(b);
    cv64 low=cvMul(cvMake(x.lo,0u),cvMake(y.lo,0u)),high=cvMul(cvMake(x.hi,0u),cvMake(y.hi,0u));
    cv64 cross=cvMul(cvMake(x.lo,0u),cvMake(y.hi,0u)),sum=cvAdd(low,cvMake(0u,cross.lo));
    high=cvAdd(high,cvMake(cross.hi,0u));high=cvAdd(high,cvMake(cvLt(sum,low),0u));low=sum;
    cross=cvMul(cvMake(x.hi,0u),cvMake(y.lo,0u));sum=cvAdd(low,cvMake(0u,cross.lo));
    high=cvAdd(high,cvMake(cross.hi,0u));high=cvAdd(high,cvMake(cvLt(sum,low),0u));low=sum;
    double_wide_words product={{low.lo,low.hi,high.lo,high.hi}};
    int top=double_wide_top(product),exponent=cvDoubleExponent(a)+cvDoubleExponent(b)-2046+top-104;
    product=double_wide_shift(product,120-top);
    if(cz!=0u)return double_wide_round(product,exponent,sign,mode);
    int ce=cvDoubleExponent(c)-1023,common=exponent>ce?exponent:ce;
    double_wide_words addend=double_wide_shift(double_wide_from(cvDoubleSignificand(c)),68);
    product=double_wide_shift(product,exponent-common);addend=double_wide_shift(addend,ce-common);
    if(sign==cs)return double_wide_round(double_wide_sum(product,addend,0u),common,sign,mode);
    if(double_wide_less(product,addend)!=0u)return double_wide_round(double_wide_sum(addend,product,1u),common,cs,mode);
    auto difference=double_wide_sum(product,addend,1u);
    if(double_wide_top(difference)<0)return cvMake(0u,mode==2u?0x80000000u:0u);
    return double_wide_round(difference,common,sign,mode);
}
CV_FN cv64 double_sqrt(cv64 a,uint mode) {
    if(cvDoubleIsNaN(a)!=0u)return cvMake(0u,0x7ff80000u);
    if((a.lo|(a.hi&0x7fffffffu))==0u)return a;
    if((a.hi&0x80000000u)!=0u)return cvMake(0u,0x7ff80000u);
    if(a.hi==0x7ff00000u)return a;
    int exponent=cvDoubleExponent(a)-1023;
    cv64 significand=cvDoubleSignificand(a);if(exponent%2!=0) {significand=cvShl(significand,1u);--exponent;}
    auto radicand=double_wide_shift(double_wide_from(significand),58);
    cv64 remainder=cvMake(0u,0u),root=cvMake(0u,0u);
    for(int bit=55;bit>=0;--bit) {
        remainder=cvShl(remainder,2u);remainder.lo|=double_wide_bit(radicand,uint(bit*2+1))*2u+double_wide_bit(radicand,uint(bit*2));
        cv64 trial=cvShl(root,2u);trial.lo|=1u;uint take=cvLt(remainder,trial)==0u?1u:0u;
        if(take!=0u)remainder=cvSub(remainder,trial);root=cvShl(root,1u);root.lo|=take;
    }
    root.lo|=(remainder.lo|remainder.hi)!=0u?1u:0u;
    return cvDoubleRound(root,exponent/2+1023,0u,mode);
}
CV_FN cv64 cvDoubleDivide(cv64 a,cv64 b) {
    if(cvDoubleIsNaN(a)!=0u || cvDoubleIsNaN(b)!=0u) return cvMake(0u,0x7ff80000u);
    cv64 ax=cvMake(a.lo,a.hi&0x7fffffffu),bx=cvMake(b.lo,b.hi&0x7fffffffu);
    uint sign=(a.hi^b.hi)&0x80000000u;
    uint zero_a=(ax.lo|ax.hi)==0u?1u:0u,zero_b=(bx.lo|bx.hi)==0u?1u:0u;
    if((zero_a!=0u && zero_b!=0u) || (ax.hi==0x7ff00000u && bx.hi==0x7ff00000u)) return cvMake(0u,0x7ff80000u);
    if(ax.hi==0x7ff00000u || zero_b!=0u) return cvMake(0u,sign|0x7ff00000u);
    if(bx.hi==0x7ff00000u || zero_a!=0u) return cvMake(0u,sign);
    int exponent=cvDoubleExponent(a)-cvDoubleExponent(b)+1023;
    cv64 remainder=cvDoubleSignificand(a),divisor=cvDoubleSignificand(b),quotient=cvMake(0u,0u);
    if(cvLt(remainder,divisor)!=0u) {remainder=cvShl(remainder,1u);--exponent;}
    // Produce 53 significand bits plus guard/round/sticky information. Each
    // remainder is less than twice the divisor, so two words suffice exactly.
    for(uint i=0u;i<56u;++i) {
        uint bit=cvLt(remainder,divisor)==0u?1u:0u;
        if(bit!=0u) remainder=cvSub(remainder,divisor);
        quotient=cvShl(quotient,1u);quotient.lo=quotient.lo|bit;
        remainder=cvShl(remainder,1u);
    }
    quotient.lo=quotient.lo|((remainder.lo|remainder.hi)!=0u?1u:0u);
    return cvDoubleRound(quotient,exponent,sign);
}
CV_FN uint cvAtomicFloatAdd(uint a,uint b,uint flush,uint mode=0u) {
    if(flush!=0u) {
        if((a&0x7f800000u)==0u)a=a&0x80000000u;
        if((b&0x7f800000u)==0u)b=b&0x80000000u;
    }
    // Three low bits carry guard, round and sticky information. Keep binary32
    // addition in one word instead of expanding through binary64 arithmetic.
    uint magnitude_a=a&0x7fffffffu,magnitude_b=b&0x7fffffffu;
    if(magnitude_a>0x7f800000u || magnitude_b>0x7f800000u)return 0x7fffffffu;
    if(magnitude_a==0x7f800000u || magnitude_b==0x7f800000u) {
        if(magnitude_a==magnitude_b && ((a^b)&0x80000000u)!=0u)return 0x7fffffffu;
        return magnitude_a==0x7f800000u?a:b;
    }
    if(magnitude_a<magnitude_b) {uint temporary=a;a=b;b=temporary;temporary=magnitude_a;magnitude_a=magnitude_b;magnitude_b=temporary;}
    uint sign=a&0x80000000u;
    if(magnitude_b==0u)return magnitude_a==0u?(mode==2u?(a|b):(a&b)):a;
    uint exponent=magnitude_a>>23u,other_exponent=magnitude_b>>23u;
    uint left=((a&0x7fffffu)|(exponent!=0u?0x800000u:0u))<<3u;
    uint right=((b&0x7fffffu)|(other_exponent!=0u?0x800000u:0u))<<3u;
    exponent=exponent!=0u?exponent:1u;other_exponent=other_exponent!=0u?other_exponent:1u;
    uint distance=exponent-other_exponent;
    if(distance>=32u)right=1u;
    else if(distance!=0u)right=(right>>distance)|((right<<(32u-distance))!=0u?1u:0u);
    uint significand;
    if(((a^b)&0x80000000u)==0u) {
        significand=left+right;
        if(significand>=0x8000000u) {significand=(significand>>1u)|(significand&1u);++exponent;}
    } else {
        significand=left-right;
        if(significand==0u)return mode==2u?0x80000000u:0u;
        while(significand<0x4000000u && exponent>1u) {significand<<=1u;--exponent;}
    }
    uint rounded=significand>>3u,tail=significand&7u;
    bool increment=mode==0u?(tail>4u || (tail==4u && (rounded&1u)!=0u)):mode==2u?(sign!=0u && tail!=0u):mode==3u?(sign==0u && tail!=0u):false;
    if(increment)++rounded;
    if(rounded>=0x1000000u) {rounded>>=1u;++exponent;}
    if(exponent>=255u)return sign|((mode==1u || (mode==2u && !sign) || (mode==3u && sign))?0x7f7fffffu:0x7f800000u);
    if(exponent==1u && rounded<0x800000u)exponent=0u;
    uint result=sign|(exponent<<23u)|(rounded&0x7fffffu);
    if(flush!=0u && (result&0x7f800000u)==0u)result=result&0x80000000u;
    return result;
}
}
#undef CV_FN
namespace cudavm {
__device__ static inline cv64 split(unsigned long long x){return cvMake(unsigned(x),unsigned(x>>32));}
__device__ static inline unsigned long long join(cv64 x){return (unsigned long long)(x.lo)|((unsigned long long)(x.hi)<<32);}
}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_integral(unsigned long long bits,unsigned mode) {
    auto sign=bits&0x8000000000000000ull, magnitude=bits&0x7fffffffffffffffull;
    unsigned exponent=unsigned(magnitude>>52);
    if(exponent>=1075u || magnitude==0ull)return bits;
    if(exponent<1023u) {
        bool increment=mode==0u?magnitude>0x3fe0000000000000ull:mode==2u?sign!=0ull:mode==3u?sign==0ull:false;
        return sign|(increment?0x3ff0000000000000ull:0ull);
    }
    unsigned shift=1075u-exponent;
    auto fraction=magnitude&((1ull<<shift)-1ull), retained=magnitude>>shift;
    if(fraction==0ull)return bits;
    auto halfway=1ull<<(shift-1u);
    bool increment=mode==0u?(fraction>halfway || (fraction==halfway && (retained&1ull))):mode==2u?sign!=0ull:mode==3u?sign==0ull:false;
    return sign|((retained+unsigned(increment))<<shift);
}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_add(unsigned long long a,unsigned long long b){using namespace cudavm;return join(cvDoubleAdd(split(a),split(b)));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_add_mode(unsigned long long a,unsigned long long b,unsigned mode){using namespace cudavm;return join(cvDoubleAdd(split(a),split(b),mode));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_mul_mode(unsigned long long a,unsigned long long b,unsigned mode){using namespace cudavm;return join(cvDoubleMultiply(split(a),split(b),mode));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_fma_mode(unsigned long long a,unsigned long long b,unsigned long long c,unsigned mode){using namespace cudavm;return join(double_fma(split(a),split(b),split(c),mode));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_sqrt_mode(unsigned long long a,unsigned mode){using namespace cudavm;return join(double_sqrt(split(a),mode));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_sub(unsigned long long a,unsigned long long b){using namespace cudavm;return join(cvDoubleSubtract(split(a),split(b)));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_mul(unsigned long long a,unsigned long long b){using namespace cudavm;return join(cvDoubleMultiply(split(a),split(b)));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_div(unsigned long long a,unsigned long long b){using namespace cudavm;return join(cvDoubleDivide(split(a),split(b)));}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f64_compare(unsigned long long a,unsigned long long b,unsigned predicate){using namespace cudavm;return cvDoubleCompare(split(a),split(b),predicate);}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_minmax(unsigned long long a,unsigned long long b,unsigned maximum){using namespace cudavm;return join(double_minmax(split(a),split(b),maximum));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f32_to_f64(unsigned bits){using namespace cudavm;return join(cvFloatBitsToDoubleBits(bits));}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f64_to_f32(unsigned long long bits,unsigned mode){using namespace cudavm;return cvDoubleBitsToFloatBitsMode(split(bits),mode);}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f32_add(unsigned a,unsigned b,unsigned flush){using namespace cudavm;return cvAtomicFloatAdd(a,b,flush);}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_f32_add_mode(unsigned a,unsigned b,unsigned mode){using namespace cudavm;return cvAtomicFloatAdd(a,b,0u,mode);}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_u64_to_f32(unsigned long long bits,unsigned mode){using namespace cudavm;return cvU64ToFloatBitsMode(split(bits),mode);}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_i64_to_f32(unsigned long long bits,unsigned mode){using namespace cudavm;return cvI64ToFloatBitsMode(split(bits),mode);}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f32_to_i64(unsigned bits,unsigned mode){using namespace cudavm;return join(cvFloatBitsTo64Mode(bits,mode));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_u64_to_f64(unsigned long long bits,unsigned mode){using namespace cudavm;return join(cvU64ToDoubleBitsMode(split(bits),mode));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_i64_to_f64(unsigned long long bits,unsigned mode){using namespace cudavm;return join(cvI64ToDoubleBitsMode(split(bits),mode));}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_f64_to_i64(unsigned long long bits,unsigned mode){using namespace cudavm;return join(cvDoubleBitsTo64Mode(split(bits),mode));}
