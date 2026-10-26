#include "device_numeric.cu"
#include "device_half.cu"
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_umulhi32(unsigned a,unsigned b) {return cudavm::cvMulHi(a,b);}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_word_smulhi32(unsigned a,unsigned b) {return cudavm::cvSMulHi32(a,b);}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_umulhi64(unsigned long long a,unsigned long long b) {
    auto result=cudavm::cvMulHi64(cudavm::cvMake(unsigned(a),unsigned(a>>32)),cudavm::cvMake(unsigned(b),unsigned(b>>32)));
    return (static_cast<unsigned long long>(result.hi)<<32)|result.lo;
}
extern "C" __device__ __attribute__((used)) inline unsigned long long __cudavm_native_word_smulhi64(unsigned long long a,unsigned long long b) {
    auto result=cudavm::cvSMulHi64(cudavm::cvMake(unsigned(a),unsigned(a>>32)),cudavm::cvMake(unsigned(b),unsigned(b>>32)));
    return (static_cast<unsigned long long>(result.hi)<<32)|result.lo;
}
#pragma clang system_header
#define CV_MATH_FN __device__ inline
#define CV_MATH_PRECISE
#define CV_MATH_CLZ(x) __builtin_clz(x)
#define CV_MATH_FMA(x,y,z) __builtin_elementwise_fma(x,y,z)
#define CV_MATH_CONTRACT _Pragma("clang fp contract(off)")
#define CV_BITS(x) __builtin_bit_cast(unsigned, (x))
#define CV_FLOAT(x) __builtin_bit_cast(float, unsigned(x))
#include "device_math_scalar.cu"
extern "C" __device__ inline float __nv_sinf(float x) {return cudavm::cvMathSin(x);}
extern "C" __device__ inline float __nv_cosf(float x) {return cudavm::cvMathCos(x);}
extern "C" __device__ inline float __nv_fabsf(float x) {return cudavm::cvMathAbs(x);}
extern "C" __device__ inline float __nv_fminf(float x,float y) {return cudavm::cvMathMin(x,y);}
extern "C" __device__ inline float __nv_copysignf(float x,float y) {return cudavm::cvMathCopySign(x,y);}
extern "C" __device__ inline float __nv_floorf(float x) {return cudavm::cvMathFloor(x);}
extern "C" __device__ inline float __nv_ceilf(float x) {return cudavm::cvMathCeil(x);}
extern "C" __device__ inline float __nv_truncf(float x) {return cudavm::cvMathTrunc(x);}
extern "C" __device__ inline float __nv_roundf(float x) {return cudavm::cvMathRound(x);}
extern "C" __device__ inline float __nv_rintf(float x) {return cudavm::cvMathRint(x);}
extern "C" __device__ inline float __nv_nearbyintf(float x) {return cudavm::cvMathRint(x);}
extern "C" __device__ inline float __nv_exp2f(float x) {return cudavm::math_exp2(x);}
extern "C" __device__ inline float __nv_expf(float x) {return cudavm::math_exp(x);}
extern "C" __device__ inline float __nv_exp10f(float x) {return cudavm::math_exp10(x);}
extern "C" __device__ inline float __nv_log2f(float x) {return cudavm::math_log2(x);}
extern "C" __device__ inline float __nv_logf(float x) {return cudavm::math_log(x);}
extern "C" __device__ inline float __nv_log10f(float x) {return cudavm::math_log10(x);}
extern "C" __device__ inline float __nv_tanhf(float x) {return cudavm::math_tanh(x);}
extern "C" __device__ inline float __nv_fmaf_rn(float x,float y,float z) {return __builtin_elementwise_fma(x,y,z);}
extern "C" __device__ inline float __nv_fsub_rn(float x,float y) {return x-y;}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_sqrt(float x) {return cudavm::cvMathSqrt(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_max(float x,float y) {return cudavm::cvMathMax(x,y);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_min(float x,float y) {return cudavm::cvMathMin(x,y);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_divide(float x,float y) {return cudavm::cvFloatDivide(x,y);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_integral(float x,unsigned mode) {return cudavm::cvMathIntegral(x,mode);}
extern "C" __device__ __attribute__((used)) inline unsigned __cudavm_native_compare(unsigned x,unsigned y,unsigned predicate) {return cudavm::cvFloatCompare(x,y,predicate);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_sin(float x) {return cudavm::cvMathSin(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_cos(float x) {return cudavm::cvMathCos(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_exp2(float x) {return cudavm::math_exp2(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_exp(float x) {return cudavm::math_exp(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_exp10(float x) {return cudavm::math_exp10(x); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_log2(float x) {return cudavm::math_log2(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_log(float x) {return cudavm::math_log(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_log10(float x) {return cudavm::math_log10(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_tanh(float x) {return cudavm::math_tanh(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_rcp(float x) {return cudavm::math_reciprocal(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_rsqrt(float x) {return cudavm::math_rsqrt(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_flush(float x) {return cudavm::math_flush(x);}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_sin(float x) {return cudavm::math_half_trig(x,0); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_cos(float x) {return cudavm::math_half_trig(x,1); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_log2(float x) {return cudavm::math_log2_mode(x,1); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_log(float x) {return cudavm::math_log2_mode(x,1)*0.6931471805599453f; }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_log10(float x) {return cudavm::math_log2_mode(x,1)*0.3010299956639812f; }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_tanh(float x) {return cudavm::math_tanh_mode(x,1); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_rcp(float x) {return cudavm::math_reciprocal_mode(x,1); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_sqrt(float x) {unsigned b=__builtin_bit_cast(unsigned,x),a=b&0x7fffffffu; if(!a || b==0x7f800000u)return x; if(a>0x7f800000u)return __builtin_bit_cast(float,b|0x00400000u); if(b&0x80000000u)return __builtin_bit_cast(float,0x7fc00000u); return __builtin_elementwise_sqrt(x); }
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_half_rsqrt(float x) {return cudavm::math_reciprocal_mode(__cudavm_native_half_sqrt(x),1); }
namespace cudavm {
// Within this range, native add/multiply cannot produce a subnormal nonzero
// result or overflow. Cancellation of two binary32 operands is at least 2^-60.
__device__ static inline bool bounded_float_arithmetic_operand(unsigned bits) {
 unsigned magnitude=bits&0x7fffffffu;
 return !magnitude || (magnitude>=(90u<<23) && magnitude<(161u<<23));
}
}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_add_f32(float a,float b) {
 unsigned x=__builtin_bit_cast(unsigned,a),y=__builtin_bit_cast(unsigned,b);
 if(cudavm::bounded_float_arithmetic_operand(x) && cudavm::bounded_float_arithmetic_operand(y))return a+b;
 return __builtin_bit_cast(float,__cudavm_native_word_f32_add(x,y,0));
}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_mul_f32(float a,float b) {
 unsigned x=__builtin_bit_cast(unsigned,a),y=__builtin_bit_cast(unsigned,b);
 if(cudavm::bounded_float_arithmetic_operand(x) && cudavm::bounded_float_arithmetic_operand(y))return a*b;
 // An addend with the product's zero sign makes exact FMA round precisely
 // once as multiplication does, including both representations of zero.
 return __builtin_bit_cast(float,__cudavm_native_word_f32_fma(x,y,(x^y)&0x80000000u));
}
extern "C" __device__ __attribute__((used)) inline float __cudavm_native_fma(float a,float b,float c) {
 unsigned x=__builtin_bit_cast(unsigned,a),y=__builtin_bit_cast(unsigned,b),z=__builtin_bit_cast(unsigned,c);
 unsigned ax=x&0x7fffffffu,ay=y&0x7fffffffu,az=z&0x7fffffffu;
 bool sx=!ax || (ax>=(90u<<23) && ax<(161u<<23));
 bool sy=!ay || (ay>=(90u<<23) && ay<(161u<<23));
 bool sz=!az || (az>=(90u<<23) && az<(161u<<23));
 if(sx && sy && sz)return __builtin_elementwise_fma(a,b,c);
 return __builtin_bit_cast(float,__cudavm_native_word_f32_fma(x,y,z));}
#undef CV_MATH_CLZ
#undef CV_MATH_FMA
#undef CV_MATH_FN
#undef CV_MATH_PRECISE
#undef CV_MATH_CONTRACT
#undef CV_BITS
#undef CV_FLOAT
