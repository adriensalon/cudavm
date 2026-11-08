#include <cuda_fp6.h>
namespace cudavm {
template<bool E2> struct fp6_slice_types;
template<> struct fp6_slice_types<true> {
    using scalar = __nv_fp6_e2m3; using pair = __nv_fp6x2_e2m3; using quad = __nv_fp6x4_e2m3;
    static constexpr auto format = __NV_E2M3;
};
template<> struct fp6_slice_types<false> {
    using scalar = __nv_fp6_e3m2; using pair = __nv_fp6x2_e3m2; using quad = __nv_fp6x4_e3m2;
    static constexpr auto format = __NV_E3M2;
};
__device__ inline auto fp6_slice_narrow(float a, cudaRoundMode r, __nv_fp6_interpretation_t f) { return __nv_cvt_float_to_fp6(a,f,r); }
__device__ inline auto fp6_slice_narrow(double a, cudaRoundMode r, __nv_fp6_interpretation_t f) { return __nv_cvt_double_to_fp6(a,f,r); }
__device__ inline auto fp6_slice_narrow(__half_raw a, cudaRoundMode r, __nv_fp6_interpretation_t f) { return __nv_cvt_halfraw_to_fp6(a,f,r); }
__device__ inline auto fp6_slice_narrow(__nv_bfloat16_raw a, cudaRoundMode r, __nv_fp6_interpretation_t f) { return __nv_cvt_bfloat16raw_to_fp6(a,f,r); }
__device__ inline auto fp6_slice_narrow(float2 a, cudaRoundMode r, __nv_fp6_interpretation_t f) { return __nv_cvt_float2_to_fp6x2(a,f,r); }
__device__ inline auto fp6_slice_narrow(double2 a, cudaRoundMode r, __nv_fp6_interpretation_t f) { return __nv_cvt_double2_to_fp6x2(a,f,r); }
__device__ inline auto fp6_slice_narrow(__half2_raw a, cudaRoundMode r, __nv_fp6_interpretation_t f) { return __nv_cvt_halfraw2_to_fp6x2(a,f,r); }
__device__ inline auto fp6_slice_narrow(__nv_bfloat162_raw a, cudaRoundMode r, __nv_fp6_interpretation_t f) { return __nv_cvt_bfloat16raw2_to_fp6x2(a,f,r); }
__device__ inline float2 fp6_slice_pair(float a,float b) { return float2{a,b}; }
__device__ inline double2 fp6_slice_pair(double a,double b) { return double2{a,b}; }
__device__ inline __half2_raw fp6_slice_pair(__half_raw a,__half_raw b) { return __half2_raw{a.x,b.x}; }
__device__ inline __nv_bfloat162_raw fp6_slice_pair(__nv_bfloat16_raw a,__nv_bfloat16_raw b) { return __nv_bfloat162_raw{a.x,b.x}; }
__device__ inline float fp6_slice_public(float a) { return a; }
__device__ inline double fp6_slice_public(double a) { return a; }
__device__ inline __half fp6_slice_public(__half_raw a) { return __half(a); }
__device__ inline __nv_bfloat16 fp6_slice_public(__nv_bfloat16_raw a) { return __nv_bfloat16(a); }
__device__ inline float2 fp6_slice_public(float2 a) { return a; }
__device__ inline double2 fp6_slice_public(double2 a) { return a; }
__device__ inline __half2 fp6_slice_public(__half2_raw a) { return __half2(a); }
__device__ inline __nv_bfloat162 fp6_slice_public(__nv_bfloat162_raw a) { return __nv_bfloat162(a); }
template<class Quad> __device__ Quad fp6_slice_quad(float a,float b,float c,float d) { return Quad(float4{a,b,c,d}); }
template<class Quad> __device__ Quad fp6_slice_quad(double a,double b,double c,double d) { return Quad(double4_16a{a,b,c,d}); }
template<class Quad,class Raw> __device__ Quad fp6_slice_quad(Raw a,Raw b,Raw c,Raw d) { return Quad(fp6_slice_public(fp6_slice_pair(a,b)),fp6_slice_public(fp6_slice_pair(c,d))); }
template<unsigned Source> __device__ auto fp6_slice_input(unsigned long long bits) {
    if constexpr(Source==1) return __uint_as_float(unsigned(bits));
    else if constexpr(Source==2) return __longlong_as_double(static_cast<long long>(bits));
    else if constexpr(Source==3) return __half_raw{static_cast<unsigned short>(bits)};
    else return __nv_bfloat16_raw{static_cast<unsigned short>(bits)};
}
template<bool E2,unsigned Source>
__device__ void fp6_slice_execute(unsigned long long* output,const unsigned long long* input,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    using types=fp6_slice_types<E2>;
    if constexpr(Source==0) {
        const unsigned bits=unsigned(input[i*3]);
        typename types::scalar scalar; scalar.__x=static_cast<unsigned char>(bits);
        typename types::pair pair; pair.__x=static_cast<unsigned short>(bits);
        typename types::quad quad; quad.__x=bits;
        const auto h=__nv_cvt_fp6_to_halfraw(scalar.__x,types::format);
        const auto bf=__nv_cvt_fp6_to_bf16raw(scalar.__x,types::format);
        const auto h2=__nv_cvt_fp6x2_to_halfraw2(pair.__x,types::format);
        const auto bf2=__nv_cvt_fp6x2_to_bf162raw(pair.__x,types::format);
        const auto f2=static_cast<float2>(pair); const auto f4=static_cast<float4>(quad);
        output+=i*12;
        output[0]=h.x; output[1]=bf.x;
        output[2]=__float_as_uint(static_cast<float>(scalar));
        output[3]=static_cast<unsigned long long>(__double_as_longlong(static_cast<double>(scalar)));
        output[4]=unsigned(h2.x)|(unsigned(h2.y)<<16); output[5]=unsigned(bf2.x)|(unsigned(bf2.y)<<16);
        output[6]=__float_as_uint(f2.x); output[7]=__float_as_uint(f2.y);
        output[8]=__float_as_uint(f4.x); output[9]=__float_as_uint(f4.y);
        output[10]=__float_as_uint(f4.z); output[11]=__float_as_uint(f4.w);
    } else {
        const auto a=fp6_slice_input<Source>(input[i*3]),b=fp6_slice_input<Source>(input[i*3+1]),c=fp6_slice_input<Source>(input[i*3+2]);
        const auto d=fp6_slice_input<Source>(input[i*3+1]^(Source==1?0x80000000ull:Source==2?0x8000000000000000ull:0x8000ull));
        const auto pair=fp6_slice_pair(a,b);
        output+=i*7;
        output[0]=fp6_slice_narrow(a,cudaRoundNearest,types::format); output[1]=fp6_slice_narrow(a,cudaRoundZero,types::format);
        output[2]=fp6_slice_narrow(pair,cudaRoundNearest,types::format); output[3]=fp6_slice_narrow(pair,cudaRoundZero,types::format);
        output[4]=typename types::scalar(fp6_slice_public(a)).__x;
        output[5]=typename types::pair(fp6_slice_public(pair)).__x;
        output[6]=fp6_slice_quad<typename types::quad>(a,b,c,d).__x;
    }
}
}
#define FP6_KERNEL(NAME,E2,SOURCE) \
extern "C" __global__ void fp6_##NAME(unsigned long long* output,const unsigned long long* input,unsigned count) { \
    cudavm::fp6_slice_execute<E2,SOURCE>(output,input,count); \
}
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 0
FP6_KERNEL(e2m3_widen,true,0)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 1
FP6_KERNEL(e3m2_widen,false,0)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 2
FP6_KERNEL(e2m3_float,true,1)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 3
FP6_KERNEL(e3m2_float,false,1)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 4
FP6_KERNEL(e2m3_double,true,2)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 5
FP6_KERNEL(e3m2_double,false,2)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 6
FP6_KERNEL(e2m3_half,true,3)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 7
FP6_KERNEL(e3m2_half,false,3)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 8
FP6_KERNEL(e2m3_bfloat,true,4)
#endif
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 9
FP6_KERNEL(e3m2_bfloat,false,4)
#endif
#undef FP6_KERNEL
#if !defined(CUDAVM_FP6_PROFILE) || CUDAVM_FP6_PROFILE == 10
static_assert(sizeof(__nv_fp6_e2m3)==1 && alignof(__nv_fp6_e2m3)==1);
static_assert(sizeof(__nv_fp6_e3m2)==1 && alignof(__nv_fp6_e3m2)==1);
static_assert(sizeof(__nv_fp6x2_e2m3)==2 && alignof(__nv_fp6x2_e2m3)==2);
static_assert(sizeof(__nv_fp6x2_e3m2)==2 && alignof(__nv_fp6x2_e3m2)==2);
static_assert(sizeof(__nv_fp6x4_e2m3)==4 && alignof(__nv_fp6x4_e2m3)==4);
static_assert(sizeof(__nv_fp6x4_e3m2)==4 && alignof(__nv_fp6x4_e3m2)==4);
extern "C" __global__ void fp6_representation(unsigned long long* output,const unsigned long long* input,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const unsigned bits=unsigned(input[i*3]);
    __nv_fp6_e2m3 a; a.__x=static_cast<unsigned char>(bits);
    __nv_fp6x2_e2m3 b; b.__x=static_cast<unsigned short>(bits);
    __nv_fp6x4_e2m3 c; c.__x=bits;
    __nv_fp6_e3m2 x; x.__x=static_cast<unsigned char>(bits);
    __nv_fp6x2_e3m2 y; y.__x=static_cast<unsigned short>(bits);
    __nv_fp6x4_e3m2 z; z.__x=bits;
    const auto ac=a; const auto bc=b; const auto cc=c; const auto xc=x; const auto yc=y; const auto zc=z;
    output+=i*6;output[0]=ac.__x;output[1]=bc.__x;output[2]=cc.__x;output[3]=xc.__x;output[4]=yc.__x;output[5]=zc.__x;
}
#endif

