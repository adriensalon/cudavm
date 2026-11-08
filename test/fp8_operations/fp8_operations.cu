#include <cuda_fp8.h>
namespace cudavm {
template<bool E4> struct fp8_slice_types;
template<> struct fp8_slice_types<true> {
    using scalar = __nv_fp8_e4m3; using pair = __nv_fp8x2_e4m3; using quad = __nv_fp8x4_e4m3;
    static constexpr auto format = __NV_E4M3;
};
template<> struct fp8_slice_types<false> {
    using scalar = __nv_fp8_e5m2; using pair = __nv_fp8x2_e5m2; using quad = __nv_fp8x4_e5m2;
    static constexpr auto format = __NV_E5M2;
};
__device__ inline auto fp8_slice_narrow(float a, __nv_saturation_t s, __nv_fp8_interpretation_t f) { return __nv_cvt_float_to_fp8(a,s,f); }
__device__ inline auto fp8_slice_narrow(double a, __nv_saturation_t s, __nv_fp8_interpretation_t f) { return __nv_cvt_double_to_fp8(a,s,f); }
__device__ inline auto fp8_slice_narrow(__half_raw a, __nv_saturation_t s, __nv_fp8_interpretation_t f) { return __nv_cvt_halfraw_to_fp8(a,s,f); }
__device__ inline auto fp8_slice_narrow(__nv_bfloat16_raw a, __nv_saturation_t s, __nv_fp8_interpretation_t f) { return __nv_cvt_bfloat16raw_to_fp8(a,s,f); }
__device__ inline auto fp8_slice_narrow(float2 a, __nv_saturation_t s, __nv_fp8_interpretation_t f) { return __nv_cvt_float2_to_fp8x2(a,s,f); }
__device__ inline auto fp8_slice_narrow(double2 a, __nv_saturation_t s, __nv_fp8_interpretation_t f) { return __nv_cvt_double2_to_fp8x2(a,s,f); }
__device__ inline auto fp8_slice_narrow(__half2_raw a, __nv_saturation_t s, __nv_fp8_interpretation_t f) { return __nv_cvt_halfraw2_to_fp8x2(a,s,f); }
__device__ inline auto fp8_slice_narrow(__nv_bfloat162_raw a, __nv_saturation_t s, __nv_fp8_interpretation_t f) { return __nv_cvt_bfloat16raw2_to_fp8x2(a,s,f); }
__device__ inline float2 fp8_slice_pair(float a,float b) { return float2{a,b}; }
__device__ inline double2 fp8_slice_pair(double a,double b) { return double2{a,b}; }
__device__ inline __half2_raw fp8_slice_pair(__half_raw a,__half_raw b) { return __half2_raw{a.x,b.x}; }
__device__ inline __nv_bfloat162_raw fp8_slice_pair(__nv_bfloat16_raw a,__nv_bfloat16_raw b) { return __nv_bfloat162_raw{a.x,b.x}; }
__device__ inline float fp8_slice_public(float a) { return a; }
__device__ inline double fp8_slice_public(double a) { return a; }
__device__ inline __half fp8_slice_public(__half_raw a) { return __half(a); }
__device__ inline __nv_bfloat16 fp8_slice_public(__nv_bfloat16_raw a) { return __nv_bfloat16(a); }
__device__ inline float2 fp8_slice_public(float2 a) { return a; }
__device__ inline double2 fp8_slice_public(double2 a) { return a; }
__device__ inline __half2 fp8_slice_public(__half2_raw a) { return __half2(a); }
__device__ inline __nv_bfloat162 fp8_slice_public(__nv_bfloat162_raw a) { return __nv_bfloat162(a); }
template<class Quad> __device__ Quad fp8_slice_quad(float a,float b,float c,float d) { return Quad(float4{a,b,c,d}); }
template<class Quad> __device__ Quad fp8_slice_quad(double a,double b,double c,double d) { return Quad(double4{a,b,c,d}); }
template<class Quad,class Raw> __device__ Quad fp8_slice_quad(Raw a,Raw b,Raw c,Raw d) { return Quad(fp8_slice_public(fp8_slice_pair(a,b)),fp8_slice_public(fp8_slice_pair(c,d))); }
template<unsigned Source> __device__ auto fp8_slice_input(unsigned long long bits) {
    if constexpr(Source==1) return __uint_as_float(unsigned(bits));
    else if constexpr(Source==2) return __longlong_as_double(static_cast<long long>(bits));
    else if constexpr(Source==3) return __half_raw{static_cast<unsigned short>(bits)};
    else return __nv_bfloat16_raw{static_cast<unsigned short>(bits)};
}
template<bool E4,unsigned Source>
__device__ void fp8_slice_execute(unsigned long long* output,const unsigned long long* input,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    using types=fp8_slice_types<E4>;
    if constexpr(Source==0) {
        const unsigned bits=unsigned(input[i*3]);
        typename types::scalar scalar; scalar.__x=static_cast<unsigned char>(bits);
        typename types::pair pair; pair.__x=static_cast<unsigned short>(bits);
        typename types::quad quad; quad.__x=bits;
        const auto h=__nv_cvt_fp8_to_halfraw(scalar.__x,types::format);
        const auto bf=__nv_cvt_fp8_to_bf16raw(scalar.__x,types::format);
        const auto h2=__nv_cvt_fp8x2_to_halfraw2(pair.__x,types::format);
        const auto bf2=__nv_cvt_fp8x2_to_bf162raw(pair.__x,types::format);
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
        const auto a=fp8_slice_input<Source>(input[i*3]),b=fp8_slice_input<Source>(input[i*3+1]),c=fp8_slice_input<Source>(input[i*3+2]);
        const auto d=fp8_slice_input<Source>(input[i*3+1]^(Source==1?0x80000000ull:Source==2?0x8000000000000000ull:0x8000ull));
        const auto pair=fp8_slice_pair(a,b);
        output+=i*7;
        output[0]=fp8_slice_narrow(a,__NV_NOSAT,types::format); output[1]=fp8_slice_narrow(a,__NV_SATFINITE,types::format);
        output[2]=fp8_slice_narrow(pair,__NV_NOSAT,types::format); output[3]=fp8_slice_narrow(pair,__NV_SATFINITE,types::format);
        output[4]=typename types::scalar(fp8_slice_public(a)).__x;
        output[5]=typename types::pair(fp8_slice_public(pair)).__x;
        output[6]=fp8_slice_quad<typename types::quad>(a,b,c,d).__x;
    }
}
}
#define FP8_KERNEL(NAME,E4,SOURCE) \
extern "C" __global__ void fp8_##NAME(unsigned long long* output,const unsigned long long* input,unsigned count) { \
    cudavm::fp8_slice_execute<E4,SOURCE>(output,input,count); \
}
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 0
FP8_KERNEL(e4m3_widen,true,0)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 1
FP8_KERNEL(e5m2_widen,false,0)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 2
FP8_KERNEL(e4m3_float,true,1)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 3
FP8_KERNEL(e5m2_float,false,1)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 4
FP8_KERNEL(e4m3_double,true,2)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 5
FP8_KERNEL(e5m2_double,false,2)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 6
FP8_KERNEL(e4m3_half,true,3)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 7
FP8_KERNEL(e5m2_half,false,3)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 8
FP8_KERNEL(e4m3_bfloat,true,4)
#endif
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 9
FP8_KERNEL(e5m2_bfloat,false,4)
#endif
#undef FP8_KERNEL
#if !defined(CUDAVM_FP8_PROFILE) || CUDAVM_FP8_PROFILE == 10
static_assert(sizeof(__nv_fp8_e4m3)==1 && alignof(__nv_fp8_e4m3)==1);
static_assert(sizeof(__nv_fp8_e5m2)==1 && alignof(__nv_fp8_e5m2)==1);
static_assert(sizeof(__nv_fp8x2_e4m3)==2 && alignof(__nv_fp8x2_e4m3)==2);
static_assert(sizeof(__nv_fp8x2_e5m2)==2 && alignof(__nv_fp8x2_e5m2)==2);
static_assert(sizeof(__nv_fp8x4_e4m3)==4 && alignof(__nv_fp8x4_e4m3)==4);
static_assert(sizeof(__nv_fp8x4_e5m2)==4 && alignof(__nv_fp8x4_e5m2)==4);
extern "C" __global__ void fp8_representation(unsigned long long* output,const unsigned long long* input,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const unsigned bits=unsigned(input[i*3]);
    __nv_fp8_e4m3 a; a.__x=static_cast<unsigned char>(bits);
    __nv_fp8x2_e4m3 b; b.__x=static_cast<unsigned short>(bits);
    __nv_fp8x4_e4m3 c; c.__x=bits;
    __nv_fp8_e5m2 x; x.__x=static_cast<unsigned char>(bits);
    __nv_fp8x2_e5m2 y; y.__x=static_cast<unsigned short>(bits);
    __nv_fp8x4_e5m2 z; z.__x=bits;
    const auto ac=a; const auto bc=b; const auto cc=c; const auto xc=x; const auto yc=y; const auto zc=z;
    output+=i*6;output[0]=ac.__x;output[1]=bc.__x;output[2]=cc.__x;output[3]=xc.__x;output[4]=yc.__x;output[5]=zc.__x;
}
#endif
