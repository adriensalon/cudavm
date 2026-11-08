#pragma clang system_header
#define __CUDACC__ 1
#define __CUDACC_RTC__ 1
#define __CUDACC_VER_MAJOR__ 13
#define __CUDACC_VER_MINOR__ 4
#define __CUDACC_VER_BUILD__ 1
#define CUDA_VERSION 13040
#define __FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__ 1
#define __FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__ 1
#if __has_include(<nv/target>)
#include <nv/target>
#endif
#include <vector_types.h>
#include <__clang_cuda_builtin_vars.h>
typedef __SIZE_TYPE__ size_t;
#include <cuda_runtime_api.h>
#include <__clang_cuda_texture_intrinsics.h>
#include <texture_indirect_functions.h>
#include <surface_indirect_functions.h>
__host__ __device__ inline float2 make_float2(float x,float y) { return float2{x,y}; }
// NVRTC supplies this utility to the CUDA numeric headers without a host C++
// standard library. The lvalue overload also coexists with a caller's utility.
namespace std { template<class T> __device__ constexpr T&& move(T& value) { return static_cast<T&&>(value); } }
#ifndef NULL
#define NULL __null
#endif
__device__ inline __cuda_builtin_threadIdx_t::operator dim3() const { return dim3(x,y,z); }
__device__ inline __cuda_builtin_threadIdx_t::operator uint3() const { return uint3{x,y,z}; }
__device__ inline __cuda_builtin_blockIdx_t::operator dim3() const { return dim3(x,y,z); }
__device__ inline __cuda_builtin_blockIdx_t::operator uint3() const { return uint3{x,y,z}; }
__device__ inline __cuda_builtin_blockDim_t::operator dim3() const { return dim3(x,y,z); }
__device__ inline __cuda_builtin_blockDim_t::operator uint3() const { return uint3{x,y,z}; }
__device__ inline __cuda_builtin_gridDim_t::operator dim3() const { return dim3(x,y,z); }
__device__ inline __cuda_builtin_gridDim_t::operator uint3() const { return uint3{x,y,z}; }
extern "C" __device__ void __cudavm_rtc_syncthreads();
#define __syncthreads() __cudavm_rtc_syncthreads()
extern "C" __device__ void __trap();
extern "C" __device__ void* malloc(__SIZE_TYPE__);
extern "C" __device__ void free(void*);
extern "C" __device__ void* __nv_aligned_device_malloc(__SIZE_TYPE__, __SIZE_TYPE__);
extern "C" __device__ int printf(const char*, ...);
extern "C" __device__ int vprintf(const char*, const char*);
extern "C" __device__ void __assertfail(const char*, const char*, unsigned, const char*, __SIZE_TYPE__);
#ifndef assert
#ifdef NDEBUG
#define assert(condition) static_cast<void>(0)
#else
#define assert(condition) ((condition) ? static_cast<void>(0) : __assertfail(#condition, __FILE__, __LINE__, __func__, sizeof(char)))
#endif
#endif
extern "C" __device__ inline void* memcpy(void* destination,const void* source,__SIZE_TYPE__ bytes) {return __builtin_memcpy(destination,source,bytes);}
extern "C" __device__ __attribute__((convergent)) void __cudavm_rtc_barrier_sync(unsigned);
extern "C" __device__ __attribute__((convergent)) void __cudavm_rtc_barrier_count(unsigned,unsigned);
__device__ inline void __barrier_sync(unsigned id) {__cudavm_rtc_barrier_sync(id);}
__device__ inline void __barrier_sync_count(unsigned id,unsigned count) {__cudavm_rtc_barrier_count(id,count);}
extern "C" __device__ void __cudavm_rtc_fence_block();
extern "C" __device__ void __cudavm_rtc_fence_device();
extern "C" __device__ void __cudavm_rtc_fence_system();
__device__ inline void __threadfence_block() {__cudavm_rtc_fence_block();}
__device__ inline void __threadfence() {__cudavm_rtc_fence_device();}
__device__ inline void __threadfence_system() {__cudavm_rtc_fence_system();}
// Legacy public pipeline headers parse timed barrier helpers even when callers
// only use the pipeline's batch operations. Keep native timing operations inline;
// a shader that reaches them still requires explicit backend lowering.
__device__ inline long long clock64() {
    long long cycles; asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles)); return cycles;
}
#if __CUDA_ARCH__ >= 700
__device__ inline void __nanosleep(unsigned nanoseconds) { asm volatile("nanosleep.u32 %0;" :: "r"(nanoseconds)); }
#endif
extern "C" __device__ __attribute__((const)) unsigned __cudavm_rtc_is_global(const void*);
extern "C" __device__ __attribute__((const)) unsigned __cudavm_rtc_is_shared(const void*);
extern "C" __device__ __attribute__((const)) unsigned __cudavm_rtc_is_local(const void*);
extern "C" __device__ __attribute__((const)) unsigned __cudavm_rtc_is_constant(const void*);
__device__ inline __attribute__((const)) unsigned __isGlobal(const void* pointer) {return __cudavm_rtc_is_global(pointer);}
__device__ inline __attribute__((const)) unsigned __isShared(const void* pointer) {return __cudavm_rtc_is_shared(pointer);}
__device__ inline __attribute__((const)) unsigned __isLocal(const void* pointer) {return __cudavm_rtc_is_local(pointer);}
__device__ inline __attribute__((const)) unsigned __isConstant(const void* pointer) {return __cudavm_rtc_is_constant(pointer);}
__device__ inline size_t __cvta_generic_to_global(const void* pointer) {
    return (size_t)(void __attribute__((address_space(1)))*)pointer;
}
__device__ inline size_t __cvta_generic_to_shared(const void* pointer) {
    return (size_t)(void __attribute__((address_space(3)))*)pointer;
}
__device__ inline size_t __cvta_generic_to_local(const void* pointer) {
    return (size_t)(void __attribute__((address_space(5)))*)pointer;
}
__device__ inline size_t __cvta_generic_to_constant(const void* pointer) {
    return (size_t)(void __attribute__((address_space(4)))*)pointer;
}
__device__ inline void* __cvta_global_to_generic(size_t pointer) {
    return (void*)(void __attribute__((address_space(1)))*)pointer;
}
__device__ inline void* __cvta_shared_to_generic(size_t pointer) {
    return (void*)(void __attribute__((address_space(3)))*)pointer;
}
__device__ inline void* __cvta_local_to_generic(size_t pointer) {
    return (void*)(void __attribute__((address_space(5)))*)pointer;
}
__device__ inline void* __cvta_constant_to_generic(size_t pointer) {
    return (void*)(void __attribute__((address_space(4)))*)pointer;
}
extern "C" __device__ __attribute__((convergent)) int __cudavm_rtc_block_popc(int);
extern "C" __device__ __attribute__((convergent)) int __cudavm_rtc_block_and(int);
extern "C" __device__ __attribute__((convergent)) int __cudavm_rtc_block_or(int);
__device__ inline int __syncthreads_count(int p) {return __cudavm_rtc_block_popc(p);}
__device__ inline int __syncthreads_and(int p) {return __cudavm_rtc_block_and(p);}
__device__ inline int __syncthreads_or(int p) {return __cudavm_rtc_block_or(p);}
// Generic Clang builtins produce portable LLVM integer intrinsics. The zero
// guards retain CUDA's defined clz/ffs results without target builtin enums.
__device__ inline int __popc(unsigned x) {return __builtin_popcount(x);}
__device__ inline int __popcll(unsigned long long x) {return __builtin_popcountll(x);}
__device__ inline int __clz(int x) {return x?__builtin_clz(unsigned(x)):32;}
__device__ inline int __clzll(long long x) {return x?__builtin_clzll(static_cast<unsigned long long>(x)):64;}
__device__ inline int __ffs(int x) {return __builtin_ffs(x);}
__device__ inline int __ffsll(long long x) {return __builtin_ffsll(x);}
__device__ inline unsigned __fns(unsigned mask,unsigned base,int offset) {
    if(offset==0)return base;
    unsigned remaining=offset<0?0u-unsigned(offset):unsigned(offset);
    for(unsigned i=0;i<32;++i) {
        unsigned lane=offset<0?base-i:base+i;
        if(lane<32u && (mask&(1u<<lane)) && --remaining==0u)return lane;
    }
    return 0xffffffffu;
}
__device__ inline unsigned __brev(unsigned x) {return __builtin_bitreverse32(x);}
__device__ inline unsigned long long __brevll(unsigned long long x) {return __builtin_bitreverse64(x);}
__device__ inline float __uint_as_float(unsigned x) {return __builtin_bit_cast(float,x);}
__device__ inline float __int_as_float(int x) {return __builtin_bit_cast(float,x);}
__device__ inline unsigned __float_as_uint(float x) {return __builtin_bit_cast(unsigned,x);}
__device__ inline int __float_as_int(float x) {return __builtin_bit_cast(int,x);}
__device__ inline double __longlong_as_double(long long x) {return __builtin_bit_cast(double,x);}
__device__ inline long long __double_as_longlong(double x) {return __builtin_bit_cast(long long,x);}
__device__ inline int __double2loint(double x) {
    unsigned word=unsigned(__builtin_bit_cast(unsigned long long,x));
    return __builtin_bit_cast(int,word);
}
__device__ inline int __double2hiint(double x) {
    unsigned word=unsigned(__builtin_bit_cast(unsigned long long,x)>>32);
    return __builtin_bit_cast(int,word);
}
__device__ inline double __hiloint2double(int hi,int lo) {
    unsigned long long bits=(static_cast<unsigned long long>(unsigned(hi))<<32)|unsigned(lo);
    return __builtin_bit_cast(double,bits);
}
// Parsed by the half header's optional transcendental wrappers. A caller using
// it still needs a device implementation, like other unresolved NVRTC calls.
extern "C" __device__ float tanhf(float);
// Bfloat16 fallbacks use the strict, FTZ-independent FMA wrapper supplied by
// the embedded device math source below.
extern "C" __device__ float __fmaf_ieee_rn(float, float, float);
extern "C" __device__ unsigned __nv_byte_perm(unsigned,unsigned,unsigned);
__device__ inline unsigned __byte_perm(unsigned x,unsigned y,unsigned s) {return __nv_byte_perm(x,y,s);}
__device__ inline unsigned __funnelshift_l(unsigned lo,unsigned hi,unsigned s) {
    unsigned result;asm("shf.l.wrap.b32 %0, %1, %2, %3;":"=r"(result):"r"(lo),"r"(hi),"r"(s));return result;
}
__device__ inline unsigned __funnelshift_lc(unsigned lo,unsigned hi,unsigned s) {
    unsigned result;asm("shf.l.clamp.b32 %0, %1, %2, %3;":"=r"(result):"r"(lo),"r"(hi),"r"(s));return result;
}
__device__ inline unsigned __funnelshift_r(unsigned lo,unsigned hi,unsigned s) {
    unsigned result;asm("shf.r.wrap.b32 %0, %1, %2, %3;":"=r"(result):"r"(lo),"r"(hi),"r"(s));return result;
}
__device__ inline unsigned __funnelshift_rc(unsigned lo,unsigned hi,unsigned s) {
    unsigned result;asm("shf.r.clamp.b32 %0, %1, %2, %3;":"=r"(result):"r"(lo),"r"(hi),"r"(s));return result;
}
// Use generic Clang atomic builtins, which emit ordinary LLVM atomics without
// depending on the target-specific NVVM builtin enum or an installed toolkit.
__device__ inline unsigned atomicAdd(unsigned* p,unsigned v) {return __atomic_fetch_add(p,v,__ATOMIC_RELAXED);}
// The public SM70/75 arrive/wait helpers require this CTA-scoped overload.
// Semantic preparation converts the assembly to an ordinary scoped LLVM atomic.
__device__ inline unsigned atomicAdd_block(unsigned* p,unsigned v) {
    unsigned previous;
    asm volatile("atom.add.relaxed.cta.u32 %0, [%1], %2;" : "=r"(previous) : "l"(p), "r"(v) : "memory");
    return previous;
}
__device__ inline unsigned atomicOr(unsigned* p,unsigned v) {return __atomic_fetch_or(p,v,__ATOMIC_RELAXED);}
__device__ inline unsigned atomicAnd(unsigned* p,unsigned v) {return __atomic_fetch_and(p,v,__ATOMIC_RELAXED);}
__device__ inline unsigned atomicCAS(unsigned* p,unsigned c,unsigned v) {__atomic_compare_exchange_n(p,&c,v,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED);return c;}
// The bfloat16 header parses optional subword atomic wrappers. Executing them
// requires a separately checked subword-atomic implementation.
__device__ unsigned short atomicCAS(unsigned short*, unsigned short, unsigned short);
__device__ inline unsigned min(unsigned a,unsigned b) {return a<b?a:b;}
__device__ inline unsigned max(unsigned a,unsigned b) {return a>b?a:b;}
__device__ inline int min(int a,int b) {return a<b?a:b;}
__device__ inline int max(int a,int b) {return a>b?a:b;}
extern "C" __device__ float __cudavm_rtc_atomic_add_float(float*,float);
extern "C" __device__ double __cudavm_rtc_atomic_add_double(double*,double);
__device__ inline float atomicAdd(float* p,float v) { return __cudavm_rtc_atomic_add_float(p,v); }
__device__ inline double atomicAdd(double* p,double v) { return __cudavm_rtc_atomic_add_double(p,v); }
__device__ inline float atomicExch(float* p,float v) { union {float f;unsigned u;} bits;bits.f=v;bits.u=__atomic_exchange_n(reinterpret_cast<unsigned*>(p),bits.u,__ATOMIC_RELAXED);return bits.f; }
__device__ inline unsigned long long atomicAdd(unsigned long long* p,unsigned long long v) { return __atomic_fetch_add(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicExch(unsigned long long* p,unsigned long long v) { return __atomic_exchange_n(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicCAS(unsigned long long* p,unsigned long long c,unsigned long long v) { __atomic_compare_exchange_n(p,&c,v,false,__ATOMIC_RELAXED,__ATOMIC_RELAXED);return c; }
__device__ inline unsigned long long atomicAnd(unsigned long long* p,unsigned long long v) { return __atomic_fetch_and(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicOr(unsigned long long* p,unsigned long long v) { return __atomic_fetch_or(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicXor(unsigned long long* p,unsigned long long v) { return __atomic_fetch_xor(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicMin(unsigned long long* p,unsigned long long v) { return __atomic_fetch_min(p,v,__ATOMIC_RELAXED); }
__device__ inline unsigned long long atomicMax(unsigned long long* p,unsigned long long v) { return __atomic_fetch_max(p,v,__ATOMIC_RELAXED); }
__device__ inline long long atomicMin(long long* p,long long v) { return __atomic_fetch_min(p,v,__ATOMIC_RELAXED); }
__device__ inline long long atomicMax(long long* p,long long v) { return __atomic_fetch_max(p,v,__ATOMIC_RELAXED); }
