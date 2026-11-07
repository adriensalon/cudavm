// Explicit arithmetic intrinsics retain their PTX rounding and contraction
// contract on native CUDA. Shader preparation imports the same instructions
// through the shared arithmetic lowering and exact numerical helpers.
#define CUDAVM_FLOAT_ARITHMETIC(MODE, PTX) \
extern "C" __device__ inline float __nv_fadd_##MODE(float x,float y) { \
    float result; asm("add." PTX ".f32 %0,%1,%2;" : "=f"(result) : "f"(x), "f"(y)); return result; } \
extern "C" __device__ inline float __nv_fsub_##MODE(float x,float y) { \
    float result; asm("sub." PTX ".f32 %0,%1,%2;" : "=f"(result) : "f"(x), "f"(y)); return result; } \
extern "C" __device__ inline float __nv_fmul_##MODE(float x,float y) { \
    float result; asm("mul." PTX ".f32 %0,%1,%2;" : "=f"(result) : "f"(x), "f"(y)); return result; } \
extern "C" __device__ inline float __nv_fmaf_##MODE(float x,float y,float z) { \
    float result; asm("fma." PTX ".f32 %0,%1,%2,%3;" : "=f"(result) : "f"(x), "f"(y), "f"(z)); return result; }
CUDAVM_FLOAT_ARITHMETIC(rn, "rn")
CUDAVM_FLOAT_ARITHMETIC(rz, "rz")
CUDAVM_FLOAT_ARITHMETIC(rd, "rm")
CUDAVM_FLOAT_ARITHMETIC(ru, "rp")
#undef CUDAVM_FLOAT_ARITHMETIC
extern "C" __device__ inline float __nv_fmaf(float x,float y,float z) {return __nv_fmaf_rn(x,y,z);}
#define CUDAVM_DOUBLE_ARITHMETIC(MODE, PTX) \
extern "C" __device__ inline double __nv_dadd_##MODE(double x,double y) { \
    double result; asm("add." PTX ".f64 %0,%1,%2;" : "=d"(result) : "d"(x), "d"(y)); return result; } \
extern "C" __device__ inline double __nv_dsub_##MODE(double x,double y) { \
    double result; asm("sub." PTX ".f64 %0,%1,%2;" : "=d"(result) : "d"(x), "d"(y)); return result; } \
extern "C" __device__ inline double __nv_dmul_##MODE(double x,double y) { \
    double result; asm("mul." PTX ".f64 %0,%1,%2;" : "=d"(result) : "d"(x), "d"(y)); return result; } \
extern "C" __device__ inline double __nv_fma_##MODE(double x,double y,double z) { \
    double result; asm("fma." PTX ".f64 %0,%1,%2,%3;" : "=d"(result) : "d"(x), "d"(y), "d"(z)); return result; }
CUDAVM_DOUBLE_ARITHMETIC(rn, "rn")
CUDAVM_DOUBLE_ARITHMETIC(rz, "rz")
CUDAVM_DOUBLE_ARITHMETIC(rd, "rm")
CUDAVM_DOUBLE_ARITHMETIC(ru, "rp")
#undef CUDAVM_DOUBLE_ARITHMETIC
extern "C" __device__ inline double __nv_fma(double x,double y,double z) {return __nv_fma_rn(x,y,z);}
#ifndef __CLANG_CUDA_DEVICE_FUNCTIONS_H__
// Clang's source AOT wrapper supplies these public names itself.
#define CUDAVM_FLOAT_ARITHMETIC(MODE) \
__device__ inline float __fadd_##MODE(float x,float y) {return __nv_fadd_##MODE(x,y);} \
__device__ inline float __fsub_##MODE(float x,float y) {return __nv_fsub_##MODE(x,y);} \
__device__ inline float __fmul_##MODE(float x,float y) {return __nv_fmul_##MODE(x,y);} \
__device__ inline float __fmaf_##MODE(float x,float y,float z) {return __nv_fmaf_##MODE(x,y,z);}
CUDAVM_FLOAT_ARITHMETIC(rn)
CUDAVM_FLOAT_ARITHMETIC(rz)
CUDAVM_FLOAT_ARITHMETIC(rd)
CUDAVM_FLOAT_ARITHMETIC(ru)
#undef CUDAVM_FLOAT_ARITHMETIC
extern "C" __device__ inline float fmaf(float x,float y,float z) {return __nv_fmaf(x,y,z);}
#define CUDAVM_DOUBLE_ARITHMETIC(MODE) \
__device__ inline double __dadd_##MODE(double x,double y) {return __nv_dadd_##MODE(x,y);} \
__device__ inline double __dsub_##MODE(double x,double y) {return __nv_dsub_##MODE(x,y);} \
__device__ inline double __dmul_##MODE(double x,double y) {return __nv_dmul_##MODE(x,y);} \
__device__ inline double __fma_##MODE(double x,double y,double z) {return __nv_fma_##MODE(x,y,z);}
CUDAVM_DOUBLE_ARITHMETIC(rn)
CUDAVM_DOUBLE_ARITHMETIC(rz)
CUDAVM_DOUBLE_ARITHMETIC(rd)
CUDAVM_DOUBLE_ARITHMETIC(ru)
#undef CUDAVM_DOUBLE_ARITHMETIC
extern "C" __device__ inline double fma(double x,double y,double z) {return __nv_fma(x,y,z);}
#endif
