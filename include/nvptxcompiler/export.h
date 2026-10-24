#pragma once
/* Export declarations owned by the nvptxcompiler module. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_NVPTXCOMPILER_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_NVPTXCOMPILER)
#  define CUDAVM_NVPTXCOMPILER_API __declspec(dllexport)
# else
#  define CUDAVM_NVPTXCOMPILER_API __declspec(dllimport)
# endif
#else
# define CUDAVM_NVPTXCOMPILER_API __attribute__((visibility("default")))
#endif
