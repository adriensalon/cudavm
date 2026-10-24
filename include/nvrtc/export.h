#pragma once
/* Export declarations owned by the nvrtc module. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_NVRTC_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_NVRTC)
#  define CUDAVM_NVRTC_API __declspec(dllexport)
# else
#  define CUDAVM_NVRTC_API __declspec(dllimport)
# endif
#else
# define CUDAVM_NVRTC_API __attribute__((visibility("default")))
#endif
