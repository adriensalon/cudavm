#pragma once
/* Export declarations owned by the nvvm module. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_NVVM_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_NVVM)
#  define CUDAVM_NVVM_API __declspec(dllexport)
# else
#  define CUDAVM_NVVM_API __declspec(dllimport)
# endif
#else
# define CUDAVM_NVVM_API __attribute__((visibility("default")))
#endif
