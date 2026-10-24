#pragma once
/* Export declarations owned by the cuda module. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_CUDA_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_CUDA)
#  define CUDAVM_CUDA_API __declspec(dllexport)
# else
#  define CUDAVM_CUDA_API __declspec(dllimport)
# endif
#else
# define CUDAVM_CUDA_API __attribute__((visibility("default")))
#endif
