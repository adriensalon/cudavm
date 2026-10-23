#pragma once
/* A single linkage mode per process; each shared component owns its exports. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_CUDA_API
# define CUDAVM_CUDART_API
# define CUDAVM_NVRTC_API
# define CUDAVM_NVJITLINK_API
# define CUDAVM_COMPILER_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_CUDA)
#  define CUDAVM_CUDA_API __declspec(dllexport)
# else
#  define CUDAVM_CUDA_API __declspec(dllimport)
# endif
# if defined(CUDAVM_BUILDING_CUDART)
#  define CUDAVM_CUDART_API __declspec(dllexport)
# else
#  define CUDAVM_CUDART_API __declspec(dllimport)
# endif
# if defined(CUDAVM_BUILDING_NVRTC)
#  define CUDAVM_NVRTC_API __declspec(dllexport)
# else
#  define CUDAVM_NVRTC_API __declspec(dllimport)
# endif
# if defined(CUDAVM_BUILDING_COMPILER)
#  define CUDAVM_COMPILER_API __declspec(dllexport)
# else
#  define CUDAVM_COMPILER_API __declspec(dllimport)
# endif
# if defined(CUDAVM_BUILDING_NVJITLINK)
#  define CUDAVM_NVJITLINK_API __declspec(dllexport)
# else
#  define CUDAVM_NVJITLINK_API __declspec(dllimport)
# endif
#else
# define CUDAVM_CUDA_API __attribute__((visibility("default")))
# define CUDAVM_CUDART_API __attribute__((visibility("default")))
# define CUDAVM_NVRTC_API __attribute__((visibility("default")))
# define CUDAVM_NVJITLINK_API __attribute__((visibility("default")))
# define CUDAVM_COMPILER_API __attribute__((visibility("default")))
#endif
