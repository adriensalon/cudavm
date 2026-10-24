#pragma once
/* Export declarations owned by the nvjitlink module. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_NVJITLINK_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_NVJITLINK)
#  define CUDAVM_NVJITLINK_API __declspec(dllexport)
# else
#  define CUDAVM_NVJITLINK_API __declspec(dllimport)
# endif
#else
# define CUDAVM_NVJITLINK_API __attribute__((visibility("default")))
#endif
