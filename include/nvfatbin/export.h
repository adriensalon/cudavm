#pragma once
/* Export declarations owned by the nvfatbin module. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_NVFATBIN_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_NVFATBIN)
#  define CUDAVM_NVFATBIN_API __declspec(dllexport)
# else
#  define CUDAVM_NVFATBIN_API __declspec(dllimport)
# endif
#else
# define CUDAVM_NVFATBIN_API __attribute__((visibility("default")))
#endif
