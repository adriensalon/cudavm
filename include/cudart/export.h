#pragma once
/* Export declarations owned by the cudart module. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_CUDART_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_CUDART)
#  define CUDAVM_CUDART_API __declspec(dllexport)
# else
#  define CUDAVM_CUDART_API __declspec(dllimport)
# endif
#else
# define CUDAVM_CUDART_API __attribute__((visibility("default")))
#endif
