#pragma once
/* Export declarations owned by the private core module. */
#if defined(CUDAVM_STATIC)
# define CUDAVM_CORE_API
#elif defined(_WIN32)
# if defined(CUDAVM_BUILDING_CORE)
#  define CUDAVM_CORE_API __declspec(dllexport)
# else
#  define CUDAVM_CORE_API __declspec(dllimport)
# endif
#else
# define CUDAVM_CORE_API __attribute__((visibility("default")))
#endif
