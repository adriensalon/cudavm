#pragma once
#include <cuda_runtime.h>
#include <cudart/cudavm_registration.h>
#include <stdexcept>
#ifndef __noinline__
#if defined(_MSC_VER)
#define __noinline__ __declspec(noinline)
#else
#define __noinline__ __attribute__((noinline))
#endif
#endif
namespace cudavm {
template<class T> inline const void* symbol_address(const T& symbol) noexcept {
    return const_cast<const void*>(static_cast<const volatile void*>(&symbol));
}
[[noreturn]] inline void direct_kernel_call() {
    throw std::logic_error("a CUDA kernel identity stub cannot be called as a host function");
}
}
