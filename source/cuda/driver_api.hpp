#pragma once
#include <cuda.h>
#include <cuda/export.h>
#include <stdint.h>
// Private Driver contract, declared and implemented by Driver. Never a facade
// dispatch table. Runtime composes public cu* calls with these registry probes.
extern "C" {
CUDAVM_CUDA_API uint64_t cudavmDriverEpoch();
CUDAVM_CUDA_API int cudavmDriverContextIsPrimary(CUcontext);
CUDAVM_CUDA_API int cudavmDriverContextExists(CUcontext);
CUDAVM_CUDA_API int cudavmDriverIsLibraryKernel(const void*);
CUDAVM_CUDA_API CUresult cudavmDriverLoadRegisteredLibrary(CUlibrary*, const void*, size_t);
CUDAVM_CUDA_API CUresult cudavmDriverLoadRegisteredRdcLibrary(CUlibrary*, const void* const*, const size_t*, size_t, int);
CUDAVM_CUDA_API CUresult cudavmDriverCreateManagedRegistration(void**, const void*, size_t, const char* const*, const void* const*, const size_t*, size_t);
CUDAVM_CUDA_API CUresult cudavmDriverLoadManagedRegistration(CUlibrary*, void*);
CUDAVM_CUDA_API CUresult cudavmDriverRetireManagedRegistration(void*);
CUDAVM_CUDA_API void cudavmDriverReleaseManagedRegistration(void*);
CUDAVM_CUDA_API CUresult cudavmDriverNvidiaImageSize(const void*, size_t*);
CUDAVM_CUDA_API CUresult cudavmDriverValidateNvidiaImage(const void*, size_t);
CUDAVM_CUDA_API CUresult cudavmDriverReleaseRegisteredModules(CUcontext, const CUlibrary*, size_t);
CUDAVM_CUDA_API int cudavmDriverPrimaryDevice(CUcontext);
CUDAVM_CUDA_API uint64_t cudavmDriverContextGeneration(CUcontext);
CUDAVM_CUDA_API int cudavmDriverIsDevicePointer(CUdeviceptr);
CUDAVM_CUDA_API int cudavmDriverIsManagedPointer(CUdeviceptr);
CUDAVM_CUDA_API int cudavmDriverIsHostPointer(CUdeviceptr);
CUDAVM_CUDA_API int cudavmDriverPhysicalDevice(int device);
CUDAVM_CUDA_API const char* cudavmDriverDiagnostic();
CUDAVM_CUDA_API void cudavmDriverSetDiagnostic(const char*);
CUDAVM_CUDA_API int cudavmDriverInHostFunction();
CUDAVM_CUDA_API CUresult cudavmDriverMemset3DAsync(CUdeviceptr, size_t pitch, size_t slice_height,
                                                 unsigned char value, size_t width, size_t height, size_t depth, CUstream);
CUDAVM_CUDA_API CUresult cudavmDriverMemset3D(CUdeviceptr, size_t pitch, size_t slice_height,
                                            unsigned char value, size_t width, size_t height, size_t depth);
}
