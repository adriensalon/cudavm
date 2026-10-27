#pragma once
#include <stddef.h>
#include <stdint.h>
#include <cuda/export.h>
#define CUDAVM_API CUDAVM_CUDA_API
#ifdef __cplusplus
extern "C" {
#endif
/* Backend tags in the program registration ABI. Device discovery and selection
   use the standard CUDA device and context APIs. */
typedef enum cudavmBackend {
    cudavmBackendAuto = -1,
    cudavmBackendCuda = 0,
    cudavmBackendOpenCL = 1,
    cudavmBackendOpenGL = 2,
    cudavmBackendVulkan = 3,
    cudavmBackendDirectX = 4
} cudavmBackend;
/* Detailed thread-local diagnostic for the preceding cudavm operation. */
CUDAVM_API const char* cudavmGetDiagnostic(void);
/* Optional dispatch profiling on the current shader context. Enable before
   recording both timing events and keep enabled until both complete. The interval sums device execution of compute
   dispatches in this context; copies and queue/host idle time are excluded.
   Concurrent streams contribute to the same context total. Handles are CUevent.
   Disabled/unavailable profiling returns CUDA_ERROR_NOT_SUPPORTED. */
CUDAVM_API int cudavmEnableComputeProfiling(int enabled);
CUDAVM_API int cudavmEventComputeElapsedTime(float* milliseconds, void* start, void* end);
/* Release all native state before unloading the DLL. Call only after other
   host threads have stopped using cudavm; outstanding handles become invalid. */
CUDAVM_API int cudavmShutdown(void);
/* Handle masks use bit (1ULL << CUexternal*HandleType). Optional facilities
   are negotiated on the current device, independently of API version strings.
   struct_size and abi_version must be supplied by the caller. */
typedef struct cudavmInteropCapabilities {
    unsigned int abi_version;
    size_t struct_size;
    uint64_t memory_handle_types;
    uint64_t semaphore_handle_types;
    uint64_t memory_metadata_required_types;
    unsigned int graphics_gl_buffers; /* explicit bridge is attached */
} cudavmInteropCapabilities;
CUDAVM_API int cudavmGetInteropCapabilities(cudavmInteropCapabilities* output);
#ifdef __cplusplus
}
#endif
