#pragma once
#include <cuda.h>
#include <cuda/cudavm_cuda.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Required when importing opaque Vulkan buffer allocations into the Vulkan
   backend. Supply the exporting allocation's memoryTypeIndex, the original
   VkBufferCreateInfo size/usage (flags must be zero), and VkPhysicalDeviceIDProperties
   deviceUUID/driverUUID. The buffer must support storage and transfer usage.
   This is allocation metadata, independent of the application's API version. */
typedef struct cudavmExternalMemoryMetadata {
    unsigned int abi_version;
    size_t struct_size;
    unsigned int memory_type_index;
    uint64_t buffer_size;
    unsigned int buffer_usage;
    unsigned char device_uuid[16];
    unsigned char driver_uuid[16];
} cudavmExternalMemoryMetadata;
CUDAVM_API int cudavmImportExternalMemory(CUexternalMemory* output,
    const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* descriptor,
    const cudavmExternalMemoryMetadata* metadata);
#ifdef __cplusplus
}
#endif
