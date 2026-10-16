DRIVER_API(cuDeviceGetUuid, (CUuuid* output, CUdevice device), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null device UUID output"); valid_device(device);
    auto value = global().provider->device_uuid(device, true); std::memcpy(output->bytes, value.data(), value.size()))
#undef cuDeviceGetUuid
DRIVER_API(cuDeviceGetUuid, (CUuuid* output, CUdevice device), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null device UUID output"); valid_device(device);
    auto value = global().provider->device_uuid(device, false); std::memcpy(output->bytes, value.data(), value.size()))
DRIVER_API(cuDeviceGetLuid, (char* output, unsigned* mask, CUdevice device), driver_ready(); require(output && mask, CUDA_ERROR_INVALID_VALUE, "null device LUID output"); valid_device(device);
    auto value = global().provider->device_luid(device); std::memcpy(output, value.bytes.data(), value.bytes.size()); *mask = value.node_mask)
DRIVER_API(cuDeviceGetProperties, (CUdevprop* output, CUdevice device), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null legacy device properties output"); valid_device(device); *output = legacy_device_properties(device))
DRIVER_API(cuDeviceComputeCapability, (int* major, int* minor, CUdevice device), driver_ready(); require(major && minor, CUDA_ERROR_INVALID_VALUE, "null compute capability output"); valid_device(device);
    auto a = global().provider->attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR); auto b = global().provider->attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR); *major = a; *minor = b)
DRIVER_API(cuDeviceGetPCIBusId, (char* output, int length, CUdevice device), driver_ready(); require(output && length > 0, CUDA_ERROR_INVALID_VALUE, "invalid PCI bus output"); valid_device(device);
    auto value = global().provider->device_pci(device); char text[32]{};
    auto bytes = std::snprintf(text, sizeof(text), "%04x:%02x:%02x.%x", value.domain, value.bus, value.device, value.function);
    require(bytes > 0 && size_t(bytes) < sizeof(text), CUDA_ERROR_UNKNOWN, "native PCI address exceeds its representation");
    std::memcpy(output, text, std::min<size_t>(length, size_t(bytes) + 1)); require(length > bytes, CUDA_ERROR_INVALID_VALUE, "PCI bus output is too small"))
DRIVER_API(cuDeviceGetByPCIBusId, (CUdevice* output, const char* text), driver_ready(); require(output, CUDA_ERROR_INVALID_VALUE, "null PCI device output"); *output = device_from_pci(text))
