// NVRTC's device Runtime instance follows the pinned SDK version. The public
// device header supplies its CDP2 forwarding wrapper and declared error type.
extern "C" __device__ __attribute__((used, always_inline)) inline cudaError_t CUDARTAPI __cudavm_rtc_runtime_version(int* version) {
    if (!version) return cudaErrorInvalidValue;
    *version = CUDART_VERSION;
    return cudaSuccess;
}
