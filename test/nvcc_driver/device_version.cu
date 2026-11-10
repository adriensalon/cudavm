#include <cuda_runtime.h>
#include <cstdio>
__global__ void runtime_version(unsigned* output) {
    int version = -1;
    output[0] = unsigned(cudaRuntimeGetVersion(&version));
    output[1] = unsigned(version);
    output[2] = unsigned(cudaRuntimeGetVersion(nullptr));
}
int main() {
    unsigned* device{}; unsigned result[3]{};
    if (cudaMalloc(&device, sizeof(result)) != cudaSuccess) return 1;
    runtime_version<<<1,1>>>(device);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess
        || cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost) != cudaSuccess
        || cudaFree(device) != cudaSuccess || cudaDeviceReset() != cudaSuccess) return 1;
    if (result[0] != cudaSuccess || result[1] != CUDART_VERSION || result[2] != cudaErrorInvalidValue) return 1;
    std::puts("PASS: installed device Runtime version helper and null-output contract"); return 0;
}
