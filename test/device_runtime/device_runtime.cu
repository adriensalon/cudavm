#include <cuda_device_runtime_api.h>
extern "C" __global__ void device_runtime_version(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output += i * 12;
    struct { unsigned before; int version; unsigned after; } local{input[i * 3], int(input[i * 3 + 2]), input[i * 3 + 1]};
    output[0] = cudaRuntimeGetVersion(&local.version);
    output[1] = unsigned(local.version); output[2] = local.before; output[3] = local.after;
    output[4] = cudaRuntimeGetVersion(nullptr);
    local.version = int(input[i * 3 + 2]);
    output[5] = cudaRuntimeGetVersion(&local.version);
    output[6] = unsigned(local.version); output[7] = input[i * 3 + 2];
    output[8] = input[i * 3]; output[9] = input[i * 3 + 2]; output[10] = input[i * 3 + 1];
    output[11] = cudaRuntimeGetVersion(reinterpret_cast<int*>(output + 9));
}
