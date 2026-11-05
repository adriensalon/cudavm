#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
extern "C" __device__ void __assertfail(const char*, const char*, unsigned, const char*, __SIZE_TYPE__);
extern "C" __global__ void assert_checked(unsigned* output, unsigned count, unsigned fail) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    if (fail && id == count - 1)
        __assertfail("device_assert exact witness", "device_assert.cu", 75, "assert_checked", 1);
    output[id] = id * 37 + count;
}
