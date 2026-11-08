#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
#if !defined(__CUDACC_RTC__)
#include <assert.h>
#endif
extern "C" __global__ void assert_checked(unsigned* output, unsigned count, unsigned fail) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    if (fail && id == count - 1) {
#line 75 "device_assert.cu"
        assert(!"device_assert exact witness");
    }
    output[id] = id * 37 + count;
}
