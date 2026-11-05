#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#include <cstdio>
#endif
extern "C" __global__ void printf_records(unsigned* output, const unsigned* input, unsigned count) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    char local[4] = {'o', 'k', char('A' + id % 26), 0};
    float value = 1.25f;
    output[id] = printf("record %u|%d|%08x|%lld|%.2f|%s|%%\n", id,
        short(-int(id) - 7), input[id], -1234567890123ll, value, local);
    local[0] = 'x';
}
extern "C" __global__ void printf_formats(unsigned* output, const char* format, unsigned value) {
    char local[] = "captured";
    output[0] = printf(format, value, local);
    local[0] = 'x';
    output[1] = printf("promoted %d|%u|%c|%.2f|%ld|%lu\n", short(-19),
        static_cast<unsigned char>(123), 'Q', 3.5f, long(-23), static_cast<unsigned long>(1234));
    output[2] = printf("literal %%\n");
    output[3] = printf(nullptr);
}
extern "C" __global__ void printf_arguments(unsigned* output) {
    output[0] = printf("arguments %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
        0u,1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,
        16u,17u,18u,19u,20u,21u,22u,23u,24u,25u,26u,27u,28u,29u,30u,31u);
}
