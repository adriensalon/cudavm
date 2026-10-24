__device__ inline int __reduce_add_sync(unsigned mask,int value) {
    return int(__cudavm_rtc_reduce_add(unsigned(value),mask));
}
