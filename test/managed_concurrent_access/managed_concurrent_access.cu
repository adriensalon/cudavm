#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
__device__ __forceinline__ unsigned managed_acquire(const unsigned* pointer) {
    unsigned value;
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(pointer) : "memory");
    return value;
}
__device__ __forceinline__ void managed_release(unsigned* pointer, unsigned value) {
    asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
}
extern "C" __global__ void managed_participant(unsigned* control, const unsigned* source,
    unsigned* destination, unsigned words, unsigned rounds, unsigned seed, unsigned handoff) {
    managed_release(control + 3, 1);
    for (unsigned round = 0; round < rounds; ++round) {
        while (managed_acquire(control) != round + 1) {
            if (managed_acquire(control + 2)) return;
        }
        for (unsigned word = 0; word < words; ++word) {
            const unsigned value = handoff ? source[word] : seed + word * 37u + round * 0x10001u;
            destination[word] = (value ^ (seed + round * 0x9e3779b9u)) + word * 17u;
        }
        managed_release(control + 1, round + 1);
    }
}
