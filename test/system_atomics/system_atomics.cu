#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
__device__ __forceinline__ unsigned system_load(const unsigned* pointer) {
    unsigned result;
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(result) : "l"(pointer) : "memory");
    return result;
}
__device__ __forceinline__ void system_store(unsigned* pointer, unsigned value) {
    asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
}
extern "C" __global__ void system_handoff(unsigned* mailbox, unsigned* tickets,
    unsigned rounds, unsigned seed, unsigned rmw) {
    volatile unsigned* payload = mailbox;
    for (unsigned round = 0; round < rounds; ++round) {
        while (system_load(mailbox) != round + 1) {
            if (system_load(mailbox + 2)) return;
        }
        for (unsigned word = 0; word < 4; ++word)
            payload[8 + word] = (payload[4 + word] ^ (seed + round * 0x10001u)) + word;
        unsigned previous = round;
        if (rmw) asm volatile("atom.add.acq_rel.sys.u32 %0, [%1], %2;"
            : "=r"(previous) : "l"(mailbox + 3), "r"(1u) : "memory");
        tickets[round] = previous;
        system_store(mailbox + 1, round + 1);
    }
}
