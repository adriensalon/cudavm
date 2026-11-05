#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
static_assert(__CUDA_ARCH__ >= 900, "128-bit CAS profile requires SM90");
struct wide_value { unsigned long long low, high; };
__device__ __forceinline__ wide_value wide_cas(wide_value* address, wide_value compare, wide_value replacement) {
    wide_value previous;
    asm volatile("{ .reg .b128 a, b, c; mov.b128 a, {%3, %4}; mov.b128 b, {%5, %6}; "
        "atom.cas.b128 c, [%2], a, b; mov.b128 {%0, %1}, c; }"
        : "=l"(previous.low), "=l"(previous.high)
        : "l"(address), "l"(compare.low), "l"(compare.high), "l"(replacement.low), "l"(replacement.high) : "memory");
    return previous;
}
__device__ __forceinline__ wide_value ticket_value(unsigned ticket) {
    return {(static_cast<unsigned long long>(ticket) << 32) | ~ticket,
        (static_cast<unsigned long long>(ticket ^ 0x9e3779b9u) << 32) | (ticket * 17u)};
}
extern "C" __global__ void wide_serial(wide_value* output, wide_value* storage, const wide_value* operations, unsigned count) {
    for (unsigned operation = 0; operation < count; ++operation)
        output[operation] = wide_cas(storage, operations[operation * 2], operations[operation * 2 + 1]);
}
extern "C" __global__ void wide_tickets(wide_value* output, wide_value* storage, unsigned rounds) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x, total = gridDim.x * blockDim.x;
    for (unsigned round = 0; round < rounds; ++round) {
        auto previous = wide_cas(storage, {0, 0}, {0, 0});
        for (;;) {
            auto expected = previous;
            unsigned ticket = unsigned(expected.low >> 32);
            previous = wide_cas(storage, expected, ticket_value(ticket + 1));
            if (previous.low == expected.low && previous.high == expected.high) break;
        }
        output[round * total + id] = previous;
    }
}
