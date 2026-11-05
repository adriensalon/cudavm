#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#include <cstdlib>
#endif
using heap_word = unsigned long long;
extern "C" __global__ void heap_allocate(unsigned* output, heap_word* pointers, unsigned count, unsigned seed) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned bytes = 17 + (id * 13 + seed) % 97;
    auto* pointer = static_cast<unsigned char*>(malloc(bytes));
    pointers[id] = reinterpret_cast<heap_word>(pointer);
    output[id * 3] = pointer != nullptr;
    output[id * 3 + 1] = unsigned(reinterpret_cast<heap_word>(pointer) & 15);
    output[id * 3 + 2] = bytes;
    if (pointer) for (unsigned i = 0; i < bytes; ++i) pointer[i] = static_cast<unsigned char>(seed + id * 17 + i * 7);
}
extern "C" __global__ void heap_consume(unsigned* output, heap_word* pointers, unsigned count, unsigned seed) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    unsigned owner = (id + 1) % count, bytes = 17 + (owner * 13 + seed) % 97;
    auto* pointer = reinterpret_cast<unsigned char*>(pointers[owner]);
    if (!pointer) { output[id * 3] = ~0u; return; }
    unsigned sum = 0;
    for (unsigned i = 0; i < bytes; ++i) sum += pointer[i];
    output[id * 3] = sum;
    output[id * 3 + 1] = pointer[0];
    output[id * 3 + 2] = pointer[bytes - 1];
    free(pointer);
    pointers[owner] = 0;
}
extern "C" __global__ void heap_reuse(unsigned* output, unsigned rounds, unsigned seed) {
    unsigned sum = 0, aligned = 1;
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned bytes = 19 + (round % 7) * 137;
        auto* pointer = static_cast<unsigned char*>(malloc(bytes));
        if (!pointer) { output[0] = ~0u; return; }
        aligned &= (reinterpret_cast<heap_word>(pointer) & 15) == 0;
        for (unsigned i = 0; i < bytes; ++i) pointer[i] = static_cast<unsigned char>(seed + round + i);
        for (unsigned i = 0; i < bytes; ++i) sum += pointer[i];
        free(pointer);
    }
    free(nullptr);
    auto* impossible = malloc(~heap_word(0));
    output[0] = sum;
    output[1] = aligned;
    output[2] = impossible == nullptr;
    if (impossible) free(impossible);
}
