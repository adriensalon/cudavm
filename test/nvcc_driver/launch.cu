#include <cuda_runtime.h>
#include "parameters.hpp"
#include <array>
#include <cstdio>

// These names also exercise isolation of generated registration storage.
unsigned image = 17, program = 23;
__global__ void transform(unsigned* output, const unsigned* input, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i] * fixture_factor + i + fixture_bias;
}
int main(int argc, char**) {
    if (argc == 2) return image + program == 40 ? 0 : 1; // No GPU initialization.
    constexpr unsigned size = 80, offset = 3, count = 53, guard = 0xa95b371d;
    std::array<unsigned, size> input{}, result{}, read_input{};
    for (unsigned i = 0; i < size; ++i) input[i] = i * 17 + 1;
    result.fill(guard);
    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) std::fprintf(stderr, "Runtime: %s\n", cudaGetErrorName(status));
        return status == cudaSuccess;
    };
    int devices = 0; auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && !devices)) return 77;
    if (!check(status)) return 1;
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
        unsigned *source{}, *output{}; cudaStream_t stream{}; cudaEvent_t event{};
        if (!check(cudaMalloc(&source, sizeof(input))) || !check(cudaMalloc(&output, sizeof(result)))) return 1;
        result.fill(guard);
        if (!check(cudaMemcpy(source, input.data(), sizeof(input), cudaMemcpyHostToDevice))
            || !check(cudaMemcpy(output, result.data(), sizeof(result), cudaMemcpyHostToDevice))
            || !check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
            || !check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming))) return 1;
        transform<<<3, 32, 0, stream>>>(output + offset, source + offset, count);
        if (!check(cudaGetLastError()) || !check(cudaEventRecord(event, stream)) || !check(cudaEventSynchronize(event))) return 1;
        if (!check(cudaMemcpy(result.data(), output, sizeof(result), cudaMemcpyDeviceToHost))
            || !check(cudaMemcpy(read_input.data(), source, sizeof(read_input), cudaMemcpyDeviceToHost))) return 1;
        if (input != read_input) return 1;
        for (unsigned i = 0; i < size; ++i)
            if (result[i] != (i >= offset && i < offset + count ? input[i] * fixture_factor + i - offset + fixture_bias : guard)) return 1;
        if (!check(cudaEventDestroy(event)) || !check(cudaStreamDestroy(stream)) || !check(cudaFree(source))
            || !check(cudaFree(output)) || !check(cudaDeviceReset())) return 1;
    }
    std::puts("PASS: public nvcc guarded launch and reset"); return 0;
}
