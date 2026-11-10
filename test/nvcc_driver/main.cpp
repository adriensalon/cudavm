#include <cuda_runtime.h>
#include <array>
#include <cstdio>
extern "C" cudaError_t launch_separate(unsigned*, const unsigned*, unsigned, cudaStream_t);
int main() {
    constexpr unsigned capacity = 80, offset = 3, count = 53, guard = 0xa95b371d;
    std::array<unsigned, capacity> input{}, output{}, read_input{};
    for (unsigned i = 0; i < capacity; ++i) input[i] = i * 17 + 1;
    auto check = [](cudaError_t status) { if (status != cudaSuccess) std::fprintf(stderr, "Runtime: %s\n", cudaGetErrorName(status)); return status == cudaSuccess; };
    int devices = 0; auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && !devices)) return 77;
    if (!check(status)) return 1;
    for (unsigned round = 0; round < 2; ++round) {
        unsigned *a{}, *b{}; cudaStream_t stream{}; output.fill(guard);
        if (!check(cudaMalloc(&a, sizeof(input))) || !check(cudaMalloc(&b, sizeof(output)))
            || !check(cudaMemcpy(a, input.data(), sizeof(input), cudaMemcpyHostToDevice))
            || !check(cudaMemcpy(b, output.data(), sizeof(output), cudaMemcpyHostToDevice))
            || !check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))
            || !check(launch_separate(b + offset, a + offset, count, stream))
            || !check(cudaStreamSynchronize(stream))
            || !check(cudaMemcpy(output.data(), b, sizeof(output), cudaMemcpyDeviceToHost))
            || !check(cudaMemcpy(read_input.data(), a, sizeof(read_input), cudaMemcpyDeviceToHost))) return 1;
        if (input != read_input) return 1;
        for (unsigned i = 0; i < capacity; ++i)
            if (output[i] != (i >= offset && i < offset + count ? input[i] * 3 + i - offset + 7 : guard)) return 1;
        if (!check(cudaStreamDestroy(stream)) || !check(cudaFree(a)) || !check(cudaFree(b)) || !check(cudaDeviceReset())) return 1;
    }
    std::puts("PASS: ordinary archive selection, device linking and fresh-process execution"); return 0;
}
