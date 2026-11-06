extern "C" __device__ unsigned archive_start(unsigned);
extern "C" __device__ unsigned archive_guard = 0x13579bdfu;
extern "C" __global__ void archive_results(unsigned* output, const unsigned* input, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned value = input[i];
    unsigned* row = output + i * 4;
    row[0] = archive_start(value);
    row[1] = archive_start(value ^ i);
    row[2] = archive_guard;
    row[3] = value ^ (i * 0x193759bdu);
}
