extern "C" __device__ unsigned archive_guard = 0xabcdef01u;
extern "C" __device__ unsigned archive_missing(unsigned);
extern "C" __global__ void archive_unused_kernel(unsigned* output) {
    *output = archive_missing(11);
}
