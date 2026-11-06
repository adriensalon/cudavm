extern "C" __device__ unsigned archive_leaf(unsigned);
extern "C" __device__ unsigned archive_start(unsigned value) {
    return archive_leaf(value * 3u + 5u) + 7u;
}
