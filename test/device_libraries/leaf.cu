#if ARCHIVE_CYCLE
extern "C" __device__ unsigned archive_tail(unsigned);
#endif
extern "C" __device__ unsigned archive_leaf(unsigned value) {
#if ARCHIVE_CYCLE
    return archive_tail(value ^ 0x193759bdu) + 11u;
#else
    return (value ^ 0x193759bdu) * 13u + 17u;
#endif
}
