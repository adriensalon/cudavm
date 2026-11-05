using object_word = unsigned long long;
struct alignas(16) device_record {
    object_word value;
    unsigned tag, salt;
    unsigned* audit;
    __device__ __forceinline__ device_record(unsigned seed, unsigned* journal)
        : value((object_word(seed) << 32) | (seed ^ 0x89abcdefu)), tag(seed + 17u), salt(seed ^ 0x13579bdfu), audit(journal) {
        *audit = (*audit * 33u) ^ seed;
    }
    __device__ __forceinline__ device_record(const device_record& other)
        : value(other.value ^ 0x100000001ull), tag(other.tag + 3u), salt(other.salt ^ 0xffffffffu), audit(other.audit) {
        *audit = (*audit * 33u) ^ 0xc001cafeu;
    }
    __device__ __forceinline__ void bump(unsigned delta) {
        value += (object_word(delta) << 32) | delta;
        tag ^= delta; salt += delta * 7u;
    }
    __device__ __forceinline__ ~device_record() {
        *audit = (*audit * 33u) ^ (unsigned(value) ^ unsigned(value >> 32) ^ tag ^ salt);
    }
};
extern "C" __global__ void object_results(unsigned* out, const unsigned* in, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned seed = in[i]; out += i * 15; out[0] = 0;
    {
        device_record original(seed, out);
        {
            device_record copy(original);
            original.bump(seed + 5u); copy.bump(seed ^ 0x10203040u);
            out[1] = unsigned(original.value); out[2] = unsigned(original.value >> 32);
            out[3] = original.tag; out[4] = original.salt;
            out[5] = unsigned(copy.value); out[6] = unsigned(copy.value >> 32);
            out[7] = copy.tag; out[8] = copy.salt;
        }
    }
    out[9] = sizeof(device_record); out[10] = alignof(device_record);
    out[11] = __builtin_offsetof(device_record, value); out[12] = __builtin_offsetof(device_record, tag);
    out[13] = __builtin_offsetof(device_record, salt); out[14] = __builtin_offsetof(device_record, audit);
}
