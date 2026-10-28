extern "C" {
__device__ unsigned mutable_words[19] = {
    0x13579bdf, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0xeca86420};
__constant__ unsigned constant_words[7] = {0, 1, 0x80000000, 0xffffffff, 0x12345678, 0xabcdef01, 17};

__global__ void read_symbols(unsigned* out, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned slot = i % 17 + 1, constant_slot = i % 7;
    unsigned value = mutable_words[slot], constant = constant_words[constant_slot];
    unsigned long long address = reinterpret_cast<unsigned long long>(&mutable_words[slot]);
    unsigned long long constant_address = reinterpret_cast<unsigned long long>(&constant_words[constant_slot]);
    out += i * 7;
    out[0] = value; out[1] = constant; out[2] = (value ^ constant) + i;
    out[3] = unsigned(address); out[4] = unsigned(address >> 32);
    out[5] = unsigned(constant_address); out[6] = unsigned(constant_address >> 32);
}
__global__ void update_symbols(unsigned salt) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 17) mutable_words[i + 1] += salt ^ constant_words[i % 7];
}
}
