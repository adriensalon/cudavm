struct __align__(16) parameter_record {
    unsigned tag;
    unsigned short small;
    unsigned char low, high;
    unsigned long long wide;
    unsigned values[3], tail;
};

extern "C" __global__ void parameter_layout(unsigned* out, unsigned char marker,
    parameter_record record, unsigned short bias, uint4 vector,
    unsigned long long salt, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out += i * 17;
    out[0] = marker;
    out[1] = unsigned(record.small) + bias;
    out[2] = record.low; out[3] = record.high;
    out[4] = record.tag ^ i;
    out[5] = unsigned(record.wide); out[6] = unsigned(record.wide >> 32);
    out[7] = record.values[0] + i;
    out[8] = record.values[1] ^ i;
    out[9] = record.values[2] - i;
    out[10] = record.tail;
    out[11] = vector.x; out[12] = vector.y;
    out[13] = vector.z; out[14] = vector.w;
    unsigned long long sum = record.wide + salt + i;
    out[15] = unsigned(sum); out[16] = unsigned(sum >> 32);
}
