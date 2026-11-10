__device__ unsigned separate_bias = 7;
__device__ unsigned transform_word(unsigned value, unsigned index) { return value * 3 + index + separate_bias; }
__device__ unsigned duplicate_definition(unsigned value) { return value + 1; }
