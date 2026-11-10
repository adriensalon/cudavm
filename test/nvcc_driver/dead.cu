// This member must stay unselected. Flattening an archive causes both duplicate
// and unresolved device definitions, even though the host program never uses it.
extern __device__ unsigned missing_device_helper(unsigned);
__device__ unsigned duplicate_definition(unsigned value) { return value + 99; }
__global__ void unused_kernel(unsigned* output) { *output = missing_device_helper(100); }
