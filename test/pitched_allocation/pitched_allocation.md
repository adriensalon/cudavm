# Pitched allocation

This format-independent M01.allocation / M01.padding witness checks Driver
`cuMemAllocPitch` with element-size hints 4, 8 and 16, byte widths 1, 3, 7, 16,
257 and 4093, and one, two or five payload rows plus two guard rows. It registers
as `conformance.pitched_allocation.<backend>` through the shared Driver API
runner. It needs an available backend endpoint and no shader input or compiler.

The returned pitch must contain the requested byte width and preserve alignment
for its element hint; the base meets the bounded 16-byte scalar/vector profile.
The test uses the returned pitch rather than requiring a particular padding
amount. The base allocation range must contain every requested pitched row, and
interior queries at the final requested byte of each row retain the same range
identity. All products are checked for overflow before allocating host buffers.

Every backing byte, including row padding and both outer guard rows, starts
with a known value. Independent host patterns populate payload rows through
public byte copies. A second phase changes interior columns of selected rows;
neighboring columns, untouched rows and padding retain their earlier contents.
Readback compares the entire backing allocation and both host guards exactly.
Upload sources remain unchanged and every allocation is freed with a checked
result after completed transfers. The API contract is the
[CUDA Driver memory reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html).

Allocation, query, transfer, free and result errors fail without endpoint
fallback. Copy operations are storage witnesses and do not close copy coverage.
Runtime pitched/3D allocation overloads, invalid dimensions/hints and overflow
validation, device limits and exhaustion, texture/copy-engine alignment limits,
kernel input paths, pending-use lifetime, static SDK linkage and other
platform/device profiles remain open obligations.
