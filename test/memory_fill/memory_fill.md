# Memory fill

This format-independent M01.fill / M01.padding witness checks Driver
`cuMemsetD8`, `cuMemsetD16`, `cuMemsetD32`, `cuMemsetD2D8`, `cuMemsetD2D16`,
`cuMemsetD2D32` and each corresponding `Async` entry point. The API runner selects
the first backend endpoint and creates an explicit context. It links only the
shared public Driver and needs no shader fixtures, input producer or compiler
components. Registration is `conformance.memory_fill.<backend>`.

Linear fills operate on aligned interior addresses of initialized allocations.
Counts 0, 1, 17, 40 and 41 cover empty, single-element, partial and exact capacity
regions. Zero, all-one, high-bit and mixed-byte patterns check each storage width.
An independent host byte oracle expands the low 8/16/32 bits in little-endian order
and preserves every other initialized byte, including both allocation guards.
Element-count units must agree across the typed APIs.

Pitched cases use five initialized rows with a 128-byte pitch. Complete rows,
interior subrectangles, one interior row, the last element of the last row and
zero-width/height rectangles check byte pitch versus element width, row bounds,
padding and untouched rows. The host oracle computes byte offsets from row and
column coordinates independently of the Driver fill implementation. Every byte
in the backing allocation and both outer guards is compared exactly.

Synchronous and asynchronous variants reuse that oracle. Async cases use an
explicit nonblocking stream and observe data after successful stream completion;
the stream is destroyed only after its work completes. Any allocation, transfer,
fill, completion, destruction or result error fails without device fallback.
The API contract is the
[CUDA Driver memory reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html).

Runtime APIs, default/PTDS stream variants, immediate return/completion behavior,
capture/graphs, pending resource retirement, invalid alignment/pitch/pointer and
overflow cases, large sizes, static SDK linkage and other platform/device profiles
remain open obligations. Storage readback is not credit for the memory-copy slice.
