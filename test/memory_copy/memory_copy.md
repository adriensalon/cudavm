# Linear memory copies

This format-independent M01.copy / M01.padding witness checks Driver
`cuMemcpyHtoD`, `cuMemcpyDtoD`, `cuMemcpyDtoH` and their `Async` variants in an
explicit context. It links only the shared Driver, uses the first backend
endpoint and registers as `conformance.memory_copy.<backend>`. Shader input
formats, module loaders and input producer provenance do not apply to these
API-only transfer cases.

Distinct source/destination patterns initialize a 137-byte payload with sixteen
guard bytes at each end. Ten source-offset/destination-offset/count combinations
cover empty and complete copies, single bytes, odd offsets, multiword interiors,
different source/destination positions and exact final-byte boundaries. Source
and destination allocations never overlap. The host oracle copies the requested
byte interval into an independent expected vector. All destination bytes,
untouched neighbors and allocation guards must match exactly, while both host
and device source contents remain unchanged. The unused host or device destination
must also retain its initialized contents.

Async cases use Driver-allocated pinned host buffers, keep them live until
explicit nonblocking-stream completion, and compare the same oracle as the
synchronous calls. The stream and pinned buffers are released with checked API
results after all work completes. Allocation and setup copies are prerequisites;
they do not close allocation or host-pinning obligations. Any transfer,
completion, release or result error fails without endpoint fallback, within the
CTest process deadline. Contracts follow the
[CUDA Driver memory reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html).

Runtime copies, generic/UVA direction inference, pageable async completion rules,
host-to-host and peer paths, pending resource retirement, invalid pointers/counts,
overflow and overlap validation, large sizes, graph/capture and default/PTDS
stream modes, kernel producer/consumer input paths, static SDK linkage and other
platform/device profiles remain open obligations. The cases establish the listed
bounded directional-copy profile rather than complete memory-copy conformance.
