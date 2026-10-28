# Device allocation

This format-independent M01.allocation / M01.validation witness checks Driver
`cuMemAlloc`, `cuMemFree` and live `cuMemGetAddressRange` queries in an explicit
context. The shared public Driver is its only cudavm link dependency. It needs
an available device/backend and does not initialize a shader compiler or require
an executable input format. CUDA/PTX/cubin/NVVM inputs and producer provenance
are inapplicable to these API-only cases. Registration is
`conformance.device_allocation.<backend>`.

Simultaneously live allocations use sizes 1, 3, 7, 16, 31, 64, 255, 256, 257 and
4093 bytes. Their requested ranges are disjoint and their base addresses meet a
bounded 16-byte scalar/vector alignment profile. Every byte is initialized from
an independent host pattern. Ordered one-byte updates at the beginning, middle
and end preserve all neighboring bytes and other live allocations. Readback has
host guards, and uploads preserve their host input. Retiring alternate owners,
allocating replacements and repeating generations check surviving storage and
legal free/reallocation lifetimes without requiring a particular reused address.

Base/interior range queries must identify the same allocation and extent,
including every combination of optional output pointers. Reported extents may
include native allocation rounding but must contain the requested bytes. A
zero-size allocation must return `CUDA_ERROR_INVALID_VALUE`; its failure output
and uninitialized allocation contents are not inspected. The contract is the
[CUDA Driver memory API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html).

The runner selects the first enumerated endpoint of the requested backend,
creates its context and executes the cases once. Missing devices return the
suite's unavailable status. Any context, allocation, transfer, query, free or
result failure fails the case without device fallback. CTest supplies a process
deadline. Transfers serve as storage witnesses and do not close their own slice.

Runtime allocation/Driver interoperation, kernel dereference and its input paths,
full alignment/size and exhaustion boundaries, memory accounting, deferred frees,
cross-context ownership, pending-use synchronization, additional error overloads,
static SDK consumption and other platform/device profiles remain open obligations.
