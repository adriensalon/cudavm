# Array storage

This format-independent slice owns array allocation, channel layouts,
descriptors, storage, memory requirements and retirement under M01.arrays,
M01.allocation and M01.validation. It uses the public Driver and Runtime APIs
without shader fixtures, a compiler or a producer toolchain.

Cases are named `conformance.array_storage.<backend>.<case>`. The slice follows
the suite's shared-library policy and links the public Driver and Runtime.
Select the slice with
`ctest -R '^conformance\.array_storage\.' --output-on-failure`; append a case
name to narrow the run.

| Case | Observable contract |
| --- | --- |
| `numeric` | Signed/unsigned integer, half and float arrays with applicable channel counts; zero/unit/odd extents; ordinary, layered, cube, surface and gather shapes; independent live allocations and replacement storage. |
| `formats` | Every normalized, packed and block-compressed format in the pinned Driver enum; exact component descriptions, fixed channel counts, compressed blocks and byte preservation. |
| `planes` | Planar and semiplanar formats, subsampled plane extents, shared aliases, independent plane contents and layer strides, and retirement of derived handles. |
| `deferred` | Metadata-only allocation, per-format size/alignment queries and whole-mipmap requirements, stable repeated queries, Driver/Runtime agreement and rejection of queries on ordinary allocations. |
| `flags` | Depth, attachment and video allocation intents; descriptor preservation and storage access; sparse admission and layout queries where the endpoint advertises that facility; incompatible mapping flags. |
| `limits` | Reported texture/surface/gather allocation limits, minimum legal shapes, rejection beyond each relevant axis, alternate 3D limits and checked size overflow. |
| `validation` | Null creation/query outputs and descriptors, invalid handles, widths, formats, channel counts, shapes, flags and Runtime channel layouts. |
| `lifetime` | Mixed-facade queued transfers, parent/plane retention, stale handles, context ownership and teardown without disturbing another context's live storage. |

The format tables enumerate the pinned `CUarray_format` entries rather than
inferring support from the ordinary scalar formats. They distinguish logical
component bits from transfer elements: compressed formats use complete storage
blocks, while packed formats copy packed elements. Byte-copy X offsets obey
element alignment; uncompressed byte widths can end within the final element.
Compressed byte widths cover complete blocks. Runtime channel descriptors
use the corresponding public kinds; legacy Driver allocation spellings remain
Driver cases and are queried and transferred through the Runtime too. Public
header bytes and the requirement inventories are not modified.

Driver `cuArrayCreate`/`cuArray3DCreate` and Runtime
`cudaMallocArray`/`cudaMalloc3DArray` are exercised where their signatures apply.
`cuArrayGetDescriptor`, `cuArray3DGetDescriptor`, `cudaArrayGetInfo` and
`cudaGetChannelDesc` agree before and after writes. Runtime info checks every
combination of optional outputs. Failed-call output values and fresh allocation
contents are not inspected; allocation handles need no particular reuse policy.

Full-volume Driver and Runtime transfers use independently constructed upload
and download descriptors and host-generated coordinate/byte patterns. The oracle
checks payload, different host row/slice strides, padding, outer guards and
upload-source preservation. Floating, packed and compressed contents are raw
storage; decoding, conversion and sampling have their own owners. Plane views
share parent backing while keeping plane offsets and layer strides distinct.

`cuArrayGetMemoryRequirements` and `cudaArrayGetMemoryRequirements` require
deferred allocation. Queries on ordinary arrays return invalid-value status.
The deferred case checks returned size against the independently computed
payload minimum, nonzero power-of-two alignment, size/alignment consistency and
stable Driver/Runtime results. The corresponding mipmapped queries check the
whole allocation, without prescribing one backend's tiling to another.
Deferred objects have no payload backing until mapping. This case retires them
unmapped; mapping and accesses after mapping belong to the virtual-memory and
sparse-array contracts.

Sparse admission follows the endpoint's existing capability. A declared absence
checks the defined rejection and is printed explicitly; it is not evidence of
implemented sparse storage. Advertised support must create the array and return
consistent Driver/Runtime sparse layout properties. Sparse tile mapping,
residency and mip-tail execution remain owned by `sparse_arrays`. Graphics and
video flag preservation does not establish interoperability with an external
graphics or video participant.

Pinned host buffers and a nonblocking stream exercise accepted uploads/readbacks
before retirement, including a plane whose parent is destroyed before stream
synchronization. Results are observed after the defined completion point.
Context teardown retires its allocations and views; foreign-context operations
must fail without changing surviving storage.

Only backend/device absence permits the runner's unavailable result. Errors in
an applicable allocation, query, transfer, destruction or oracle fail that case
under its CTest deadline. The suite uses the
[pinned Driver memory reference](https://docs.nvidia.com/cuda/archive/13.4.1/cuda-driver-api/group__CUDA__MEM.html),
[Runtime memory reference](https://docs.nvidia.com/cuda/archive/13.4.1/cuda-runtime-api/group__CUDART__MEMORY.html)
and the pinned public channel descriptors.

Mip-level data transfers, texture/surface access, sampling/decoding, external
sharing and sparse residency have separate catalog owners. Full-volume storage
witnesses do not replace partial-copy slices. A passing run supplies evidence
for its configured backend and selected endpoint; it does
not complete the broader M01 family or certify every platform/device profile.
