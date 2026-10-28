# Two-dimensional memory copies

This format-independent M01.copy / M01.padding witness checks Driver `cuMemcpy2D`
and `cuMemcpy2DAsync` across every host, pitched-device and array source/destination
pair. The shared Driver API runner supplies the first backend endpoint and an
explicit context. Registration is `conformance.memory_copy_2d.<backend>`; shader
formats, module loaders and input producer provenance are inapplicable.

Host buffers are pinned, have a 140-byte row pitch and include sixteen outer
guard bytes at each end. Device endpoints use the actual pitch returned by
`cuMemAllocPitch`, ensuring the synchronous copy-engine pitch precondition.
Arrays have 29 by 7 elements, each two unsigned 16-bit channels (four bytes),
so byte X coordinates must remain distinct from element indices. All endpoints
have 116 logical row bytes and seven rows, with distinct initialized source and
destination patterns. Every byte of linear padding is initialized as well.

Six rectangles cover complete resources, one element, an interior subrectangle,
the bottom-right boundary and zero width/height. All array coordinates and widths
are multiples of the four-byte element size. The independent host oracle assigns
bytes from source row/column coordinates to destination row/column coordinates.
Complete destination storage, untouched rows, linear row padding and host guards
are checked; the complete source remains unchanged. Setup/readback operations use
full-resource coordinates and do not replace the partial-region result oracle.

Sync and async cases reuse the same oracle. Async operations use an explicit
nonblocking stream and retain pinned buffers, arrays and device allocations until
successful stream completion. Resources and stream are released with checked
results. Any allocation, transfer, completion, release or result error fails
without backend endpoint fallback, under the CTest process deadline. Contracts
follow the [CUDA Driver memory reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html).

Runtime and unaligned-copy APIs, generic/unified and peer routes, other array
formats, channel counts, mip/layered resources, invalid alignment/pitch/extents
and overlap, maximum sizes, pending-use retirement, default/PTDS stream modes,
capture/graphs, kernel producer/consumer input paths, static SDK linkage and
other platform/device profiles remain open obligations. Allocation and array
creation are prerequisites and do not close their respective slices.
