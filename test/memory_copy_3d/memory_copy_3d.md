# Three-dimensional memory copies

This format-independent M01.copy / M01.padding witness checks Driver `cuMemcpy3D`
and `cuMemcpy3DAsync` across every host, pitched-device and 3D-array source and
destination pairing. It registers as `conformance.memory_copy_3d.<backend>` with
the shared Driver API runner and needs no compiler, shader input or producer.

The logical volume has eleven four-byte elements per row, five rows per slice
and four slices. Arrays have two unsigned 16-bit channels per element. Host
storage is pinned with a 60-byte pitch and seven storage rows per slice; device
storage uses the pitch returned by `cuMemAllocPitch` with nine storage rows per
slice. Arrays have compact logical slice geometry. Consequently, row pitch,
storage height and slice stride are independent quantities. Host storage also
has sixteen outer guard bytes at each end.

Distinct XYZ-dependent patterns initialize source and destination payloads,
with every padding byte and unused storage row initialized separately. Seven
regions cover the complete volume, its final element, an interior box, differing
source/destination XYZ positions and zero width/height/depth. X coordinates and
widths remain byte units and respect the four-byte array element size. The host
oracle assigns bytes using separate row and slice coordinates; every destination
byte, untouched slice/row, padding byte and host guard is checked, and complete
source contents remain unchanged. Setup/readback uses full-volume coordinates.

Sync/async cases share the same oracle. Async calls use an explicit nonblocking
stream, preserve all resources until successful completion and release them with
checked results. Importer/module dimensions are inapplicable. API, allocation,
completion, destruction or result errors fail without endpoint fallback, under
the CTest deadline. Contracts follow the
[CUDA Driver memory reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html).

Runtime element-based extents, peer descriptors, generic/unified endpoints,
other formats/channels, mip/layered/cubemap arrays, invalid alignment/pitch/height,
overflow and overlap validation, maximum volumes, pending retirement,
default/PTDS stream modes, capture/graphs, kernel input paths, static SDK linkage
and other platform/device profiles remain open obligations. Array/allocation
setup does not close those slices.
