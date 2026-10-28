# Warp synchronization

This D03.barriers / D01.divergence witness checks `__syncwarp(mask)`,
PTX `bar.warp.sync` and NVVM `llvm.nvvm.bar.warp.sync` in the SM75 profile.
Only nonempty, explicitly named live participants execute a barrier. Masks are
intersected with the actual partial warp before use; an empty mask makes every
thread return. CUDA source, independently authored PTX and NVVM IR, and a genuine
NVIDIA SM75 cubin reuse the same host oracle through both Driver memory loaders.
The original assembler command and input hashes are in `provenance.json`;
producer tools are maintenance dependencies only.

Each participant writes a per-thread shared word and a guarded global scratch
word, waits, and reads the next named participant's words. A second barrier
protects those reads before the next generation overwrites storage. Every round
is checked separately against integer expectations computed from the preserved
host input, including the final global scratch contents. Zero rounds, one round
and repeated rounds check untouched records and reuse. Full, alternating,
sparse, lane-31-only and empty masks execute with single-thread, partial-warp,
multiple-warp and multidimensional block shapes in several blocks. The exact
shapes are `(1,1,1)`, `(13,1,1)`, `(32,1,1)`, `(45,1,1)`, `(8,3,3)` and
`(8,4,3)`, within the suite's common emitted launch profile.

A disjoint-group mode splits each mask by even/odd lanes and gives each group
and neighboring warp different iteration counts. Groups share physical storage
but access only their own initialized elements. A divergent mode publishes
through unequal instruction paths: odd lanes perform a volatile update loop,
even lanes compute the same delta directly, and the paths meet through distinct
warp barrier sites. Both paths subsequently read their named peer and rendezvous
again before reuse. The explicit PTX/NVVM barriers retain this distinction even
if a source compiler can merge equivalent calls. Allocation guards, inactive
records, unused output phases and input words are checked exactly.

Import, compilation, loading, launch and result errors fail. Availability comes
from the harness's independent execution probe, and each CTest process has a
deadline. The publication and participation rules follow the
[PTX warp barrier specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync).

Progressive retirement, divergent device-call contexts, all admitted block/resource
limits, older architecture restrictions, Runtime/AOT/RDC/static linkage,
file/fresh-process loaders, volatile global/generic/local accesses, shared volatile
widths other than scalar 32-bit, volatile strong-system global instruction encodings,
and other memory-space or system-scope publication
remain open obligations. Passing this bounded execution profile does not close
the family inventory.
