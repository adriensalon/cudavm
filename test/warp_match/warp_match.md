# Warp matching

This bounded D03.collectives witness checks `__match_any_sync` and
`__match_all_sync` for `int`, `unsigned`, `long long`, `unsigned long long`,
`float` and `double`, including both returned mask and the written all-match
predicate. PTX checks `match.any/all.sync.b32/b64`; NVVM uses the corresponding
32/64-bit intrinsics with explicit pair extraction. A genuine NVIDIA SM75 cubin
is assembled from the independent PTX, with its command and hashes recorded in
`provenance.json`. Producer tools are maintenance dependencies only.

The host oracle compares integer encodings and enumerates participating lanes.
It checks uniform values, unique values, equivalence classes, matching low words
with different high words, low-word distinctions, signed zeros, subnormals,
infinities, and quiet/signaling NaN payloads. Floating operands compare their
bits rather than numerical equality. Full, alternating, sparse, lane-31-only
and empty masks check every returned bit. Predicate storage starts nonboolean
and must become exactly zero or one. A second kernel repeats the operations
while lanes progressively exit; exited lanes remain named in the mask but
are excluded from the expected result.

Single-thread, partial and multiple warps execute in several blocks, with guards
around both allocations, preserved source bytes and untouched nonparticipant
and retired records. All embedded inputs and both Driver memory loaders reuse
the same oracle. The harness's independent execution probe establishes backend
availability; any slice import, compilation, load, launch or result error fails.
Applicability is the SM75 execution profile, with only legal named participants.
The instruction contract is the
[PTX match specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync).

`long`/`unsigned long` host-ABI variants, disjoint/nested divergent match sites,
predicate pointers in global/shared storage, AOT/RDC/static linkage, Runtime
launches, file/fresh-process loading and other architecture profiles remain
open obligations. This execution witness does not close the family inventory.
