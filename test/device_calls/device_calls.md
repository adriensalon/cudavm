# Device calls

This bounded D01.calls witness checks direct device calls with 32/64-bit unsigned
arguments and return values, nested calls, repeated calls from a loop and
divergent call sites. A 64-bit helper rotates and adds its arguments; a 32-bit
helper returns through either branch of an integer fold; a third helper retains
its caller state across calls to both. The kernel observes every result and
calls the fold again from different branches with different argument orders.

CUDA C++ uses explicitly non-inlined device functions. Independent PTX spells
out mixed-width parameter/return slots and direct calls; independent NVVM IR
uses non-inlined typed functions. The genuine NVIDIA SM75 cubin is assembled
from PTX, with its producer command and exact input hashes in `provenance.json`.
The nested helper's machine calls target earlier leaf definitions, exercising
negative relative displacements as well as forward kernel calls. The binary
must import and execute, including its machine call/return ABI.
Producer tools are fixture-maintenance dependencies only.

The host oracle uses unsigned modular arithmetic and the standard host rotate
operation. Zero/all-one operands, sign/word boundaries and deterministic raw-word
samples exercise low/high returns, carries, shift zero and every rotation count.
Different loop counts per invocation require independent call frames and preserve
values across repeated nested calls. Single-thread, partial and multiple warps
execute in several blocks. Zero, one, partial, full and beyond-launch element
counts check allocation guards and untouched inactive rows, while source words
and guards remain unchanged.

Every embedded input uses the same exact oracle through both Driver memory
loaders. The applicability rule is the harness's executable SM75 profile with
bounded, acyclic direct call graphs and defined unsigned operations. Import,
compilation, loading, launch and result errors fail; unavailable hardware is
identified only by the independent execution probe.

Indirect calls/function pointers, recursion, aggregate/vector/narrow/float
argument and result ABIs, explicit stack limits, exceptions, divergent calls
containing collectives, cross-translation-unit linking, Runtime/AOT/RDC/static
linkage, file/fresh-process loading and other architectures remain open
obligations. This fixture does not establish complete device-call conformance.
