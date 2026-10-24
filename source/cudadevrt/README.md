This module owns `include/cudadevrt` and private CUDA C++ device helper sources.
The compiler embeds the `.cu` sources and `.cu.in` specialization templates,
compiles their device definitions into LLVM IR and lowers their service primitives.
These sources provide allocation, output, assertions, numeric helpers and frontend
wrappers. They do not build a separate host library or expose a host API.

Public device Runtime execution, including child launches, remains planned.
Pinned public headers stay unchanged under `include/cudadevrt`.
