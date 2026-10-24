This module owns `include/nvptxcompiler` and the future exact NVIDIA PTX compiler
API facade. No public placeholder success exports are provided.

The private `cudavm_core` service is owned and built by `source/core`.
It imports PTX for the Driver's shader adapters and has no dependency on this
public API module.
