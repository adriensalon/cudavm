This module owns `include/nvptxcompiler` and the future exact NVIDIA PTX compiler
API and assembly implementation. No public placeholder success exports are provided.

Compiler builds also produce the private `cudavm_ptx_jit` service for importing a
bounded scalar PTX profile into the shared LLVM IR and emitting native shaders.
The Driver discovers this service only when loading descriptor-free PTX on a
shader adapter. Its versioned internal C entry point does not produce NVIDIA
cubins or implement the public nvPTXCompiler API.
