This module owns `include/nvvm` and the future exact NVVM API facade.
Implementation remains pending; no placeholder library or success exports are provided.

The versioned NVVM importer belongs to `source/cudavm` and its compiler
module. This facade will use the matching shared/static compiler boundary
without linking LLVM itself.
