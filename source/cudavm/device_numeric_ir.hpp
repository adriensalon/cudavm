#pragma once
namespace llvm { class Module; }
namespace cudavm {
// Materialize the same embedded numeric source used by both CUDA frontends.
void link_device_numeric_ir(llvm::Module&);
}
