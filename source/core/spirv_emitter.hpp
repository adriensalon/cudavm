#pragma once
#include "compiler.hpp"
namespace cudavm {
// A Vulkan shader uses the existing packet/allocation ABI and has no native
// pointer arguments. The shader's CFG and values remain ordinary LLVM IR.
std::unique_ptr<llvm::Module> lower_native_spirv(const llvm::Module&, const std::string& kernel, bool opencl = false);
std::string emit_native_spirv(const llvm::Module&);
std::string emit_native_spirv_payload(const llvm::Module&, llvm::json::Array* inventory = nullptr);
}
