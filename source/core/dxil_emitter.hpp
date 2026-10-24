#pragma once
#include "shader_lowering.hpp"
namespace cudavm {
std::unique_ptr<llvm::Module> lower_native_dxil(const llvm::Module&, const std::string& kernel,
    const std::array<unsigned,3>& block = {1,1,1});
std::string emit_native_dxil(const llvm::Module&);
std::string emit_native_dxil_payload(const llvm::Module&, llvm::json::Array* inventory = nullptr);
}
