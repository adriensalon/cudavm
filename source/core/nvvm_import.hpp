#pragma once
#include "nvvm_api.hpp"
#include <memory>
#include <string_view>
namespace llvm { class LLVMContext; class Module; }
namespace cudavm {
std::unique_ptr<llvm::Module> import_nvvm(llvm::LLVMContext&, std::string_view, const char*, const char*, bool nvidia_lto = false);
void compile_nvvm(const compiler_nvvm_request&, compiler_result&);
}
