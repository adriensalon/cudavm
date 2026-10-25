#pragma once
#include <memory>
#include <string_view>
namespace llvm { class LLVMContext; class Module; }
namespace cudavm {
// Bounded ELF64 / SM75 machine-code import. The source image contains no PTX.
std::unique_ptr<llvm::Module> import_cubin(llvm::LLVMContext&, std::string_view, std::string_view architecture = {});
}
