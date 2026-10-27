#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
namespace llvm { class LLVMContext; class Module; class CallInst; class Value; }
namespace cudavm {
struct ptx_import_error : std::runtime_error {
    bool unsupported;
    bool unsupported_version;
    bool architecture_mismatch;
    ptx_import_error(bool missing, const std::string& diagnostic, bool version = false, bool architecture = false)
        : std::runtime_error(diagnostic), unsupported(missing), unsupported_version(version), architecture_mismatch(architecture) {}
};
// Imports an explicitly bounded PTX instruction profile. No input descriptors,
// source compiler or host registration metadata are required.
std::unique_ptr<llvm::Module> import_ptx(llvm::LLVMContext&, std::string_view, std::string_view architecture = {}, bool relocatable = false);
// Register-constrained assembly blocks share the external import semantics.
// Native CUDA retains the original assembly; these operate on shader clones.
llvm::Value* import_inline_ptx(llvm::CallInst&);
void lower_inline_ptx(llvm::Module&);
}
