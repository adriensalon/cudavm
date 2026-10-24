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
    ptx_import_error(bool missing, const std::string& diagnostic, bool version = false)
        : std::runtime_error(diagnostic), unsupported(missing), unsupported_version(version) {}
};
// Imports an explicitly bounded PTX instruction profile. No input descriptors,
// source compiler or host registration metadata are required.
std::unique_ptr<llvm::Module> import_ptx(llvm::LLVMContext&, std::string_view, std::string_view architecture = {});
// Checked, scalar-result numeric assembly from the pinned CUDA headers.
llvm::Value* import_numeric_inline_ptx(llvm::CallInst&);
}
