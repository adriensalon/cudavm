#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
namespace llvm { class LLVMContext; class Module; }
namespace cudavm {
struct ptx_import_error : std::runtime_error {
    bool unsupported;
    ptx_import_error(bool missing, const std::string& diagnostic)
        : std::runtime_error(diagnostic), unsupported(missing) {}
};
// Imports an explicitly bounded PTX instruction profile. No input descriptors,
// source compiler or host registration metadata are required.
std::unique_ptr<llvm::Module> import_ptx(llvm::LLVMContext&, std::string_view);
}
