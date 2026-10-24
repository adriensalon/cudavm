#pragma once
#include "compiler.hpp"
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>
#include <stdexcept>

namespace cudavm {
// Adopt helper-only LLVM device libraries without depending on compiler-private
// cudavm metadata. The selected source TU supplies the exact target/source ABI;
// no host registration is invented for an input with no original host object.
inline std::string import_external_library(llvm::Module& module, llvm::MemoryBufferRef input,
        const llvm::Module& prototype, const compiler_options& settings) {
    auto fail = [&](const std::string& reason) {
        throw std::runtime_error("incompatible external LLVM device library " + input.getBufferIdentifier().str() + ": " + reason);
    };
    for (const auto& metadata : module.named_metadata())
        if (metadata.getName().starts_with("cudavm.")) fail("partial or unrecognized cudavm metadata");
    auto* abi = prototype.getNamedMetadata("cudavm.source.abi");
    if (!abi || abi->getNumOperands() != 1 || abi->getOperand(0)->getNumOperands() != 6)
        fail("a raw cudavm source ABI is required");
    if (module.getTargetTriple() != prototype.getTargetTriple()
            || module.getTargetTriple().getArch() != llvm::Triple::nvptx64)
        fail("target triple must match the nvptx64 source ABI");
    if (module.getDataLayoutStr().empty() || module.getDataLayoutStr() != prototype.getDataLayoutStr())
        fail("device data layout must match the source ABI");
    if (!module.getModuleInlineAsm().empty() || !module.ifunc_empty())
        fail("module assembly and indirect resolvers require separate lowering");
    for (const auto& function : module) {
        if (function.getCallingConv() == llvm::CallingConv::PTX_Kernel || function.hasFnAttribute("cudavm.kernel"))
            fail("only helper/data libraries are supported; kernel entries need host ownership");
        if (function.hasAvailableExternallyLinkage()) fail("available_externally bodies are not persistent library definitions");
        auto architecture = function.getFnAttribute("target-cpu");
        if (architecture.isValid() && architecture.getValueAsString() != settings.architecture)
            fail("helper architecture must match --arch");
    }
    for (const auto& variable : module.globals()) {
        if (variable.isThreadLocal()) fail("thread-local storage is unsupported");
        if ((variable.getName() == "llvm.global_ctors" || variable.getName() == "llvm.global_dtors")
                && variable.hasInitializer() && !variable.getInitializer()->isNullValue())
            fail("device constructors/destructors need runtime initialization");
        if (variable.hasAvailableExternallyLinkage()) fail("available_externally storage is not a persistent definition");
    }
    if (auto* annotations = module.getNamedMetadata("nvvm.annotations"))
        for (auto* row : annotations->operands())
            for (unsigned i = 1; i < row->getNumOperands(); i += 2)
                if (auto* name = llvm::dyn_cast_or_null<llvm::MDString>(row->getOperand(i)); name && name->getString() == "kernel")
                    fail("kernel annotations require source/host ownership");

    llvm::SHA256 digest;
    digest.update(input.getBuffer());
    std::string identity;
    constexpr char hex[] = "0123456789abcdef";
    for (auto byte : digest.final()) { identity += hex[byte >> 4]; identity += hex[byte & 15]; }
    auto& context = module.getContext();
    auto text = [&](llvm::StringRef value) { return llvm::MDString::get(context, value); };
    // Reuse the public source-ABI tuple, rather than assuming that an external
    // library was compiled with the CLI's default C++ standard or host profile.
    llvm::SmallVector<llvm::Metadata*> fields;
    for (const auto& field : abi->getOperand(0)->operands()) {
        auto* value = llvm::dyn_cast_or_null<llvm::MDString>(field);
        if (!value) fail("invalid source ABI tuple");
        fields.push_back(text(value->getString()));
    }
    module.getOrInsertNamedMetadata("cudavm.source.abi")->addOperand(llvm::MDNode::get(context, fields));
    const auto origin = "external-llvm:" + identity;
    module.getOrInsertNamedMetadata("cudavm.translation.units")->addOperand(
        llvm::MDNode::get(context, {text(identity), text(origin)}));
    module.getOrInsertNamedMetadata("cudavm.external.library.v1")->addOperand(
        llvm::MDNode::get(context, {text(identity), text("llvm-bitcode-helper-data-v1")}));
    module.getOrInsertNamedMetadata("cudavm.ir.stage")->addOperand(
        llvm::MDNode::get(context, text("clang.cuda.tu")));
    module.setModuleIdentifier(origin);
    module.setSourceFileName(origin);
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    llvm::WriteBitcodeToFile(module, stream);
    return bytes;
}
}
