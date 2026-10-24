#pragma once
#include <clang/AST/Attr.h>
#include <clang/AST/GlobalDecl.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <stdexcept>
#include <vector>

namespace cudavm {
struct source_data_alias {
    std::string name, target;
    llvm::Type* type;
    unsigned address_space;
    llvm::GlobalValue::LinkageTypes linkage;
};
struct data_alias_consumer : clang::ASTConsumer, clang::RecursiveASTVisitor<data_alias_consumer> {
    clang::CodeGenerator& generator;
    std::vector<source_data_alias>& aliases;
    data_alias_consumer(clang::CodeGenerator& g, std::vector<source_data_alias>& a) : generator(g), aliases(a) {}
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool VisitFunctionDecl(clang::FunctionDecl* function) {
        auto* attr = function->getAttr<clang::AliasAttr>();
        if (!attr || !function->hasAttr<clang::CUDAGlobalAttr>() || function->getType()->isDependentType()) return true;
        // Clang can omit a CUDA global alias together with its device stub.
        // Recover the actual device entry signature from CodeGen, as for data.
        clang::GlobalDecl declaration(function, clang::KernelReferenceKind::Kernel);
        auto name = generator.GetMangledName(declaration).str();
        for (const auto& alias : aliases) if (alias.name == name) return true;
        auto* value = llvm::dyn_cast<llvm::GlobalValue>(generator.GetAddrOfGlobal(declaration, false)->stripPointerCasts());
        if (!value) throw std::runtime_error("CUDA kernel alias has no device type: " + name);
        auto linkage = function->hasExternalFormalLinkage() ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::InternalLinkage;
        if (function->hasAttr<clang::WeakAttr>()) linkage = llvm::GlobalValue::WeakAnyLinkage;
        aliases.push_back({name, attr->getAliasee().str(), value->getValueType(), value->getAddressSpace(), linkage});
        return true;
    }
    bool VisitVarDecl(clang::VarDecl* variable) {
        auto* attr = variable->getAttr<clang::AliasAttr>();
        if (!attr || !variable->isFileVarDecl() || variable->getType()->isDependentType()
            || (!variable->hasAttr<clang::CUDADeviceAttr>() && !variable->hasAttr<clang::CUDAConstantAttr>())) return true;
        auto name = generator.GetMangledName(clang::GlobalDecl(variable)).str();
        for (const auto& alias : aliases) if (alias.name == name) return true;
        auto* value = llvm::dyn_cast<llvm::GlobalValue>(generator.GetAddrOfGlobal(clang::GlobalDecl(variable), false)->stripPointerCasts());
        if (!value) throw std::runtime_error("CUDA data alias has no device type: " + name);
        auto linkage = variable->hasExternalFormalLinkage() ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::InternalLinkage;
        if (variable->hasAttr<clang::WeakAttr>()) linkage = llvm::GlobalValue::WeakAnyLinkage;
        aliases.push_back({name, attr->getAliasee().str(), value->getValueType(),
            variable->hasAttr<clang::CUDAConstantAttr>() ? 4u : 1u, linkage});
        return true;
    }
    void HandleTranslationUnit(clang::ASTContext& context) override {
        if (!context.getDiagnostics().hasErrorOccurred()) TraverseDecl(context.getTranslationUnitDecl());
    }
};
inline void materialize_data_aliases(llvm::Module& module, const std::vector<source_data_alias>& aliases) {
    // Clang emits variable aliases in generic AS0, skips constant aliases and
    // can omit CUDA kernel aliases. The device AST supplies their actual ABI.
    for (const auto& alias : aliases) {
        auto* target = module.getNamedValue(alias.target);
        auto* old = module.getNamedValue(alias.name);
        if (!target || target == old) throw std::runtime_error("CUDA data alias requires a defined target: " + alias.name);
        if (old && !llvm::isa<llvm::GlobalAlias>(old) && !old->isDeclarationForLinker())
            throw std::runtime_error("CUDA data alias conflicts with storage: " + alias.name);
        auto* pointer = llvm::PointerType::get(module.getContext(), alias.address_space);
        auto linkage = old && llvm::isa<llvm::GlobalAlias>(old) ? old->getLinkage() : alias.linkage;
        auto* replacement = llvm::GlobalAlias::create(alias.type, alias.address_space, linkage, "",
            llvm::ConstantExpr::getPointerCast(target, pointer), &module);
        if (old) {
            if (auto* previous = llvm::dyn_cast<llvm::GlobalAlias>(old)) replacement->copyAttributesFrom(previous);
            else {
                replacement->setVisibility(old->getVisibility());
                replacement->setDLLStorageClass(old->getDLLStorageClass());
                replacement->setDSOLocal(old->isDSOLocal());
            }
            replacement->takeName(old);
            old->replaceAllUsesWith(llvm::ConstantExpr::getPointerCast(replacement, old->getType()));
            old->eraseFromParent();
        } else replacement->setName(alias.name);
    }
}
}
