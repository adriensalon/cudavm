#pragma once
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudavm {
// Group selection precedes individual strong/weak selection. Otherwise an alias
// can preserve a losing member, or a weak symbol can split a selected group.
struct comdat_selection {
    struct variant {
        std::string identity;
        llvm::Comdat::SelectionKind kind;
        uint64_t bytes = 0;
        std::string initializer;
        bool data_key = false;
        std::set<std::string> definitions;
        size_t members = 0;
    };
    struct choice {
        std::vector<variant> variants;
        size_t winner = 0;
        llvm::Comdat::SelectionKind kind = llvm::Comdat::Any;
        bool coherent = false;
    };
    std::map<std::string, choice> groups;
    static const llvm::Comdat* group(const llvm::GlobalValue& value) { return value.getComdat(); }
    void discover(const llvm::Module& module, const std::string& identity) {
        for (const auto& [name, comdat] : module.getComdatSymbolTable()) {
            variant value;
            value.identity = identity;
            value.kind = comdat.getSelectionKind();
            for (const auto& member : module.global_values()) {
                if (group(member) != &comdat || member.isDeclarationForLinker()) continue;
                if (!llvm::isa<llvm::GlobalAlias>(member)) ++value.members;
                if (!member.hasLocalLinkage()) value.definitions.insert(member.getName().str());
            }
            if (!value.members) continue;
            auto* key = module.getNamedValue(name);
            if (auto* alias = llvm::dyn_cast_or_null<llvm::GlobalAlias>(key)) key = alias->getAliaseeObject();
            if (auto* storage = llvm::dyn_cast_or_null<llvm::GlobalVariable>(key); storage && storage->hasInitializer()) {
                auto size = module.getDataLayout().getTypeAllocSize(storage->getValueType());
                if (!size.isScalable()) {
                    value.data_key = true;
                    value.bytes = size.getFixedValue();
                    llvm::raw_string_ostream stream(value.initializer);
                    storage->getInitializer()->printAsOperand(stream, true);
                }
            }
            groups[name.str()].variants.push_back(std::move(value));
        }
    }
    void select() {
        for (auto& [name, selected] : groups) {
            auto fail = [&](const char* reason) { throw std::runtime_error("device COMDAT " + name + ": " + reason); };
            selected.kind = selected.variants.front().kind;
            for (const auto& value : selected.variants) {
                selected.coherent |= value.members > 1 || value.kind != llvm::Comdat::Any;
                if (value.kind == selected.kind) continue;
                bool compatible = (value.kind == llvm::Comdat::Any || value.kind == llvm::Comdat::Largest)
                    && (selected.kind == llvm::Comdat::Any || selected.kind == llvm::Comdat::Largest);
                if (!compatible) fail("incompatible selection kinds");
                selected.kind = llvm::Comdat::Largest;
            }
            const auto& first = selected.variants.front();
            std::set<std::string> definitions;
            for (size_t i = 0; i < selected.variants.size(); ++i) {
                const auto& value = selected.variants[i];
                if (selected.kind == llvm::Comdat::NoDeduplicate) {
                    for (const auto& symbol : value.definitions)
                        if (!definitions.insert(symbol).second) fail("nodeduplicate has multiply defined members");
                } else if (selected.kind != llvm::Comdat::Any) {
                    if (!first.data_key || !value.data_key) fail("data-dependent selection requires a defined storage key");
                    if (selected.kind == llvm::Comdat::SameSize && first.bytes != value.bytes) fail("samesize key sizes differ");
                    if (selected.kind == llvm::Comdat::ExactMatch && first.initializer != value.initializer) fail("exactmatch key contents differ");
                    if (selected.kind == llvm::Comdat::Largest && value.bytes > selected.variants[selected.winner].bytes)
                        selected.winner = i;
                }
            }
        }
    }
    bool selected(const llvm::GlobalValue& value, const std::string& identity) const {
        auto* member_group = group(value);
        if (!member_group) return true;
        const auto& choice = groups.at(member_group->getName().str());
        return !choice.coherent || choice.kind == llvm::Comdat::NoDeduplicate
            || choice.variants[choice.winner].identity == identity;
    }
    bool coherent(const llvm::GlobalValue& value) const {
        auto* member_group = group(value);
        return member_group && groups.at(member_group->getName().str()).coherent;
    }
    void apply(llvm::Module& module, const std::string& identity) const {
        std::vector<llvm::GlobalValue*> kept, removed;
        for (auto& value : module.global_values())
            if (coherent(value)) (selected(value, identity) ? kept : removed).push_back(&value);
        std::vector<llvm::GlobalObject*> removed_objects;
        for (auto* value : removed)
            if (auto* object = llvm::dyn_cast<llvm::GlobalObject>(value)) removed_objects.push_back(object);
        for (auto* value : kept) if (auto* object = llvm::dyn_cast<llvm::GlobalObject>(value)) {
            auto* member_group = group(*value);
            if (value->getName() == member_group->getName() && groups.at(member_group->getName().str()).kind == llvm::Comdat::Largest)
                object->setMetadata("cudavm.largest.group.key", llvm::MDNode::get(module.getContext(), {}));
        }
        for (auto* object : removed_objects) {
            auto* member_group = group(*object);
            if (object->getName() == member_group->getName() && groups.at(member_group->getName().str()).kind == llvm::Comdat::Largest)
                object->setMetadata("cudavm.largest.group.key", llvm::MDNode::get(module.getContext(), {}));
        }
        // Aliases must be handled before changing their targets' COMDAT/body.
        for (auto* value : removed) if (auto* alias = llvm::dyn_cast<llvm::GlobalAlias>(value)) {
            llvm::GlobalValue* declaration;
            if (auto* type = llvm::dyn_cast<llvm::FunctionType>(alias->getValueType())) {
                auto* function = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage, alias->getAddressSpace(), "", &module);
                if (auto* target = llvm::dyn_cast_or_null<llvm::Function>(alias->getAliaseeObject())) function->copyAttributesFrom(target);
                declaration = function;
            } else {
                auto* target = llvm::dyn_cast_or_null<llvm::GlobalVariable>(alias->getAliaseeObject());
                declaration = new llvm::GlobalVariable(module, alias->getValueType(), target && target->isConstant(),
                    llvm::GlobalValue::ExternalLinkage, nullptr, "", nullptr, llvm::GlobalVariable::NotThreadLocal, alias->getAddressSpace());
            }
            declaration->takeName(alias);
            alias->replaceAllUsesWith(declaration);
            alias->eraseFromParent();
        }
        for (auto* object : removed_objects) {
            object->setComdat(nullptr);
            object->setMetadata("cudavm.discarded.group.definition", llvm::MDNode::get(module.getContext(), {}));
            if (auto* function = llvm::dyn_cast<llvm::Function>(object)) function->deleteBody();
            else {
                auto* variable = llvm::cast<llvm::GlobalVariable>(object);
                variable->setInitializer(nullptr);
            }
            object->setLinkage(llvm::GlobalValue::ExternalLinkage);
        }
        for (auto* value : kept) {
            if (auto* object = llvm::dyn_cast<llvm::GlobalObject>(value)) {
                object->setMetadata("cudavm.selected.group.definition", llvm::MDNode::get(module.getContext(), {}));
                object->setComdat(nullptr);
            }
            if (!value->hasLocalLinkage()) value->setLinkage(llvm::GlobalValue::ExternalLinkage);
        }
        // LLVM must not perform a second selection using stale group keys after
        // the complete selected input set has already supplied the decision.
        std::vector<std::string> erased;
        for (const auto& [name, group] : module.getComdatSymbolTable())
            if (auto found = groups.find(name.str()); found != groups.end() && found->second.coherent) erased.push_back(name.str());
        for (const auto& name : erased) module.getComdatSymbolTable().erase(name);
    }
};
}
