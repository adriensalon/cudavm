#include "compiler.hpp"
#include "program.hpp"
#include "program_image.hpp"
#include "../nvfatbin/container_api.hpp"
#include "device_conversions.hpp"
#include "device_integers.hpp"
#include "device_simd.hpp"
#include "device_dpx.hpp"
#include "device_math.hpp"
#include "device_services.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Object/Archive.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/SCCP.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include "host_link.hpp"
#include "external_library.hpp"
#include "comdat_selection.hpp"
#include "address_initializer.hpp"
#include <llvm/IR/ValueHandle.h>
namespace cudavm {
namespace {
constexpr auto stage_key = "cudavm.ir.stage";
void check_ir(const llvm::Module& module) {
    std::string error;
    llvm::raw_string_ostream stream(error);
    if (llvm::verifyModule(module, &stream)) throw std::runtime_error("IR contract failed:\n" + error);
}
std::string stage(const llvm::Module& module) {
    auto* data = module.getNamedMetadata(stage_key);
    if (!data || data->getNumOperands() != 1 || data->getOperand(0)->getNumOperands() != 1)
        throw std::runtime_error("missing or ambiguous cudavm IR stage");
    auto* value = llvm::dyn_cast_or_null<llvm::MDString>(data->getOperand(0)->getOperand(0));
    if (!value) throw std::runtime_error("invalid cudavm IR stage metadata");
    return value->getString().str();
}
void set_stage(llvm::Module& module, const char* name) {
    auto* data = module.getOrInsertNamedMetadata(stage_key);
    data->clearOperands();
    data->addOperand(llvm::MDNode::get(module.getContext(), llvm::MDString::get(module.getContext(), name)));
    check_ir(module);
}
void advance(llvm::Module& module, const char* from, const char* to) {
    if (stage(module) != from) throw std::runtime_error(std::string("IR transition requires ") + from);
    set_stage(module, to);
}
void append(llvm::Module& module, const char* key, llvm::ArrayRef<llvm::Metadata*> fields) {
    module.getOrInsertNamedMetadata(key)->addOperand(llvm::MDNode::get(module.getContext(), fields));
}
llvm::MDString* string(llvm::Module& module, llvm::StringRef value) { return llvm::MDString::get(module.getContext(), value); }
void trace(llvm::Module& module, const char* pass) { append(module, "cudavm.pipeline", {string(module, pass)}); }
void optimize_closed(llvm::Module& module, bool specialize) {
    llvm::LoopAnalysisManager loops;
    llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager calls;
    llvm::ModuleAnalysisManager modules;
    llvm::PassBuilder builder;
    builder.registerModuleAnalyses(modules); builder.registerCGSCCAnalyses(calls);
    builder.registerFunctionAnalyses(functions); builder.registerLoopAnalyses(loops);
    builder.crossRegisterProxies(loops, functions, calls, modules);
    llvm::ModulePassManager passes;
    if (specialize) passes.addPass(llvm::IPSCCPPass());
    passes.addPass(llvm::GlobalDCEPass());
    passes.run(module, modules);
}
void close_graph(llvm::Module& module) {
    if (auto* output = module.getFunction("vprintf")) {
        // CUDA defines the null-format return value. Host stdio declarations
        // can otherwise attach nonnull to Clang's lowered device call.
        output->removeParamAttr(0, llvm::Attribute::NonNull);
        for (auto* user : output->users()) if (auto* call = llvm::dyn_cast<llvm::CallBase>(user))
            call->removeParamAttr(0, llvm::Attribute::NonNull);
    }
    validate_device_services(module);
    for (auto& function : module) {
        if (function.isDeclaration() || function.hasFnAttribute("cudavm.kernel")) continue;
        // Device linking is complete. Only kernels are public roots; LLVM's
        // reachability includes direct calls, function values and static data.
        function.setLinkage(llvm::GlobalValue::InternalLinkage);
        function.setComdat(nullptr);
    }
    optimize_closed(module, false);
    for (auto& function : module) {
        if (!function.isDeclaration() || function.use_empty() || function.isIntrinsic()
            || function.hasFnAttribute("cudavm.cuda.origin") || device_service(function.getName().str())) continue;
        throw std::runtime_error("unresolved device symbol: " + function.getName().str());
    }
    for (auto& function : module)
        if (!function.isDeclaration() && function.getName().starts_with("__cudaCDP2"))
            throw std::runtime_error("CUDA device runtime operation requires device-runtime lowering: " + function.getName().str());
    for (auto& variable : module.globals())
        if (variable.isDeclaration() && !variable.use_empty()
            && (variable.getAddressSpace() == 1 || variable.getAddressSpace() == 4))
            throw std::runtime_error("unresolved device global: " + variable.getName().str());
}
bool same_storage_type(llvm::Type* a, llvm::Type* b) {
    if (a == b) return true;
    if (a->getTypeID() != b->getTypeID()) return false;
    if (auto* array = llvm::dyn_cast<llvm::ArrayType>(a)) {
        auto* other = llvm::cast<llvm::ArrayType>(b);
        return array->getNumElements() == other->getNumElements() && same_storage_type(array->getElementType(), other->getElementType());
    }
    if (auto* record = llvm::dyn_cast<llvm::StructType>(a)) {
        auto* other = llvm::cast<llvm::StructType>(b);
        if (record->isOpaque() || other->isOpaque() || record->isPacked() != other->isPacked()
            || record->getNumElements() != other->getNumElements()) return false;
        for (unsigned i = 0; i < record->getNumElements(); ++i)
            if (!same_storage_type(record->getElementType(i), other->getElementType(i))) return false;
        return true;
    }
    return false;
}
const llvm::Function* alias_target(const llvm::GlobalAlias& alias, bool final_link) {
    auto fail = [&](const char* reason) -> const llvm::Function* {
        throw std::runtime_error("unsupported device function alias " + alias.getName().str() + ": " + reason);
    };
    std::set<const llvm::GlobalAlias*> visited;
    const llvm::Constant* value = &alias;
    while (auto* current = llvm::dyn_cast<llvm::GlobalAlias>(value)) {
        if (!visited.insert(current).second) return fail("cyclic aliases");
        if (!current->hasLocalLinkage() && !current->hasExternalLinkage()
            && !(final_link && (current->hasWeakLinkage() || current->hasLinkOnceLinkage())))
            return fail("weak or replaceable aliases require a separate link contract");
        if (current != &alias && current->isInterposable())
            return fail("intermediate weak aliases require a separate link contract");
        if (!current->getAliasee()) return fail("missing target");
        value = current->getAliasee()->stripPointerCasts();
    }
    auto* function = llvm::dyn_cast<llvm::Function>(value);
    if (!function || function->isDeclaration()) return fail("target must be a defined device function");
    if (!function->hasLocalLinkage() && !function->hasExternalLinkage()
        && !(final_link && (function->hasWeakLinkage() || function->hasLinkOnceLinkage())))
        return fail("weak or replaceable targets require a separate link contract");
    if (alias.getAddressSpace() != function->getAddressSpace() || alias.getValueType() != function->getFunctionType())
        return fail("incompatible helper signature or address space");
    return function;
}
struct data_view {
    const llvm::GlobalVariable* target;
    uint64_t offset, bytes, alignment;
};
data_view data_alias_target(const llvm::GlobalAlias& alias, bool final_link = false) {
    auto fail = [&](const char* reason) -> data_view {
        throw std::runtime_error("unsupported device data alias " + alias.getName().str() + ": " + reason);
    };
    const auto& dl = alias.getParent()->getDataLayout();
    std::set<const llvm::GlobalAlias*> visited;
    const llvm::Constant* value = &alias;
    llvm::APInt offset(64, 0);
    while (true) {
        if (auto* current = llvm::dyn_cast<llvm::GlobalAlias>(value)) {
            if (!visited.insert(current).second) return fail("cyclic aliases");
            if (!current->hasLocalLinkage() && !current->hasExternalLinkage()
                && !(final_link && (current->hasWeakLinkage() || current->hasLinkOnceLinkage())))
                return fail("weak or replaceable aliases require a separate storage contract");
            if (current != &alias && current->isInterposable())
                return fail("intermediate weak aliases require a separate storage contract");
            if (current->getAddressSpace() != alias.getAddressSpace() || !current->getAliasee())
                return fail("missing target or incompatible address space");
            value = current->getAliasee();
        } else if (auto* gep = llvm::dyn_cast<llvm::GEPOperator>(value)) {
            llvm::APInt addend(64, 0);
            if (!gep->accumulateConstantOffset(dl, addend)) return fail("nonconstant storage offset");
            bool overflow = false;
            offset = offset.sadd_ov(addend, overflow);
            if (overflow) return fail("storage offset overflow");
            value = llvm::cast<llvm::Constant>(gep->getPointerOperand());
        } else if (auto* cast = llvm::dyn_cast<llvm::ConstantExpr>(value);
                   cast && (cast->getOpcode() == llvm::Instruction::BitCast
                         || cast->getOpcode() == llvm::Instruction::AddrSpaceCast)) {
            value = llvm::cast<llvm::Constant>(cast->getOperand(0));
        } else break;
    }
    auto* target = llvm::dyn_cast<llvm::GlobalVariable>(value);
    if (!target || !target->hasInitializer()) return fail("target must be defined module storage");
    if (target->getAddressSpace() != alias.getAddressSpace()
        || (alias.getAddressSpace() != 1 && alias.getAddressSpace() != 4))
        return fail("incompatible global/constant address space");
    if ((!target->hasLocalLinkage() && !target->hasExternalLinkage()
            && !(final_link && (target->hasWeakLinkage() || target->hasLinkOnceLinkage()))) || target->isThreadLocal())
        return fail("weak or replaceable targets require a separate storage contract");
    if (!alias.getValueType()->isSized() || !target->getValueType()->isSized())
        return fail("unsized storage type");
    auto bytes = dl.getTypeAllocSize(alias.getValueType());
    auto capacity = dl.getTypeAllocSize(target->getValueType());
    if (bytes.isScalable() || capacity.isScalable() || offset.isNegative() || !bytes.getFixedValue()
        || offset.getZExtValue() > capacity.getFixedValue()
        || bytes.getFixedValue() > capacity.getFixedValue() - offset.getZExtValue())
        return fail("storage view is outside its target");
    auto alignment = llvm::commonAlignment(target->getAlign().value_or(dl.getABITypeAlign(target->getValueType())), offset.getZExtValue());
    if (alignment < dl.getABITypeAlign(alias.getValueType())) return fail("misaligned storage view");
    return {target, offset.getZExtValue(), bytes.getFixedValue(), alignment.value()};
}
bool same_alias_attributes(llvm::AttributeList a, llvm::AttributeList b, unsigned parameters) {
    // Opaque pointers do not encode by-value pointee layout or ABI attributes.
    auto same_type = [](llvm::Type* left, llvm::Type* right) {
        return left && right ? same_storage_type(left, right) : left == right;
    };
    for (auto kind : {llvm::Attribute::SExt, llvm::Attribute::ZExt, llvm::Attribute::InReg})
        if (a.hasRetAttr(kind) != b.hasRetAttr(kind)) return false;
    for (unsigned i = 0; i < parameters; ++i) {
        for (auto kind : {llvm::Attribute::SExt, llvm::Attribute::ZExt, llvm::Attribute::InReg,
                llvm::Attribute::ByVal, llvm::Attribute::ByRef, llvm::Attribute::StructRet,
                llvm::Attribute::InAlloca, llvm::Attribute::Preallocated, llvm::Attribute::Nest,
                llvm::Attribute::SwiftSelf, llvm::Attribute::SwiftError})
            if (a.hasParamAttr(i, kind) != b.hasParamAttr(i, kind)) return false;
        if (!same_type(a.getParamByValType(i), b.getParamByValType(i)) || !same_type(a.getParamByRefType(i), b.getParamByRefType(i))
            || !same_type(a.getParamStructRetType(i), b.getParamStructRetType(i))
            || !same_type(a.getParamPreallocatedType(i), b.getParamPreallocatedType(i))
            || !same_type(a.getParamInAllocaType(i), b.getParamInAllocaType(i))) return false;
        if ((a.hasParamAttr(i, llvm::Attribute::ByVal) || a.hasParamAttr(i, llvm::Attribute::ByRef)
                || a.hasParamAttr(i, llvm::Attribute::Preallocated) || a.hasParamAttr(i, llvm::Attribute::InAlloca))
            && a.getParamAlignment(i) != b.getParamAlignment(i)) return false;
    }
    return true;
}
bool same_alias_abi(const llvm::Function& a, const llvm::Function& b) {
    // Clang represents an unimplemented __global__ prototype with C convention
    // and assigns PTX_Kernel only to its definition. It is still a launch ABI.
    auto kernel = [](const llvm::Function& f) {
        return f.hasFnAttribute("cudavm.kernel") || f.getCallingConv() == llvm::CallingConv::PTX_Kernel;
    };
    bool convention = a.getCallingConv() == b.getCallingConv()
        || (a.isDeclaration() && a.getCallingConv() == llvm::CallingConv::C && kernel(b))
        || (b.isDeclaration() && b.getCallingConv() == llvm::CallingConv::C && kernel(a));
    return a.getAddressSpace() == b.getAddressSpace()
        && a.getFunctionType() == b.getFunctionType() && convention
        && same_alias_attributes(a.getAttributes(), b.getAttributes(), a.arg_size());
}
void freeze_weak_alias_targets(llvm::Module& module) {
    std::map<const llvm::Function*, llvm::Function*> copies;
    std::vector<std::pair<llvm::GlobalAlias*, const llvm::Function*>> aliases;
    for (auto& alias : module.aliases()) {
        if (!alias.getValueType()->isFunctionTy()) continue;
        auto* target = alias_target(alias, true);
        if (target->getLinkage() == llvm::GlobalValue::WeakAnyLinkage
            || target->getLinkage() == llvm::GlobalValue::LinkOnceAnyLinkage)
            aliases.emplace_back(&alias, target);
    }
    // LLVM can replace an incoming weak aliasee with an already selected strong
    // definition. An alias denotes its TU's original body, so preserve that body
    // before either link arrival order can redirect it. Ordinary references to
    // the weak name still participate in normal strong/weak resolution.
    for (auto [alias, target] : aliases) {
        auto [found, inserted] = copies.emplace(target, nullptr);
        if (inserted) {
            llvm::ValueToValueMapTy map;
            auto* copy = llvm::CloneFunction(const_cast<llvm::Function*>(target), map);
            copy->setName(target->getName() + ".cudavm.alias.target");
            copy->setLinkage(llvm::GlobalValue::InternalLinkage);
            copy->setComdat(nullptr);
            copy->setVisibility(llvm::GlobalValue::DefaultVisibility);
            copy->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
            found->second = copy;
        }
        alias->setAliasee(found->second);
    }
    check_ir(module);
}
void freeze_weak_data_alias_targets(llvm::Module& module, const std::set<std::string>& replaced, llvm::StringRef identity) {
    std::map<const llvm::GlobalVariable*, llvm::GlobalVariable*> copies;
    std::vector<std::pair<llvm::GlobalAlias*, data_view>> aliases;
    for (auto& alias : module.aliases()) {
        if (alias.getValueType()->isFunctionTy()) continue;
        auto view = data_alias_target(alias, true);
        if ((view.target->getLinkage() == llvm::GlobalValue::WeakAnyLinkage
                || view.target->getLinkage() == llvm::GlobalValue::LinkOnceAnyLinkage)
            && replaced.contains(view.target->getName().str()))
            aliases.emplace_back(&alias, view);
    }
    // An alias binds its TU's original storage, while ordinary uses of the weak
    // name can bind a selected override. Preserve one backing object per target
    // before LLVM can replace it, including all interior views and alias chains.
    for (auto [alias, view] : aliases) {
        auto [found, inserted] = copies.emplace(view.target, nullptr);
        if (inserted) {
            auto* target = const_cast<llvm::GlobalVariable*>(view.target);
            auto* copy = new llvm::GlobalVariable(module, target->getValueType(), target->isConstant(),
                llvm::GlobalValue::InternalLinkage, target->getInitializer(),
                target->getName() + "__cudavm_alias_storage__" + identity, nullptr, target->getThreadLocalMode(),
                target->getAddressSpace());
            copy->copyAttributesFrom(target);
            copy->setLinkage(llvm::GlobalValue::InternalLinkage);
            copy->setComdat(nullptr);
            copy->setVisibility(llvm::GlobalValue::DefaultVisibility);
            copy->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
            // Its address is observable through an exported view.
            copy->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::None);
            found->second = copy;
        }
        llvm::Constant* pointer = found->second;
        if (view.offset)
            pointer = llvm::ConstantExpr::getGetElementPtr(llvm::Type::getInt8Ty(module.getContext()), pointer,
                llvm::ConstantInt::get(llvm::Type::getInt64Ty(module.getContext()), view.offset));
        alias->setAliasee(pointer);
    }
    check_ir(module);
}
void normalize_odr_alias_comdats(llvm::Module& module, const std::set<std::string>& replaced) {
    // LLVM places an alias in its target's COMDAT. Discarding an equivalent
    // inline target can otherwise discard an unrelated strong alias name,
    // leaving host/device declarations unresolved. Singleton `any` groups can
    // use ordinary ODR symbol coalescing without changing group semantics.
    std::set<llvm::GlobalObject*> targets;
    for (auto& alias : module.aliases()) {
        auto* target = alias.getValueType()->isFunctionTy()
            ? static_cast<const llvm::GlobalObject*>(alias_target(alias, true))
            : static_cast<const llvm::GlobalObject*>(data_alias_target(alias, true).target);
        if (target->hasLinkOnceODRLinkage() || target->hasWeakODRLinkage())
            targets.insert(const_cast<llvm::GlobalObject*>(target));
        if (!alias.getValueType()->isFunctionTy() && (target->hasLinkOnceODRLinkage() || target->hasWeakODRLinkage())
                && !replaced.contains(alias.getName().str())) {
            const auto view = data_alias_target(alias, true);
            append(module, "cudavm.odr.data.bindings.v1", {string(module, alias.getName()), string(module, target->getName()),
                string(module, std::to_string(view.offset))});
        }
    }
    for (auto* target : targets) {
        auto* group = target->getComdat();
        if (!group) continue;
        if (group->getSelectionKind() != llvm::Comdat::Any)
            throw std::runtime_error("ODR alias requires a singleton any COMDAT: " + target->getName().str());
        for (auto& value : module.global_objects())
            if (&value != target && value.getComdat() == group)
                throw std::runtime_error("ODR alias COMDAT includes other definitions: " + target->getName().str());
        target->setComdat(nullptr);
    }
}
void rebind_odr_data_aliases(llvm::Module& module) {
    // IRMover can preserve a private copy for an alias of a discarded ODR
    // definition. ODR storage is one program object: aliases of equivalent
    // definitions must follow the coalesced object rather than split its address.
    auto* bindings = module.getNamedMetadata("cudavm.odr.data.bindings.v1");
    if (!bindings) return;
    for (auto* row : bindings->operands()) {
        if (row->getNumOperands() != 3) throw std::runtime_error("invalid ODR alias binding metadata");
        auto* name = llvm::dyn_cast_or_null<llvm::MDString>(row->getOperand(0));
        auto* target_name = llvm::dyn_cast_or_null<llvm::MDString>(row->getOperand(1));
        auto* offset_text = llvm::dyn_cast_or_null<llvm::MDString>(row->getOperand(2));
        if (!name || !target_name || !offset_text) throw std::runtime_error("invalid ODR alias binding metadata");
        uint64_t offset = 0;
        auto* alias = module.getNamedAlias(name->getString());
        auto* target = module.getGlobalVariable(target_name->getString(), true);
        if (!alias || !target || !target->hasInitializer() || offset_text->getString().getAsInteger(10, offset))
            throw std::runtime_error("ODR alias lost its selected storage");
        llvm::Constant* pointer = target;
        if (offset) pointer = llvm::ConstantExpr::getGetElementPtr(llvm::Type::getInt8Ty(module.getContext()), pointer,
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(module.getContext()), offset));
        alias->setAliasee(pointer);
        (void)data_alias_target(*alias, true);
    }
}
struct link_diagnostics {
    llvm::LLVMContext& context;
    std::unique_ptr<llvm::DiagnosticHandler> previous;
    link_diagnostics(llvm::LLVMContext& c, std::string& output) : context(c), previous(c.getDiagnosticHandler()) {
        context.setDiagnosticHandler(std::make_unique<llvm::DiagnosticHandler>());
        context.setDiagnosticHandlerCallBack([](const llvm::DiagnosticInfo* info, void* data) {
            llvm::raw_string_ostream stream(*static_cast<std::string*>(data));
            llvm::DiagnosticPrinterRawOStream printer(stream);
            info->print(printer);
            stream << '\n';
        }, &output);
    }
    ~link_diagnostics() { context.setDiagnosticHandler(std::move(previous)); }
};
constexpr auto archive_key = "cudavm.link.archive.v1";
std::string archive_digest(llvm::StringRef bytes) {
    llvm::SHA256 hash;
    hash.update(bytes);
    std::string result;
    constexpr char alphabet[] = "0123456789abcdef";
    for (auto byte : hash.final()) { result += alphabet[byte >> 4]; result += alphabet[byte & 15]; }
    return result;
}
llvm::StringRef metadata_string(const llvm::MDNode& row, unsigned index) {
    auto* value = index < row.getNumOperands() ? llvm::dyn_cast_or_null<llvm::MDString>(row.getOperand(index)) : nullptr;
    if (!value) throw std::runtime_error("invalid device link archive metadata");
    return value->getString();
}
std::string unit_identity(const llvm::Module& module) {
    auto* units = module.getNamedMetadata("cudavm.translation.units");
    if (!units || units->getNumOperands() != 1 || units->getOperand(0)->getNumOperands() != 2)
        throw std::runtime_error("raw device link input requires one translation-unit identity");
    auto identity = metadata_string(*units->getOperand(0), 0);
    if (identity.size() != 64 || identity.find_first_not_of("0123456789abcdef") != llvm::StringRef::npos)
        throw std::runtime_error("invalid device translation-unit identity");
    return identity.str();
}
struct archived_unit { std::string bytes, digest, origin; bool raw_seen; };
struct library_unit { std::string identity; archived_unit input; std::set<std::string> definitions; };
std::map<std::string, archived_unit> collect_link_inputs(const compiler_options& values, std::vector<library_unit>& libraries) {
    std::map<std::string, archived_unit> units;
    std::vector<std::pair<std::string, std::string>> candidates;
    auto add = [&](const std::string& identity, llvm::StringRef bytes, const std::string& origin, bool composed) {
        auto digest = archive_digest(bytes);
        auto found = units.find(identity);
        if (found != units.end()) {
            if (found->second.digest != digest || found->second.bytes != bytes)
                throw std::runtime_error("translation-unit identity reused for different contents: " + identity + " in " + origin);
            if (!composed && found->second.raw_seen)
                throw std::runtime_error("multiply defined translation unit input: " + identity);
            found->second.raw_seen |= !composed;
        } else units.emplace(identity, archived_unit{bytes.str(), std::move(digest), origin, !composed});
    };
    auto image_bitcode = [&](llvm::MemoryBufferRef buffer) -> std::string {
        auto bytes = buffer.getBuffer();
        auto view = std::string_view(bytes.data(), bytes.size());
        if (!has_program_image(view) && !has_fatbin(view)) return bytes.str();
        compiler_input_request_v3 request;
        request.base.base = {3, has_fatbin(view) ? compiler_input_kind::fatbinary : compiler_input_kind::program_image, bytes.data(), bytes.size(), nullptr, nullptr};
        request.base.architecture = values.architecture.c_str(); request.relocatable = 1;
        compiler_result diagnostics;
        auto program = import_compiler_program(request.base.base, diagnostics);
        if (stage(*program->module) == "cudavm.portable" && !program->module->getNamedMetadata(archive_key)) {
            compiler_options settings;
            settings.input = program->module->getModuleIdentifier(); settings.bitcode = program_input_identity(*program->module);
            settings.standard = "c++17"; settings.architecture = program->architecture;
            mark_translation_unit(*program->module, settings, {});
        }
        std::string result;
        llvm::raw_string_ostream stream(result);
        llvm::WriteBitcodeToFile(*program->module, stream);
        return result;
    };
    auto candidate = [&](llvm::MemoryBufferRef buffer, const std::string& origin) {
        candidates.emplace_back(image_bitcode(buffer), origin);
    };
    auto parse_candidate = [&](llvm::MemoryBufferRef buffer, const std::string& origin) {
        llvm::LLVMContext discovery;
        auto parsed = llvm::parseBitcodeFile(buffer, discovery);
        if (!parsed) throw std::runtime_error("device library member requires compile-tu bitcode " + origin + ": " + llvm::toString(parsed.takeError()));
        auto& module = **parsed;
        check_ir(module);
        std::string imported;
        if (!module.getNamedMetadata(stage_key)) {
            if (units.empty()) throw std::runtime_error("external LLVM device libraries require an explicit cudavm translation unit: " + origin);
            llvm::LLVMContext prototype_context;
            const auto& input = units.begin()->second;
            auto prototype = llvm::parseBitcodeFile(llvm::MemoryBufferRef(input.bytes, input.origin), prototype_context);
            if (!prototype) throw std::runtime_error("invalid source ABI prototype: " + llvm::toString(prototype.takeError()));
            imported = import_external_library(module, buffer, **prototype, values);
            check_ir(module);
            buffer = llvm::MemoryBufferRef(imported, origin);
        }
        if (stage(module) != "clang.cuda.tu" || module.getNamedMetadata(archive_key))
            throw std::runtime_error("device library member requires a raw compile-tu artifact: " + origin);
        library_unit value{unit_identity(module), {buffer.getBuffer().str(), archive_digest(buffer.getBuffer()), origin, false}, {}};
        for (const auto& definition : module.global_values())
            if (!definition.isDeclarationForLinker() && !definition.hasLocalLinkage() && !definition.hasAppendingLinkage())
                value.definitions.insert(definition.getName().str());
        libraries.push_back(std::move(value));
    };
    std::vector<std::pair<std::unique_ptr<llvm::MemoryBuffer>, std::string>> buffers;
    for (const auto& path : values.inputs) {
        auto file = llvm::MemoryBuffer::getFile(path);
        if (!file) throw std::runtime_error("cannot read " + path + ": " + file.getError().message());
        auto bytes = (*file)->getBuffer();
        if (has_program_image(std::string_view(bytes.data(), bytes.size())) || has_fatbin(std::string_view(bytes.data(), bytes.size()))) {
            auto normalized = image_bitcode((*file)->getMemBufferRef());
            buffers.emplace_back(llvm::MemoryBuffer::getMemBufferCopy(normalized, path), path);
        } else buffers.emplace_back(std::move(*file), path);
    }
    for (const auto& input : values.ir_inputs)
        buffers.emplace_back(llvm::MemoryBuffer::getMemBufferCopy(input.bytes, input.name), input.name);
    for (const auto& [file, path] : buffers) {
        if (file->getBuffer().starts_with("!<arch>\n") || file->getBuffer().starts_with("!<thin>\n")) {
            auto archive = llvm::object::Archive::create(file->getMemBufferRef());
            if (!archive) throw std::runtime_error("invalid device library " + path + ": " + llvm::toString(archive.takeError()));
            if ((*archive)->isThin()) throw std::runtime_error("thin device libraries are unsupported: " + path);
            llvm::Error error = llvm::Error::success();
            for (const auto& child : (*archive)->children(error)) {
                auto name = child.getName();
                if (!name) throw std::runtime_error("invalid device library member: " + llvm::toString(name.takeError()));
                auto origin = path + "(" + name->str() + ")";
                auto buffer = child.getMemoryBufferRef();
                if (!buffer) throw std::runtime_error("invalid device library member " + origin + ": " + llvm::toString(buffer.takeError()));
                candidate(*buffer, origin);
            }
            if (error) throw std::runtime_error("invalid device library " + path + ": " + llvm::toString(std::move(error)));
            continue;
        }
        // Discovery must not pollute the final context's named record types.
        // Replay the canonical TU order in that context after collecting bytes.
        llvm::LLVMContext discovery;
        auto parsed = llvm::parseBitcodeFile(file->getMemBufferRef(), discovery);
        if (!parsed) throw std::runtime_error("invalid device bitcode " + path + ": " + llvm::toString(parsed.takeError()));
        auto& module = **parsed;
        check_ir(module);
        auto input_stage = stage(module);
        if (input_stage == "clang.cuda.tu") {
            if (module.getNamedMetadata(archive_key)) throw std::runtime_error("raw TU cannot contain a device link archive: " + path);
            add(unit_identity(module), file->getBuffer(), path, false);
            continue;
        }
        auto* archive = module.getNamedMetadata(archive_key);
        if ((input_stage != "cudavm.portable" && input_stage != "clang.cuda.linked") || !archive || !archive->getNumOperands())
            throw std::runtime_error("device link accepts compile-tu artifacts or portable artifacts with a raw-TU archive: " + path);
        std::set<std::string> archived, declared;
        for (auto* row : archive->operands()) {
            if (row->getNumOperands() != 3) throw std::runtime_error("invalid device link archive metadata: " + path);
            auto identity = metadata_string(*row, 0).str();
            auto digest = metadata_string(*row, 1);
            auto bytes = metadata_string(*row, 2);
            if (digest != archive_digest(bytes)) throw std::runtime_error("device link archive digest mismatch: " + path);
            if (!archived.insert(identity).second) throw std::runtime_error("duplicate translation unit in device link archive: " + path);
            add(identity, bytes, path, true);
        }
        auto* membership = module.getNamedMetadata("cudavm.translation.units");
        if (!membership) throw std::runtime_error("missing device link archive membership: " + path);
        for (auto* row : membership->operands()) {
            if (row->getNumOperands() != 2) throw std::runtime_error("invalid device link archive membership: " + path);
            declared.insert(metadata_string(*row, 0).str());
        }
        if (declared != archived) throw std::runtime_error("device link archive membership mismatch: " + path);
    }
    for (const auto& path : values.library_inputs) {
        auto file = llvm::MemoryBuffer::getFile(path);
        if (!file) throw std::runtime_error("cannot read " + path + ": " + file.getError().message());
        candidate((*file)->getMemBufferRef(), path);
    }
    // Defer discovery until explicit inputs have supplied the source ABI. This
    // preserves archive/member order even when libraries precede the root TU.
    for (const auto& [bytes, origin] : candidates) parse_candidate(llvm::MemoryBufferRef(bytes, origin), origin);
    return units;
}
std::set<std::string> live_undefined(const llvm::Module& module) {
    // Match final closure roots without lowering or diagnosing missing symbols.
    // Dead helpers must not pull otherwise unused library members into a link.
    auto live = llvm::CloneModule(module);
    resolve_device_function_aliases(*live, true);
    resolve_device_data_aliases(*live, true);
    for (auto& function : *live) {
        if (function.isDeclaration() || function.hasFnAttribute("cudavm.kernel")) continue;
        function.setLinkage(llvm::GlobalValue::InternalLinkage);
        function.setComdat(nullptr);
    }
    optimize_closed(*live, false);
    std::set<std::string> unresolved;
    for (const auto& value : live->global_values()) {
        if (!value.isDeclaration() || value.use_empty() || value.hasExternalWeakLinkage()) continue;
        if (auto* function = llvm::dyn_cast<llvm::Function>(&value); function && function->isIntrinsic()) continue;
        unresolved.insert(value.getName().str());
    }
    return unresolved;
}
struct parameter { uint64_t size, alignment, offset; bool pointer; };
std::vector<parameter> parameter_layout(const llvm::Function& function) {
    const auto& layout = function.getParent()->getDataLayout();
    std::vector<parameter> result;
    uint64_t offset = 0;
    for (unsigned i = 0; i < function.arg_size(); ++i) {
        auto* type = function.getParamByValType(i);
        bool by_value = type != nullptr;
        if (!type) type = function.getArg(i)->getType();
        auto size = layout.getTypeAllocSize(type);
        if (size.isScalable()) throw std::runtime_error("scalable kernel parameters are unsupported");
        auto alignment = layout.getABITypeAlign(type).value();
        if (by_value && function.getParamAlign(i)) alignment = function.getParamAlign(i)->value();
        offset = llvm::alignTo(offset, alignment);
        result.push_back({size.getFixedValue(), alignment, offset, type->isPointerTy()});
        offset += size.getFixedValue();
    }
    return result;
}
void freeze_kernel_layout(llvm::Module& module) {
    for (auto& function : module) {
        if (!function.hasFnAttribute("cudavm.kernel")) continue;
        llvm::SmallVector<llvm::Metadata*> parameters;
        for (const auto& value : parameter_layout(function)) {
            llvm::SmallVector<llvm::Metadata*> fields;
            for (uint64_t number : {value.size, value.alignment, value.offset})
                fields.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt64Ty(module.getContext()), number)));
            parameters.push_back(llvm::MDNode::get(module.getContext(), fields));
        }
        function.setMetadata("cudavm.kernel.layout", llvm::MDNode::get(module.getContext(), parameters));
    }
}
std::string semantic_name(llvm::StringRef original) {
    constexpr const char* native[] = {"tid", "ctaid", "ntid", "nctaid"};
    constexpr const char* portable[] = {"thread_id", "block_id", "block_size", "grid_size"};
    for (unsigned i = 0; i < 4; ++i) {
        std::string prefix = std::string("llvm.nvvm.read.ptx.sreg.") + native[i] + ".";
        if (original.starts_with(prefix)) return std::string("cudavm.op.") + portable[i] + "." + original.drop_front(prefix.size()).str();
    }
    if (original == "llvm.nvvm.barrier0") return "cudavm.op.barrier.block";
    if (original == "llvm.nvvm.read.ptx.sreg.warpsize") return "cudavm.op.warp.size";
    // Other NVVM operations remain compact CUDA semantic identities. Their
    // exact origin is preserved; shader backends must implement each explicitly.
    return "cudavm.cuda." + original.drop_front(10).str();
}
void import_semantics(llvm::Module& module) {
    std::vector<llvm::Function*> intrinsics;
    for (auto& function : module) {
        function.removeFnAttr("target-cpu"); function.removeFnAttr("target-features");
        function.removeFnAttr("tune-cpu");
        if (function.hasFnAttribute("cudavm.kernel")) function.setCallingConv(llvm::CallingConv::C);
        if (function.getName().starts_with("llvm.nvvm.")) intrinsics.push_back(&function);
    }
    for (auto* native : intrinsics) {
        auto original = native->getName().str();
        auto* operation = llvm::Function::Create(native->getFunctionType(), llvm::GlobalValue::ExternalLinkage,
            semantic_name(original), module);
        operation->copyAttributesFrom(native);
        operation->addFnAttr("cudavm.cuda.origin", original);
        for (unsigned i = 0; i < native->arg_size(); ++i) operation->removeParamAttr(i, llvm::Attribute::ImmArg);
        for (auto* user : native->users()) {
            if (auto* call = llvm::dyn_cast<llvm::CallBase>(user))
                for (unsigned i = 0; i < call->arg_size(); ++i) call->removeParamAttr(i, llvm::Attribute::ImmArg);
        }
        native->replaceAllUsesWith(operation);
        native->eraseFromParent();
    }
    if (auto* annotations = module.getNamedMetadata("nvvm.annotations")) {
        auto* semantic = module.getOrInsertNamedMetadata("cudavm.cuda.annotations");
        for (auto* node : annotations->operands()) semantic->addOperand(node);
        module.eraseNamedMetadata(annotations);
    }
    module.setTargetTriple(llvm::Triple());
    // Keep the frontend data layout as the source object representation. CUDA
    // address spaces 0/1/3/4/5 remain generic/global/shared/constant/local domains.
}
}
void mark_translation_unit(llvm::Module& module, const compiler_options& values, const host_extraction& host) {
    module.setModuleIdentifier(values.input);
    for (auto& function : module)
        if (function.getCallingConv() == llvm::CallingConv::PTX_Kernel) function.addFnAttr("cudavm.kernel");
    append(module, "cudavm.source.abi", {string(module, module.getTargetTriple().str()), string(module, values.standard),
        string(module, values.architecture), string(module, "CUDA-13.4.1"), string(module, "clang-sdk-feature-level-12.8"),
        string(module, "cudavm-host-registration-v1")});
    append(module, "cudavm.translation.units", {string(module, translation_unit_id(values)), string(module, values.input)});
    for (const auto& entry : host.registrations) {
        auto symbol = *entry.getAsObject()->getString("device_symbol");
        auto* value = module.getNamedValue(symbol);
        auto* alias = llvm::dyn_cast_or_null<llvm::GlobalAlias>(value);
        auto* function = alias ? alias_target(*alias, true) : llvm::dyn_cast_or_null<llvm::Function>(value);
        if (!function || function->isDeclaration() || !function->hasFnAttribute("cudavm.kernel"))
            throw std::runtime_error("host kernel registration has no matching device definition: " + symbol.str());
        append(module, "cudavm.host.registrations", {string(module, translation_unit_id(values)), string(module, symbol)});
    }
    llvm::SmallVector<llvm::GlobalValue*> symbols;
    for (const auto& entry : host.symbols) {
        auto* object = entry.getAsObject();
        auto symbol = *object->getString("device_symbol");
        auto* variable = module.getNamedValue(*object->getString("ir_symbol"));
        // Clang adds a local-variable suffix when its device AST sees a host
        // reference. __CUDA_ARCH__ branches can hide that reference, so accept
        // either exact CodeGen name before imposing our stable linked identity.
        if (!variable) variable = module.getNamedValue(symbol);
        if (!variable || variable->getValueType()->isFunctionTy()
            || (variable->getAddressSpace() != 1 && variable->getAddressSpace() != 4))
            throw std::runtime_error("host symbol registration has no matching device global: " + symbol.str());
        variable->setName(symbol);
        symbols.push_back(variable);
        append(module, "cudavm.host.symbols", {string(module, translation_unit_id(values)), string(module, symbol)});
    }
    // Taking a symbol's address on the host must retain even an unused global.
    llvm::appendToCompilerUsed(module, symbols);
    set_stage(module, "clang.cuda.tu");
}
std::unique_ptr<llvm::Module> link_device(llvm::LLVMContext& context, const compiler_options& values) {
    link_diagnostics diagnostics(context, values.diagnostics);
    std::vector<library_unit> libraries;
    auto units = collect_link_inputs(values, libraries);
    host_link host(values.host_inputs);
    auto select_host = [&] {
        std::set<std::string> identities;
        for (const auto& [id, input] : units) identities.insert(id);
        bool changed = false;
        for (const auto& id : host.select(identities)) {
            if (units.contains(id)) continue;
            auto found = std::find_if(libraries.begin(), libraries.end(), [&](const auto& value) { return value.identity == id; });
            if (found == libraries.end()) throw std::runtime_error("host-selected translation unit has no device input: " + id);
            units.emplace(id, std::move(found->input)); libraries.erase(found); changed = true;
        }
        return changed;
    };
    while (select_host()) {}
    if (units.empty()) throw std::runtime_error("device link requires at least one explicit translation unit or host-selected member");
    auto replay = [&](llvm::LLVMContext& replay_context) {
        // Data addresses are observable: copying a winning weak definition would
        // split its identity from its aliases. Decide providers over the entire
        // selected set before preserving only the replaced original backings.
        std::map<std::string, std::pair<std::string, unsigned>> storage_providers;
        comdat_selection groups;
        for (const auto& [identity, input] : units) {
            llvm::LLVMContext discovery;
            auto parsed = llvm::parseBitcodeFile(llvm::MemoryBufferRef(input.bytes, input.origin), discovery);
            if (!parsed) throw std::runtime_error("invalid device bitcode: " + llvm::toString(parsed.takeError()));
            groups.discover(**parsed, identity);
        }
        groups.select();
        for (const auto& [identity, input] : units) {
            llvm::LLVMContext discovery;
            auto parsed = llvm::parseBitcodeFile(llvm::MemoryBufferRef(input.bytes, input.origin), discovery);
            if (!parsed) throw std::runtime_error("invalid device bitcode: " + llvm::toString(parsed.takeError()));
            for (const auto& value : (*parsed)->global_values()) {
                if (value.getValueType()->isFunctionTy() || value.isDeclarationForLinker()
                    || value.hasLocalLinkage() || value.hasAppendingLinkage()) continue;
                if (!groups.selected(value, identity)) continue;
                // LLVM prefers external over common over weak over linkonce.
                // Equal-strength definitions retain the canonical first TU.
                unsigned rank = groups.coherent(value) ? 5u : value.hasCommonLinkage() ? 3u : value.hasWeakLinkage() ? 2u
                    : value.hasLinkOnceLinkage() ? 1u : 4u;
                auto [found, inserted] = storage_providers.emplace(value.getName().str(), std::pair{identity, rank});
                if (!inserted && rank > found->second.second) found->second = {identity, rank};
            }
        }
        std::unique_ptr<llvm::Module> result;
        for (const auto& [identity, input] : units) {
            const auto& path = input.origin;
            auto parsed = llvm::parseBitcodeFile(llvm::MemoryBufferRef(input.bytes, path), replay_context);
            if (!parsed) throw std::runtime_error("invalid device bitcode " + path + ": " + llvm::toString(parsed.takeError()));
            auto module = std::move(*parsed);
            if (stage(*module) != "clang.cuda.tu") throw std::runtime_error("device link accepts only compile-tu artifacts: " + path);
            if (unit_identity(*module) != identity || module->getNamedMetadata(archive_key))
                throw std::runtime_error("device link archive translation-unit identity mismatch: " + path);
            validate_device_function_aliases(*module, true);
            validate_device_data_aliases(*module, true);
            groups.apply(*module, identity);
            freeze_weak_alias_targets(*module);
            std::set<std::string> replaced;
            for (const auto& [name, provider] : storage_providers)
                if (provider.first != identity) replaced.insert(name);
            normalize_odr_alias_comdats(*module, replaced);
            freeze_weak_data_alias_targets(*module, replaced, identity);
            auto* abi = module->getNamedMetadata("cudavm.source.abi");
            if (!abi || abi->getNumOperands() != 1 || abi->getOperand(0)->getNumOperands() != 6)
                throw std::runtime_error("unsupported CUDA compiler source/host ABI; recompile this translation unit: " + path);
            for (const auto& field : abi->getOperand(0)->operands())
                if (!llvm::dyn_cast_or_null<llvm::MDString>(field))
                    throw std::runtime_error("invalid CUDA compiler source/host ABI metadata: " + path);
            if (metadata_string(*abi->getOperand(0), 5) != "cudavm-host-registration-v1")
                throw std::runtime_error("unsupported CUDA compiler source/host ABI; recompile this translation unit: " + path);
            if (metadata_string(*abi->getOperand(0), 2) != values.architecture)
                throw std::runtime_error("incompatible CUDA compilation settings: --arch must match the source ABI in " + path);
            if (!result) { result = std::move(module); continue; }
            if (result->getDataLayoutStr() != module->getDataLayoutStr() || result->getTargetTriple() != module->getTargetTriple())
                throw std::runtime_error("incompatible CUDA source ABI in " + path);
            auto* a = result->getNamedMetadata("cudavm.source.abi");
            auto* b = module->getNamedMetadata("cudavm.source.abi");
            if (!a || !b || a->getOperand(0) != b->getOperand(0))
                throw std::runtime_error("incompatible CUDA compilation settings in " + path);
            // Check both declaration/alias arrival orders before LLVM replaces
            // either value. An opaque called pointer can hide a wrong signature.
            for (auto& incoming : module->global_values()) {
                if (incoming.hasLocalLinkage() || incoming.hasAppendingLinkage()) continue;
                auto* existing = result->getNamedValue(incoming.getName());
                if (!existing || existing->hasLocalLinkage()) continue;
                auto* incoming_alias = llvm::dyn_cast<llvm::GlobalAlias>(&incoming);
                auto* existing_alias = llvm::dyn_cast<llvm::GlobalAlias>(existing);
                if (!incoming.getValueType()->isFunctionTy() && !existing->getValueType()->isFunctionTy()) continue;
                auto* incoming_function = llvm::dyn_cast<llvm::Function>(&incoming);
                auto* existing_function = llvm::dyn_cast<llvm::Function>(existing);
                bool weak_helper = incoming.isWeakForLinker() || existing->isWeakForLinker();
                bool external_library = module->getNamedMetadata("cudavm.external.library.v1")
                    || result->getNamedMetadata("cudavm.external.library.v1");
                bool group_definition = (incoming_function && (incoming_function->getMetadata("cudavm.discarded.group.definition")
                        || incoming_function->getMetadata("cudavm.selected.group.definition")))
                    || (existing_function && (existing_function->getMetadata("cudavm.discarded.group.definition")
                        || existing_function->getMetadata("cudavm.selected.group.definition")));
                auto* left = incoming_alias ? alias_target(*incoming_alias, true) : incoming_function;
                auto* right = existing_alias ? alias_target(*existing_alias, true) : existing_function;
                bool ptx_kind_mismatch = left && right && (module->getNamedMetadata("cudavm.import.architecture")
                        || result->getNamedMetadata("cudavm.import.architecture"))
                    && left->hasFnAttribute("cudavm.kernel") != right->hasFnAttribute("cudavm.kernel");
                if (!left || !right || !same_alias_abi(*left, *right) || ptx_kind_mismatch)
                    throw std::runtime_error(std::string(incoming_alias || existing_alias
                        ? "incompatible device function alias: " : weak_helper || external_library || group_definition
                            ? "incompatible device helper override: " : "incompatible device function declaration: ")
                        + incoming.getName().str() + " in " + path);
            }
            // Opaque LLVM pointers can hide incompatible declarations when the
            // linker replaces a global. Check storage contracts before replacement.
            for (auto& incoming : module->global_values()) {
                if (incoming.hasLocalLinkage() || incoming.hasAppendingLinkage()) continue;
                if (incoming.getValueType()->isFunctionTy()) continue;
                auto* existing = result->getNamedValue(incoming.getName());
                if (!existing || existing->hasLocalLinkage()) continue;
                auto* variable = llvm::dyn_cast<llvm::GlobalVariable>(existing);
                auto* incoming_variable = llvm::dyn_cast<llvm::GlobalVariable>(&incoming);
                auto byte_key = [](const llvm::GlobalVariable* value) {
                    auto* array = value ? llvm::dyn_cast<llvm::ArrayType>(value->getValueType()) : nullptr;
                    return array && array->getElementType()->isIntegerTy(8) && value->getMetadata("cudavm.largest.group.key");
                };
                if (byte_key(variable) && byte_key(incoming_variable)) {
                    if (variable->getAddressSpace() != incoming_variable->getAddressSpace()
                        || variable->isConstant() != incoming_variable->isConstant()
                        || variable->getThreadLocalMode() != incoming_variable->getThreadLocalMode())
                        throw std::runtime_error("incompatible device global storage contract: " + incoming.getName().str() + " in " + path);
                    auto* definition = variable->hasInitializer() ? variable : incoming_variable;
                    auto* other = definition == variable ? incoming_variable : variable;
                    if (other->getAlign().valueOrOne() > definition->getAlign().valueOrOne())
                        throw std::runtime_error("incompatible device global alignment: " + incoming.getName().str() + " in " + path);
                    continue;
                }
                if (existing->getAddressSpace() != incoming.getAddressSpace()
                    || !same_storage_type(existing->getValueType(), incoming.getValueType()))
                    throw std::runtime_error("incompatible device global: " + incoming.getName().str() + " in " + path);
                auto* alias = llvm::dyn_cast<llvm::GlobalAlias>(variable ? &incoming : existing);
                auto* declaration = variable ? variable : incoming_variable;
                if (alias && declaration && declaration->getAlign().valueOrOne().value() > data_alias_target(*alias, true).alignment)
                    throw std::runtime_error("incompatible device global alignment: " + incoming.getName().str() + " in " + path);
                if (alias && declaration && declaration->hasInitializer()
                    && (alias->isWeakForLinker() || declaration->isWeakForLinker())) {
                    auto* storage = data_alias_target(*alias, true).target;
                    if (storage->isConstant() != declaration->isConstant()
                        || storage->getThreadLocalMode() != declaration->getThreadLocalMode())
                        throw std::runtime_error("incompatible device global storage contract: " + incoming.getName().str() + " in " + path);
                }
                if (!variable || !incoming_variable) continue;
                // Discarded weak/ODR definitions still supplied load/store ABI
                // promises to their callers. LLVM's merge alone can hide them.
                bool discarded_group = variable->getMetadata("cudavm.discarded.group.definition")
                    || incoming_variable->getMetadata("cudavm.discarded.group.definition");
                if ((variable->hasInitializer() || variable->getMetadata("cudavm.discarded.group.definition"))
                    && (incoming_variable->hasInitializer() || incoming_variable->getMetadata("cudavm.discarded.group.definition"))
                    && (variable->isWeakForLinker() || incoming_variable->isWeakForLinker() || discarded_group)
                    && (variable->getAlign() != incoming_variable->getAlign()
                        || variable->isConstant() != incoming_variable->isConstant()
                        || variable->getThreadLocalMode() != incoming_variable->getThreadLocalMode()))
                    throw std::runtime_error("incompatible device global storage contract: " + incoming.getName().str() + " in " + path);
                auto* definition = variable->hasInitializer() ? variable : incoming_variable->hasInitializer() ? incoming_variable : nullptr;
                declaration = definition == variable ? incoming_variable : variable;
                if (definition && declaration->isDeclaration()
                    && declaration->getAlign().valueOrOne() > definition->getAlign().valueOrOne())
                    throw std::runtime_error("incompatible device global alignment: " + incoming.getName().str() + " in " + path);
            }
            module->eraseNamedMetadata(module->getNamedMetadata(stage_key));
            if (llvm::Linker(*result).linkInModule(std::move(module))) throw std::runtime_error("device link failed: " + path);
        }
        rebind_odr_data_aliases(*result);
        return result;
    };
    while (!libraries.empty()) {
        if (select_host()) continue;
        // Each discovery round is isolated, just like archive discovery. Named
        // record types from trial links must not change canonical final replay.
        llvm::LLVMContext selection_context;
        link_diagnostics selection_diagnostics(selection_context, values.diagnostics);
        auto trial = replay(selection_context);
        auto unresolved = live_undefined(*trial);
        auto selected = libraries.end();
        for (auto candidate = libraries.begin(); candidate != libraries.end(); ++candidate) {
            if (auto existing = units.find(candidate->identity); existing != units.end()) {
                if (existing->second.bytes != candidate->input.bytes)
                    throw std::runtime_error("translation-unit identity reused for different contents: " + candidate->identity + " in " + candidate->input.origin);
                continue;
            }
            if (std::any_of(candidate->definitions.begin(), candidate->definitions.end(),
                    [&](const auto& name) { return unresolved.contains(name); })) {
                selected = candidate;
                break;
            }
        }
        if (selected == libraries.end()) break;
        units.emplace(selected->identity, std::move(selected->input));
        libraries.erase(selected);
    }
    while (select_host()) {}
    auto result = replay(context);
    for (const auto& [identity, input] : units)
        append(*result, archive_key, {string(*result, identity), string(*result, input.digest), string(*result, input.bytes)});
    result->setModuleIdentifier("cudavm.linked.device");
    set_stage(*result, "clang.cuda.linked");
    trace(*result, "device-link");
    return result;
}
void lower_device_conversions(llvm::Module& module) {
    for(auto name:{"int_as_float","uint_as_float","float_as_int","float_as_uint","longlong_as_double","double_as_longlong"}) {
        auto* function=module.getFunction(std::string("__nv_")+name);
        if(!function || !function->isDeclaration())continue;
        auto spelling=std::string_view(name);
        bool to_float=spelling.ends_with("_as_float") || spelling=="longlong_as_double";
        bool wide=spelling=="longlong_as_double" || spelling=="double_as_longlong";
        auto* word=llvm::IntegerType::get(module.getContext(),wide?64:32);
        auto* real=wide?llvm::Type::getDoubleTy(module.getContext()):llvm::Type::getFloatTy(module.getContext());
        auto* signature=llvm::FunctionType::get(to_float?real:word,{to_float?word:real},false);
        if(function->getFunctionType()!=signature)throw std::runtime_error("invalid CUDA bitcast signature: "+function->getName().str());
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
        builder.CreateRet(builder.CreateBitCast(function->getArg(0),function->getReturnType()));
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    for(auto conversion:device_conversions) {
        auto* function=module.getFunction(std::string("__nv_")+conversion.name);
        if(!function || !function->isDeclaration()) continue;
        auto* real=conversion.real_bits==64?llvm::Type::getDoubleTy(module.getContext()):llvm::Type::getFloatTy(module.getContext());
        auto* integer=llvm::Type::getIntNTy(module.getContext(),conversion.integer_bits);
        auto* signature=llvm::FunctionType::get(conversion.to_real?real:integer,
            {conversion.to_real?integer:real},false);
        if(function->getFunctionType()!=signature)
            throw std::runtime_error("invalid CUDA conversion signature: "+function->getName().str());
        // Use the named intrinsic with its checked signature. Clang's public
        // device wrapper calls libdevice; these operations need no device library.
        auto native=module.getOrInsertFunction(conversion.intrinsic,signature);
        function->replaceAllUsesWith(native.getCallee());
        function->eraseFromParent();
    }
    for(auto conversion:double_conversions) {
        auto* function=module.getFunction(std::string("__nv_")+conversion.name);
        if(!function || !function->isDeclaration()) continue;
        auto* signature=llvm::FunctionType::get(llvm::Type::getFloatTy(module.getContext()),
            {llvm::Type::getDoubleTy(module.getContext())},false);
        if(function->getFunctionType()!=signature)
            throw std::runtime_error("invalid CUDA conversion signature: "+function->getName().str());
        auto native=module.getOrInsertFunction(conversion.intrinsic,signature);
        function->replaceAllUsesWith(native.getCallee());
        function->eraseFromParent();
    }
    check_ir(module);
}
void lower_half_math_intrinsics(llvm::Module& module) {
    std::vector<llvm::CallInst*> calls;
    for(auto& function:module)for(auto& block:function)for(auto& instruction:block)
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&instruction);call && call->getCalledFunction() && call->getCalledFunction()->isIntrinsic() && call->getType()->isHalfTy())calls.push_back(call);
    for(auto* call:calls) {
        const char* operation=nullptr;
        switch(call->getCalledFunction()->getIntrinsicID()) {
        case llvm::Intrinsic::sqrt:operation="half_sqrt";break;
        case llvm::Intrinsic::sin:operation="half_sin";break;
        case llvm::Intrinsic::cos:operation="half_cos";break;
        case llvm::Intrinsic::exp:operation="exp";break;
        case llvm::Intrinsic::exp2:operation="exp2";break;
        case llvm::Intrinsic::log:operation="half_log";break;
        case llvm::Intrinsic::log2:operation="half_log2";break;
        case llvm::Intrinsic::log10:operation="half_log10";break;
        default:break;
        }
        if(!operation)continue;
        auto require=[&](const std::string& name) {
            auto* function=module.getFunction("__cudavm_native_"+name);
            if(!function || function->isDeclaration())throw std::runtime_error("missing half math helper: "+name);
            return function;
        };
        llvm::IRBuilder<> b(call);
        auto* bits=b.CreateZExt(b.CreateBitCast(call->getArgOperand(0),b.getInt16Ty()),b.getInt32Ty());
        auto* x=b.CreateBitCast(b.CreateCall(require("word_f16_to_f32"),{bits}),b.getFloatTy());
        auto* value=b.CreateCall(require(operation),{x});
        auto* result=b.CreateCall(require("word_f32_to_f16"),{b.CreateBitCast(value,b.getInt32Ty()),b.getInt32(0)});
        call->replaceAllUsesWith(b.CreateBitCast(b.CreateTrunc(result,b.getInt16Ty()),b.getHalfTy()));call->eraseFromParent();
    }
}
void lower_device_math(llvm::Module& module) {
    // Both source frontends enter here. PTX import does not, so ABI-specific
    // pinned-header mappings cannot replace arbitrary imported definitions.
    for(auto& function:module)if(!function.isDeclaration())function.addFnAttr("cudavm.cuda.source");
    lower_half_math_intrinsics(module);
    for(auto operation:device_math_operations) {
        auto* function=module.getFunction(std::string("__nv_")+operation.name);
        if(!function) continue;
        auto* real=llvm::Type::getFloatTy(module.getContext());
        auto* signature=llvm::FunctionType::get(real,std::vector<llvm::Type*>(operation.arguments,real),false);
        if(function->getFunctionType()!=signature)
            throw std::runtime_error("invalid CUDA math signature: "+function->getName().str());
        if(operation.intrinsic && function->isDeclaration()) {
            auto native=module.getOrInsertFunction(operation.intrinsic,signature);
            function->replaceAllUsesWith(native.getCallee());
            function->eraseFromParent();
        } else if(!operation.intrinsic && !function->isDeclaration()) {
            // The embedded frontend header defines full-range trig functions.
            // Shader emission calls the same source directly instead of expanding
            // its range-reduction loops into the kernel's PC dispatcher.
            function->addFnAttr("cudavm.math",operation.shader_helper);
        }
    }
    check_ir(module);
}
void lower_device_integers(llvm::Module& module) {
    for(auto name:{"min","max","umin","umax","llmin","llmax","ullmin","ullmax"}) {
        auto* function=module.getFunction(std::string("__nv_")+name);
        if(!function || !function->isDeclaration())continue;
        bool wide=std::string_view(name).find("ll")!=std::string_view::npos;
        auto* integer=llvm::IntegerType::get(module.getContext(),wide?64:32);
        if(function->getFunctionType()!=llvm::FunctionType::get(integer,{integer,integer},false)) throw std::runtime_error("invalid CUDA integer min/max signature");
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
        bool unsign=std::string_view(name).starts_with('u'),minimum=std::string_view(name).ends_with("min");
        auto predicate=unsign?(minimum?llvm::CmpInst::ICMP_ULT:llvm::CmpInst::ICMP_UGT):(minimum?llvm::CmpInst::ICMP_SLT:llvm::CmpInst::ICMP_SGT);
        builder.CreateRet(builder.CreateSelect(builder.CreateICmp(predicate,function->getArg(0),function->getArg(1)),function->getArg(0),function->getArg(1)));
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    for(auto operation:device_dpx_operations) {
        std::string name=std::string("__")+operation.name;
        // Clang's pristine CUDA header uses static Itanium-mangled wrappers.
        std::string mangled="_ZL"+std::to_string(name.size())+name+
            std::string(operation.arguments,operation.bits==32 && operation.signed_value?'i':'j');
        if(operation.predicate)mangled+=operation.bits==16?"PbS_":"Pb";
        for(const auto& symbol:{mangled,std::string("__cudavm_rtc_dpx_")+operation.name}) {
            auto* function=module.getFunction(symbol);if(!function)continue;
            auto* word=llvm::Type::getInt32Ty(module.getContext());
            std::vector<llvm::Type*> parameters(operation.arguments,word);
            if(operation.predicate)for(unsigned i=0;i<32/operation.bits;++i)
                parameters.push_back(llvm::PointerType::getUnqual(module.getContext()));
            if(function->getFunctionType()!=llvm::FunctionType::get(word,parameters,false))
                throw std::runtime_error("invalid CUDA DPX signature: "+symbol);
            auto linkage=function->getLinkage();function->deleteBody();function->setLinkage(linkage);
            function->removeFnAttr(llvm::Attribute::NoInline);
            function->removeFnAttr(llvm::Attribute::OptimizeNone);
            llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
            auto lane=[&](llvm::Value* value,unsigned index) -> llvm::Value* {
                if(operation.bits==32)return value;
                value=builder.CreateAnd(builder.CreateLShr(value,index*16),builder.getInt32(65535));
                return operation.signed_value?builder.CreateAShr(builder.CreateShl(value,16),16):value;
            };
            auto compare=[&](llvm::Value* a,llvm::Value* b) -> llvm::Value* {
                if(operation.signed_value)return operation.maximum?builder.CreateICmpSGE(a,b):builder.CreateICmpSLE(a,b);
                return operation.maximum?builder.CreateICmpUGE(a,b):builder.CreateICmpULE(a,b);
            };
            llvm::Value* result=builder.getInt32(0);
            llvm::Value* predicates[2]{};
            for(unsigned i=0;i<32/operation.bits;++i) {
                auto* a=lane(function->getArg(0),i);auto* b=lane(function->getArg(1),i);
                llvm::Value* value=nullptr;
                if(operation.add) {
                    // LLVM add has no overflow flags. Packed sums wrap before
                    // comparison, independently of the other halfword.
                    auto* sum=builder.CreateAdd(a,b);
                    if(operation.bits==16)sum=lane(sum,0);
                    auto* c=lane(function->getArg(2),i);
                    value=builder.CreateSelect(compare(sum,c),sum,c);
                } else {
                    predicates[i]=compare(a,b);value=builder.CreateSelect(predicates[i],a,b);
                    if(operation.arguments==3) {
                        auto* c=lane(function->getArg(2),i);
                        value=builder.CreateSelect(compare(value,c),value,c);
                    }
                }
                if(operation.relu)value=builder.CreateSelect(builder.CreateICmpSGT(value,builder.getInt32(0)),value,builder.getInt32(0));
                if(operation.bits==16)value=builder.CreateShl(builder.CreateAnd(value,builder.getInt32(65535)),i*16);
                result=builder.CreateOr(result,value);
            }
            // The public packed signature lists high then low predicates.
            // Preserve that store order when both destinations alias.
            if(operation.predicate)for(unsigned i=0;i<32/operation.bits;++i)
                builder.CreateStore(builder.CreateZExt(predicates[32/operation.bits-1-i],builder.getInt8Ty()),function->getArg(operation.arguments+i));
            builder.CreateRet(result);function->addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }
    // Snapshot source assembly before defining native helper bodies. Import
    // only the wrapper's exact register, merge and side-effect contract.
    std::vector<llvm::CallInst*> simd_calls;
    for(auto& function:module)for(auto& block:function)for(auto& instruction:block)
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&instruction);call && call->isInlineAsm())simd_calls.push_back(call);
    auto* simd_word=llvm::Type::getInt32Ty(module.getContext());
    auto simd_function=[&](const device_simd_operation& operation) {
        auto symbol=std::string("__nv_")+operation.name;
        auto* signature=llvm::FunctionType::get(simd_word,std::vector<llvm::Type*>(operation.arguments,simd_word),false);
        auto* function=module.getFunction(symbol);
        if(function && function->getFunctionType()!=signature)
            throw std::runtime_error("invalid CUDA SIMD signature: "+symbol);
        if(!function)function=llvm::Function::Create(signature,llvm::GlobalValue::ExternalLinkage,symbol,module);
        if(function->isDeclaration()) {
            llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
            llvm::Value* a=function->getArg(0);
            if(operation.operation==7) {
                auto* b=function->getArg(1);
                unsigned mask=operation.bits==16?0x7fff7fff:0x7f7f7f7f;
                builder.CreateRet(builder.CreateAdd(builder.CreateAnd(builder.CreateLShr(builder.CreateXor(a,b),1),builder.getInt32(mask)),builder.CreateAnd(a,b)));
            } else {
                llvm::Value* b=operation.arguments==2?static_cast<llvm::Value*>(function->getArg(1)):builder.getInt32(0);
                if(operation.operation==11 || operation.operation==12){b=a;a=builder.getInt32(0);}
                auto* asm_signature=llvm::FunctionType::get(simd_word,{simd_word,simd_word,simd_word},false);
                const char* spelling=operation.assembly;
                if(!spelling)for(auto candidate:device_simd_operations)
                    if(candidate.operation+10==operation.operation && candidate.bits==operation.bits)spelling=candidate.assembly;
                if(!spelling)throw std::runtime_error("missing CUDA SIMD assembly: "+symbol);
                auto* assembly=llvm::InlineAsm::get(asm_signature,spelling,"=r,r,r,r",false);
                llvm::Value* result=builder.CreateCall(assembly,{a,b,builder.getInt32(0)});
                if(operation.operation>=31)
                    result=builder.CreateMul(result,builder.getInt32((1u<<operation.bits)-1u));
                // Match Clang's workaround for ptxas signed halfword max when
                // both low lanes are negative. Keep the packed selection exact.
                if(operation.operation==17 && operation.bits==16) {
                    auto* compare=llvm::InlineAsm::get(asm_signature,"vset2.s32.s32.gt $0,$1,$2,$3;","=r,r,r,r",false);
                    auto* mask=builder.CreateMul(builder.CreateCall(compare,{a,b,builder.getInt32(0)}),builder.getInt32(65535));
                    auto* selected=builder.CreateOr(builder.CreateAnd(a,mask),builder.CreateAnd(b,builder.CreateNot(mask)));
                    auto* both=builder.CreateICmpNE(builder.CreateAnd(builder.CreateAnd(a,b),builder.getInt32(0x8000)),builder.getInt32(0));
                    result=builder.CreateSelect(both,selected,result);
                }
                builder.CreateRet(result);
                function->addFnAttr("cudavm.simd",operation.name);
            }
            function->addFnAttr(llvm::Attribute::AlwaysInline);
        }
        return function;
    };
    for(auto* call:simd_calls) {
        auto* assembly=llvm::cast<llvm::InlineAsm>(call->getCalledOperand());
        if(call->getFunctionType()!=llvm::FunctionType::get(simd_word,{simd_word,simd_word,simd_word},false) ||
            assembly->getConstraintString()!="=r,r,r,r" || assembly->hasSideEffects() || assembly->isAlignStack() ||
            assembly->getDialect()!=llvm::InlineAsm::AD_ATT)continue;
        auto* merge=llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
        if(!merge || !merge->isZero())continue;
        for(auto operation:device_simd_operations) {
            // Abs and neg share their binary instruction. Keep those generic
            // forms; the only unary-only instruction here is saturated abs.
            if(!operation.assembly || (operation.arguments==1 && operation.operation!=1) ||
                assembly->getAsmString()!=operation.assembly)continue;
            std::vector<llvm::Value*> arguments{call->getArgOperand(0)};
            if(operation.arguments==2)arguments.push_back(call->getArgOperand(1));
            else {
                auto* zero=llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
                if(!zero || !zero->isZero())continue;
            }
            llvm::IRBuilder<> builder(call);
            call->replaceAllUsesWith(builder.CreateCall(simd_function(operation),arguments));call->eraseFromParent();break;
        }
    }
    for(auto operation:device_simd_operations)
        if(auto* function=module.getFunction(std::string("__nv_")+operation.name);function && function->isDeclaration())simd_function(operation);
    for(auto operation:device_dot_operations) {
        auto* function=module.getFunction(operation.symbol);
        if(!function || !function->isDeclaration())continue;
        auto* word=llvm::Type::getInt32Ty(module.getContext());
        auto* signature=llvm::FunctionType::get(word,{word,word,word},false);
        if(function->getFunctionType()!=signature)
            throw std::runtime_error("invalid CUDA dot product signature: "+std::string(operation.symbol));
        std::vector<llvm::Type*> parameters{word,word};
        if(operation.high>=0)parameters.push_back(llvm::Type::getInt1Ty(module.getContext()));
        parameters.push_back(word);
        auto native=module.getOrInsertFunction(operation.intrinsic,llvm::FunctionType::get(word,parameters,false));
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
        std::vector<llvm::Value*> arguments{function->getArg(0),function->getArg(1)};
        if(operation.high>=0)arguments.push_back(builder.getInt1(operation.high!=0));
        arguments.push_back(function->getArg(2));
        builder.CreateRet(builder.CreateCall(native,arguments));
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    for(auto operation:device_integer_operations) {
        auto symbol=std::string("__nv_")+operation.name;
        auto* function=module.getFunction(symbol);
        if(!function || !function->isDeclaration())continue;
        auto* word=llvm::Type::getIntNTy(module.getContext(),operation.bits);
        auto* signature=llvm::FunctionType::get(word,std::vector<llvm::Type*>(operation.arguments,word),false);
        if(function->getFunctionType()!=signature)throw std::runtime_error("invalid CUDA integer intrinsic signature: "+symbol);
        auto native=module.getOrInsertFunction(operation.intrinsic,signature);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
        std::vector<llvm::Value*> arguments;
        for(auto& argument:function->args())arguments.push_back(&argument);
        builder.CreateRet(builder.CreateCall(native,arguments));
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    for(auto name:{"hadd","rhadd","uhadd","urhadd"}) {
        auto symbol=std::string("__nv_")+name;
        auto* function=module.getFunction(symbol);
        if(!function || !function->isDeclaration())continue;
        auto* word=llvm::Type::getInt32Ty(module.getContext());
        if(function->getFunctionType()!=llvm::FunctionType::get(word,{word,word},false))
            throw std::runtime_error("invalid CUDA integer intrinsic signature: "+symbol);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
        auto spelling=std::string_view(name);
        bool rounded=spelling=="rhadd" || spelling=="urhadd";
        auto* a=function->getArg(0);auto* b=function->getArg(1);
        auto* different=builder.CreateXor(a,b);
        auto* half=spelling.starts_with("u")?builder.CreateLShr(different,1):builder.CreateAShr(different,1);
        // Compute floor/ceil of the mathematical sum/2, including negative
        // odd sums, without overflowing an intermediate signed sum.
        builder.CreateRet(rounded?builder.CreateSub(builder.CreateOr(a,b),half):
            builder.CreateAdd(builder.CreateAnd(a,b),half));
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    // CUDA's byte-permutation wrapper ignores bit 3 in each selector nibble;
    // raw PTX prmt uses that bit for sign replication instead.
    if(auto* function=module.getFunction("__nv_byte_perm");function && function->isDeclaration()) {
        auto* word=llvm::Type::getInt32Ty(module.getContext());
        auto* signature=llvm::FunctionType::get(word,{word,word,word},false);
        if(function->getFunctionType()!=signature)throw std::runtime_error("invalid CUDA byte permutation signature");
        auto native=module.getOrInsertFunction("llvm.nvvm.prmt",signature);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
        builder.CreateRet(builder.CreateCall(native,{function->getArg(0),function->getArg(1),
            builder.CreateAnd(function->getArg(2),builder.getInt32(0x7777))}));
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    // Clang's CUDA header spells funnel shifts as these four exact inline-asm
    // forms. Import only their checked scalar signature and register contract.
    std::vector<llvm::CallInst*> funnels;
    for(auto& function:module)for(auto& block:function)for(auto& instruction:block)
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&instruction);call && call->isInlineAsm())funnels.push_back(call);
    for(auto* call:funnels) {
        auto* assembly=llvm::cast<llvm::InlineAsm>(call->getCalledOperand());
        auto text=assembly->getAsmString();
        bool left=text=="shf.l.wrap.b32 $0, $1, $2, $3;" || text=="shf.l.clamp.b32 $0, $1, $2, $3;";
        bool right=text=="shf.r.wrap.b32 $0, $1, $2, $3;" || text=="shf.r.clamp.b32 $0, $1, $2, $3;";
        if(!left && !right)continue;
        auto* word=llvm::Type::getInt32Ty(module.getContext());
        auto* signature=llvm::FunctionType::get(word,{word,word,word},false);
        if(call->getFunctionType()!=signature || assembly->getConstraintString()!="=r,r,r,r" ||
            assembly->hasSideEffects() || assembly->isAlignStack() || assembly->getDialect()!=llvm::InlineAsm::AD_ATT)
            continue;
        llvm::IRBuilder<> builder(call);
        auto native=module.getOrInsertFunction(left?"llvm.fshl.i32":"llvm.fshr.i32",signature);
        auto* low=call->getArgOperand(0);auto* high=call->getArgOperand(1);auto* shift=call->getArgOperand(2);
        llvm::Value* result=builder.CreateCall(native,{high,low,shift});
        if(text.find(".clamp.")!=std::string::npos)
            result=builder.CreateSelect(builder.CreateICmpULT(shift,builder.getInt32(32)),result,left?low:high);
        call->replaceAllUsesWith(result);call->eraseFromParent();
    }
    // Clang's CUDA wrappers call these libdevice symbols. Define their exact
    // integer semantics before closing the device graph; no libdevice is needed.
    for(auto name:{"popc","clz","ffs","brev"})for(unsigned bits:{32u,64u}) {
        auto symbol=std::string("__nv_")+name+(bits==64?"ll":"");
        auto* function=module.getFunction(symbol);
        if(!function || !function->isDeclaration())continue;
        bool reverse=std::string_view(name)=="brev";
        auto* word=llvm::Type::getIntNTy(module.getContext(),bits);
        auto* result=reverse?word:llvm::Type::getInt32Ty(module.getContext());
        if(function->getFunctionType()!=llvm::FunctionType::get(result,{word},false))
            throw std::runtime_error("invalid CUDA integer intrinsic signature: "+symbol);
        auto operation=std::string_view(name);
        auto intrinsic=std::string("llvm.")+(reverse?"bitreverse":operation=="popc"?"ctpop":operation=="clz"?"ctlz":"cttz")+".i"+std::to_string(bits);
        bool zero_flag=operation=="clz" || operation=="ffs";
        std::vector<llvm::Type*> parameters{word};
        if(zero_flag)parameters.push_back(llvm::Type::getInt1Ty(module.getContext()));
        auto native=module.getOrInsertFunction(intrinsic,llvm::FunctionType::get(word,parameters,false));
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(module.getContext(),"entry",function));
        auto* operand=function->getArg(0);
        std::vector<llvm::Value*> arguments{operand};
        if(zero_flag)arguments.push_back(builder.getFalse());
        llvm::Value* value=builder.CreateCall(native,arguments);
        if(operation=="ffs")value=builder.CreateSelect(builder.CreateICmpEQ(operand,llvm::ConstantInt::get(word,0)),
            llvm::ConstantInt::get(word,0),builder.CreateAdd(value,llvm::ConstantInt::get(word,1)));
        if(value->getType()!=result)value=builder.CreateTrunc(value,result);
        builder.CreateRet(value);
        function->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    check_ir(module);
}
void validate_device_function_aliases(const llvm::Module& module, bool final_link) {
    for (const auto& alias : module.aliases())
        if (alias.getValueType()->isFunctionTy()) (void)alias_target(alias, final_link);
    for (const auto& function : module) for (const auto& block : function) for (const auto& instruction : block) {
        auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        if (!call) continue;
        auto* alias = llvm::dyn_cast<llvm::GlobalAlias>(call->getCalledOperand()->stripPointerCasts());
        if (!alias) continue;
        auto* target = alias_target(*alias, final_link);
        if (target->hasFnAttribute("cudavm.kernel") || target->getCallingConv() == llvm::CallingConv::PTX_Kernel)
            throw std::runtime_error("device calls to a kernel alias are unsupported: " + alias->getName().str());
        if (call->getFunctionType() != target->getFunctionType() || call->getCallingConv() != target->getCallingConv()
            || !same_alias_attributes(call->getAttributes(), target->getAttributes(), target->arg_size()))
            throw std::runtime_error("incompatible device function alias call: " + alias->getName().str());
    }
}
void resolve_device_function_aliases(llvm::Module& module, bool final_link, bool native_exports) {
    validate_device_function_aliases(module, final_link);
    std::vector<std::pair<llvm::GlobalAlias*, llvm::Function*>> aliases;
    for (auto& alias : module.aliases())
        if (alias.getValueType()->isFunctionTy())
            aliases.emplace_back(&alias, const_cast<llvm::Function*>(alias_target(alias, final_link)));
    std::vector<std::pair<llvm::GlobalAlias*, llvm::Function*>> replacements;
    for (auto [alias, target] : aliases) {
        bool kernel = target->hasFnAttribute("cudavm.kernel") || target->getCallingConv() == llvm::CallingConv::PTX_Kernel;
        // PTX .alias is legal only for non-weak helpers. NVRTC keeps those
        // exports, and materializes the other names as independently callable
        // entries. AOT helper calls can simply bind the selected target.
        if (native_exports && !kernel && !alias->isWeakForLinker() && !target->isWeakForLinker()) continue;
        if (kernel || native_exports) {
            llvm::ValueToValueMapTy map;
            auto* entry = llvm::CloneFunction(target, map);
            entry->setLinkage(alias->getLinkage());
            entry->setComdat(nullptr);
            entry->setVisibility(alias->getVisibility());
            entry->setDLLStorageClass(alias->getDLLStorageClass());
            entry->setDSOLocal(alias->isDSOLocal());
            entry->takeName(alias);
            if (kernel) entry->addFnAttr("cudavm.kernel");
            // Launch bounds and other NVVM annotations belong to each exported
            // entry, in addition to the cloned parameter/function attributes.
            for (auto key : {"nvvm.annotations", "cudavm.cuda.annotations"}) {
                auto* annotations = module.getNamedMetadata(key);
                if (!annotations) continue;
                const unsigned count = annotations->getNumOperands();
                for (unsigned i = 0; i < count; ++i) {
                    auto* row = annotations->getOperand(i);
                    auto* value = row->getNumOperands() ? llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(row->getOperand(0)) : nullptr;
                    if (!value || value->getValue() != target) continue;
                    llvm::SmallVector<llvm::Metadata*> fields;
                    for (const auto& field : row->operands()) fields.push_back(field.get());
                    fields[0] = llvm::ValueAsMetadata::get(entry);
                    annotations->addOperand(llvm::MDNode::get(module.getContext(), fields));
                }
            }
            replacements.emplace_back(alias, entry);
        } else replacements.emplace_back(alias, target);
    }
    // Resolve all chains and create entries before replacing uses. Cloned
    // bodies can themselves reference another helper alias in this set.
    for (auto [alias, target] : replacements) alias->replaceAllUsesWith(target);
    for (auto [alias, target] : replacements) alias->eraseFromParent();
    check_ir(module);
}
void validate_device_data_aliases(const llvm::Module& module, bool final_link) {
    for (const auto& alias : module.aliases())
        if (!alias.getValueType()->isFunctionTy()) (void)data_alias_target(alias, final_link);
}
void resolve_device_data_aliases(llvm::Module& module, bool final_link) {
    validate_device_data_aliases(module, final_link);
    std::vector<llvm::GlobalAlias*> aliases;
    llvm::SmallVector<llvm::GlobalValue*> roots;
    for (auto& alias : module.aliases()) {
        if (alias.getValueType()->isFunctionTy()) continue;
        auto view = data_alias_target(alias, final_link);
        append(module, "cudavm.data.views.v1", {string(module, alias.getName()), string(module, view.target->getName()),
            string(module, std::to_string(view.offset)), string(module, std::to_string(view.bytes))});
        roots.push_back(const_cast<llvm::GlobalVariable*>(view.target));
        aliases.push_back(&alias);
    }
    // Exported views retain the backing object without creating another allocation.
    llvm::appendToCompilerUsed(module, roots);
    for (auto* alias : aliases) alias->replaceAllUsesWith(alias->getAliasee());
    for (auto* alias : aliases) alias->eraseFromParent();
}
std::string emit_data_views(const llvm::Module& module) {
    std::string output;
    if (auto* views = module.getNamedMetadata("cudavm.data.views.v1")) {
        std::map<std::string, std::string> rows;
        auto hex = [](llvm::StringRef value) {
            std::string result;
            constexpr char digits[] = "0123456789abcdef";
            for (unsigned char byte : value) { result += digits[byte >> 4]; result += digits[byte & 15]; }
            return result;
        };
        for (auto* row : views->operands()) {
            if (row->getNumOperands() != 4) throw std::runtime_error("invalid device data view metadata");
            auto name = metadata_string(*row, 0);
            auto target = metadata_string(*row, 1);
            auto* storage = module.getGlobalVariable(target, true);
            if (!storage || !storage->hasInitializer()) throw std::runtime_error("device data view lost its storage");
            uint64_t offset = 0, bytes = 0;
            auto capacity = module.getDataLayout().getTypeAllocSize(storage->getValueType());
            if (name.empty() || name == target || name.contains('\0') || target.contains('\0')
                || metadata_string(*row, 2).getAsInteger(10, offset) || metadata_string(*row, 3).getAsInteger(10, bytes)
                || capacity.isScalable() || !bytes || offset > capacity.getFixedValue()
                || bytes > capacity.getFixedValue() - offset)
                throw std::runtime_error("invalid device data view metadata");
            auto line = "// CUDAVM-DATA-VIEW-V1 " + hex(name) + " " + hex(target) + " "
                + metadata_string(*row, 2).str() + " " + metadata_string(*row, 3).str() + "\n";
            if (!rows.emplace(name.str(), line).second) throw std::runtime_error("duplicate device data view");
        }
        for (const auto& [name, line] : rows) output += line;
    }
    return output;
}
void normalize_device_relocations(llvm::Module& module) {
    for (auto& variable : module.globals())
        if (variable.hasInitializer() && (variable.getAddressSpace() == 1 || variable.getAddressSpace() == 4))
            variable.setInitializer(address_initializer{module.getDataLayout(), variable.getName().str()}.normalize(variable.getInitializer()));
}
void resolve_weak_device_references(llvm::Module& module) {
    std::vector<llvm::GlobalValue*> missing;
    for (auto& value : module.global_values())
        if (value.hasExternalWeakLinkage() && value.isDeclarationForLinker()) missing.push_back(&value);
    if (missing.empty()) return;
    // Optional references never extract an archive member solely for themselves.
    // This runs only after final provider selection (or non-RDC source closure).
    auto observed_address = [&](auto&& visit, llvm::Value* value, std::set<llvm::Value*>& seen) -> bool {
        if (!seen.insert(value).second) return false;
        for (auto* user : value->users()) {
            if (auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(user)) {
                if (visit(visit, expression, seen)) return true;
            } else if (llvm::isa<llvm::CastInst>(user) || llvm::isa<llvm::GetElementPtrInst>(user)) {
                if (visit(visit, user, seen)) return true;
            } else if (auto* call = llvm::dyn_cast<llvm::CallBase>(user)) {
                if (call->getCalledOperand() != value) return true;
            } else if (auto* load = llvm::dyn_cast<llvm::LoadInst>(user)) {
                if (load->getPointerOperand() != value) return true;
            } else if (auto* store = llvm::dyn_cast<llvm::StoreInst>(user)) {
                if (store->getValueOperand() == value) return true;
            } else if (auto* global = llvm::dyn_cast<llvm::GlobalVariable>(user)) {
                if (global->getName() != "llvm.used" && global->getName() != "llvm.compiler.used") return true;
            } else return true;
        }
        return false;
    };
    for (auto* value : missing) {
        if (auto* symbols = module.getNamedMetadata("cudavm.host.symbols"))
            for (auto* row : symbols->operands())
                if (row->getNumOperands() == 2 && metadata_string(*row, 1) == value->getName())
                    throw std::runtime_error("unresolved weak host symbol requires defined storage: " + value->getName().str());
        std::set<llvm::Value*> seen;
        if (!value->use_empty() && !observed_address(observed_address, value, seen))
            throw std::runtime_error(std::string(value->getValueType()->isFunctionTy()
                ? "unresolved device symbol: " : "unresolved device global: ") + value->getName().str());
    }
    llvm::removeFromUsedLists(module, [&](llvm::Constant* value) {
        auto* symbol = llvm::dyn_cast<llvm::GlobalValue>(value->stripPointerCasts());
        return symbol && symbol->hasExternalWeakLinkage() && symbol->isDeclarationForLinker();
    });
    for (auto* value : missing) {
        // CUDA null pointers remain zero through the generic/global/constant
        // address-space casts emitted by Clang. Fold those constant cast chains
        // explicitly instead of leaving target-dependent null cvta expressions.
        std::vector<llvm::WeakTrackingVH> casts;
        std::set<llvm::Value*> seen;
        auto collect = [&](auto&& visit, llvm::Value* input) -> void {
            if (!seen.insert(input).second) return;
            for (auto* user : input->users()) if (auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(user);
                    expression && expression->isCast() && expression->getOpcode() != llvm::Instruction::IntToPtr) {
                visit(visit, expression);
                casts.emplace_back(expression);
            }
        };
        collect(collect, value);
        for (auto& handle : casts)
            if (auto* expression = llvm::dyn_cast_or_null<llvm::ConstantExpr>(static_cast<llvm::Value*>(handle)))
                expression->replaceAllUsesWith(llvm::Constant::getNullValue(expression->getType()));
        value->replaceAllUsesWith(llvm::Constant::getNullValue(value->getType()));
        value->eraseFromParent();
    }
    optimize_closed(module, true);
    check_ir(module);
}
bool lower_device_atomic_assembly(llvm::CallInst& call,llvm::StringRef text) {
    llvm::IRBuilder<> b(&call);
    for(auto scope:{"cta","gpu"}) for(unsigned bits:{32u,64u}) {
        auto scope_id=b.getContext().getOrInsertSyncScopeID(std::string_view(scope)=="cta"?"block":"device");
        auto* integer=b.getIntNTy(bits);
        auto pointer=[&]() -> llvm::Value* {
            auto* value=call.getArgOperand(0);
            return value->getType()->isPointerTy()?value:b.CreateIntToPtr(value,b.getPtrTy());
        };
        auto size=std::to_string(bits);
        if(call.arg_size()==1 && call.getType()==integer && (call.getArgOperand(0)->getType()->isPointerTy() || call.getArgOperand(0)->getType()->isIntegerTy(64))) {
            for(auto order:{"relaxed","acquire"}) if(text==std::string("ld.")+order+"."+scope+".u"+size+"$0,[$1];") {
                auto* load=b.CreateAlignedLoad(integer,pointer(),llvm::Align(bits/8));
                load->setAtomic(std::string_view(order)=="acquire"?llvm::AtomicOrdering::Acquire:llvm::AtomicOrdering::Monotonic,scope_id);
                call.replaceAllUsesWith(load);call.eraseFromParent();return true;
            }
        }
        if(call.arg_size()!=2 || call.getArgOperand(1)->getType()!=integer || (!call.getArgOperand(0)->getType()->isPointerTy() && !call.getArgOperand(0)->getType()->isIntegerTy(64)))continue;
        if(call.getType()->isVoidTy()) for(auto order:{"relaxed","release"}) if(text==std::string("st.")+order+"."+scope+".u"+size+"[$0],$1;") {
            auto* store=b.CreateAlignedStore(call.getArgOperand(1),pointer(),llvm::Align(bits/8));
            store->setAtomic(std::string_view(order)=="release"?llvm::AtomicOrdering::Release:llvm::AtomicOrdering::Monotonic,scope_id);
            call.eraseFromParent();return true;
        }
        for(auto op:{"add","or","and","xor"}) for(auto order:{"relaxed","acquire","release","acq_rel"}) {
            bool reduction=call.getType()->isVoidTy();
            if(!reduction && call.getType()!=integer)continue;
            std::string prefix=std::string(reduction?"red.":"atom.")+op+"."+order+"."+scope;
            auto suffix=size+(reduction?"[$0],$1;":"$0,[$1],$2;");
            if(text!=prefix+".u"+suffix && text!=prefix+".b"+suffix)continue;
            auto mode=std::string_view(order)=="acquire"?llvm::AtomicOrdering::Acquire:std::string_view(order)=="release"?llvm::AtomicOrdering::Release:std::string_view(order)=="acq_rel"?llvm::AtomicOrdering::AcquireRelease:llvm::AtomicOrdering::Monotonic;
            auto operation=std::string_view(op)=="add"?llvm::AtomicRMWInst::Add:std::string_view(op)=="or"?llvm::AtomicRMWInst::Or:std::string_view(op)=="and"?llvm::AtomicRMWInst::And:llvm::AtomicRMWInst::Xor;
            auto* rmw=b.CreateAtomicRMW(operation,pointer(),call.getArgOperand(1),llvm::Align(bits/8),mode,scope_id);
            if(!reduction)call.replaceAllUsesWith(rmw);
            call.eraseFromParent();return true;
        }
    }
    return false;
}
// Recognize only scalar CTA synchronization assembly. Keep all other assembly
// for the backend's explicit unsupported-operation diagnostic.
void lower_device_synchronization(llvm::Module& module) {
    std::vector<llvm::CallInst*> calls;
    for(auto& f:module) for(auto& bb:f) for(auto& i:bb)
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&i)) calls.push_back(call);
    for(auto* call:calls) {
        llvm::IRBuilder<> b(call);
        std::string operation;
        std::vector<llvm::Value*> args;
        if(auto* function=call->getCalledFunction()) {
            auto name=function->getName();
            if(name=="__trap") {b.CreateIntrinsic(llvm::Intrinsic::trap,llvm::ArrayRef<llvm::Type*>{},llvm::ArrayRef<llvm::Value*>{});call->eraseFromParent();continue;}
            if(name=="__cudavm_rtc_barrier_sync") operation="llvm.nvvm.barrier.cta.sync.aligned.all";
            if(name=="__cudavm_rtc_barrier_count") operation="llvm.nvvm.barrier.cta.sync.aligned.count";
            if(name=="__cudavm_rtc_fence_block") operation="llvm.nvvm.membar.cta";
            if(name=="__cudavm_rtc_fence_device") operation="llvm.nvvm.membar.gl";
            if(name=="__cudavm_rtc_fence_system") operation="llvm.nvvm.membar.sys";
            if(operation.empty()) continue;
            for(auto& arg:call->args()) args.push_back(arg.get());
        } else if(call->isInlineAsm()) {
            auto* assembly=llvm::cast<llvm::InlineAsm>(call->getCalledOperand());
            if(assembly->isAlignStack() || assembly->getDialect()!=llvm::InlineAsm::AD_ATT) continue;
            std::string text;
            for(char c:assembly->getAsmString()) if(c!=' ' && c!='\t' && c!='\n' && c!='\r') text+=c;
            if(assembly->hasSideEffects() && lower_device_atomic_assembly(*call,text))continue;
            if(call->getType()->isIntegerTy(32) && call->arg_empty() && assembly->getConstraintString()=="=r") {
                for(auto reg:{"laneid","lanemask_lt","lanemask_eq","lanemask_le","lanemask_gt","lanemask_ge"}) {
                    if(text!=std::string("mov.u32$0,%")+reg+";") continue;
                    std::string name=reg;std::replace(name.begin(),name.end(),'_','.');
                    auto callee=module.getOrInsertFunction("llvm.nvvm.read.ptx.sreg."+name,llvm::FunctionType::get(b.getInt32Ty(),{},false));
                    call->replaceAllUsesWith(b.CreateCall(callee));operation="register";break;
                }
                if(operation=="register") {call->eraseFromParent();continue;}
            }
            if(!call->getType()->isVoidTy())continue;
            if(text=="membar.cta;") operation="llvm.nvvm.membar.cta";
            else if(text=="membar.gl;") operation="llvm.nvvm.membar.gl";
            else if(text=="membar.sys;") operation="llvm.nvvm.membar.sys";
            else {
                bool arrive=false,aligned=false;size_t prefix=0;
                for(auto spelling:{"bar.sync", "bar.arrive", "barrier.sync", "barrier.arrive", "barrier.cta.sync", "barrier.cta.arrive"})
                    if(text.starts_with(spelling)) {prefix=std::string(spelling).size();arrive=std::string(spelling).ends_with("arrive");aligned=std::string_view(spelling).starts_with("bar.");break;}
                if(!prefix || text.back()!=';') continue;
                if(text.substr(prefix).starts_with(".aligned")){prefix+=8;aligned=true;}
                auto operands=text.substr(prefix,text.size()-prefix-1);
                bool valid=true;
                while(!operands.empty()) {
                    auto comma=operands.find(',');auto token=operands.substr(0,comma);
                    unsigned number=0;
                    if(token.starts_with('$') && !llvm::StringRef(token).drop_front().getAsInteger(10,number) && number<call->arg_size()) args.push_back(call->getArgOperand(number));
                    else if(!llvm::StringRef(token).getAsInteger(10,number)) args.push_back(b.getInt32(number));
                    else {valid=false;break;}
                    if(comma==std::string::npos)break;
                    operands.erase(0,comma+1);
                }
                if(!valid || args.empty() || args.size()>2 || (arrive && args.size()!=2)) continue;
                operation=std::string("llvm.nvvm.barrier.cta.")+(arrive?"arrive":"sync")+(aligned?".aligned.":".")+(args.size()==2?"count":"all");
            }
        } else continue;
        if(operation.empty())continue;
        std::vector<llvm::Type*> types;
        for(auto* arg:args) {if(!arg->getType()->isIntegerTy(32)) throw std::runtime_error("CTA barrier requires 32-bit operands");types.push_back(arg->getType());}
        auto callee=module.getOrInsertFunction(operation,llvm::FunctionType::get(b.getVoidTy(),types,false));
        auto* replacement=b.CreateCall(callee,args);replacement->setConvergent();call->eraseFromParent();
    }
}
void prepare_portable(llvm::Module& module) {
    if (stage(module) != "clang.cuda.linked") throw std::runtime_error("portable preparation requires linked CUDA IR");
    resolve_device_function_aliases(module, true);
    resolve_device_data_aliases(module, true);
    resolve_weak_device_references(module);
    normalize_device_relocations(module);
    // The closed program has selected all storage providers. Emit fixed module
    // definitions; raw weak/ODR forms remain in the archive for later replay.
    for (auto& variable : module.globals())
        if (variable.hasInitializer() && (variable.getAddressSpace() == 1 || variable.getAddressSpace() == 4)
            && variable.isWeakForLinker()) {
            variable.setLinkage(llvm::GlobalValue::ExternalLinkage);
            variable.setComdat(nullptr);
        }
    lower_device_conversions(module);
    lower_device_math(module);
    lower_device_integers(module);
    lower_device_synchronization(module);
    import_semantics(module);
    trace(module, "cuda-semantic-import");
    advance(module, "clang.cuda.linked", "cudavm.semantic");
    close_graph(module);
    trace(module, "closed-device-graph");
    advance(module, "cudavm.semantic", "cudavm.closed");
    optimize_closed(module, true);
    check_ir(module);
    trace(module, "closed-program-specialization");
    // Freeze reachable function identities without expanding recursion or
    // indirect calls. A selected backend owns those implementations.
    std::vector<llvm::Function*> functions;
    for (auto& function : module) if (!function.isDeclaration()) functions.push_back(&function);
    std::sort(functions.begin(), functions.end(), [](auto* a, auto* b) { return a->getName() < b->getName(); });
    for (size_t i = 0; i < functions.size(); ++i) functions[i]->addFnAttr("cudavm.function.id", std::to_string(i + 1));
    trace(module, "freeze-function-identities");
    freeze_kernel_layout(module);
    trace(module, "kernel-parameter-layout");
    advance(module, "cudavm.closed", "cudavm.portable");
    for (auto& function : module)
        if (function.getName().starts_with("llvm.nvvm.")) throw std::runtime_error("native NVVM operation escaped semantic import");
}
llvm::json::Object describe(const llvm::Module& module) {
    llvm::json::Array entries, operations, pipeline, source_abis;
    for (const auto& function : module) {
        if (function.hasFnAttribute("cudavm.cuda.origin")) {
            operations.push_back(llvm::json::Object{{"name", function.getName().str()},
                {"cuda_origin", function.getFnAttribute("cudavm.cuda.origin").getValueAsString().str()}});
        }
        if (!function.hasFnAttribute("cudavm.kernel")) continue;
        llvm::json::Array parameters;
        uint64_t offset = 0;
        auto values = parameter_layout(function);
        for (unsigned i = 0; i < values.size(); ++i) {
            const auto& value = values[i];
            parameters.push_back(llvm::json::Object{{"index", int64_t(i)}, {"size", int64_t(value.size)},
                {"alignment", int64_t(value.alignment)}, {"offset", int64_t(value.offset)}, {"pointer", value.pointer}});
            offset = value.offset + value.size;
        }
        entries.push_back(llvm::json::Object{{"symbol", function.getName().str()}, {"parameters", std::move(parameters)}, {"parameter_bytes", int64_t(offset)}});
    }
    for (const char* key : {"cudavm.pipeline", "cudavm.source.abi"}) {
        if (auto* data = module.getNamedMetadata(key)) for (auto* row : data->operands()) {
            if (std::string(key) == "cudavm.pipeline") pipeline.push_back(llvm::cast<llvm::MDString>(row->getOperand(0))->getString().str());
            else {
                llvm::json::Array fields;
                for (auto& field : row->operands()) fields.push_back(llvm::cast<llvm::MDString>(field)->getString().str());
                source_abis.push_back(std::move(fields));
            }
        }
    }
    llvm::json::Array archived;
    if (auto* archive = module.getNamedMetadata(archive_key)) for (auto* row : archive->operands())
        archived.push_back(llvm::json::Object{{"translation_unit", metadata_string(*row, 0).str()}, {"sha256", metadata_string(*row, 1).str()}});
    return llvm::json::Object{{"schema_version", 1}, {"baseline", "13.4.1"}, {"stage", stage(module)},
        {"kernels", std::move(entries)}, {"semantic_operations", std::move(operations)},
        {"passes", std::move(pipeline)}, {"source_abis", std::move(source_abis)},
        {"device_link_archive", std::move(archived)}, {"runtime_launch_ready", false}, {"full_cuda_conformance", false}};
}
}
