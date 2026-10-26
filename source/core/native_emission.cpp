#include "native_emission.hpp"
#include "device_services.hpp"
#include "spirv_emitter.hpp"
#include "dxil_emitter.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace cudavm {
namespace {
void verify(const llvm::Module& module) {
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    if (llvm::verifyModule(module, &stream))
        throw std::runtime_error("invalid native emission input:\n" + diagnostic);
}
void validate_layout(const llvm::Module& module, const llvm::json::Array& kernels) {
    for (const auto& entry : kernels) {
        const auto& kernel = *entry.getAsObject();
        auto symbol = *kernel.getString("symbol");
        const auto& function = *module.getFunction(symbol);
        if (function.isDeclaration() || function.isVarArg() || !function.getReturnType()->isVoidTy())
            throw std::runtime_error("invalid native kernel signature: " + symbol.str());
        auto* frozen = function.getMetadata("cudavm.kernel.layout");
        const auto& parameters = *kernel.getArray("parameters");
        if (!frozen || frozen->getNumOperands() != parameters.size())
            throw std::runtime_error("missing frozen kernel ABI: " + symbol.str());
        for (unsigned i = 0; i < parameters.size(); ++i) {
            auto* row = llvm::dyn_cast_or_null<llvm::MDNode>(frozen->getOperand(i));
            if (!row || row->getNumOperands() != 3)
                throw std::runtime_error("invalid frozen kernel ABI: " + symbol.str());
            unsigned field = 0;
            for (const char* name : {"size", "alignment", "offset"}) {
                auto* constant = llvm::dyn_cast_or_null<llvm::ConstantAsMetadata>(row->getOperand(field++));
                auto* number = constant ? llvm::dyn_cast<llvm::ConstantInt>(constant->getValue()) : nullptr;
                if (!number || number->getBitWidth() != 64
                    || number->getZExtValue() != uint64_t(*parameters[i].getAsObject()->getInteger(name)))
                    throw std::runtime_error("changed frozen kernel ABI: " + symbol.str() + " parameter " + std::to_string(i));
            }
        }
    }
}
llvm::json::Array array(const std::set<std::string>& values) {
    llvm::json::Array result;
    for (const auto& value : values) result.push_back(value);
    return result;
}
const char* triple(native_target target) {
    switch (target) {
    case native_target::nvptx: return "nvptx64-nvidia-cuda";
    case native_target::spirv: return "spirv64-unknown-unknown";
    case native_target::dxil: return "dxil-pc-shadermodel6.0-compute";
    }
    throw std::runtime_error("unknown native emission target");
}
void cuda_lower(llvm::Module& module) {
    module.setTargetTriple(llvm::Triple(triple(native_target::nvptx)));
    std::vector<llvm::Function*> operations;
    for (auto& function : module) {
        if (function.hasFnAttribute("cudavm.kernel")) function.setCallingConv(llvm::CallingConv::PTX_Kernel);
        if (function.hasFnAttribute("cudavm.cuda.origin")) operations.push_back(&function);
    }
    for (auto* function : operations) {
        auto name = function->getFnAttribute("cudavm.cuda.origin").getValueAsString().str();
        // A portable module cannot also contain a conflicting native identity.
        if (module.getNamedValue(name)) throw std::runtime_error("duplicate native CUDA operation: " + name);
        auto* native = llvm::Function::Create(function->getFunctionType(), llvm::GlobalValue::ExternalLinkage, name, module);
        native->copyAttributesFrom(function);
        native->removeFnAttr("cudavm.cuda.origin");
        auto intrinsic = native->getIntrinsicID();
        if (intrinsic == llvm::Intrinsic::not_intrinsic)
            throw std::runtime_error("unknown native CUDA operation: " + name);
        // LLVM selection depends on canonical intrinsic memory/convergence
        // attributes, including for valid portable IR assembled by a client.
        native->addFnAttrs(llvm::AttrBuilder(module.getContext(), llvm::Intrinsic::getFnAttributes(module.getContext(), intrinsic)));
        function->replaceAllUsesWith(native);
        function->eraseFromParent();
    }
    if (auto* annotations = module.getNamedMetadata("cudavm.cuda.annotations")) {
        for (auto* node : annotations->operands()) module.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(node);
        module.eraseNamedMetadata(annotations);
    }
    auto* stage = module.getOrInsertNamedMetadata("cudavm.ir.stage");
    stage->clearOperands();
    stage->addOperand(llvm::MDNode::get(module.getContext(), llvm::MDString::get(module.getContext(), "cudavm.native.nvptx")));
}
}
const char* native_target_name(native_target target) {
    switch (target) {
    case native_target::nvptx: return "nvptx";
    case native_target::spirv: return "spirv";
    case native_target::dxil: return "dxil";
    }
    throw std::runtime_error("unknown native emission target");
}
void validate_native_kernel_abi(const llvm::Module& module) {
    auto description = describe(module);
    validate_layout(module, *description.getArray("kernels"));
}
llvm::json::Object inspect_native(const llvm::Module& portable, native_target target) {
    auto name = native_target_name(target);
    verify(portable);
    const auto& layout = portable.getDataLayout();
    if (portable.getDataLayoutStr().empty() || !layout.isLittleEndian()
        || layout.getPointerSizeInBits(0) != 64 || layout.isNonIntegralAddressSpace(0)
        || layout.getPointerSizeInBits(1) != 64 || layout.isNonIntegralAddressSpace(1)
        || layout.getPointerSizeInBits(4) != 64 || layout.isNonIntegralAddressSpace(4))
        throw std::runtime_error("native emission requires little-endian integral 64-bit CUDA addresses");
    auto description = describe(portable);
    if (description.getString("stage") != "cudavm.portable")
        throw std::runtime_error("native emission requires portable IR");
    validate_layout(portable, *description.getArray("kernels"));
    std::set<std::string> requirements, scopes, orders;
    llvm::json::Array globals;
    for (const auto& variable : portable.globals()) {
        if (variable.isDeclaration() && !variable.use_empty() && variable.getAddressSpace() != 3)
            throw std::runtime_error("unresolved native device global: " + variable.getName().str());
        if (variable.getName().starts_with("llvm.")) continue;
        auto bytes = layout.getTypeAllocSize(variable.getValueType());
        if (bytes.isScalable()) throw std::runtime_error("scalable native global: " + variable.getName().str());
        globals.push_back(llvm::json::Object{{"symbol", variable.getName().str()},
            {"address_space", int64_t(variable.getAddressSpace())}, {"bytes", int64_t(bytes.getFixedValue())},
            {"alignment", int64_t(variable.getAlign().value_or(layout.getABITypeAlign(variable.getValueType())).value())},
            {"declaration", variable.isDeclaration()}, {"constant", variable.isConstant()}});
    }
    llvm::SmallVector<llvm::StringRef> scope_names;
    portable.getContext().getSyncScopeNames(scope_names);
    for (const auto& function : portable) {
        if (function.getName().starts_with("llvm.nvvm."))
            throw std::runtime_error("native NVVM operation in portable IR: " + function.getName().str());
        if (function.isDeclaration() && !function.use_empty() && !function.isIntrinsic()
            && !function.hasFnAttribute("cudavm.cuda.origin") && !device_service(function.getName().str()))
            throw std::runtime_error("unresolved native device function: " + function.getName().str());
        if (function.hasFnAttribute("cudavm.cuda.origin")) {
            requirements.insert("cuda-semantic-operations");
            auto origin = function.getFnAttribute("cudavm.cuda.origin").getValueAsString();
            if (origin.contains("barrier")) requirements.insert("cuda-barriers");
            if (origin.contains("shfl") || origin.contains("vote") || origin.contains("match")
                || origin.contains("redux") || origin.contains("warp") || origin.contains("activemask")) requirements.insert("cuda-warp-semantics");
        }
        for (const auto& block : function) for (const auto& instruction : block) {
            if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
                if (call->isInlineAsm()) requirements.insert("cuda-inline-assembly");
                else if (!llvm::isa<llvm::Function>(call->getCalledOperand()->stripPointerCasts())) requirements.insert("indirect-calls");
            }
            if (const auto* fp = llvm::dyn_cast<llvm::FPMathOperator>(&instruction); fp && fp->getFastMathFlags().any())
                requirements.insert("instruction-fast-math");
            auto atomic = [&](llvm::AtomicOrdering order, llvm::SyncScope::ID scope) {
                requirements.insert("atomics-and-fences");
                orders.insert(llvm::toIRString(order));
                if (scope >= scope_names.size()) throw std::runtime_error("invalid native atomic scope");
                scopes.insert(scope_names[scope].empty() ? "system" : scope_names[scope].str());
            };
            if (const auto* rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&instruction)) atomic(rmw->getOrdering(), rmw->getSyncScopeID());
            if (const auto* cmp = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&instruction)) {
                atomic(cmp->getSuccessOrdering(), cmp->getSyncScopeID()); atomic(cmp->getFailureOrdering(), cmp->getSyncScopeID());
            }
            if (const auto* fence = llvm::dyn_cast<llvm::FenceInst>(&instruction)) atomic(fence->getOrdering(), fence->getSyncScopeID());
            if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction); load && load->isAtomic()) atomic(load->getOrdering(), load->getSyncScopeID());
            if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction); store && store->isAtomic()) atomic(store->getOrdering(), store->getSyncScopeID());
        }
    }
    bool cuda = target == native_target::nvptx;
    return llvm::json::Object{{"schema_version", 1}, {"target", name}, {"input_stage", "cudavm.portable"},
        {"target_triple", triple(target)}, {"source_data_layout", portable.getDataLayoutStr()},
        {"lowering_available", true}, {"artifact_format", cuda ? "ptx" : target == native_target::spirv ? "spirv" : "dxil"},
        {"kernel_abi", "frozen-cuda-parameters-v1"}, {"kernels", std::move(*description.getArray("kernels"))},
        {"globals", std::move(globals)}, {"semantic_operations", std::move(*description.getArray("semantic_operations"))},
        {"requirements", array(requirements)}, {"atomic_orderings", array(orders)}, {"atomic_scopes", array(scopes)},
        {"address_contract", cuda ? "native-cuda-addresses" : "cuda-virtual-addresses-require-resource-resolution"},
        {"address_spaces", llvm::json::Object{{"generic", 0}, {"global", 1}, {"shared", 3}, {"constant", 4}, {"local", 5}, {"parameter", 6}}},
        {"shared_contract", "per-block-static-and-dynamic-storage"},
        {"warp_contract", "cuda-32-lanes-active-mask-and-reconvergence"},
        {"barrier_contract", "cuda-execution-scope-and-memory-order"},
        {"math_contract", "preserve-instruction-flags-and-explicit-cuda-operations"},
        {"symbol_owner", "existing-device-linker-and-module-registry"},
        {"device_capabilities_verified", false}, {"full_cuda_conformance", false}};
}
native_program prepare_native(const llvm::Module& portable, const native_request& request) {
    auto contract = inspect_native(portable, request.target);
    if (request.target == native_target::spirv || request.target == native_target::dxil) {
        auto symbol = request.kernel;
        if (symbol.empty()) {
            const auto& kernels = *contract.getArray("kernels");
            if (kernels.size() != 1) throw std::runtime_error("native shader emission requires one selected kernel");
            symbol = kernels.front().getAsObject()->getString("symbol")->str();
        }
        auto module = request.target == native_target::spirv ? lower_native_spirv(portable, symbol) : lower_native_dxil(portable, symbol, request.block);
        contract["selected_kernel"] = symbol;
        contract["target_stage"] = request.target == native_target::spirv ? "cudavm.native.spirv" : "cudavm.native.dxil";
        contract["target_triple"] = module->getTargetTriple().str();
        contract["target_data_layout"] = module->getDataLayoutStr();
        contract["resource_abi"] = "cuda-packet-v3-word-memory";
        return {std::move(module), std::move(contract)};
    }
    if (request.target != native_target::nvptx)
        throw std::runtime_error(std::string("native ") + native_target_name(request.target)
            + " CUDA ABI lowering is unavailable; use the existing adapter fallback until "
            + (request.target == native_target::spirv ? "BACKEND-2" : "BACKEND-3"));
    auto cpu = llvm::StringRef(request.architecture);
    if (cpu.ends_with("a") || cpu.ends_with("f")) cpu = cpu.drop_back();
    if (!cpu.starts_with("sm_") || cpu.size() <= 3
        || !std::all_of(cpu.begin() + 3, cpu.end(), [](unsigned char c) { return c >= '0' && c <= '9'; }))
        throw std::runtime_error("native NVPTX architecture requires sm_<number> with optional a/f suffix");
    auto module = llvm::CloneModule(portable);
    cuda_lower(*module);
    verify(*module);
    validate_layout(*module, *contract.getArray("kernels"));
    contract["architecture"] = request.architecture;
    contract["target_stage"] = "cudavm.native.nvptx";
    return {std::move(module), std::move(contract)};
}
native_artifact emit_native(const llvm::Module& portable, const native_request& request) {
    auto program = prepare_native(portable, request);
    native_artifact artifact;
    if (request.target == native_target::spirv || request.target == native_target::dxil) {
        artifact.bytes = request.target == native_target::spirv ? emit_native_spirv(*program.module) : emit_native_dxil(*program.module);
        artifact.contract = std::move(program.contract);
        return artifact;
    }
    std::string target_layout;
    artifact.bytes = emit_cuda_ir(*program.module, request.architecture, &target_layout);
    artifact.contract = std::move(program.contract);
    artifact.contract["target_data_layout"] = std::move(target_layout);
    if (request.shader_artifacts) {
        artifact.bytes += emit_native_module(portable);
        llvm::json::Array spirv_inventory;
        artifact.bytes += emit_native_spirv_payload(portable, &spirv_inventory);
        artifact.contract["native_spirv_kernels"] = std::move(spirv_inventory);
        llvm::json::Array dxil_inventory;
        artifact.bytes += emit_native_dxil_payload(portable, &dxil_inventory);
        artifact.contract["native_dxil_kernels"] = std::move(dxil_inventory);
    }
    artifact.contract["shader_artifacts_requested"] = request.shader_artifacts;
    return artifact;
}
}
