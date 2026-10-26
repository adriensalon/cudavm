#include "compiler.hpp"
#include "native_emission.hpp"
#include <stdexcept>
#include <optional>
#include <mutex>
#include <string_view>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Utils/Cloning.h>
extern "C" void LLVMInitializeNVPTXTargetInfo();
extern "C" void LLVMInitializeNVPTXTarget();
extern "C" void LLVMInitializeNVPTXTargetMC();
extern "C" void LLVMInitializeNVPTXAsmPrinter();
namespace cudavm {
std::string emit_cuda(const llvm::Module& portable, const std::string& architecture) {
    return emit_native(portable, {native_target::nvptx, architecture, true, {}}).bytes;
}
std::string emit_cuda_ir(const llvm::Module& input, const std::string& architecture, std::string* target_data_layout) {
    auto module = llvm::CloneModule(input);
    // Imported extern shared storage has an undef initializer so the portable
    // linker owns its per-kernel identity. NVPTX must emit an unsized external
    // declaration: a defined zero-sized global becomes one byte of static
    // storage and does not bind the launch's dynamic shared allocation.
    for (auto& storage : module->globals()) if (storage.getAddressSpace() == 3 &&
        module->getDataLayout().getTypeAllocSize(storage.getValueType()).getFixedValue() == 0) {
        storage.setInitializer(nullptr);
        storage.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }
    // Clang lowers printf to the driver's vprintf protocol. Preserve CUDA's
    // null-format result even when that lower-level service expects a string.
    // Device malloc also cannot represent an allocation whose size wraps its
    // mandatory sixteen-byte alignment. Guard before calling the native service.
    for (const char* name : {"vprintf", "malloc"}) if (auto* native = module->getFunction(name); native && native->isDeclaration()) {
        bool output = std::string_view(name) == "vprintf";
        auto* signature = native->getFunctionType();
        bool supported = !signature->isVarArg() && (output
            ? signature->getNumParams() == 2 && signature->getReturnType()->isIntegerTy(32) && signature->getParamType(0)->isPointerTy() && signature->getParamType(1)->isPointerTy()
            : signature->getNumParams() == 1 && signature->getReturnType()->isPointerTy() && signature->getParamType(0)->isIntegerTy(64));
        if (!supported) continue;
        std::vector<llvm::CallInst*> calls;
        for (auto* user : native->users()) if (auto* call = llvm::dyn_cast<llvm::CallInst>(user); call && call->getCalledFunction() == native) calls.push_back(call);
        if (calls.empty()) continue;
        if (output) { native->removeParamAttr(0, llvm::Attribute::NonNull); for (auto* call : calls) call->removeParamAttr(0, llvm::Attribute::NonNull); }
        auto* checked = llvm::Function::Create(signature, llvm::GlobalValue::InternalLinkage, std::string("__cudavm_native_checked_") + name, *module);
        checked->setCallingConv(native->getCallingConv()); checked->addFnAttr(llvm::Attribute::AlwaysInline);
        auto* first = llvm::BasicBlock::Create(module->getContext(), "entry", checked);
        auto* failed = llvm::BasicBlock::Create(module->getContext(), "unavailable", checked);
        auto* invoke = llvm::BasicBlock::Create(module->getContext(), "native", checked);
        llvm::IRBuilder<> builder(first);
        auto* invalid = output ? builder.CreateICmpEQ(checked->getArg(0), llvm::Constant::getNullValue(checked->getArg(0)->getType()))
            : builder.CreateICmpUGT(checked->getArg(0), builder.getInt64(UINT64_MAX - 15));
        builder.CreateCondBr(invalid, failed, invoke);
        builder.SetInsertPoint(failed); builder.CreateRet(output ? static_cast<llvm::Value*>(builder.getInt32(-1)) : llvm::Constant::getNullValue(signature->getReturnType()));
        builder.SetInsertPoint(invoke);
        std::vector<llvm::Value*> arguments; for (auto& argument : checked->args()) arguments.push_back(&argument);
        auto* result = builder.CreateCall(native, arguments); result->setCallingConv(native->getCallingConv()); builder.CreateRet(result);
        for (auto* call : calls) call->setCalledFunction(checked);
    }
    lower_half_math_intrinsics(*module);
    for(const auto& function:*module)for(const auto& block:function)for(const auto& instruction:block)
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&instruction);call && call->getType()->isHalfTy() && call->getCalledFunction() && call->getCalledFunction()->isIntrinsic()) {
            auto id=call->getCalledFunction()->getIntrinsicID();
            if(id!=llvm::Intrinsic::fma && id!=llvm::Intrinsic::fabs && id!=llvm::Intrinsic::copysign)
                throw std::runtime_error("unsupported native half intrinsic: "+call->getCalledFunction()->getName().str());
        }
    // PTX cannot alias entry points or weak functions. Export those names as
    // real bodies while retaining native .alias for ordinary strong helpers.
    resolve_device_function_aliases(*module, true, true);
    resolve_device_data_aliases(*module, true);
    normalize_device_relocations(*module);
    // NVPTX's legacy alloca-hoisting pass moves every constant-count alloca
    // to the entry, including repeated allocations whose live addresses escape
    // a loop iteration. Keep those counts in a native register so they retain
    // PTX's dynamic stack-allocation semantics at their original program point.
    for (auto& function : *module) for (auto& block : function) for (auto& instruction : block)
        if (auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(&instruction); allocation && !allocation->isStaticAlloca())
            if (auto* count = llvm::dyn_cast<llvm::ConstantInt>(allocation->getArraySize())) {
                if (count->getValue().getActiveBits() > 64) throw std::runtime_error("native stack allocation count exceeds 64 bits");
                llvm::IRBuilder<> builder(allocation);
                auto* signature = llvm::FunctionType::get(builder.getInt64Ty(), {builder.getInt64Ty()}, false);
                auto* move = llvm::InlineAsm::get(signature, "mov.b64 $0, $1;", "=l,l", false);
                allocation->setOperand(0, builder.CreateCall(move, {builder.getInt64(count->getZExtValue())}));
            }
    // Keep an explicit floating-point register before NVPTX splits a packed
    // float store into bytes. Reusing the integer result register directly for
    // add.f32 and byte stores produces a one-bit error in the NVIDIA PTX JIT.
    std::vector<llvm::StoreInst*> packed_float_stores;
    for(auto& f:*module)for(auto& b:f)for(auto& i:b)
        if(auto* s=llvm::dyn_cast<llvm::StoreInst>(&i);s && s->getValueOperand()->getType()->isFloatTy() && s->getAlign().value()<4)
            packed_float_stores.push_back(s);
    for(auto* s:packed_float_stores) {
        llvm::IRBuilder<> builder(s);
        auto* signature=llvm::FunctionType::get(builder.getFloatTy(),{builder.getFloatTy()},false);
        auto* move=llvm::InlineAsm::get(signature,"{ .reg .f32 cvPackedFloat; mov.b32 cvPackedFloat, $1; mov.b32 $0, cvPackedFloat; }","=f,f",false);
        s->setOperand(0,builder.CreateCall(move,{s->getValueOperand()}));
    }
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        LLVMInitializeNVPTXTargetInfo(); LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetMC(); LLVMInitializeNVPTXAsmPrinter();
    });
    std::string error;
    auto* target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
    if (!target) throw std::runtime_error(error);
    llvm::TargetOptions settings;
    // Inline instructions from CUDA headers also impose a PTX dialect floor.
    // Packed half comparisons need PTX 6.5; the pinned tanh assembly needs
    // PTX 7.0 even on SM 75. Dynamic alloca needs PTX 7.3 and SM 52.
    // Never lower a newer architecture's default floor.
    unsigned dialect=0;
    for(const auto& function:*module)for(const auto& block:function)for(const auto& instruction:block) {
        if (auto* allocation=llvm::dyn_cast<llvm::AllocaInst>(&instruction); allocation && !allocation->isStaticAlloca()) dialect=std::max(dialect,73u);
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&instruction);call && call->isInlineAsm()) {
            auto text=llvm::cast<llvm::InlineAsm>(call->getCalledOperand())->getAsmString();
            if(text.contains(".u32.f16x2"))dialect=std::max(dialect,65u);
            if(text.contains("tanh.approx."))dialect=std::max(dialect,70u);
        }
    }
    std::string features;
    if (architecture.starts_with("sm_")) {
        auto sm=std::stoi(architecture.substr(3));
        if (dialect>=73 && sm<52) throw std::runtime_error("dynamic invocation-private storage requires SM 52 or later");
        if (dialect && sm<(dialect>=73?87:80)) features="+ptx"+std::to_string(dialect);
    }
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(module->getTargetTriple(), architecture, features, settings, std::nullopt));
    if (!machine) throw std::runtime_error("NVPTX target machine creation failed");
    module->setDataLayout(machine->createDataLayout());
    if (target_data_layout) {
        validate_native_kernel_abi(*module);
        *target_data_layout = module->getDataLayoutStr();
    }
    std::string diagnostics;
    llvm::raw_string_ostream diagnostic_stream(diagnostics);
    if (llvm::verifyModule(*module, &diagnostic_stream))
        throw std::runtime_error("invalid CUDA backend IR:\n" + diagnostics);
    llvm::SmallVector<char, 0> code;
    llvm::raw_svector_ostream stream(code);
    llvm::legacy::PassManager passes;
    if (machine->addPassesToEmitFile(passes, stream, nullptr, llvm::CodeGenFileType::AssemblyFile))
        throw std::runtime_error("NVPTX assembly emission is unavailable");
    passes.run(*module);
    return std::string(code.begin(), code.end()) + emit_data_views(*module);
}
}
