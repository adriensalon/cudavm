#include "compiler.hpp"
#include "native_emission.hpp"
#include <stdexcept>
#include <optional>
#include <mutex>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InlineAsm.h>
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
    // PTX cannot alias entry points or weak functions. Export those names as
    // real bodies while retaining native .alias for ordinary strong helpers.
    resolve_device_function_aliases(*module, true, true);
    resolve_device_data_aliases(*module, true);
    normalize_device_relocations(*module);
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
    // Packed half comparisons with integer results were introduced in PTX 6.5,
    // after the default 6.3 dialect for SM 75. Preserve lower defaults elsewhere.
    std::string features;
    for(const auto& function:*module)for(const auto& block:function)for(const auto& instruction:block)
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&instruction);call && call->isInlineAsm())
            if(architecture.starts_with("sm_") && std::stoi(architecture.substr(3))<80
                && llvm::cast<llvm::InlineAsm>(call->getCalledOperand())->getAsmString().find(".u32.f16x2")!=std::string::npos)features="+ptx65";
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
