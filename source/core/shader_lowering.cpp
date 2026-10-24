#include "shader_lowering.hpp"
#include "shader_collectives.hpp"
#include "shader_local.hpp"
#include "shader_calls.hpp"
#include "device_services.hpp"
#include "device_conversions.hpp"
#include "ptx_import.hpp"

#include <map>
#include <tuple>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsSPIRV.h>
#include <llvm/IR/IntrinsicsDirectX.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/KnownBits.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Scalar/TailRecursionElimination.h>
#include <llvm/Transforms/Scalar/Scalarizer.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Scalar/Reg2Mem.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/CallPromotionUtils.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/LowerMemIntrinsics.h>
#include <llvm/Transforms/Utils/FixIrreducible.h>
#include <llvm/Transforms/Utils/Local.h>
namespace cudavm {
namespace {
#include "shader_half.inl"
void scalar_math(llvm::Module& module) {
    auto* divide = module.getFunction("__cudavm_native_divide");
    auto* compare = module.getFunction("__cudavm_native_compare");
    auto* square_root = module.getFunction("__cudavm_native_sqrt");
    auto* maximum = module.getFunction("__cudavm_native_max");
    auto* integral = module.getFunction("__cudavm_native_integral");
    auto numeric=[&](const std::string& name){
        auto* function=module.getFunction("__cudavm_native_word_"+name);
        if(name.find("f16")!=std::string::npos && (!function || function->isDeclaration()))
            throw std::runtime_error("missing half numeric helper: "+name);
        return function;
    };
    std::vector<llvm::Instruction*> work;
    for (auto& function : module) for (auto& bb : function) for (auto& instruction : bb) work.push_back(&instruction);
    for (auto* instruction : work) {
        llvm::IRBuilder<> builder(instruction); llvm::Value* replacement = nullptr;
        auto bits=[&](llvm::Value* value){return builder.CreateBitCast(value,value->getType()->isDoubleTy()?builder.getInt64Ty():builder.getInt32Ty());};
        auto half_bits=[&](llvm::Value* value){return builder.CreateZExt(builder.CreateBitCast(value,builder.getInt16Ty()),builder.getInt32Ty());};
        auto half_value=[&](llvm::Value* value){return builder.CreateBitCast(builder.CreateTrunc(value,builder.getInt16Ty()),builder.getHalfTy());};
        auto half_float=[&](llvm::Value* value){return builder.CreateBitCast(builder.CreateCall(numeric("f16_to_f32"),{half_bits(value)}),builder.getFloatTy());};
        if(instruction->getOpcode()==llvm::Instruction::FNeg && instruction->getType()->isHalfTy())
            replacement=half_value(builder.CreateXor(half_bits(instruction->getOperand(0)),builder.getInt32(0x8000)));
        if(auto* binary=llvm::dyn_cast<llvm::BinaryOperator>(instruction);binary && binary->getType()->isHalfTy()) {
            auto* left=half_bits(binary->getOperand(0));auto* right=half_bits(binary->getOperand(1));
            if(binary->getOpcode()==llvm::Instruction::FSub)right=builder.CreateXor(right,builder.getInt32(0x8000));
            if(binary->getOpcode()==llvm::Instruction::FAdd || binary->getOpcode()==llvm::Instruction::FSub || binary->getOpcode()==llvm::Instruction::FMul || binary->getOpcode()==llvm::Instruction::FDiv)
                replacement=half_value(builder.CreateCall(numeric(binary->getOpcode()==llvm::Instruction::FDiv?"f16_div":binary->getOpcode()==llvm::Instruction::FMul?"f16_mul":"f16_add"),{left,right}));
        }
        if(auto* operation=llvm::dyn_cast<llvm::FCmpInst>(instruction);operation && operation->getOperand(0)->getType()->isHalfTy()) {
            if(!compare || compare->isDeclaration())throw std::runtime_error("missing half comparison helper");
            replacement=builder.CreateICmpNE(builder.CreateCall(compare,{bits(half_float(operation->getOperand(0))),bits(half_float(operation->getOperand(1))),builder.getInt32(operation->getPredicate())}),builder.getInt32(0));
        }
        if(instruction->getOpcode()==llvm::Instruction::FNeg && instruction->getType()->isDoubleTy()) replacement=builder.CreateBitCast(builder.CreateXor(bits(instruction->getOperand(0)),builder.getInt64(0x8000000000000000ull)),builder.getDoubleTy());
        if(auto* binary=llvm::dyn_cast<llvm::BinaryOperator>(instruction);binary && binary->getType()->isDoubleTy()) {
            const char* operation=binary->getOpcode()==llvm::Instruction::FAdd?"add":binary->getOpcode()==llvm::Instruction::FSub?"sub":binary->getOpcode()==llvm::Instruction::FMul?"mul":binary->getOpcode()==llvm::Instruction::FDiv?"div":nullptr;
            if(operation) if(auto* helper=numeric(std::string("f64_")+operation)) replacement=builder.CreateBitCast(builder.CreateCall(helper,{bits(binary->getOperand(0)),bits(binary->getOperand(1))}),builder.getDoubleTy());
        }
        if(auto* operation=llvm::dyn_cast<llvm::FCmpInst>(instruction);operation && operation->getOperand(0)->getType()->isDoubleTy())
            if(auto* helper=numeric("f64_compare")) replacement=builder.CreateICmpNE(builder.CreateCall(helper,{bits(operation->getOperand(0)),bits(operation->getOperand(1)),builder.getInt32(operation->getPredicate())}),builder.getInt32(0));
        if(auto* cast=llvm::dyn_cast<llvm::CastInst>(instruction)) {
            auto* source=cast->getOperand(0);
            if(cast->getOpcode()==llvm::Instruction::FPExt && source->getType()->isHalfTy() && cast->getType()->isFloatTy())replacement=half_float(source);
            if(cast->getOpcode()==llvm::Instruction::FPTrunc && source->getType()->isFloatTy() && cast->getType()->isHalfTy())
                replacement=half_value(builder.CreateCall(numeric("f32_to_f16"),{bits(source),builder.getInt32(0)}));
            if(cast->getOpcode()==llvm::Instruction::FPTrunc && source->getType()->isDoubleTy() && cast->getType()->isHalfTy())
                replacement=half_value(builder.CreateCall(numeric("f64_to_f16"),{bits(source),builder.getInt32(0)}));
            if(cast->getOpcode()==llvm::Instruction::FPExt && source->getType()->isHalfTy() && cast->getType()->isDoubleTy())
                replacement=builder.CreateBitCast(builder.CreateCall(numeric("f32_to_f64"),{builder.CreateBitCast(half_float(source),builder.getInt32Ty())}),builder.getDoubleTy());
            if((cast->getOpcode()==llvm::Instruction::SIToFP || cast->getOpcode()==llvm::Instruction::UIToFP) && cast->getType()->isHalfTy()) {
                bool unsign=cast->getOpcode()==llvm::Instruction::UIToFP;
                auto* wide=unsign?builder.CreateZExtOrTrunc(source,builder.getInt64Ty()):builder.CreateSExtOrTrunc(source,builder.getInt64Ty());
                auto* negative=unsign?builder.getFalse():builder.CreateICmpSLT(wide,builder.getInt64(0));
                replacement=half_value(builder.CreateCall(numeric("f16_round"),{builder.CreateSelect(negative,builder.CreateNeg(wide),wide),builder.getInt32(0),builder.CreateSelect(negative,builder.getInt32(0x8000),builder.getInt32(0)),builder.getInt32(0)}));
            }
            if((cast->getOpcode()==llvm::Instruction::FPToSI || cast->getOpcode()==llvm::Instruction::FPToUI) && source->getType()->isHalfTy())
                replacement=builder.CreateTruncOrBitCast(builder.CreateCall(numeric("f16_to_integer"),{half_bits(source),builder.getInt32(1),builder.getInt32(cast->getType()->getIntegerBitWidth()),builder.getInt32(cast->getOpcode()==llvm::Instruction::FPToUI)}),cast->getType());
            if(cast->getOpcode()==llvm::Instruction::FPExt && source->getType()->isFloatTy() && cast->getType()->isDoubleTy())
                if(auto* helper=numeric("f32_to_f64")) replacement=builder.CreateBitCast(builder.CreateCall(helper,{bits(source)}),builder.getDoubleTy());
            if(cast->getOpcode()==llvm::Instruction::FPTrunc && source->getType()->isDoubleTy() && cast->getType()->isFloatTy())
                if(auto* helper=numeric("f64_to_f32")) replacement=builder.CreateBitCast(builder.CreateCall(helper,{bits(source),builder.getInt32(0)}),builder.getFloatTy());
            if((cast->getOpcode()==llvm::Instruction::SIToFP || cast->getOpcode()==llvm::Instruction::UIToFP) && (cast->getType()->isFloatTy() || cast->getType()->isDoubleTy())) {
                auto signed_value=cast->getOpcode()==llvm::Instruction::SIToFP;
                auto* helper=numeric(std::string(signed_value?"i":"u")+"64_to_"+(cast->getType()->isDoubleTy()?"f64":"f32"));
                if(helper) {
                    auto* extended=signed_value?builder.CreateSExtOrTrunc(source,builder.getInt64Ty()):builder.CreateZExtOrTrunc(source,builder.getInt64Ty());
                    replacement=builder.CreateBitCast(builder.CreateCall(helper,{extended,builder.getInt32(0)}),cast->getType());
                }
            }
            if((cast->getOpcode()==llvm::Instruction::FPToSI || cast->getOpcode()==llvm::Instruction::FPToUI) && (source->getType()->isFloatTy() || source->getType()->isDoubleTy()))
                if(auto* helper=numeric(source->getType()->isDoubleTy()?"f64_to_i64":"f32_to_i64")) replacement=builder.CreateTruncOrBitCast(builder.CreateCall(helper,{bits(source),builder.getInt32(1)}),cast->getType());
        }
        // Only these mode helpers contain a guarded native division. Half
        // callers select normal-range operands; strict FP32 callers select
        // cvFloatDivide. Constant inlining removes the unused route.
        auto parent=instruction->getFunction()->getName();
        bool bounded_division=parent.starts_with("_ZN6cudavm14math_log2_modeEfj") || parent.starts_with("_ZN6cudavm14math_tanh_modeEfj") || parent.starts_with("_ZN6cudavm20math_reciprocal_modeEfj");
        if (divide && !divide->isDeclaration() && !bounded_division && instruction->getOpcode() == llvm::Instruction::FDiv && instruction->getType()->isFloatTy())
            replacement = builder.CreateCall(divide,{instruction->getOperand(0),instruction->getOperand(1)});
        if (compare && !compare->isDeclaration()) if (auto* operation = llvm::dyn_cast<llvm::FCmpInst>(instruction); operation && operation->getOperand(0)->getType()->isFloatTy())
            replacement = builder.CreateICmpNE(builder.CreateCall(compare,{builder.CreateBitCast(operation->getOperand(0),builder.getInt32Ty()),builder.CreateBitCast(operation->getOperand(1),builder.getInt32Ty()),builder.getInt32(operation->getPredicate())}),builder.getInt32(0));
        if (auto* call = llvm::dyn_cast<llvm::CallInst>(instruction); call && call->getCalledFunction()) {
            auto* function = call->getCalledFunction();
            auto name = function->hasFnAttribute("cudavm.cuda.origin") ? function->getFnAttribute("cudavm.cuda.origin").getValueAsString() : function->getName();
            if((name=="__cudavm_native_sin" || name=="__cudavm_native_cos") && call->arg_size()==1)
                if(auto* cast=llvm::dyn_cast<llvm::BitCastInst>(call->getArgOperand(0)))
                    if(auto* conversion=llvm::dyn_cast<llvm::CallInst>(cast->getOperand(0));conversion && conversion->getCalledFunction() && conversion->getCalledFunction()->getName()=="__cudavm_native_word_f16_to_f32") {
                        auto* helper=module.getFunction(name=="__cudavm_native_sin"?"__cudavm_native_half_sin":"__cudavm_native_half_cos");
                        if(!helper || helper->isDeclaration())throw std::runtime_error("missing bounded half trig helper");
                        replacement=builder.CreateCall(helper,{call->getArgOperand(0)});
                    }
            if(function->isIntrinsic() && function->getIntrinsicID()==llvm::Intrinsic::fma && call->getType()->isHalfTy())
                replacement=half_value(builder.CreateCall(numeric("f16_fma"),{half_bits(call->getArgOperand(0)),half_bits(call->getArgOperand(1)),half_bits(call->getArgOperand(2))}));
            if(integral && function->isIntrinsic() && call->getType()->isFloatTy()) {
                unsigned mode=0;
                switch(function->getIntrinsicID()) {
                case llvm::Intrinsic::floor:mode=1;break;
                case llvm::Intrinsic::ceil:mode=2;break;
                case llvm::Intrinsic::trunc:mode=0;break;
                case llvm::Intrinsic::round:mode=3;break;
                case llvm::Intrinsic::roundeven:case llvm::Intrinsic::rint:case llvm::Intrinsic::nearbyint:mode=4;break;
                default:mode=5;break;
                }
                if(mode<5) replacement=builder.CreateCall(integral,{call->getArgOperand(0),builder.getInt32(mode)});
            }
            if (square_root && !square_root->isDeclaration() && name == "llvm.nvvm.sqrt.rn.f") replacement = builder.CreateCall(square_root,{call->getArgOperand(0)});
            if (maximum && !maximum->isDeclaration() && name == "llvm.nvvm.fmax.f") replacement = builder.CreateCall(maximum,{call->getArgOperand(0),call->getArgOperand(1)});
            for(const auto& conversion:device_conversions) if(name==conversion.intrinsic) {
                auto real=conversion.real_bits==64?"f64":"f32";
                if(conversion.to_real) {
                    auto* integer=conversion.signed_value?builder.CreateSExtOrTrunc(call->getArgOperand(0),builder.getInt64Ty()):builder.CreateZExtOrTrunc(call->getArgOperand(0),builder.getInt64Ty());
                    if(auto* helper=numeric(std::string(conversion.signed_value?"i64_to_":"u64_to_")+real)) replacement=builder.CreateBitCast(builder.CreateCall(helper,{integer,builder.getInt32(conversion.mode)}),call->getType());
                } else if(auto* helper=numeric(std::string(real)+"_to_i64")) replacement=builder.CreateTruncOrBitCast(builder.CreateCall(helper,{bits(call->getArgOperand(0)),builder.getInt32(conversion.mode)}),call->getType());
            }
            for(const auto& conversion:double_conversions) if(name==conversion.intrinsic)
                if(auto* helper=numeric("f64_to_f32")) replacement=builder.CreateBitCast(builder.CreateCall(helper,{bits(call->getArgOperand(0)),builder.getInt32(conversion.mode)}),builder.getFloatTy());
        }
        if (replacement) { instruction->replaceAllUsesWith(replacement); instruction->eraseFromParent(); }
    }
}
// A closed SSA set of function pointers needs no device code addresses. Carry
// an integer discriminator through the same selects/PHIs and inline each arm.
bool lower_local_calls(llvm::Module& module) {
    std::vector<llvm::CallInst*> pending;
    for (auto& function : module) for (auto& block : function) for (auto& instruction : block)
        if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction); call && !call->getCalledFunction() && !call->isInlineAsm()) pending.push_back(call);
    bool changed = false;
    for (auto* call : pending) {
        std::set<llvm::Value*> visited;
        std::vector<llvm::Function*> targets;
        std::function<bool(llvm::Value*)> collect = [&](llvm::Value* value) {
            if (!visited.insert(value).second) return true;
            if (auto* target = llvm::dyn_cast<llvm::Function>(value)) {
                if (target->isDeclaration() || target->getFunctionType() != call->getFunctionType() || !llvm::isLegalToPromote(*call, target)) return false;
                if (std::find(targets.begin(), targets.end(), target) == targets.end()) targets.push_back(target);
                return true;
            }
            if (auto* select = llvm::dyn_cast<llvm::SelectInst>(value)) return collect(select->getTrueValue()) && collect(select->getFalseValue());
            if (auto* phi = llvm::dyn_cast<llvm::PHINode>(value)) return llvm::all_of(phi->incoming_values(), [&](auto& incoming) { return collect(incoming.get()); });
            return false;
        };
        if (!collect(call->getCalledOperand()) || targets.empty()) continue;
        std::map<llvm::Value*, llvm::Value*> tags;
        llvm::IRBuilder<> builder(call);
        std::function<llvm::Value*(llvm::Value*)> discriminator = [&](llvm::Value* value) -> llvm::Value* {
            if (auto found = tags.find(value); found != tags.end()) return found->second;
            if (auto* target = llvm::dyn_cast<llvm::Function>(value))
                return builder.getInt32(unsigned(std::find(targets.begin(), targets.end(), target) - targets.begin()));
            if (auto* phi = llvm::dyn_cast<llvm::PHINode>(value)) {
                auto* tag = llvm::PHINode::Create(builder.getInt32Ty(), phi->getNumIncomingValues(), "callee.tag", phi->getIterator());
                tags[value] = tag;
                for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) tag->addIncoming(discriminator(phi->getIncomingValue(i)), phi->getIncomingBlock(i));
                return tag;
            }
            auto* select = llvm::cast<llvm::SelectInst>(value);
            auto* yes = discriminator(select->getTrueValue()); auto* no = discriminator(select->getFalseValue());
            llvm::IRBuilder<> at(select);
            return tags[value] = at.CreateSelect(select->getCondition(), yes, no, "callee.tag");
        };
        auto* tag = discriminator(call->getCalledOperand());
        auto* before = call->getParent(); auto* function = before->getParent();
        auto* after = before->splitBasicBlock(call, "call.return"); before->getTerminator()->eraseFromParent();
        auto* invalid = llvm::BasicBlock::Create(module.getContext(), "call.invalid", function, after);
        llvm::IRBuilder<>(invalid).CreateUnreachable();
        builder.SetInsertPoint(before); auto* dispatch = builder.CreateSwitch(tag, invalid, unsigned(targets.size()));
        auto* result = call->getType()->isVoidTy() ? nullptr : llvm::PHINode::Create(call->getType(), unsigned(targets.size()), "call.result", call->getIterator());
        for (unsigned i = 0; i < targets.size(); ++i) {
            auto* arm = llvm::BasicBlock::Create(module.getContext(), "call.target", function, after);
            auto* direct = llvm::cast<llvm::CallInst>(call->clone()); direct->setCalledFunction(targets[i]); direct->setTailCallKind(llvm::CallInst::TCK_None);
            builder.SetInsertPoint(arm); builder.Insert(direct); builder.CreateBr(after);
            dispatch->addCase(builder.getInt32(i), arm); if (result) result->addIncoming(direct, arm);
        }
        if (result) call->replaceAllUsesWith(result);
        call->eraseFromParent(); changed = true;
        // Dead pointer PHI/select cycles are not trivially dead instruction
        // chains. Remove the closed dead set so no function addresses reach
        // target lowering; retain nodes with any remaining observable user.
        std::set<llvm::Instruction*> dead;
        for (auto* value : visited) if (auto* instruction = llvm::dyn_cast<llvm::Instruction>(value)) dead.insert(instruction);
        bool removed;
        do {
            removed = false;
            for (auto at = dead.begin(); at != dead.end();) {
                if (llvm::any_of((*at)->users(), [&](auto* user) { return !dead.contains(llvm::dyn_cast<llvm::Instruction>(user)); })) {
                    at = dead.erase(at); removed = true;
                } else ++at;
            }
        } while (removed);
        for (auto* instruction : dead) instruction->dropAllReferences();
        for (auto* instruction : dead) instruction->eraseFromParent();
    }
    return changed;
}
void optimize(llvm::Module& module) {
    llvm::LoopAnalysisManager loops; llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager calls; llvm::ModuleAnalysisManager modules;
    llvm::PassBuilder pb;
    pb.registerModuleAnalyses(modules); pb.registerCGSCCAnalyses(calls);
    pb.registerFunctionAnalyses(functions); pb.registerLoopAnalyses(loops);
    pb.crossRegisterProxies(loops, functions, calls, modules);
    for (auto& function : module) function.removeFnAttr(llvm::Attribute::OptimizeNone);
    llvm::ModulePassManager tail_calls;
    tail_calls.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::TailCallElimPass()));
    tail_calls.run(module, modules);
    // Do not partially inline recursive SCCs. Inliner use-list order otherwise
    // changes the residual call graph after an IR serialization round trip.
    llvm::CallGraph graph(module);
    std::set<llvm::Function*> recursive;
    for (auto component = llvm::scc_begin(&graph); !component.isAtEnd(); ++component)
        if (component.hasCycle()) for (auto* node : *component)
            if (auto* function = node->getFunction()) recursive.insert(function);
    for (auto& f : module) if (!f.isDeclaration()) {
        f.removeFnAttr(llvm::Attribute::OptimizeNone); f.removeFnAttr(llvm::Attribute::NoInline);
        if(shader_native_helper(f) || recursive.contains(&f)) {
            f.removeFnAttr(llvm::Attribute::AlwaysInline); f.addFnAttr(llvm::Attribute::NoInline);
        } else if (!f.hasFnAttribute("cudavm.kernel") && !f.hasFnAttribute("hlsl.shader") && f.getCallingConv() != llvm::CallingConv::PTX_Kernel && f.getCallingConv() != llvm::CallingConv::SPIR_KERNEL)
            f.addFnAttr(llvm::Attribute::AlwaysInline);
    }
    llvm::ModulePassManager passes;
    passes.addPass(llvm::AlwaysInlinerPass());
    llvm::FunctionPassManager fp;
    fp.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG)); fp.addPass(llvm::PromotePass());
    fp.addPass(llvm::InstCombinePass()); fp.addPass(llvm::SimplifyCFGPass());
    llvm::ScalarizerPassOptions scalar; scalar.ScalarizeLoadStore = true;
    fp.addPass(llvm::ScalarizerPass(scalar));
    // Scalarization exposes copies and bitfields in CUDA group objects. Fold
    // their initialized tags before lowering a generic group's dispatch, so a
    // legal tile does not retain unreachable grid/abort implementations.
    fp.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));fp.addPass(llvm::PromotePass());
    fp.addPass(llvm::InstCombinePass());fp.addPass(llvm::SCCPPass());fp.addPass(llvm::SimplifyCFGPass());
    passes.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fp)));
    passes.run(module, modules);
}
struct lowering {
    llvm::Module& source;
    llvm::Function& kernel;
    const llvm::DataLayout& dl;
    std::unique_ptr<llvm::Module> result;
    llvm::IRBuilder<> b;
    llvm::Function *entry = nullptr, *resolve = nullptr, *read = nullptr, *write = nullptr;
    llvm::Function *output_read = nullptr, *output_write = nullptr;
    std::map<llvm::Function*, llvm::Value*> output_routes;
    llvm::GlobalVariable* shared = nullptr;
    llvm::Constant* names[packet_resource_count]{};
    std::map<const llvm::Value*, llvm::Value*> values;
    std::map<const llvm::BasicBlock*, llvm::BasicBlock*> blocks;
    std::map<const llvm::BasicBlock*, llvm::BasicBlock*> tails;
    std::map<const llvm::GlobalVariable*, uint32_t> globals, shared_offsets;
    std::map<const llvm::AllocaInst*, uint32_t> local_offsets;
    std::map<const llvm::Argument*, uint32_t> byval_offsets;
    llvm::GlobalVariable* local_storage = nullptr;
    llvm::AllocaInst* opencl_local = nullptr;
    llvm::AllocaInst* stack_top = nullptr;
    std::map<llvm::Function*, llvm::Value*> invocation_stack_bases;
    uint32_t local_bytes = 0;
    bool arena_local = false;
    bool partial_stores = false;
    bool dynamic_shared = false, collective_storage = false, named_barriers = false;
    llvm::GlobalVariable* named_barrier_state = nullptr;
    llvm::GlobalVariable* collective_data = nullptr;
    llvm::GlobalVariable *collective_pc = nullptr, *collective_masks = nullptr, *collective_control = nullptr;
    llvm::GlobalVariable* collective_frames = nullptr;
    std::map<unsigned,llvm::Function*> collective_helpers;
    std::map<std::tuple<unsigned,unsigned,unsigned,unsigned,unsigned>, llvm::Function*> atomic_helpers;
    uint32_t shared_bytes = 0;
    bool opencl;
    bool dxil;
    std::array<unsigned,3> group;
    bool kernel_entry;
    bool assert_cancellation = false;
    unsigned service_shared_words;
    lowering(llvm::Module& module, llvm::Function& f, shader_environment environment, const std::array<unsigned,3>& block, unsigned shared_words = 0) : source(module), kernel(f), dl(module.getDataLayout()),
        result(std::make_unique<llvm::Module>("cudavm.native.shader." + f.getName().str(), module.getContext())), b(module.getContext()),
        opencl(environment == shader_environment::opencl), dxil(environment == shader_environment::dxil), group(block),
        kernel_entry(f.hasFnAttribute("cudavm.kernel") || f.getCallingConv()==llvm::CallingConv::PTX_Kernel), service_shared_words(shared_words) {}
    [[noreturn]] void fail(const llvm::Value& value, const char* reason) {
        std::string text; llvm::raw_string_ostream out(text); value.print(out);
        throw std::runtime_error(std::string("native ") + (dxil ? "DXIL" : "SPIR-V") + " kernel " + kernel.getName().str() + ": " + reason + "\n" + text);
    }
    llvm::Type* type(llvm::Type* input) {
        if (input->isPointerTy()) return b.getInt64Ty();
        if(input->isFloatTy()) return b.getInt32Ty();
        if(input->isDoubleTy()) return b.getInt64Ty();
        if(input->isHalfTy()) return b.getInt32Ty();
        if (auto* integer = llvm::dyn_cast<llvm::IntegerType>(input)) {
            auto bits = integer->getBitWidth();
            if (bits > 1 && bits < 32) return b.getInt32Ty();
            if (bits > 32 && bits < 64) return b.getInt64Ty();
        }
        if (auto* structure = llvm::dyn_cast<llvm::StructType>(input)) {
            std::vector<llvm::Type*> elements; for (auto* element : structure->elements()) elements.push_back(type(element));
            return llvm::StructType::get(b.getContext(), elements, structure->isPacked());
        }
        return input;
    }
    llvm::Value* canonical(llvm::Value* output, llvm::Type* source_type) {
        if (auto* integer = llvm::dyn_cast<llvm::IntegerType>(source_type)) {
            auto bits = integer->getBitWidth(), storage = output->getType()->getIntegerBitWidth();
            if (bits > 1 && bits < storage)
                return b.CreateAnd(output, llvm::ConstantInt::get(output->getType(), llvm::APInt::getLowBitsSet(storage, bits)));
        }
        return output;
    }
    llvm::Value* signed_value(const llvm::Value* input) {
        auto* output = value(input); auto bits = input->getType()->isPointerTy() ? 64u : input->getType()->getIntegerBitWidth();
        auto storage = output->getType()->getIntegerBitWidth();
        if (bits > 1 && bits < storage) {
            auto* shift = llvm::ConstantInt::get(output->getType(), storage - bits);
            return b.CreateAShr(b.CreateShl(output, shift), shift);
        }
        return output;
    }
    llvm::Type* resource_pointer_type() { return llvm::PointerType::get(b.getContext(), 1); }
    std::vector<llvm::Type*> arguments(std::initializer_list<llvm::Type*> base) {
        std::vector<llvm::Type*> args(base);
        if (opencl) for (unsigned i=0;i<packet_resource_count;++i) args.push_back(resource_pointer_type());
        return args;
    }
    llvm::Value* call_helper(llvm::Function* function, std::initializer_list<llvm::Value*> base) {
        auto first = base.begin();
        if (arena_local && (function == read || function == write)) {
            auto known = llvm::computeKnownBits(*first, result->getDataLayout());
            llvm::APInt mask(64, 0xffffffffc0000000ull), local(64, 0xffffffff80000000ull);
            if ((known.One & mask) == local && (known.Zero & mask) == (mask ^ local)) {
                // Proven private addresses need neither allocation lookup nor
                // resource dispatch. Preserve the ordinary private word/mask
                // semantics while binding the invocation's stack directly.
                auto* offset = b.CreateAnd(b.CreateTrunc(*first, b.getInt32Ty()), b.getInt32(0x3fffffffu));
                offset = b.CreateURem(b.CreateLShr(offset, b.getInt32(2)), b.getInt32(std::max(2u, (local_bytes + 3) / 4)));
                auto* pointer = stack_word(nullptr, offset);
                auto* previous = b.CreateAlignedLoad(b.getInt32Ty(), pointer, llvm::Align(4));
                if (function == read) return previous;
                return b.CreateAlignedStore(b.CreateOr(b.CreateAnd(previous, b.CreateNot(*(first + 2))), b.CreateAnd(*(first + 1), *(first + 2))), pointer, llvm::Align(4));
            }
        }
        if (function == read || function == write) {
            auto known = llvm::computeKnownBits(*first, result->getDataLayout());
            llvm::APInt mask(64, 0xffffffff00000000ull), tag(64, device_output_address);
            if ((known.One & mask) == tag && (known.Zero & mask) == (mask ^ tag)) {
                auto*& helper = function == read ? output_read : output_write;
                if (!helper) helper = make_output_access(function == write);
                function = helper;
            }
        }
        std::vector<llvm::Value*> args{b.CreateTrunc(*first, b.getInt32Ty()), b.CreateTrunc(b.CreateLShr(*first, b.getInt64(32)), b.getInt32Ty())};
        args.insert(args.end(), first + 1, base.end());
        if (function->hasFnAttribute("cudavm.native.local")) {
            auto* caller = b.GetInsertBlock()->getParent();
            if (caller == entry) args.push_back(b.CreateInBoundsGEP(opencl_local->getAllocatedType(),opencl_local,{b.getInt32(0),b.getInt32(0)}));
            else args.push_back(caller->getArg(unsigned(caller->arg_size()-packet_resource_count-1)));
        }
        if (function->hasFnAttribute("cudavm.native.output")) {
            auto* caller = b.GetInsertBlock()->getParent();
            auto [at, inserted] = output_routes.try_emplace(caller);
            if (inserted) {
                auto saved = b.saveIP(); b.SetInsertPoint(&caller->getEntryBlock(), caller->getEntryBlock().getFirstInsertionPt());
                at->second = b.CreateAlignedLoad(b.getInt32Ty(), word_pointer(0, b.getInt32(7)), llvm::Align(4)); b.restoreIP(saved);
            }
            args.push_back(at->second);
        }
        if (opencl) {
            auto* caller = b.GetInsertBlock()->getParent();
            for (unsigned i=0;i<packet_resource_count;++i)
                args.push_back(caller->getArg(unsigned(caller->arg_size() - packet_resource_count + i)));
        }
        return b.CreateCall(function, args);
    }
    llvm::Function* make_output_access(bool store) {
        auto saved = b.saveIP();
        auto signature = store ? arguments({b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty()})
            : arguments({b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty()});
        auto* function = llvm::Function::Create(llvm::FunctionType::get(store ? b.getVoidTy() : b.getInt32Ty(), signature, false),
            llvm::GlobalValue::InternalLinkage, store ? "cudavm.output.store" : "cudavm.output.load", *result);
        function->addFnAttr("cudavm.native.output");
        b.SetInsertPoint(llvm::BasicBlock::Create(b.getContext(), "entry", function));
        auto* route = function->getArg(store ? 4 : 2);
        auto* offset = b.CreateAdd(b.CreateShl(b.CreateZExt(b.CreateAnd(route, b.getInt32(0x3fffffffu)), b.getInt64Ty()), b.getInt64(2)),
            b.CreateZExt(function->getArg(0), b.getInt64Ty()));
        auto* physical = b.CreateOr(offset, b.CreateShl(b.CreateZExt(b.CreateLShr(route, b.getInt32(30)), b.getInt64Ty()), b.getInt64(packet_binding_shift)));
        access_resolved(physical, [&](auto*, const auto& pointer) {
            auto* at = pointer(); auto* previous = b.CreateAlignedLoad(b.getInt32Ty(), at, llvm::Align(4));
            if (store) {
                // A published FIFO record has one owner until its release.
                // Its byte updates need no shared-word CAS or address lookup.
                b.CreateAlignedStore(b.CreateOr(b.CreateAnd(previous, b.CreateNot(function->getArg(3))),
                    b.CreateAnd(function->getArg(2), function->getArg(3))), at, llvm::Align(4)); b.CreateRetVoid();
            } else b.CreateRet(previous);
        });
        b.restoreIP(saved); return function;
    }
    llvm::Value* word_pointer(unsigned binding, llvm::Value* index) {
        if (dxil) {
            auto* resource = llvm::TargetExtType::get(b.getContext(), "dx.RawBuffer", {b.getInt8Ty()}, {binding == 1 ? 0u : 1u, 0});
            auto* handle = b.CreateIntrinsic(resource, llvm::Intrinsic::dx_resource_handlefrombinding,
                {b.getInt32(0), b.getInt32(binding < 2 ? 0 : binding-1), b.getInt32(1), b.getInt32(0), llvm::ConstantPointerNull::get(b.getPtrTy())});
            return b.CreateIntrinsic(b.getPtrTy(), llvm::Intrinsic::dx_resource_getpointer, {handle, b.CreateMul(b.CreateZExtOrTrunc(index, b.getInt32Ty()), b.getInt32(4))});
        }
        if (opencl) {
            auto* function = b.GetInsertBlock()->getParent();
            return b.CreateGEP(b.getInt32Ty(), function->getArg(unsigned(function->arg_size() - packet_resource_count + binding)), index);
        }
        auto* buffer = llvm::TargetExtType::get(b.getContext(), "spirv.VulkanBuffer", {llvm::ArrayType::get(b.getInt32Ty(), 0)}, {12, 1});
        if (!names[binding]) names[binding] = b.CreateGlobalString("cudavm.resource." + std::to_string(binding));
        auto* handle = b.CreateIntrinsic(buffer, llvm::Intrinsic::spv_resource_handlefrombinding,
            {b.getInt32(0), b.getInt32(binding), b.getInt32(1), b.getInt32(0), names[binding]});
        return b.CreateIntrinsic(llvm::PointerType::get(b.getContext(), 11), llvm::Intrinsic::spv_resource_getpointer,
            {handle, b.CreateZExtOrTrunc(index, b.getInt32Ty())});
    }
    llvm::Value* packet(llvm::Value* index) { return b.CreateAlignedLoad(b.getInt32Ty(), word_pointer(1, index), llvm::Align(4)); }
    llvm::Value* pair(llvm::Value* lo, llvm::Value* hi) {
        return b.CreateOr(b.CreateZExt(lo, b.getInt64Ty()), b.CreateShl(b.CreateZExt(hi, b.getInt64Ty()), b.getInt64(32)));
    }
    llvm::Value* packet64(llvm::Value* word) { return pair(packet(word), packet(b.CreateAdd(word, b.getInt32(1)))); }
    llvm::Value* shared_pointer(llvm::Value* offset) {
        auto words=llvm::cast<llvm::ArrayType>(shared->getValueType())->getNumElements();
        // Other tagged address spaces also pass through this helper's CFG.
        // Keep their unreachable indices legal for GLSL's static array checks.
        auto* index=b.CreateURem(b.CreateLShr(offset,b.getInt32(2)),b.getInt32(uint32_t(words)));
        return b.CreateInBoundsGEP(shared->getValueType(), shared, {b.getInt32(0),index});
    }
    // Branch on resource identity before forming pointers. No pointer PHI or
    // Vulkan variable-pointer capability is needed for native resource access.
    void access_resolved(llvm::Value* physical, const std::function<void(llvm::BasicBlock*, const std::function<llvm::Value*()>&)>& access) {
        auto* function = b.GetInsertBlock()->getParent();
        auto* offset = b.CreateAnd(physical, b.getInt64(packet_offset_mask));
        auto* binding = b.CreateTrunc(b.CreateLShr(physical, b.getInt64(packet_binding_shift)), b.getInt32Ty());
        auto* invalid = llvm::BasicBlock::Create(b.getContext(), "resource.invalid", function);
        auto* dispatch = b.CreateSwitch(binding, invalid, packet_direct_bindings+1);
        for (unsigned i=0;i<=packet_direct_bindings;++i) {
            auto* block = llvm::BasicBlock::Create(b.getContext(), "resource.access", function);
            dispatch->addCase(b.getInt32(i), block); b.SetInsertPoint(block);
            access(block, [&, i] { return word_pointer(i ? i+1 : 0, b.CreateLShr(offset, b.getInt64(2))); });
        }
        b.SetInsertPoint(invalid);
        if (function->getReturnType()->isVoidTy()) b.CreateRetVoid();
        else b.CreateRet(llvm::Constant::getNullValue(function->getReturnType()));
    }
    void make_resolver() {
        resolve = llvm::Function::Create(llvm::FunctionType::get(b.getInt64Ty(), arguments({b.getInt32Ty(), b.getInt32Ty()}), false), llvm::GlobalValue::InternalLinkage, "cudavm.resolve.address", *result);
        auto* first = llvm::BasicBlock::Create(b.getContext(), "entry", resolve);
        auto* allocations = llvm::BasicBlock::Create(b.getContext(), "lookup", resolve);
        auto* service = llvm::BasicBlock::Create(b.getContext(), "service", resolve);
        auto* loop = llvm::BasicBlock::Create(b.getContext(), "allocations", resolve);
        auto* test = llvm::BasicBlock::Create(b.getContext(), "range", resolve);
        auto* next = llvm::BasicBlock::Create(b.getContext(), "next", resolve);
        auto* found = llvm::BasicBlock::Create(b.getContext(), "found", resolve);
        auto* absent = llvm::BasicBlock::Create(b.getContext(), "absent", resolve);
        b.SetInsertPoint(first);
        auto* address = pair(resolve->getArg(0), resolve->getArg(1));
        auto* heap = b.CreateICmpEQ(resolve->getArg(1), b.getInt32(uint32_t(device_heap_address >> 32)));
        auto* output = b.CreateICmpEQ(resolve->getArg(1), b.getInt32(uint32_t(device_output_address >> 32)));
        b.CreateCondBr(b.CreateOr(heap, output), service, allocations);
        b.SetInsertPoint(service);
        auto* route = b.CreateAlignedLoad(b.getInt32Ty(), word_pointer(0, b.CreateSelect(heap, b.getInt32(6), b.getInt32(7))), llvm::Align(4));
        auto* byte_offset = b.CreateAdd(b.CreateShl(b.CreateZExt(b.CreateAnd(route, b.getInt32(0x3fffffffu)), b.getInt64Ty()), b.getInt64(2)), b.CreateZExt(resolve->getArg(0), b.getInt64Ty()));
        b.CreateRet(b.CreateOr(byte_offset, b.CreateShl(b.CreateZExt(b.CreateLShr(route, b.getInt32(30)), b.getInt64Ty()), b.getInt64(packet_binding_shift))));
        b.SetInsertPoint(allocations);
        auto* table = b.CreateLShr(packet(b.getInt32(6)), b.getInt32(2));
        auto* count = packet(b.getInt32(7)); b.CreateBr(loop);
        b.SetInsertPoint(loop);
        auto* low = b.CreatePHI(b.getInt32Ty(), 2); low->addIncoming(b.getInt32(0), allocations);
        auto* high = b.CreatePHI(b.getInt32Ty(), 2); high->addIncoming(count, allocations);
        b.CreateCondBr(b.CreateICmpULT(low, high), test, absent);
        b.SetInsertPoint(test);
        auto* middle = b.CreateAdd(low, b.CreateLShr(b.CreateSub(high, low), b.getInt32(1)));
        auto* row = b.CreateAdd(table, b.CreateMul(middle, b.getInt32(packet_allocation_words)));
        auto* base = packet64(row); auto* delta = b.CreateSub(address, base);
        auto* size = packet64(b.CreateAdd(row, b.getInt32(2)));
        b.CreateCondBr(b.CreateAnd(b.CreateICmpUGE(address, base), b.CreateICmpULT(delta, size)), found, next);
        b.SetInsertPoint(next);
        auto* before = b.CreateICmpULT(address, base);
        low->addIncoming(b.CreateSelect(before, low, b.CreateAdd(middle, b.getInt32(1))), next);
        high->addIncoming(b.CreateSelect(before, middle, high), next); b.CreateBr(loop);
        b.SetInsertPoint(found);
        auto* offset = b.CreateAdd(packet64(b.CreateAdd(row, b.getInt32(4))), delta);
        auto* binding = b.CreateZExt(packet(b.CreateAdd(row, b.getInt32(6))), b.getInt64Ty());
        b.CreateRet(b.CreateOr(offset, b.CreateShl(binding, b.getInt64(packet_binding_shift))));
        b.SetInsertPoint(absent); b.CreateRet(b.getInt64(0));
    }
    llvm::Function* make_access(bool store) {
        auto base = store ? std::vector<llvm::Type*>{b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty()}
            : std::vector<llvm::Type*>{b.getInt32Ty(),b.getInt32Ty()};
        if (opencl && local_bytes && !arena_local) base.push_back(b.getPtrTy());
        if (opencl) for (unsigned i=0;i<packet_resource_count;++i) base.push_back(resource_pointer_type());
        auto* function = llvm::Function::Create(llvm::FunctionType::get(store ? b.getVoidTy() : b.getInt32Ty(),
            base, false),
            llvm::GlobalValue::InternalLinkage, store ? "cudavm.store.word" : "cudavm.load.word", *result);
        if (opencl && local_bytes && !arena_local) function->addFnAttr("cudavm.native.local");
        auto* first = llvm::BasicBlock::Create(b.getContext(), "entry", function);
        auto* private_space = llvm::BasicBlock::Create(b.getContext(), "private.space", function);
        auto* global_space = llvm::BasicBlock::Create(b.getContext(), "global", function);
        auto* shared_space = llvm::BasicBlock::Create(b.getContext(), "shared", function);
        auto* parameter_space = llvm::BasicBlock::Create(b.getContext(), "parameter", function);
        auto* local_space = local_bytes ? llvm::BasicBlock::Create(b.getContext(), "local", function) : nullptr;
        auto* invalid = llvm::BasicBlock::Create(b.getContext(), "invalid", function);
        b.SetInsertPoint(first);
        auto* lo = function->getArg(0); auto* hi = function->getArg(1); auto* address = pair(lo, hi);
        b.CreateCondBr(b.CreateICmpEQ(hi, b.getInt32(0xffffffffu)), private_space, global_space);
        b.SetInsertPoint(private_space); auto* tag = b.CreateAnd(lo, b.getInt32(0xc0000000u));
        auto* dispatch = b.CreateSwitch(tag, invalid, 3); dispatch->addCase(b.getInt32(0x40000000u), shared_space); dispatch->addCase(b.getInt32(0xc0000000u), parameter_space); if (local_space) dispatch->addCase(b.getInt32(0x80000000u), local_space);
        auto access = [&](llvm::BasicBlock* block, const std::function<llvm::Value*()>& pointer) {
            b.SetInsertPoint(block);
            if (store && !partial_stores) { b.CreateAlignedStore(function->getArg(2), pointer(), llvm::Align(4),true); b.CreateRetVoid(); }
            else if (store) {
                auto* whole = llvm::BasicBlock::Create(b.getContext(), "whole.word", function);
                auto* partial = llvm::BasicBlock::Create(b.getContext(), "partial.word", function);
                auto* retry = llvm::BasicBlock::Create(b.getContext(), "retry", function);
                auto* done = llvm::BasicBlock::Create(b.getContext(), "done", function);
                auto* mask = function->getArg(3);
                b.CreateCondBr(b.CreateICmpEQ(mask, b.getInt32(0xffffffffu)), whole, partial);
                b.SetInsertPoint(whole); b.CreateAlignedStore(function->getArg(2), pointer(), llvm::Align(4),true); b.CreateRetVoid();
                b.SetInsertPoint(partial);
                auto scope = dxil?llvm::SyncScope::System:b.getContext().getOrInsertSyncScopeID("device");
                auto* initial = b.CreateExtractValue(b.CreateAtomicCmpXchg(pointer(),b.getInt32(0),b.getInt32(0),llvm::Align(4),llvm::AtomicOrdering::Monotonic,llvm::AtomicOrdering::Monotonic,scope),0);
                b.CreateBr(retry);
                b.SetInsertPoint(retry);
                auto* expected = b.CreatePHI(b.getInt32Ty(), 2); expected->addIncoming(initial, partial);
                auto* desired = b.CreateOr(b.CreateAnd(expected, b.CreateNot(mask)), b.CreateAnd(function->getArg(2), mask));
                auto* exchange = b.CreateAtomicCmpXchg(pointer(), expected, desired, llvm::Align(4), llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering::Monotonic, scope);
                expected->addIncoming(b.CreateExtractValue(exchange, 0), retry);
                b.CreateCondBr(b.CreateExtractValue(exchange, 1), done, retry);
                b.SetInsertPoint(done); b.CreateRetVoid();
            }
            else b.CreateRet(b.CreateAlignedLoad(b.getInt32Ty(), pointer(), llvm::Align(4),true));
        };
        b.SetInsertPoint(global_space); access_resolved(call_helper(resolve, {address}), access);
        access(shared_space, [&] { return shared_pointer(b.CreateAnd(lo, b.getInt32(0x3fffffffu))); });
        if (local_space) {
          b.SetInsertPoint(local_space);
          auto* offset = b.CreateURem(b.CreateLShr(b.CreateAnd(lo,b.getInt32(0x3fffffffu)),b.getInt32(2)),b.getInt32(std::max(2u,(local_bytes+3)/4)));
          llvm::Value* ptr = nullptr;
          if (arena_local) {
              ptr = stack_word(nullptr, offset);
          } else ptr = opencl ? b.CreateInBoundsGEP(b.getInt32Ty(),function->getArg(store ? 4 : 2),offset)
              : b.CreateInBoundsGEP(local_storage->getValueType(),local_storage,{b.getInt32(0),offset});
          if (store) {
            auto* previous = b.CreateAlignedLoad(b.getInt32Ty(), ptr, llvm::Align(4));
            auto* mask = function->getArg(3);
            b.CreateAlignedStore(b.CreateOr(b.CreateAnd(previous, b.CreateNot(mask)), b.CreateAnd(function->getArg(2), mask)), ptr, llvm::Align(4));
            b.CreateRetVoid();
          } else b.CreateRet(b.CreateAlignedLoad(b.getInt32Ty(), ptr, llvm::Align(4)));
        }
        if (!store) access(parameter_space, [&] { return word_pointer(1, b.CreateAdd(b.getInt32(8), b.CreateLShr(b.CreateAnd(lo, b.getInt32(0x3fffffffu)), b.getInt32(2)))); });
        else { b.SetInsertPoint(parameter_space); b.CreateBr(invalid); }
        b.SetInsertPoint(invalid); if (store) b.CreateRetVoid(); else b.CreateRet(b.getInt32(0));
        return function;
    }
    llvm::Value* allocate_local(const llvm::AllocaInst& allocation) {
        if (allocation.getArraySize()->getType()->getIntegerBitWidth() > 64)
            fail(allocation, "invocation-private allocation count exceeds 64 bits");
        auto alignment = allocation.getAlign().value();
        if (alignment > shader_local_capacity) fail(allocation, "invocation-private alignment exceeds 16 KiB");
        auto size = dl.getTypeAllocSize(allocation.getAllocatedType()).getFixedValue();
        auto* count = b.CreateZExtOrTrunc(value(allocation.getArraySize()), b.getInt64Ty());
        auto* top = b.CreateZExt(b.CreateLoad(b.getInt32Ty(), stack_top), b.getInt64Ty());
        auto* aligned = b.CreateAnd(b.CreateAdd(top, b.getInt64(alignment - 1)), b.getInt64(~uint64_t(alignment - 1)));
        auto* fits = b.CreateICmpULE(aligned, b.getInt64(shader_local_capacity));
        if (size) {
            // Check before multiplying, so a wrapped size cannot alias a frame.
            auto* available = b.CreateSub(b.getInt64(shader_local_capacity), b.CreateSelect(fits, aligned, b.getInt64(shader_local_capacity)));
            fits = b.CreateAnd(fits, b.CreateICmpULE(count, b.CreateUDiv(available, b.getInt64(size))));
        }
        auto* end = b.CreateAdd(aligned, b.CreateMul(count, b.getInt64(size)));
        b.CreateStore(b.CreateSelect(fits, b.CreateTrunc(end, b.getInt32Ty()), b.getInt32(shader_local_capacity + 1)), stack_top);
        auto scope = dxil ? llvm::SyncScope::System : b.getContext().getOrInsertSyncScopeID("device");
        b.CreateAtomicRMW(llvm::AtomicRMWInst::Or, word_pointer(0, b.getInt32(packet_stack_fault_word)), b.CreateZExt(b.CreateNot(fits), b.getInt32Ty()),
            llvm::Align(4), llvm::AtomicOrdering::Monotonic, scope);
        return b.CreateSelect(fits, b.CreateAdd(b.getInt64(0xffffffff80000000ull), aligned), b.getInt64(0xffffffff00000000ull));
    }
    llvm::Value* stack_word(llvm::Value* thread, llvm::Value* offset) {
        auto* function = b.GetInsertBlock()->getParent();
        if (!thread) if (auto found = invocation_stack_bases.find(function); found != invocation_stack_bases.end())
            return word_pointer(0, b.CreateAdd(found->second, offset));
        auto saved = b.saveIP();
        bool invocation = !thread;
        if (invocation) b.SetInsertPoint(&function->getEntryBlock(), function->getEntryBlock().getFirstInsertionPt());
        auto index = [&](bool block, unsigned axis) {
            auto intrinsic = dxil ? llvm::Intrinsic::ID(block ? llvm::Intrinsic::dx_group_id : llvm::Intrinsic::dx_thread_id_in_group)
                : llvm::Intrinsic::ID(block ? llvm::Intrinsic::spv_group_id : llvm::Intrinsic::spv_thread_id_in_group);
            return b.CreateZExtOrTrunc(b.CreateIntrinsic(opencl ? b.getInt64Ty() : b.getInt32Ty(), intrinsic, {b.getInt32(axis)}), b.getInt32Ty());
        };
        auto* block = b.CreateAdd(index(true,0), b.CreateMul(packet(b.getInt32(0)), b.CreateAdd(index(true,1), b.CreateMul(packet(b.getInt32(1)), index(true,2)))));
        if (!thread) thread = linear_thread();
        auto* linear = b.CreateAdd(b.CreateMul(block, thread_count()), thread);
        llvm::Value* base = b.CreateLoad(b.getInt32Ty(), word_pointer(0, b.getInt32(packet_stack_base_word)));
        llvm::Value* stride = b.getInt32(shader_local_capacity / 4);
        if (kernel.hasFnAttribute("cudavm.service.storage")) {
            stride = b.CreateShl(b.getInt32(1), b.CreateAnd(base, b.getInt32(15)));
            base = b.CreateLShr(b.CreateAnd(base, b.getInt32(~15u)), b.getInt32(2));
        }
        auto* first = b.CreateAdd(base, b.CreateMul(linear, stride));
        if (invocation) { invocation_stack_bases[function] = first; b.restoreIP(saved); }
        return word_pointer(0, b.CreateAdd(first, offset));
    }
    llvm::Value* gep(const llvm::GEPOperator& pointer) {
        llvm::SmallMapVector<llvm::Value*, llvm::APInt, 4> variables; llvm::APInt constant(64, 0);
        if (!pointer.collectOffset(dl, 64, variables, constant)) fail(pointer, "nonlinear pointer offset");
        auto* offset = static_cast<llvm::Value*>(llvm::ConstantInt::get(b.getInt64Ty(), constant));
        for (auto& [index, scale] : variables)
            offset = b.CreateAdd(offset, b.CreateMul(b.CreateSExtOrTrunc(signed_value(index), b.getInt64Ty()), llvm::ConstantInt::get(b.getInt64Ty(), scale)));
        return b.CreateAdd(value(pointer.getPointerOperand()), offset);
    }
    llvm::Value* value(const llvm::Value* input) {
        if (auto found = values.find(input); found != values.end()) return found->second;
        if (auto* global = llvm::dyn_cast<llvm::GlobalVariable>(input)) {
            if (auto at = shared_offsets.find(global); at != shared_offsets.end()) return b.getInt64(0xffffffff40000000ull + at->second);
            if (auto at = globals.find(global); at != globals.end()) {
                auto* start = b.CreateAdd(b.CreateLShr(packet(b.getInt32(6)), b.getInt32(2)), b.CreateMul(packet(b.getInt32(7)), b.getInt32(packet_allocation_words)));
                return packet64(b.CreateAdd(start, b.getInt32(at->second * 2)));
            }
            fail(*input, "unavailable global storage");
        }
        if (auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(input)) {
            if (auto* pointer = llvm::dyn_cast<llvm::GEPOperator>(expression)) return gep(*pointer);
            if (expression->getOpcode() == llvm::Instruction::AddrSpaceCast || expression->getOpcode() == llvm::Instruction::BitCast) return value(expression->getOperand(0));
            if (expression->getOpcode() == llvm::Instruction::PtrToInt) {
                auto* address = value(expression->getOperand(0));
                if (expression->getOperand(0)->getType()->getPointerAddressSpace() == 3 || expression->getOperand(0)->getType()->getPointerAddressSpace() == 5)
                    address = b.CreateAnd(address, b.getInt64(0x3fffffffull));
                return canonical(b.CreateZExtOrTrunc(address, type(expression->getType())), expression->getType());
            }
            if (expression->getOpcode() == llvm::Instruction::IntToPtr) {
                auto* address = b.CreateZExtOrTrunc(value(expression->getOperand(0)), b.getInt64Ty());
                if (auto space = expression->getType()->getPointerAddressSpace(); space == 3 || space == 5)
                    address = b.CreateOr(address, b.getInt64(space == 3 ? 0xffffffff40000000ull : 0xffffffff80000000ull));
                return address;
            }
            if (expression->getOpcode() == llvm::Instruction::Add || expression->getOpcode() == llvm::Instruction::Sub) {
                auto* left = b.CreateZExtOrTrunc(value(expression->getOperand(0)), type(expression->getType()));
                auto* right = b.CreateZExtOrTrunc(value(expression->getOperand(1)), type(expression->getType()));
                auto* result = expression->getOpcode() == llvm::Instruction::Add ? b.CreateAdd(left, right) : b.CreateSub(left, right);
                return canonical(result, expression->getType());
            }
            fail(*input, "unsupported constant expression");
        }
        if (llvm::isa<llvm::ConstantPointerNull>(input)) return b.getInt64(0);
        if (auto* integer = llvm::dyn_cast<llvm::ConstantInt>(input)) return llvm::ConstantInt::get(type(integer->getType()), integer->getValue().zextOrTrunc(type(integer->getType())->getIntegerBitWidth()));
        if(auto* floating=llvm::dyn_cast<llvm::ConstantFP>(input);floating && (floating->getType()->isDoubleTy() || floating->getType()->isFloatTy() || floating->getType()->isHalfTy()))
            return llvm::ConstantInt::get(type(floating->getType()),floating->getValueAPF().bitcastToAPInt().zextOrTrunc(type(floating->getType())->getIntegerBitWidth()));
        if (llvm::isa<llvm::UndefValue>(input) || llvm::isa<llvm::PoisonValue>(input)) return canonical(llvm::UndefValue::get(type(input->getType())), input->getType());
        if (auto* constant = llvm::dyn_cast<llvm::Constant>(input); constant && !constant->getType()->isPointerTy()) return const_cast<llvm::Constant*>(constant);
        fail(*input, "unsupported or non-dominating value");
    }
    llvm::Value* load(llvm::Value* address, llvm::Type* t, unsigned alignment = 4) {
        auto bits = t->isPointerTy() ? 64u : t->isIntegerTy(1) ? 8u : t->getPrimitiveSizeInBits().getFixedValue();
        if (bits != 8 && bits != 16 && bits != 32 && bits != 64) fail(kernel, "memory accesses require 8/16/32/64-bit scalar words");
        llvm::Value* loaded;
        if (bits >= 32 && alignment >= 4) {
            auto* lo = call_helper(read, {address}); loaded = bits == 64 ? packet_pair(address, lo) : lo;
        } else {
            auto* integer = llvm::cast<llvm::IntegerType>(type(llvm::IntegerType::get(b.getContext(), unsigned(bits)))); loaded = llvm::ConstantInt::get(integer, 0);
            for (unsigned i = 0; i < bits / 8; ++i) {
                auto* byte_address = b.CreateAdd(address, b.getInt64(i));
                auto* shift = byte_shift(byte_address);
                auto* byte = b.CreateAnd(b.CreateLShr(call_helper(read, {byte_address}), shift), b.getInt32(255));
                loaded = b.CreateOr(loaded, b.CreateShl(b.CreateZExtOrTrunc(byte, integer), llvm::ConstantInt::get(integer, i * 8)));
            }
        }
        if(t->isIntegerTy(1)) return b.CreateICmpNE(b.CreateAnd(loaded,b.getInt32(1)),b.getInt32(0));
        if (t->isPointerTy() || t->isIntegerTy()) return loaded;
        if(t->isDoubleTy() || t->isFloatTy() || t->isHalfTy()) return loaded;
        if (t->isFloatingPointTy()) return b.CreateBitCast(loaded, t);
        fail(kernel, "aggregate or vector memory access");
    }
    llvm::Value* packet_pair(llvm::Value* address, llvm::Value* lo) { return pair(lo, call_helper(read, {b.CreateAdd(address, b.getInt64(4))})); }
    llvm::Value* byte_shift(llvm::Value* address) {
        auto* lo = b.CreateTrunc(address, b.getInt32Ty());
        auto known = llvm::computeKnownBits(address, result->getDataLayout());
        auto high = known.One.lshr(32).getZExtValue();
        if (known.Zero.lshr(32).getZExtValue() == (uint32_t(~high)) && (high == packet_private_high_word || high == uint32_t(device_output_address >> 32)))
            return b.CreateMul(b.CreateAnd(lo, b.getInt32(3)), b.getInt32(8));
        auto* hi = b.CreateTrunc(b.CreateLShr(address, b.getInt64(32)), b.getInt32Ty());
        auto* physical = b.CreateSelect(b.CreateICmpEQ(hi, b.getInt32(0xffffffffu)), lo, b.CreateTrunc(call_helper(resolve, {address}), b.getInt32Ty()));
        return b.CreateMul(b.CreateAnd(physical, b.getInt32(3)), b.getInt32(8));
    }
    void store(llvm::Value* address, llvm::Value* stored, unsigned alignment = 4, llvm::Type* source_type = nullptr) {
        auto* t = stored->getType(); auto* memory_type = source_type ? source_type : t;
        auto bits = memory_type->isPointerTy() ? 64u : memory_type->isIntegerTy(1) ? 8u : memory_type->getPrimitiveSizeInBits().getFixedValue();
        if (bits != 8 && bits != 16 && bits != 32 && bits != 64) fail(kernel, "memory stores require 8/16/32/64-bit scalar words");
        if (t->isFloatingPointTy()) stored = b.CreateBitCast(stored, llvm::IntegerType::get(b.getContext(), unsigned(bits)));
        if (!stored->getType()->isIntegerTy()) fail(kernel, "aggregate or vector memory store");
        if (bits >= 32 && alignment >= 4) {
            call_helper(write, {address, b.CreateTruncOrBitCast(stored, b.getInt32Ty()), b.getInt32(0xffffffffu)});
            if (bits == 64) call_helper(write, {b.CreateAdd(address, b.getInt64(4)), b.CreateTrunc(b.CreateLShr(stored, b.getInt64(32)), b.getInt32Ty()), b.getInt32(0xffffffffu)});
        } else for (unsigned i = 0; i < bits / 8; ++i) {
            auto* byte_address = b.CreateAdd(address, b.getInt64(i));
            auto* shift = byte_shift(byte_address);
            auto* byte = b.CreateZExtOrTrunc(b.CreateLShr(stored, llvm::ConstantInt::get(stored->getType(), i * 8)), b.getInt32Ty());
            auto* mask = b.CreateShl(b.getInt32(255), shift);
            call_helper(write, {byte_address, b.CreateShl(byte, shift), mask});
        }
    }
    llvm::Function* atomic_helper(unsigned operation, unsigned bits, llvm::AtomicOrdering order, llvm::AtomicOrdering failure, llvm::SyncScope::ID requested_scope) {
        auto key = std::make_tuple(operation,bits,unsigned(order),unsigned(failure),unsigned(requested_scope));
        if (auto found = atomic_helpers.find(key); found != atomic_helpers.end()) return found->second;
        auto saved = b.saveIP();
        auto* integer = llvm::IntegerType::get(b.getContext(), bits);
        std::vector<llvm::Type*> signature{b.getInt32Ty(),b.getInt32Ty(),integer,integer};
        if (opencl && local_bytes && !arena_local && bits==64) signature.push_back(b.getPtrTy());
        if (opencl) for (unsigned i=0;i<packet_resource_count;++i) signature.push_back(resource_pointer_type());
        auto* function = llvm::Function::Create(llvm::FunctionType::get(integer,
            signature,false), llvm::GlobalValue::InternalLinkage,
            "cudavm.atomic." + std::to_string(operation) + "." + std::to_string(bits), *result);
        if (opencl && local_bytes && !arena_local && bits==64) function->addFnAttr("cudavm.native.local");
        atomic_helpers[key] = function;
        auto* first = llvm::BasicBlock::Create(b.getContext(), "entry", function); b.SetInsertPoint(first);
        auto* address = pair(function->getArg(0), function->getArg(1));
        auto scope = dxil?llvm::SyncScope::System:requested_scope;
        if(!opencl && order==llvm::AtomicOrdering::SequentiallyConsistent) {
            // A common device mutex orders SC operations and fences across all
            // locations and widths. Word 1 is in the reserved transport prefix;
            // the 64-bit representation mutex remains independently at word 0.
            auto* retry=llvm::BasicBlock::Create(b.getContext(),"sc.acquire",function);
            auto* critical=llvm::BasicBlock::Create(b.getContext(),"sc.operation",function);
            auto* joined=llvm::BasicBlock::Create(b.getContext(),"sc.released",function);
            auto* done=llvm::BasicBlock::Create(b.getContext(),"sc.done",function);
            b.CreateBr(retry);b.SetInsertPoint(retry);
            auto gate_scope=dxil?llvm::SyncScope::System:b.getContext().getOrInsertSyncScopeID("device");
            auto* lock=b.CreateAtomicCmpXchg(word_pointer(0,b.getInt32(1)),b.getInt32(0),b.getInt32(1),llvm::Align(4),llvm::AtomicOrdering::Acquire,llvm::AtomicOrdering::Monotonic,gate_scope);
            b.CreateCondBr(b.CreateExtractValue(lock,1),critical,joined);b.SetInsertPoint(critical);
            memory_fence();
            llvm::Value* value=llvm::ConstantInt::get(integer,0);
            if(operation!=101) {
                auto* inner=atomic_helper(operation,bits,llvm::AtomicOrdering::AcquireRelease,failure==llvm::AtomicOrdering::SequentiallyConsistent?llvm::AtomicOrdering::Acquire:failure,requested_scope);
                value=call_helper(inner,{address,function->getArg(2),function->getArg(3)});
            }
            memory_fence();
            b.CreateAtomicCmpXchg(word_pointer(0,b.getInt32(1)),b.getInt32(1),b.getInt32(0),llvm::Align(4),llvm::AtomicOrdering::Release,llvm::AtomicOrdering::Monotonic,gate_scope);
            b.CreateBr(joined);b.SetInsertPoint(joined);
            auto* acquired=b.CreatePHI(b.getInt1Ty(),2);acquired->addIncoming(b.getFalse(),retry);acquired->addIncoming(b.getTrue(),critical);
            auto* observed=b.CreatePHI(integer,2);observed->addIncoming(llvm::ConstantInt::get(integer,0),retry);observed->addIncoming(value,critical);
            b.CreateCondBr(acquired,done,retry);b.SetInsertPoint(done);b.CreateRet(observed);
            b.restoreIP(saved);return function;
        }
        // DXIL interlocked operations are relaxed. Callers supply the release
        // and acquire fences; its legacy bitcode accepts only system scope.
        auto native_order=dxil?llvm::AtomicOrdering::Monotonic:order;
        auto native_failure=dxil?llvm::AtomicOrdering::Monotonic:failure;
        if (bits == 64 && opencl) {
            // Physical OpenCL pointers support complete 64-bit atomics. Avoid
            // a software lock whose progress depends on SIMD scheduling.
            auto* global_space=llvm::BasicBlock::Create(b.getContext(),"global",function);
            auto* shared_space=llvm::BasicBlock::Create(b.getContext(),"shared",function);
            b.CreateCondBr(b.CreateICmpEQ(function->getArg(1),b.getInt32(0xffffffffu)),shared_space,global_space);
            auto access=[&](llvm::BasicBlock* block,bool group_memory,const std::function<llvm::Value*()>& global_pointer) {
                b.SetInsertPoint(block);
                auto pointer=[&]() -> llvm::Value* {
                    if(group_memory)return shared_pointer(b.CreateAnd(function->getArg(0),b.getInt32(0x3fffffffu)));
                    return global_pointer();
                };
                if(operation==llvm::AtomicRMWInst::FAdd || operation==llvm::AtomicRMWInst::FSub) {
                    auto* initial=b.CreateExtractValue(b.CreateAtomicCmpXchg(pointer(),b.getInt64(0),b.getInt64(0),llvm::Align(8),llvm::AtomicOrdering::Monotonic,llvm::AtomicOrdering::Monotonic,scope),0);
                    auto* retry=llvm::BasicBlock::Create(b.getContext(),"add.retry",function);
                    auto* done=llvm::BasicBlock::Create(b.getContext(),"add.done",function);
                    b.CreateBr(retry);b.SetInsertPoint(retry);
                    auto* expected=b.CreatePHI(b.getInt64Ty(),2);expected->addIncoming(initial,block);
                    auto* next=numeric_call(operation==llvm::AtomicRMWInst::FAdd?"__cudavm_native_word_f64_add":"__cudavm_native_word_f64_sub",b.getInt64Ty(),{expected,function->getArg(2)});
                    auto* exchange=b.CreateAtomicCmpXchg(pointer(),expected,next,llvm::Align(8),native_order,native_failure,scope);
                    expected->addIncoming(b.CreateExtractValue(exchange,0),retry);
                    b.CreateCondBr(b.CreateExtractValue(exchange,1),done,retry);b.SetInsertPoint(done);b.CreateRet(expected);
                } else if(operation==100) b.CreateRet(b.CreateExtractValue(b.CreateAtomicCmpXchg(pointer(),function->getArg(3),function->getArg(2),llvm::Align(8),native_order,native_failure,scope),0));
                else b.CreateRet(b.CreateAtomicRMW(llvm::AtomicRMWInst::BinOp(operation),pointer(),function->getArg(2),llvm::Align(8),native_order,scope));
            };
            b.SetInsertPoint(global_space); access_resolved(call_helper(resolve,{address}), [&](auto* block, const auto& pointer) { access(block,false,pointer); });
            access(shared_space,true,{});
        } else if (bits == 32) {
            if(operation==llvm::AtomicRMWInst::FAdd || operation==llvm::AtomicRMWInst::FSub) {
                // Share the arithmetic across resource bindings. Duplicating a
                // numerical retry loop in every buffer-switch arm makes native
                // shader compilation grow unnecessarily with the binding count.
                auto* reader=atomic_helper(100,32,llvm::AtomicOrdering::Monotonic,llvm::AtomicOrdering::Monotonic,requested_scope);
                auto* exchange=atomic_helper(100,32,order,failure,requested_scope);
                auto* initial=call_helper(reader,{address,b.getInt32(0),b.getInt32(0)});
                auto* flush=b.CreateZExt(b.CreateICmpNE(function->getArg(1),b.getInt32(0xffffffffu)),b.getInt32Ty());
                auto* operand=operation==llvm::AtomicRMWInst::FSub?b.CreateXor(function->getArg(2),b.getInt32(0x80000000u)):function->getArg(2);
                auto* retry=llvm::BasicBlock::Create(b.getContext(),"add.retry",function);
                auto* done=llvm::BasicBlock::Create(b.getContext(),"add.done",function);
                b.CreateBr(retry);b.SetInsertPoint(retry);
                auto* expected=b.CreatePHI(b.getInt32Ty(),2);expected->addIncoming(initial,first);
                auto* next=numeric_call("__cudavm_native_word_f32_add",b.getInt32Ty(),{expected,operand,flush});
                auto* observed=call_helper(exchange,{address,next,expected});
                expected->addIncoming(observed,retry);
                b.CreateCondBr(b.CreateICmpEQ(observed,expected),done,retry);
                b.SetInsertPoint(done);b.CreateRet(expected);
                b.restoreIP(saved);return function;
            }
            auto* global_space = llvm::BasicBlock::Create(b.getContext(), "global", function);
            auto* shared_space = llvm::BasicBlock::Create(b.getContext(), "shared", function);
            b.CreateCondBr(b.CreateICmpEQ(function->getArg(1), b.getInt32(0xffffffffu)), shared_space, global_space);
            auto access = [&](llvm::BasicBlock* block, bool group_memory, const std::function<llvm::Value*()>& global_pointer) {
                b.SetInsertPoint(block);
                if((dxil && group_memory && operation!=100) ||
                        (!opencl && !dxil && (operation==llvm::AtomicRMWInst::Min || operation==llvm::AtomicRMWInst::Max)) ||
                        operation==llvm::AtomicRMWInst::UIncWrap || operation==llvm::AtomicRMWInst::UDecWrap) {
                    // Shader targets have no wrapping increment/decrement RMW;
                    // DXIL also needs CAS for its legacy shared RMW encoding.
                    // GLSL's uint transport lvalues select unsigned min/max,
                    // so preserve signed comparisons explicitly in a CAS loop.
                    auto pointer=[&](){return group_memory?shared_pointer(b.CreateAnd(function->getArg(0),b.getInt32(0x3fffffffu))):global_pointer();};
                    auto* initial=b.CreateExtractValue(b.CreateAtomicCmpXchg(pointer(),b.getInt32(0),b.getInt32(0),llvm::Align(4),llvm::AtomicOrdering::Monotonic,llvm::AtomicOrdering::Monotonic,scope),0);
                    auto* retry=llvm::BasicBlock::Create(b.getContext(),"shared.retry",function);
                    auto* done=llvm::BasicBlock::Create(b.getContext(),"shared.done",function);
                    b.CreateBr(retry);b.SetInsertPoint(retry);
                    auto* expected=b.CreatePHI(b.getInt32Ty(),2);expected->addIncoming(initial,block);
                    auto* operand=function->getArg(2);llvm::Value* next=nullptr;
                    switch(operation) {
                    case llvm::AtomicRMWInst::Xchg:next=operand;break;
                    case llvm::AtomicRMWInst::Add:next=b.CreateAdd(expected,operand);break;
                    case llvm::AtomicRMWInst::Sub:next=b.CreateSub(expected,operand);break;
                    case llvm::AtomicRMWInst::And:next=b.CreateAnd(expected,operand);break;
                    case llvm::AtomicRMWInst::Or:next=b.CreateOr(expected,operand);break;
                    case llvm::AtomicRMWInst::Xor:next=b.CreateXor(expected,operand);break;
                    case llvm::AtomicRMWInst::Min:next=b.CreateSelect(b.CreateICmpSLT(expected,operand),expected,operand);break;
                    case llvm::AtomicRMWInst::Max:next=b.CreateSelect(b.CreateICmpSGT(expected,operand),expected,operand);break;
                    case llvm::AtomicRMWInst::UMin:next=b.CreateSelect(b.CreateICmpULT(expected,operand),expected,operand);break;
                    case llvm::AtomicRMWInst::UMax:next=b.CreateSelect(b.CreateICmpUGT(expected,operand),expected,operand);break;
                    case llvm::AtomicRMWInst::UIncWrap:next=b.CreateSelect(b.CreateICmpUGE(expected,operand),b.getInt32(0),b.CreateAdd(expected,b.getInt32(1)));break;
                    case llvm::AtomicRMWInst::UDecWrap:next=b.CreateSelect(b.CreateOr(b.CreateICmpEQ(expected,b.getInt32(0)),b.CreateICmpUGT(expected,operand)),operand,b.CreateSub(expected,b.getInt32(1)));break;
                    default:fail(kernel,"unsupported native 32-bit atomic operation");
                    }
                    auto* exchange=b.CreateAtomicCmpXchg(pointer(),expected,next,llvm::Align(4),native_order,native_failure,scope);
                    expected->addIncoming(b.CreateExtractValue(exchange,0),retry);b.CreateCondBr(b.CreateExtractValue(exchange,1),done,retry);
                    b.SetInsertPoint(done);b.CreateRet(expected);return;
                }
                auto* ptr = group_memory ? shared_pointer(b.CreateAnd(function->getArg(0),b.getInt32(0x3fffffffu)))
                    : global_pointer();
                if (operation == 100) b.CreateRet(b.CreateExtractValue(b.CreateAtomicCmpXchg(ptr,function->getArg(3),function->getArg(2),llvm::Align(4),native_order,native_failure,scope),0));
                else b.CreateRet(b.CreateAtomicRMW(llvm::AtomicRMWInst::BinOp(operation),ptr,function->getArg(2),llvm::Align(4),native_order,scope));
            };
            b.SetInsertPoint(global_space); access_resolved(call_helper(resolve,{address}), [&](auto* block, const auto& pointer) { access(block,false,pointer); });
            access(shared_space,true,{});
        } else {
            // A device-scoped 32-bit lock protects each complete 64-bit update,
            // including the compare and both words. Release occurs before retry
            // lanes return to the acquisition loop; no native 64-bit atomic
            // instruction or stronger shader model is required. Acquire/release
            // supplies the lock ordering; Vulkan forbids SPIR-V SeqCst semantics.
            auto* retry = llvm::BasicBlock::Create(b.getContext(),"acquire",function);
            auto* critical = llvm::BasicBlock::Create(b.getContext(),"update",function);
            auto* joined = llvm::BasicBlock::Create(b.getContext(),"released",function);
            auto* done = llvm::BasicBlock::Create(b.getContext(),"done",function);
            b.CreateBr(retry); b.SetInsertPoint(retry);
            auto representation_scope=dxil?llvm::SyncScope::System:b.getContext().getOrInsertSyncScopeID("device");
            auto* lock = b.CreateAtomicCmpXchg(word_pointer(0,b.getInt32(0)),b.getInt32(0),b.getInt32(1),llvm::Align(4),llvm::AtomicOrdering::Acquire,llvm::AtomicOrdering::Monotonic,representation_scope);
            b.CreateCondBr(b.CreateExtractValue(lock,1),critical,joined); b.SetInsertPoint(critical);
            memory_fence();
            auto* previous = pair(call_helper(read,{address}),call_helper(read,{b.CreateAdd(address,b.getInt64(4))}));
            auto* operand = function->getArg(2); llvm::Value* next = nullptr;
            switch (operation) {
            case 100: next=b.CreateSelect(b.CreateICmpEQ(previous,function->getArg(3)),operand,previous); break;
            case llvm::AtomicRMWInst::Xchg: next=operand; break;
            case llvm::AtomicRMWInst::Add: next=b.CreateAdd(previous,operand); break;
            case llvm::AtomicRMWInst::Sub: next=b.CreateSub(previous,operand); break;
            case llvm::AtomicRMWInst::And: next=b.CreateAnd(previous,operand); break;
            case llvm::AtomicRMWInst::Or: next=b.CreateOr(previous,operand); break;
            case llvm::AtomicRMWInst::Xor: next=b.CreateXor(previous,operand); break;
            case llvm::AtomicRMWInst::Max: next=b.CreateSelect(b.CreateICmpSGT(previous,operand),previous,operand); break;
            case llvm::AtomicRMWInst::Min: next=b.CreateSelect(b.CreateICmpSLT(previous,operand),previous,operand); break;
            case llvm::AtomicRMWInst::UMax: next=b.CreateSelect(b.CreateICmpUGT(previous,operand),previous,operand); break;
            case llvm::AtomicRMWInst::UMin: next=b.CreateSelect(b.CreateICmpULT(previous,operand),previous,operand); break;
            case llvm::AtomicRMWInst::FAdd: next=numeric_call("__cudavm_native_word_f64_add",b.getInt64Ty(),{previous,operand});break;
            case llvm::AtomicRMWInst::FSub: next=numeric_call("__cudavm_native_word_f64_sub",b.getInt64Ty(),{previous,operand});break;
            default: fail(kernel,"unsupported native 64-bit atomic operation");
            }
            call_helper(write,{address,b.CreateTrunc(next,b.getInt32Ty()),b.getInt32(0xffffffffu)});
            call_helper(write,{b.CreateAdd(address,b.getInt64(4)),b.CreateTrunc(b.CreateLShr(next,b.getInt64(32)),b.getInt32Ty()),b.getInt32(0xffffffffu)});
            memory_fence();
            b.CreateAtomicCmpXchg(word_pointer(0,b.getInt32(0)),b.getInt32(1),b.getInt32(0),llvm::Align(4),llvm::AtomicOrdering::Release,llvm::AtomicOrdering::Monotonic,representation_scope);
            b.CreateBr(joined); b.SetInsertPoint(joined);
            auto* acquired=b.CreatePHI(b.getInt1Ty(),2);acquired->addIncoming(b.getFalse(),retry);acquired->addIncoming(b.getTrue(),critical);
            auto* observed=b.CreatePHI(b.getInt64Ty(),2);observed->addIncoming(b.getInt64(0),retry);observed->addIncoming(previous,critical);
            b.CreateCondBr(acquired,done,retry);b.SetInsertPoint(done);b.CreateRet(observed);
        }
        b.restoreIP(saved); return function;
    }
    llvm::SyncScope::ID memory_scope(llvm::SyncScope::ID input) {
        llvm::SmallVector<llvm::StringRef,8> names;b.getContext().getSyncScopeNames(names);
        auto name=names[input];
        if(name=="block" || name=="cta" || name=="workgroup") return b.getContext().getOrInsertSyncScopeID("workgroup");
        if(input==llvm::SyncScope::SingleThread) return input;
        if(name=="system" || name=="sys" || name=="cluster") fail(kernel,"shader system/cluster atomic coherence requires a native memory mechanism");
        // Unqualified CUDA LLVM atomics operate on the shader's device arena.
        if(name.empty() || name=="device" || name=="gpu" || name=="agent") return b.getContext().getOrInsertSyncScopeID("device");
        fail(kernel,"unrecognized shader atomic synchronization scope");
    }
    llvm::Value* atomic(unsigned operation, llvm::Value* address, llvm::Value* operand, llvm::Value* expected = nullptr,
        llvm::AtomicOrdering order=llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering failure=llvm::AtomicOrdering::Monotonic,
        llvm::SyncScope::ID input_scope=llvm::SyncScope::System) {
        if(order==llvm::AtomicOrdering::Unordered)order=llvm::AtomicOrdering::Monotonic;
        if(failure==llvm::AtomicOrdering::Unordered)failure=llvm::AtomicOrdering::Monotonic;
        bool real=operand->getType()->isFloatTy(); if(real) operand=b.CreateBitCast(operand,b.getInt32Ty());
        auto* t = operand->getType();
        if (!t->isIntegerTy(32) && !t->isIntegerTy(64)) fail(kernel,"native atomics require 32/64-bit integer storage");
        auto scope=memory_scope(input_scope);
        auto* function = atomic_helper(operation,t->getIntegerBitWidth(),order,failure,scope);
        if(dxil && llvm::isReleaseOrStronger(order)) memory_fence(llvm::AtomicOrdering::Release,scope);
        auto* out=call_helper(function,{address,operand,expected ? expected : llvm::ConstantInt::get(t,0)});
        if(dxil && (llvm::isAcquireOrStronger(order) || llvm::isAcquireOrStronger(failure))) memory_fence(llvm::AtomicOrdering::Acquire,scope);
        return real?b.CreateBitCast(out,b.getFloatTy()):out;
    }
    llvm::Value* numeric_call(const std::string& name,llvm::Type* output,std::initializer_list<llvm::Value*> inputs) {
        auto* original=source.getFunction(name);if(!original || original->isDeclaration()) fail(kernel,"exact native numeric helper is unavailable");
        std::vector<llvm::Value*> args(inputs);std::vector<llvm::Type*> signature;for(auto* arg:args)signature.push_back(arg->getType());
        if(opencl) {auto* caller=b.GetInsertBlock()->getParent();for(unsigned i=0;i<packet_resource_count;++i) {signature.push_back(resource_pointer_type());args.push_back(caller->getArg(unsigned(caller->arg_size()-packet_resource_count+i)));}}
        auto callee=result->getOrInsertFunction("cudavm.native.helper."+name,llvm::FunctionType::get(output,signature,false));
        llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr("cudavm.source.helper",name);return b.CreateCall(callee,args);
    }
    llvm::Value* thread_axis(unsigned axis) {
        auto id=dxil?llvm::Intrinsic::ID(llvm::Intrinsic::dx_thread_id_in_group):llvm::Intrinsic::ID(llvm::Intrinsic::spv_thread_id_in_group);
        return b.CreateZExtOrTrunc(b.CreateIntrinsic(opencl?b.getInt64Ty():b.getInt32Ty(),id,{b.getInt32(axis)}),b.getInt32Ty());
    }
    void memory_fence(llvm::AtomicOrdering order=llvm::AtomicOrdering::AcquireRelease,llvm::SyncScope::ID scope=llvm::SyncScope::System) {
        scope=memory_scope(scope);
        if(dxil) {
            auto callee=result->getOrInsertFunction("dx.op.barrier",b.getVoidTy(),b.getInt32Ty(),b.getInt32Ty());
            auto* function=llvm::cast<llvm::Function>(callee.getCallee());
            function->addFnAttr(llvm::Attribute::Convergent);function->addFnAttr(llvm::Attribute::NoUnwind);
            b.CreateCall(callee,{b.getInt32(80),b.getInt32(scope==b.getContext().getOrInsertSyncScopeID("workgroup")?12:10)});
        } else b.CreateFence(order,scope);
    }
    llvm::Value* linear_thread() { return b.CreateAdd(thread_axis(0),b.CreateMul(packet(b.getInt32(3)),b.CreateAdd(thread_axis(1),b.CreateMul(packet(b.getInt32(4)),thread_axis(2))))); }
    llvm::Value* thread_count() { return b.CreateMul(packet(b.getInt32(3)),b.CreateMul(packet(b.getInt32(4)),packet(b.getInt32(5)))); }
    void group_barrier() {
        if(dxil) {
            auto callee=result->getOrInsertFunction("dx.op.barrier",b.getVoidTy(),b.getInt32Ty(),b.getInt32Ty());
            auto* function=llvm::cast<llvm::Function>(callee.getCallee()); function->addFnAttr(llvm::Attribute::Convergent); function->addFnAttr(llvm::Attribute::NoUnwind);
            b.CreateCall(callee,{b.getInt32(80),b.getInt32(11)});
        } else b.CreateIntrinsic(llvm::Intrinsic::spv_group_memory_barrier_with_group_sync,llvm::ArrayRef<llvm::Type*>{},llvm::ArrayRef<llvm::Value*>{});
    }
    #include "shader_continuations.inl"
    llvm::Value* collective(const llvm::CallInst& call,unsigned operation) {
        if(operation>=32 && operation<35 && call.arg_size()>2) fail(call,"counted block reductions require separate lowering");
        if(operation>=32 && operation<36 && call.arg_size()>(operation==35?0u:1u)) if(auto* id=llvm::dyn_cast<llvm::ConstantInt>(call.getArgOperand(0));!id || !id->isZero()) fail(call,"named block reductions require separate lowering");
        bool shuffle_predicate=operation<4 && call.getType()->isStructTy();
        auto helper_key=operation+(shuffle_predicate?64:0);
        auto saved=b.saveIP(); llvm::Function* helper=nullptr;
        if(auto found=collective_helpers.find(helper_key);found!=collective_helpers.end()) helper=found->second;
        else {
            helper=llvm::Function::Create(llvm::FunctionType::get(shuffle_predicate || operation==10 || operation==11?b.getInt64Ty():b.getInt32Ty(),arguments({b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty()}),false),llvm::GlobalValue::InternalLinkage,"cudavm.collective."+std::to_string(helper_key),*result);
            helper->addFnAttr(llvm::Attribute::Convergent); collective_helpers[helper_key]=helper;
            helper->addFnAttr("cudavm.collective",std::to_string(operation));
            auto* first=llvm::BasicBlock::Create(b.getContext(),"entry",helper); b.SetInsertPoint(first);
            auto* id=linear_thread(); auto* lane=b.CreateAnd(id,b.getInt32(31)); auto* base=b.CreateAnd(id,b.getInt32(~31u));
            auto pointer=[&](llvm::Value* index){return b.CreateInBoundsGEP(collective_data->getValueType(),collective_data,{b.getInt32(0),index});};
            llvm::Value* answer=b.getInt32(0);
            if(operation<4) {
                auto* mask=b.CreateAnd(b.CreateLShr(helper->getArg(3),b.getInt32(8)),b.getInt32(31));
                auto* minimum=b.CreateAnd(lane,mask); auto* maximum=b.CreateOr(minimum,b.CreateAnd(b.CreateAnd(helper->getArg(3),b.getInt32(31)),b.CreateNot(mask)));
                auto* offset=b.CreateAnd(helper->getArg(2),b.getInt32(31)); llvm::Value* peer=nullptr; llvm::Value* eligible=nullptr;
                if(operation==0) peer=b.CreateOr(minimum,b.CreateAnd(offset,b.CreateNot(mask)));
                if(operation==1) peer=b.CreateSub(lane,offset);
                if(operation==2) peer=b.CreateAdd(lane,offset);
                if(operation==3) peer=b.CreateXor(lane,offset);
                eligible=operation==1?b.CreateAnd(b.CreateICmpUGE(lane,offset),b.CreateICmpUGE(peer,maximum)):b.CreateICmpULE(peer,maximum);
                auto* selected=b.CreateSelect(eligible,b.CreateAdd(base,peer),id);
                answer=b.CreateAlignedLoad(b.getInt32Ty(),pointer(selected),llvm::Align(4));
                if(shuffle_predicate) answer=pair(answer,b.CreateZExt(eligible,b.getInt32Ty()));
            } else if(operation==7 || operation==35 || operation>=36) answer=b.getInt32(0);
            else if(operation==20) answer=helper->getArg(0);
            else {
                auto* loop=llvm::BasicBlock::Create(b.getContext(),"scan",helper); auto* body=llvm::BasicBlock::Create(b.getContext(),"participant",helper); auto* done=llvm::BasicBlock::Create(b.getContext(),"done",helper);
                auto* count=thread_count(); auto* start=operation>=32?b.getInt32(0):base;
                auto* end=operation>=32?count:b.CreateSelect(b.CreateICmpULT(b.CreateAdd(base,b.getInt32(32)),count),b.CreateAdd(base,b.getInt32(32)),count);
                b.CreateBr(loop); b.SetInsertPoint(loop);
                auto* index=b.CreatePHI(b.getInt32Ty(),2); index->addIncoming(start,first);
                auto initial=operation==5 || operation==33?1u:operation==13?0x7fffffffu:operation==14?0x80000000u:operation==15 || operation==17?0xffffffffu:0u;
                auto* total=b.CreatePHI(b.getInt32Ty(),2); total->addIncoming(b.getInt32(initial),first);
                b.CreateCondBr(b.CreateICmpULT(index,end),body,done); b.SetInsertPoint(body);
                auto* bit=b.CreateShl(b.getInt32(1),b.CreateAnd(index,b.getInt32(31)));
                auto* participates=operation>=32?b.CreateICmpNE(collective_load(collective_pointer(collective_pc,index)),b.getInt32(0)):b.CreateICmpNE(b.CreateAnd(helper->getArg(0),bit),b.getInt32(0));
                auto* sample_block=llvm::BasicBlock::Create(b.getContext(),"sample",helper);
                auto* advance=llvm::BasicBlock::Create(b.getContext(),"advance",helper);
                b.CreateCondBr(participates,sample_block,advance); b.SetInsertPoint(sample_block);
                auto* sample=b.CreateAlignedLoad(b.getInt32Ty(),pointer(index),llvm::Align(4));
                auto* predicate=b.CreateICmpNE(sample,b.getInt32(0));
                llvm::Value* next=nullptr;
                if(operation==4) next=b.CreateOr(total,b.CreateSelect(b.CreateAnd(participates,predicate),bit,b.getInt32(0)));
                else if(operation>=8 && operation<=11) {
                    llvm::Value* equal=b.CreateICmpEQ(sample,helper->getArg(1));
                    if(operation==9 || operation==11) equal=b.CreateAnd(equal,b.CreateICmpEQ(b.CreateAlignedLoad(b.getInt32Ty(),pointer(b.CreateAdd(index,b.getInt32(1024))),llvm::Align(4)),helper->getArg(2)));
                    next=b.CreateOr(total,b.CreateSelect(b.CreateAnd(participates,equal),bit,b.getInt32(0)));
                }
                else if(operation>=12 && operation<=19) {
                    llvm::Value* value=nullptr;
                    if(operation==12) value=b.CreateAdd(total,sample);
                    if(operation==13) value=b.CreateSelect(b.CreateICmpSLT(total,sample),total,sample);
                    if(operation==14) value=b.CreateSelect(b.CreateICmpSGT(total,sample),total,sample);
                    if(operation==15) value=b.CreateSelect(b.CreateICmpULT(total,sample),total,sample);
                    if(operation==16) value=b.CreateSelect(b.CreateICmpUGT(total,sample),total,sample);
                    if(operation==17) value=b.CreateAnd(total,sample);
                    if(operation==18) value=b.CreateOr(total,sample);
                    if(operation==19) value=b.CreateXor(total,sample);
                    next=b.CreateSelect(participates,value,total);
                }
                else if(operation==5 || operation==33) next=b.CreateAnd(total,b.CreateZExt(b.CreateOr(b.CreateNot(participates),predicate),b.getInt32Ty()));
                else if(operation==32) next=b.CreateAdd(total,b.CreateZExt(predicate,b.getInt32Ty()));
                else next=b.CreateOr(total,b.CreateZExt(b.CreateAnd(participates,predicate),b.getInt32Ty()));
                b.CreateBr(advance); b.SetInsertPoint(advance);
                auto* merged=b.CreatePHI(b.getInt32Ty(),2);merged->addIncoming(total,body);merged->addIncoming(next,sample_block);
                index->addIncoming(b.CreateAdd(index,b.getInt32(1)),advance); total->addIncoming(merged,advance); b.CreateBr(loop); b.SetInsertPoint(done); answer=total;
                if(operation==10 || operation==11) {
                    auto* all=b.CreateICmpEQ(total,helper->getArg(0));
                    answer=pair(b.CreateSelect(all,total,b.getInt32(0)),b.CreateZExt(all,b.getInt32Ty()));
                }
            }
            b.CreateRet(answer);
        }
        b.restoreIP(saved);
        bool reduce=operation>=12 && operation<=19;
        llvm::Value* mask=operation>=32?b.getInt32(0xffffffffu):operation==20?b.getInt32(0):value(call.getArgOperand(reduce?1:0));
        llvm::Value* data=operation==7 || operation==20 || operation>=35?b.getInt32(0):value(call.getArgOperand(operation>=32?unsigned(call.arg_size()-1):reduce?0:1));
        if(operation>=36) {
            mask=call.arg_empty()?b.getInt32(0):value(call.getArgOperand(0));
            if(auto* id=llvm::dyn_cast<llvm::ConstantInt>(mask);id && id->getZExtValue()>=16) fail(call,"CTA barrier ID must be below 16");
            if(call.arg_size()>1) data=value(call.getArgOperand(1));
        }
        auto* high=operation==9 || operation==11?b.CreateTrunc(b.CreateLShr(data,b.getInt64(32)),b.getInt32Ty()):b.getInt32(0);
        if(data->getType()->isFloatTy()) data=b.CreateBitCast(data,b.getInt32Ty()); else data=b.CreateZExtOrTrunc(data,b.getInt32Ty());
        std::vector<llvm::Value*> args{mask,data,operation<4?value(call.getArgOperand(2)):high,operation<4?value(call.getArgOperand(3)):b.getInt32(0)};
        if (collective_frames) {
            auto context = call.getOperandBundle("cudavm.call.frame");
            if (!context) fail(call, "collective is missing its device call frame");
            collective_store(b.CreateAnd(b.CreateTrunc(value(context->Inputs[0].get()), b.getInt32Ty()), b.getInt32(0x3fffffffu)),
                collective_pointer(collective_frames, linear_thread()));
        }
        if(opencl) for (unsigned i=0;i<packet_resource_count;++i) args.push_back(entry->getArg(unsigned(entry->arg_size()-packet_resource_count+i)));
        auto* out=b.CreateCall(helper,args); out->setConvergent();
        if(call.getType()->isVoidTy()) return nullptr;
        if(shuffle_predicate || operation==10 || operation==11) return b.CreateInsertValue(b.CreateInsertValue(llvm::UndefValue::get(type(call.getType())),b.CreateTrunc(out,b.getInt32Ty()),0),b.CreateICmpNE(b.CreateLShr(out,b.getInt64(32)),b.getInt64(0)),1);
        return call.getType()->isIntegerTy(1)?b.CreateICmpNE(out,b.getInt32(0)):out;
    }
    #include "shader_texture_filter.inl"
    #include "shader_surface.inl"
    // Checked forms emitted by Clang's pinned texture/surface headers. Only
    // the shader clone interprets the object's GPU descriptor; NVPTX keeps PTX.
    bool image_operation(const llvm::CallInst& instruction, llvm::Value*& output) {
        if (!instruction.isInlineAsm()) return false;
        auto* assembly = llvm::cast<llvm::InlineAsm>(instruction.getCalledOperand());
        if (assembly->isAlignStack() || assembly->getDialect() != llvm::InlineAsm::AD_ATT) return false;
        std::string text; for (unsigned char c : assembly->getAsmString()) if (!std::isspace(c)) text += char(c);
        if (surface_operation(instruction, *assembly, text, output)) return true;
        bool fetch = text == "tex.1d.v4.f32.s32{$0,$1,$2,$3},[$4,{$5}];" ||
            text == "tex.1d.v4.s32.s32{$0,$1,$2,$3},[$4,{$5}];" || text == "tex.1d.v4.u32.s32{$0,$1,$2,$3},[$4,{$5}];";
        bool float_texture = text == "tex.2d.v4.f32.f32{$0,$1,$2,$3},[$4,{$5,$6}];" || (fetch && text.find(".f32.") != std::string::npos);
        bool integer_texture = text == "tex.2d.v4.s32.f32{$0,$1,$2,$3},[$4,{$5,$6}];" ||
            text == "tex.2d.v4.u32.f32{$0,$1,$2,$3},[$4,{$5,$6}];";
        bool texture = float_texture || integer_texture || fetch;
        if (!texture) return false;
        entry->addFnAttr("cudavm.image.operations");
        if (dxil) {
            read->addFnAttr(llvm::Attribute::AlwaysInline);
            resolve->addFnAttr(llvm::Attribute::AlwaysInline);
        }
        if (instruction.arg_size() != (fetch ? 2u : 3u) || !instruction.getArgOperand(0)->getType()->isIntegerTy(64))
            fail(instruction, "invalid image instruction operands");
        auto* structure = llvm::dyn_cast<llvm::StructType>(instruction.getType());
        std::string constraints = float_texture ? "=f,=f,=f,=f,l," : "=r,=r,=r,=r,l,";
        constraints += fetch ? "r" : "f,f";
        if (!structure || structure->getNumElements() != 4 ||
            !llvm::all_of(structure->elements(), [&](auto* type) { return float_texture ? type->isFloatTy() : type->isIntegerTy(32); }) ||
            !(fetch ? instruction.getArgOperand(1)->getType()->isIntegerTy(32) :
                instruction.getArgOperand(1)->getType()->isFloatTy() && instruction.getArgOperand(2)->getType()->isFloatTy()) ||
            assembly->getConstraintString() != constraints) fail(instruction, "invalid texture instruction signature");
        auto* object = value(instruction.getArgOperand(0));
        object = b.CreateZExt(call_helper(resolve, {object}), b.getInt64Ty());
        auto descriptor_word = [&](unsigned offset) { return image_descriptor_word(object, offset); };
        auto* base = b.CreateZExt(call_helper(resolve, {pair(descriptor_word(0), descriptor_word(4))}), b.getInt64Ty());
        auto* width = descriptor_word(8);
        auto* height = descriptor_word(12);
        auto* texel_bytes = descriptor_word(52);
        auto* pitch = descriptor_word(64);
        llvm::Value *x = value(instruction.getArgOperand(1)), *y = fetch ? b.getInt32(0) : value(instruction.getArgOperand(2));
        llvm::Value* valid = b.getTrue();
        if (fetch) {
            valid = b.CreateICmpULT(x, width); x = b.CreateMul(x, texel_bytes);
        } else {
            auto* normalized = b.CreateICmpNE(b.CreateAnd(descriptor_word(20), b.getInt32(1)), b.getInt32(0));
            auto coordinate = [&](llvm::Value* bits, llvm::Value* extent, unsigned axis) {
                // Exact floor of a nonnegative binary32 coordinate, followed
                // by integer clamping. Masked shifts avoid poison in inactive
                // selects, and no rounded float extent or FP-to-int overflow
                // can move a boundary sample to the wrong texel.
                auto* exponent = b.CreateAnd(b.CreateLShr(bits, b.getInt32(23)), b.getInt32(255));
                auto* mantissa = b.CreateOr(b.CreateAnd(bits, b.getInt32(0x7fffff)), b.getInt32(0x800000));
                auto* left = b.CreateShl(mantissa, b.CreateAnd(b.CreateSub(exponent, b.getInt32(150)), b.getInt32(31)));
                auto* right = b.CreateLShr(mantissa, b.CreateAnd(b.CreateSub(b.getInt32(150), exponent), b.getInt32(31)));
                auto* integer = b.CreateSelect(b.CreateICmpUGE(exponent, b.getInt32(150)), left, right);
                integer = b.CreateSelect(b.CreateICmpULT(exponent, b.getInt32(127)), b.getInt32(0), integer);
                integer = b.CreateSelect(b.CreateICmpUGE(exponent, b.getInt32(158)), b.getInt32(UINT32_MAX), integer);
                integer = b.CreateSelect(b.CreateICmpSLT(bits, b.getInt32(0)), b.getInt32(0), integer);
                auto* maximum = b.CreateSub(extent, b.getInt32(1));
                auto* clamped = b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, integer, maximum);

                // Reduce normalized coordinates before scaling so even the
                // largest finite binary32 values cannot overflow. Integer
                // significand products preserve texel boundaries and tiny
                // negative coordinates without shader FP contraction/FTZ.
                // Native texture coordinates flush subnormals to signed zero.
                auto* shift = b.CreateSub(b.getInt32(150), exponent);
                auto* fraction_mask = b.CreateSub(b.CreateShl(b.getInt32(1), b.CreateAnd(shift, b.getInt32(31))), b.getInt32(1));
                auto* fraction = b.CreateSelect(b.CreateICmpULT(exponent, b.getInt32(127)), mantissa, b.CreateAnd(mantissa, fraction_mask));
                fraction = b.CreateSelect(b.CreateOr(b.CreateICmpEQ(exponent, b.getInt32(0)), b.CreateICmpUGE(exponent, b.getInt32(150))), b.getInt32(0), fraction);
                auto* product = b.CreateMul(b.CreateZExt(fraction, b.getInt64Ty()), b.CreateZExt(extent, b.getInt64Ty()));
                auto* safe_shift = b.CreateZExt(b.CreateAnd(shift, b.getInt32(63)), b.getInt64Ty());
                auto* whole = b.CreateSelect(b.CreateICmpULT(shift, b.getInt32(64)), b.CreateLShr(product, safe_shift), b.getInt64(0));
                auto* remainder = b.CreateICmpNE(product, b.CreateShl(whole, safe_shift));
                auto* fractional_index = b.CreateTrunc(whole, b.getInt32Ty());
                auto* ceiling = b.CreateAdd(fractional_index, b.CreateZExt(remainder, b.getInt32Ty()));
                auto* reflected_index = b.CreateSub(extent, ceiling);
                auto* odd = b.CreateAnd(b.CreateICmpUGE(exponent, b.getInt32(127)), b.CreateICmpULE(exponent, b.getInt32(150)));
                odd = b.CreateAnd(odd, b.CreateICmpNE(b.CreateAnd(b.CreateLShr(mantissa, b.CreateAnd(shift, b.getInt32(31))), b.getInt32(1)), b.getInt32(0)));
                auto* negative = b.CreateAnd(b.CreateICmpSLT(bits, b.getInt32(0)), b.CreateICmpNE(exponent, b.getInt32(0)));
                auto* mode = descriptor_word(24 + axis * 4);
                auto* wrap = b.CreateICmpEQ(mode, b.getInt32(0));
                auto* mirror = b.CreateICmpEQ(mode, b.getInt32(2));
                auto* wrapped = b.CreateSelect(b.CreateAnd(negative, b.CreateICmpNE(fraction, b.getInt32(0))), reflected_index, fractional_index);
                // Mirror the integer texel index after floor, rather than
                // reflecting a floating coordinate. This preserves CUDA's
                // orientation at exact negative/reversed texel boundaries.
                auto* period = b.CreateMul(extent, b.getInt32(2));
                auto* positive_index = b.CreateAdd(b.CreateSelect(odd, extent, b.getInt32(0)), fractional_index);
                auto* negative_distance = b.CreateAdd(b.CreateSelect(odd, extent, b.getInt32(0)), ceiling);
                auto* negative_index = b.CreateSelect(b.CreateICmpEQ(negative_distance, b.getInt32(0)), b.getInt32(0), b.CreateSub(period, negative_distance));
                auto* mirror_index = b.CreateSelect(negative, negative_index, positive_index);
                auto* mirrored = b.CreateSelect(b.CreateICmpULT(mirror_index, extent), mirror_index, b.CreateSub(b.CreateSub(period, b.getInt32(1)), mirror_index));
                auto* repeated = b.CreateSelect(mirror, mirrored, wrapped);
                auto* normalized_clamp = b.CreateSelect(b.CreateICmpUGE(exponent, b.getInt32(127)), maximum, fractional_index);
                normalized_clamp = b.CreateSelect(negative, b.getInt32(0), normalized_clamp);
                auto* normalized_index = b.CreateSelect(b.CreateOr(wrap, mirror), repeated, normalized_clamp);
                auto* in_range = b.CreateAnd(b.CreateNot(negative), b.CreateSelect(normalized,
                    b.CreateICmpULT(exponent, b.getInt32(127)), b.CreateAnd(b.CreateICmpULT(exponent, b.getInt32(158)), b.CreateICmpULT(integer, extent))));
                valid = b.CreateAnd(valid, b.CreateOr(b.CreateICmpNE(mode, b.getInt32(3)), in_range));
                return b.CreateSelect(normalized, normalized_index, clamped);
            };
            x = coordinate(x, width, 0); y = coordinate(y, height, 1);
            x = b.CreateMul(x, texel_bytes);
        }
        auto* offset = b.CreateAdd(b.CreateMul(b.CreateZExt(y, b.getInt64Ty()), b.CreateZExt(pitch, b.getInt64Ty())), b.CreateZExt(x, b.getInt64Ty()));
        // The invalid private tag routes zero-mode accesses to the memory
        // helper's no-op/zero path, without touching a real allocation.
        auto* address = b.CreateSelect(valid, b.CreateAdd(base, offset), b.getInt64(0xffffffff00000000ull));
        llvm::Value* tuple = llvm::UndefValue::get(type(instruction.getType()));
        for (unsigned i = 0; i < 4; ++i) tuple = b.CreateInsertValue(tuple, texture_component(object, address, i, float_texture), i);
        output = float_texture && !fetch ? filter_texture(object, value(instruction.getArgOperand(1)), value(instruction.getArgOperand(2)), tuple, instruction) : tuple;
        return true;
    }
    llvm::Value* call(const llvm::CallInst& instruction) {
        llvm::Value* image = nullptr; if (image_operation(instruction, image)) return image;
        auto* function = instruction.getCalledFunction();
        if (!function) fail(instruction, "indirect calls or CUDA inline assembly require fallback");
        if (function->getName() == "cudavm.shader.stack.valid")
            return stack_top ? b.CreateICmpULE(b.CreateLoad(b.getInt32Ty(), stack_top), b.getInt32(shader_local_capacity)) : b.getTrue();
        if (function->getName() == "__cudavm_service_buffer") {
            auto* index = llvm::dyn_cast<llvm::ConstantInt>(instruction.getArgOperand(0));
            if (!index || index->getZExtValue() > 1) fail(instruction, "invalid device service buffer");
            return b.getInt64(index->isZero() ? device_heap_address : device_output_address);
        }
        if (function->getName() == "__cudavm_service_id") {
            auto* index = value(instruction.getArgOperand(0));
            llvm::Value* coordinate = b.getInt32(0);
            for (unsigned axis = 0; axis < 6; ++axis) {
                auto id = dxil ? llvm::Intrinsic::ID(axis < 3 ? llvm::Intrinsic::dx_group_id : llvm::Intrinsic::dx_thread_id_in_group)
                    : llvm::Intrinsic::ID(axis < 3 ? llvm::Intrinsic::spv_group_id : llvm::Intrinsic::spv_thread_id_in_group);
                auto* component = b.CreateZExtOrTrunc(b.CreateIntrinsic(opencl ? b.getInt64Ty() : b.getInt32Ty(), id, {b.getInt32(axis % 3)}), b.getInt32Ty());
                coordinate = b.CreateSelect(b.CreateICmpEQ(index, b.getInt32(axis)), component, coordinate);
            }
            return coordinate;
        }
        if (function->getName() == "__cudavm_service_assert") {
            b.CreateAtomicRMW(llvm::AtomicRMWInst::Or, word_pointer(0, b.getInt32(device_assert_fault_word)), b.getInt32(1),
                llvm::Align(4), llvm::AtomicOrdering::Monotonic, b.getContext().getOrInsertSyncScopeID("device"));
            return nullptr;
        }
        if(instruction.getType()->isHalfTy() || llvm::any_of(instruction.args(),[](const auto& argument){return argument->getType()->isHalfTy();}))
            fail(instruction,"remaining half operation requires explicit native shader lowering");
        auto name = function->hasFnAttribute("cudavm.cuda.origin") ? function->getFnAttribute("cudavm.cuda.origin").getValueAsString() : function->getName();
        if(auto operation=native_collective(name);operation>=0) return collective(instruction,unsigned(operation));
        if(name=="llvm.nvvm.membar.cta" || name=="llvm.nvvm.membar.gl") {
            auto scope=b.getContext().getOrInsertSyncScopeID(name.ends_with("cta")?"workgroup":"device");
            if(opencl) memory_fence(llvm::AtomicOrdering::SequentiallyConsistent,scope);
            else atomic(101,b.getInt64(0),b.getInt32(0),nullptr,llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::Monotonic,scope);
            return nullptr;
        }
        if(name=="llvm.nvvm.membar.sys") fail(instruction,"shader system fence requires host/device coherent memory");
        if(!function->isDeclaration() && shader_native_helper(*function)) {
            std::vector<llvm::Type*> signature; std::vector<llvm::Value*> args;
            for(const auto& argument:instruction.args()) {
                if((argument->getType()->isPointerTy() && !function->hasFnAttribute("cudavm.service.helper")) || !argument->getType()->isSingleValueType()) fail(instruction,"native helper requires scalar arguments");
                signature.push_back(type(argument->getType())); args.push_back(value(argument.get()));
            }
            if(opencl) for(unsigned i=0;i<packet_resource_count;++i) {signature.push_back(resource_pointer_type());args.push_back(entry->getArg(unsigned(entry->arg_size()-packet_resource_count+i)));}
            auto callee=result->getOrInsertFunction("cudavm.native.helper."+function->getName().str(),llvm::FunctionType::get(type(function->getReturnType()),signature,false));
            llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr("cudavm.source.helper",function->getName());
            return b.CreateCall(callee,args);
        }
        if (name.starts_with("llvm.nvvm.read.ptx.sreg.")) {
            auto tail = name.drop_front(24);
            if (tail == "warpsize") return b.getInt32(32);
            if(tail=="laneid" || tail.starts_with("lanemask.")) {
                auto* lane=b.CreateAnd(linear_thread(),b.getInt32(31));
                auto* bit=b.CreateShl(b.getInt32(1),lane);
                if(tail=="laneid")return lane;
                if(tail=="lanemask.eq")return bit;
                auto* below=b.CreateSub(bit,b.getInt32(1));
                if(tail=="lanemask.lt")return below;
                if(tail=="lanemask.le")return b.CreateOr(below,bit);
                if(tail=="lanemask.gt")return b.CreateNot(b.CreateOr(below,bit));
                if(tail=="lanemask.ge")return b.CreateNot(below);
            }
            if (tail.size() < 3 || tail.back() < 'x' || tail.back() > 'z') fail(instruction, "unsupported CUDA index builtin");
            auto axis = unsigned(tail.back() - 'x');
            if (tail.starts_with("tid.") || tail.starts_with("ctaid.")) {
                llvm::Intrinsic::ID id = dxil ? llvm::Intrinsic::ID(tail.starts_with("tid.") ? llvm::Intrinsic::dx_thread_id_in_group : llvm::Intrinsic::dx_group_id)
                    : llvm::Intrinsic::ID(tail.starts_with("tid.") ? llvm::Intrinsic::spv_thread_id_in_group : llvm::Intrinsic::spv_group_id);
                return b.CreateZExtOrTrunc(b.CreateIntrinsic(opencl ? b.getInt64Ty() : b.getInt32Ty(), id, {b.getInt32(axis)}), b.getInt32Ty());
            }
            if (tail.starts_with("ntid.")) return packet(b.getInt32(3 + axis));
            if (tail.starts_with("nctaid.")) return packet(b.getInt32(axis));
        }
        if (function->isIntrinsic()) {
            if(instruction.getType()->isDoubleTy()) fail(instruction,"native double intrinsic requires an exact portable numeric helper");
            if (name.starts_with("llvm.nvvm.")) fail(instruction, "CUDA operation requires explicit native shader lowering");
            if (function->getIntrinsicID() == llvm::Intrinsic::lifetime_start || function->getIntrinsicID() == llvm::Intrinsic::lifetime_end ||
                function->getIntrinsicID() == llvm::Intrinsic::assume || function->getIntrinsicID() == llvm::Intrinsic::experimental_noalias_scope_decl) return nullptr;
            switch (function->getIntrinsicID()) {
            case llvm::Intrinsic::stacksave:
                return stack_top ? b.CreateOr(b.getInt64(0xffffffff80000000ull), b.CreateZExt(b.CreateLoad(b.getInt32Ty(), stack_top), b.getInt64Ty()))
                    : b.getInt64(0xffffffff80000000ull + local_bytes);
            case llvm::Intrinsic::stackrestore:
                if (stack_top) b.CreateStore(b.CreateAnd(b.CreateTrunc(value(instruction.getArgOperand(0)), b.getInt32Ty()), b.getInt32(0x3fffffffu)), stack_top);
                return nullptr;
            case llvm::Intrinsic::is_fpclass: {
                // Optimized bit tests on atomic floating results can become
                // classification intrinsics. Classify the transported bits so
                // target FTZ modes cannot change zero/subnormal/NaN classes.
                auto* input = instruction.getArgOperand(0);
                unsigned width = input->getType()->getScalarSizeInBits();
                if (!input->getType()->isHalfTy() && !input->getType()->isFloatTy() && !input->getType()->isDoubleTy())
                    fail(instruction, "floating classification requires scalar IEEE storage");
                unsigned fraction = width == 16 ? 10 : width == 32 ? 23 : 52;
                auto* bits = value(input);
                auto constant = [&](uint64_t x) { return llvm::ConstantInt::get(bits->getType(), x); };
                uint64_t sign = uint64_t(1) << (width - 1), infinity = (sign - 1) & ~((uint64_t(1) << fraction) - 1);
                auto* magnitude = b.CreateAnd(bits, constant(sign - 1));
                auto* negative = b.CreateICmpNE(b.CreateAnd(bits, constant(sign)), constant(0));
                auto* quiet = b.CreateICmpNE(b.CreateAnd(bits, constant(uint64_t(1) << (fraction - 1))), constant(0));
                auto* nan = b.CreateICmpUGT(magnitude, constant(infinity));
                auto* zero = b.CreateICmpEQ(magnitude, constant(0));
                auto* subnormal = b.CreateAnd(b.CreateICmpULT(magnitude, constant(uint64_t(1) << fraction)), b.CreateNot(zero));
                auto* normal = b.CreateAnd(b.CreateICmpUGE(magnitude, constant(uint64_t(1) << fraction)), b.CreateICmpULT(magnitude, constant(infinity)));
                auto* infinite = b.CreateICmpEQ(magnitude, constant(infinity));
                unsigned mask = llvm::cast<llvm::ConstantInt>(instruction.getArgOperand(1))->getZExtValue();
                llvm::Value* result = b.getFalse();
                auto include = [&](unsigned flag, llvm::Value* matches) { if (mask & flag) result = b.CreateOr(result, matches); };
                include(llvm::fcSNan, b.CreateAnd(nan, b.CreateNot(quiet))); include(llvm::fcQNan, b.CreateAnd(nan, quiet));
                auto signed_class = [&](unsigned positive, unsigned negative_class, llvm::Value* matches) {
                    include(positive, b.CreateAnd(matches, b.CreateNot(negative))); include(negative_class, b.CreateAnd(matches, negative));
                };
                signed_class(llvm::fcPosInf, llvm::fcNegInf, infinite); signed_class(llvm::fcPosNormal, llvm::fcNegNormal, normal);
                signed_class(llvm::fcPosSubnormal, llvm::fcNegSubnormal, subnormal); signed_class(llvm::fcPosZero, llvm::fcNegZero, zero);
                return result;
            }
            case llvm::Intrinsic::fabs: case llvm::Intrinsic::copysign: {
                auto* magnitude=b.CreateBitCast(value(instruction.getArgOperand(0)),b.getInt32Ty());
                auto* bits=b.CreateAnd(magnitude,b.getInt32(0x7fffffffu));
                if(function->getIntrinsicID()==llvm::Intrinsic::copysign) {
                    auto* sign=b.CreateBitCast(value(instruction.getArgOperand(1)),b.getInt32Ty());
                    bits=b.CreateOr(bits,b.CreateAnd(sign,b.getInt32(0x80000000u)));
                }
                return bits;
            }
            case llvm::Intrinsic::uadd_sat: case llvm::Intrinsic::usub_sat: {
                auto* left=value(instruction.getArgOperand(0));
                auto* right=value(instruction.getArgOperand(1));
                if(function->getIntrinsicID()==llvm::Intrinsic::usub_sat)
                    return b.CreateSelect(b.CreateICmpUGE(left,right),b.CreateSub(left,right),llvm::ConstantInt::get(left->getType(),0));
                auto* sum=canonical(b.CreateAdd(left,right),instruction.getType());
                auto maximum=llvm::APInt::getMaxValue(instruction.getType()->getIntegerBitWidth()).zextOrTrunc(left->getType()->getIntegerBitWidth());
                return b.CreateSelect(b.CreateICmpULT(sum,left),llvm::ConstantInt::get(left->getType(),maximum),sum);
            }
            case llvm::Intrinsic::smin: case llvm::Intrinsic::smax:
            case llvm::Intrinsic::umin: case llvm::Intrinsic::umax: {
                auto id = function->getIntrinsicID();
                bool signed_op=id==llvm::Intrinsic::smin || id==llvm::Intrinsic::smax;
                auto* left = signed_op?signed_value(instruction.getArgOperand(0)):value(instruction.getArgOperand(0));
                auto* right = signed_op?signed_value(instruction.getArgOperand(1)):value(instruction.getArgOperand(1));
                auto predicate = id == llvm::Intrinsic::smin ? llvm::CmpInst::ICMP_SLT : id == llvm::Intrinsic::smax ? llvm::CmpInst::ICMP_SGT
                    : id == llvm::Intrinsic::umin ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_UGT;
                return canonical(b.CreateSelect(b.CreateICmp(predicate, left, right), left, right),instruction.getType());
            }
            case llvm::Intrinsic::ctpop: case llvm::Intrinsic::ctlz: case llvm::Intrinsic::cttz:
            case llvm::Intrinsic::bswap: case llvm::Intrinsic::bitreverse: case llvm::Intrinsic::fshl: case llvm::Intrinsic::fshr:
            case llvm::Intrinsic::abs:
            case llvm::Intrinsic::minnum: case llvm::Intrinsic::maxnum: case llvm::Intrinsic::minimum: case llvm::Intrinsic::maximum:
            case llvm::Intrinsic::floor: case llvm::Intrinsic::ceil: case llvm::Intrinsic::trunc:
            case llvm::Intrinsic::round: case llvm::Intrinsic::roundeven: case llvm::Intrinsic::rint: case llvm::Intrinsic::nearbyint:
            case llvm::Intrinsic::sqrt: case llvm::Intrinsic::fma: {
                std::vector<llvm::Value*> args; for (const auto& argument : instruction.args()) {
                    auto* operand=value(argument.get());
                    args.push_back(argument->getType()->isFloatTy()?b.CreateBitCast(operand,b.getFloatTy()):operand);
                }
                auto id = function->getIntrinsicID();
                if(id==llvm::Intrinsic::fma && instruction.getType()->isFloatTy() && (dxil || kernel.getName()!="__cudavm_native_fma"))
                    return numeric_call(dxil?"__cudavm_native_word_f32_fma":"__cudavm_native_fma",b.getInt32Ty(),
                        {value(instruction.getArgOperand(0)),value(instruction.getArgOperand(1)),value(instruction.getArgOperand(2))});
                if (dxil && (id==llvm::Intrinsic::ctlz || id==llvm::Intrinsic::cttz)) {
                    auto bits=instruction.getType()->getIntegerBitWidth();
                    bool leading=id==llvm::Intrinsic::ctlz;
                    auto scan=[&](llvm::Value* word) {
                        // DXIL FirstbitHi counts from the MSB; GLSL FindUMsb
                        // instead returns an index measured from the LSB.
                        return b.CreateIntrinsic(b.getInt32Ty(),leading?llvm::Intrinsic::dx_firstbituhigh:llvm::Intrinsic::dx_firstbitlow,{word});
                    };
                    auto* low=b.CreateZExtOrTrunc(args[0],b.getInt32Ty());
                    llvm::Value* count=scan(low);
                    if(bits>32) {
                        auto* high=b.CreateTrunc(b.CreateLShr(args[0],b.getInt64(32)),b.getInt32Ty());
                        auto* high_count=scan(high);
                        auto* primary=leading?high:low;
                        auto* first=leading?high_count:count;
                        auto* second=leading?count:high_count;
                        count=b.CreateSelect(b.CreateICmpEQ(primary,b.getInt32(0)),b.CreateAdd(second,b.getInt32(32)),first);
                    }
                    if (leading && bits != 32 && bits != 64) count = b.CreateSub(count, b.getInt32((bits > 32 ? 64 : 32) - bits));
                    count=b.CreateZExtOrTrunc(count,type(instruction.getType()));
                    auto* zero=llvm::cast<llvm::ConstantInt>(instruction.getArgOperand(1))->isOne()
                        ?llvm::PoisonValue::get(count->getType()):llvm::ConstantInt::get(count->getType(),bits);
                    return canonical(b.CreateSelect(b.CreateICmpEQ(args[0],llvm::Constant::getNullValue(args[0]->getType())),zero,count),instruction.getType());
                }
                auto* integer_type = llvm::dyn_cast<llvm::IntegerType>(instruction.getType());
                unsigned bits = integer_type ? integer_type->getBitWidth() : 0;
                auto storage = integer_type ? type(integer_type)->getIntegerBitWidth() : 0;
                if (integer_type && bits > 1 && bits < storage && (id == llvm::Intrinsic::fshl || id == llvm::Intrinsic::fshr)) {
                    auto* width = llvm::ConstantInt::get(type(integer_type), bits);
                    auto* shift = b.CreateURem(args[2], width);
                    auto* complement = b.CreateSub(width, shift);
                    return canonical(id == llvm::Intrinsic::fshl
                        ? b.CreateOr(b.CreateShl(args[0], shift), b.CreateLShr(args[1], complement))
                        : b.CreateOr(b.CreateShl(args[0], complement), b.CreateLShr(args[1], shift)), integer_type);
                }
                if (integer_type && id == llvm::Intrinsic::abs) args[0] = signed_value(instruction.getArgOperand(0));
                if (!opencl && !dxil && storage == 64
                    && (id==llvm::Intrinsic::ctpop || id==llvm::Intrinsic::bitreverse)) {
                    // The SPIR-V is also consumed by OpenGL, whose integer64
                    // extension does not supply 64-bit count/reverse overloads.
                    // Keep both halves on native 32-bit bit instructions.
                    auto* low=b.CreateTrunc(args[0],b.getInt32Ty());
                    auto* high=b.CreateTrunc(b.CreateLShr(args[0],b.getInt64(32)),b.getInt32Ty());
                    auto* low_result=b.CreateIntrinsic(b.getInt32Ty(),id,{low});
                    auto* high_result=b.CreateIntrinsic(b.getInt32Ty(),id,{high});
                    if(id==llvm::Intrinsic::ctpop) return b.CreateZExt(b.CreateAdd(low_result,high_result),b.getInt64Ty());
                    auto* reversed = b.CreateOr(b.CreateShl(b.CreateZExt(low_result,b.getInt64Ty()),b.getInt64(32)),b.CreateZExt(high_result,b.getInt64Ty()));
                    return bits < 64 ? b.CreateLShr(reversed, b.getInt64(64 - bits)) : reversed;
                }
                auto* out = b.CreateIntrinsic(instruction.getType()->isFloatTy()?b.getFloatTy():type(instruction.getType()),id,args);
                if(instruction.getType()->isFloatTy()) return b.CreateBitCast(out,b.getInt32Ty());
                if (integer_type && bits > 1 && bits < storage) {
                    auto* padding = llvm::ConstantInt::get(out->getType(), storage - bits);
                    if (id==llvm::Intrinsic::ctlz) return b.CreateSub(out,padding);
                    if (id==llvm::Intrinsic::cttz) return b.CreateSelect(b.CreateICmpEQ(args[0],llvm::ConstantInt::get(args[0]->getType(),0)),llvm::ConstantInt::get(out->getType(),bits),out);
                    if (id==llvm::Intrinsic::bitreverse || id==llvm::Intrinsic::bswap) return b.CreateLShr(out,padding);
                }
                return canonical(out,instruction.getType());
            }
            default: fail(instruction, "LLVM intrinsic requires explicit native shader lowering");
            }
        }
        fail(instruction, "remaining device operation or recursive helper requires fallback");
    }
    void validate_atomic_storage(const llvm::Instruction& input,llvm::Type* storage,llvm::Align alignment) {
        auto bits=storage->isPointerTy()?64u:unsigned(storage->getPrimitiveSizeInBits().getFixedValue());
        if(bits!=32 && bits!=64)fail(input,"shader atomics require 32/64-bit scalar storage");
        if(alignment.value()<bits/8)fail(input,"shader atomic storage is underaligned");
    }
    void instruction(const llvm::Instruction& input) {
        if (input.getType()->isVectorTy()) fail(input, "vector values require fallback in this native profile");
        if (auto* integer = llvm::dyn_cast<llvm::IntegerType>(input.getType()); integer && integer->getBitWidth() > 64)
            fail(input, "unsupported native integer width");
        if (llvm::isa<llvm::PHINode>(input)) return;
        if (auto* frozen = llvm::dyn_cast<llvm::FreezeInst>(&input)) { values[&input] = value(frozen->getOperand(0)); return; }
        if(auto* fence=llvm::dyn_cast<llvm::FenceInst>(&input)) {
            if(!opencl && fence->getOrdering()==llvm::AtomicOrdering::SequentiallyConsistent)
                atomic(101,b.getInt64(0),b.getInt32(0),nullptr,fence->getOrdering(),llvm::AtomicOrdering::Monotonic,fence->getSyncScopeID());
            else memory_fence(fence->getOrdering(),fence->getSyncScopeID());
            return;
        }
        if (auto* operation = llvm::dyn_cast<llvm::AtomicRMWInst>(&input)) {
            validate_atomic_storage(input,operation->getValOperand()->getType(),operation->getAlign());
            values[&input] = atomic(operation->getOperation(),value(operation->getPointerOperand()),value(operation->getValOperand()),nullptr,operation->getOrdering(),llvm::AtomicOrdering::Monotonic,operation->getSyncScopeID()); return;
        }
        if (auto* operation = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&input)) {
            validate_atomic_storage(input,operation->getCompareOperand()->getType(),operation->getAlign());
            auto* expected = value(operation->getCompareOperand());
            auto* previous = atomic(100,value(operation->getPointerOperand()),value(operation->getNewValOperand()),expected,operation->getSuccessOrdering(),operation->getFailureOrdering(),operation->getSyncScopeID());
            values[&input] = b.CreateInsertValue(b.CreateInsertValue(llvm::UndefValue::get(type(input.getType())),previous,0),b.CreateICmpEQ(previous,expected),1); return;
        }
        if (auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(&input)) {
            if (auto found = local_offsets.find(allocation); found != local_offsets.end())
                values[&input] = b.getInt64(0xffffffff80000000ull + found->second);
            else values[&input] = allocate_local(*allocation);
            return;
        }
        if (auto* branch = llvm::dyn_cast<llvm::BranchInst>(&input)) {
            if (assert_cancellation) {
                auto* ready = llvm::BasicBlock::Create(b.getContext(), "assert.continue", entry);
                auto* stopped = llvm::BasicBlock::Create(b.getContext(), "assert.stop", entry);
                auto* observed = b.CreateAtomicCmpXchg(word_pointer(0, b.getInt32(device_assert_fault_word)), b.getInt32(0), b.getInt32(0),
                    llvm::Align(4), llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering::Monotonic, b.getContext().getOrInsertSyncScopeID("device"));
                b.CreateCondBr(b.CreateICmpEQ(b.CreateExtractValue(observed, 0), b.getInt32(0)), ready, stopped);
                b.SetInsertPoint(stopped); b.CreateRetVoid(); b.SetInsertPoint(ready); tails[branch->getParent()] = ready;
            }
            if (branch->isConditional()) b.CreateCondBr(value(branch->getCondition()), blocks.at(branch->getSuccessor(0)), blocks.at(branch->getSuccessor(1)));
            else b.CreateBr(blocks.at(branch->getSuccessor(0))); return;
        }
        if (auto* returned=llvm::dyn_cast<llvm::ReturnInst>(&input)) { if(returned->getReturnValue()) b.CreateRet(value(returned->getReturnValue()));else b.CreateRetVoid(); return; }
        if (llvm::isa<llvm::UnreachableInst>(input)) {
            if (entry->getReturnType()->isVoidTy()) b.CreateRetVoid(); else b.CreateRet(llvm::Constant::getNullValue(entry->getReturnType()));
            return;
        }
        if (auto* pointer = llvm::dyn_cast<llvm::GetElementPtrInst>(&input)) { values[&input] = gep(*llvm::cast<llvm::GEPOperator>(pointer)); return; }
        if (auto* cast = llvm::dyn_cast<llvm::CastInst>(&input); cast && (input.getType()->isPointerTy() || cast->getOperand(0)->getType()->isPointerTy())) {
            auto* address = value(cast->getOperand(0));
            // PTX shared/local integers are byte offsets. Generic pointers keep
            // the transport tag; explicit space-specific ptr/int conversions
            // strip or restore it for both 32-bit and 64-bit address operands.
            auto source_space = cast->getOperand(0)->getType()->isPointerTy() ? cast->getOperand(0)->getType()->getPointerAddressSpace() : 0;
            auto destination_space = input.getType()->isPointerTy() ? input.getType()->getPointerAddressSpace() : 0;
            if (cast->getOpcode() == llvm::Instruction::PtrToInt && (source_space == 3 || source_space == 5))
                address = b.CreateAnd(address, b.getInt64(0x3fffffffull));
            if (cast->getOpcode() == llvm::Instruction::IntToPtr && (destination_space == 3 || destination_space == 5))
                address = b.CreateOr(b.CreateZExtOrTrunc(address, b.getInt64Ty()), b.getInt64(destination_space == 3 ? 0xffffffff40000000ull : 0xffffffff80000000ull));
            values[&input] = canonical(b.CreateZExtOrTrunc(address, type(input.getType())), input.getType()); return;
        }
        if (auto* select = llvm::dyn_cast<llvm::SelectInst>(&input)) { values[&input] = b.CreateSelect(value(select->getCondition()), value(select->getTrueValue()), value(select->getFalseValue())); return; }
        if(auto* cast=llvm::dyn_cast<llvm::BitCastInst>(&input)) {
            auto* source=value(cast->getOperand(0)); values[&input]=source->getType()==type(cast->getType())?source:b.CreateBitCast(source,type(cast->getType())); return;
        }
        if(input.getOpcode()==llvm::Instruction::FNeg && input.getType()->isDoubleTy()) {
            values[&input]=b.CreateXor(value(input.getOperand(0)),b.getInt64(0x8000000000000000ull));return;
        }
        if(input.getOpcode()==llvm::Instruction::FNeg && input.getType()->isFloatTy()) {
            values[&input]=b.CreateXor(value(input.getOperand(0)),b.getInt32(0x80000000u));return;
        }
        if(auto* binary=llvm::dyn_cast<llvm::BinaryOperator>(&input);binary && input.getType()->isFloatTy()) {
            auto* left=b.CreateBitCast(value(binary->getOperand(0)),b.getFloatTy());
            auto* right=b.CreateBitCast(value(binary->getOperand(1)),b.getFloatTy());
            auto* operation=b.CreateBinOp(binary->getOpcode(),left,right);
            if(auto* produced=llvm::dyn_cast<llvm::Instruction>(operation)) produced->copyIRFlags(binary);
            values[&input]=b.CreateBitCast(operation,b.getInt32Ty());return;
        }
        if(auto* compare=llvm::dyn_cast<llvm::FCmpInst>(&input);compare && compare->getOperand(0)->getType()->isFloatTy()) {
            values[&input]=b.CreateFCmp(compare->getPredicate(),b.CreateBitCast(value(compare->getOperand(0)),b.getFloatTy()),b.CreateBitCast(value(compare->getOperand(1)),b.getFloatTy()));return;
        }
        if(input.getType()->isDoubleTy() && llvm::isa<llvm::FPMathOperator>(input)) fail(input,"native double arithmetic requires an exact portable numeric helper");
        if (auto* memory = llvm::dyn_cast<llvm::LoadInst>(&input)) {
            if (memory->isAtomic()) {
                validate_atomic_storage(input,memory->getType(),memory->getAlign());
                auto* integer=memory->getType()->isFloatTy()?b.getInt32Ty():llvm::cast<llvm::IntegerType>(type(memory->getType()));
                auto* zero=llvm::ConstantInt::get(integer,0);
                auto* previous=atomic(100,value(memory->getPointerOperand()),zero,zero,memory->getOrdering(),memory->getOrdering(),memory->getSyncScopeID());
                values[&input]=previous;return;
            }
            values[&input] = load(value(memory->getPointerOperand()), memory->getType(), unsigned(memory->getAlign().value())); return;
        }
        if (auto* memory = llvm::dyn_cast<llvm::StoreInst>(&input)) {
            if (memory->isAtomic()) {validate_atomic_storage(input,memory->getValueOperand()->getType(),memory->getAlign());atomic(llvm::AtomicRMWInst::Xchg,value(memory->getPointerOperand()),value(memory->getValueOperand()),nullptr,memory->getOrdering(),llvm::AtomicOrdering::Monotonic,memory->getSyncScopeID());return;}
            store(value(memory->getPointerOperand()), value(memory->getValueOperand()), unsigned(memory->getAlign().value()), memory->getValueOperand()->getType()); return;
        }
        if (auto* operation = llvm::dyn_cast<llvm::CallInst>(&input)) { if (auto* out = call(*operation)) values[&input] = canonical(out,input.getType()); return; }
        if (auto* control = llvm::dyn_cast<llvm::SwitchInst>(&input)) {
            auto* out = b.CreateSwitch(value(control->getCondition()), blocks.at(control->getDefaultDest()), control->getNumCases());
            for (auto item : control->cases()) out->addCase(llvm::cast<llvm::ConstantInt>(value(item.getCaseValue())), blocks.at(item.getCaseSuccessor())); return;
        }
        if (llvm::isa<llvm::UnreachableInst>(input)) { b.CreateUnreachable(); return; }
        if (auto* cast = llvm::dyn_cast<llvm::CastInst>(&input); cast && cast->getSrcTy()->isIntegerTy() && cast->getDestTy()->isIntegerTy()) {
            auto* source = cast->getOpcode() == llvm::Instruction::SExt ? signed_value(cast->getOperand(0)) : value(cast->getOperand(0));
            auto* extended=cast->getOpcode()==llvm::Instruction::SExt?b.CreateSExtOrTrunc(source,type(cast->getDestTy())):b.CreateZExtOrTrunc(source,type(cast->getDestTy()));
            values[&input] = canonical(extended, cast->getDestTy()); return;
        }
        if(auto* cast=llvm::dyn_cast<llvm::CastInst>(&input);cast && cast->getSrcTy()->isIntegerTy() && cast->getDestTy()->isFloatingPointTy()) {
            auto* converted=cast->getOpcode()==llvm::Instruction::SIToFP?b.CreateSIToFP(signed_value(cast->getOperand(0)),cast->getDestTy()):b.CreateUIToFP(value(cast->getOperand(0)),cast->getDestTy());
            values[&input]=b.CreateBitCast(converted,type(cast->getDestTy()));return;
        }
        if(auto* cast=llvm::dyn_cast<llvm::CastInst>(&input);cast && cast->getSrcTy()->isFloatingPointTy() && cast->getDestTy()->isIntegerTy()) {
            auto* real=b.CreateBitCast(value(cast->getOperand(0)),cast->getSrcTy());
            auto* converted=cast->getOpcode()==llvm::Instruction::FPToSI?b.CreateFPToSI(real,type(cast->getDestTy())):b.CreateFPToUI(real,type(cast->getDestTy()));
            values[&input]=canonical(converted,cast->getDestTy()); return;
        }
        if (auto* binary = llvm::dyn_cast<llvm::BinaryOperator>(&input); binary && input.getType()->isIntegerTy()) {
            bool signed_op = binary->getOpcode() == llvm::Instruction::SDiv || binary->getOpcode() == llvm::Instruction::SRem || binary->getOpcode() == llvm::Instruction::AShr;
            auto* left = signed_op ? signed_value(binary->getOperand(0)) : value(binary->getOperand(0));
            auto* right = signed_op && binary->getOpcode() != llvm::Instruction::AShr ? signed_value(binary->getOperand(1)) : value(binary->getOperand(1));
            if(binary->getOpcode()==llvm::Instruction::SRem) {
                auto* zero=llvm::ConstantInt::get(left->getType(),0);
                auto* negative=b.CreateICmpSLT(left,zero);
                auto* numerator=b.CreateSelect(negative,b.CreateNeg(left),left);
                auto* denominator=b.CreateSelect(b.CreateICmpSLT(right,zero),b.CreateNeg(right),right);
                auto* remainder=b.CreateURem(numerator,denominator);
                values[&input]=canonical(b.CreateSelect(negative,b.CreateNeg(remainder),remainder),input.getType());return;
            }
            values[&input] = canonical(b.CreateBinOp(binary->getOpcode(), left, right), input.getType()); return;
        }
        if (auto* compare = llvm::dyn_cast<llvm::ICmpInst>(&input)) {
            values[&input] = b.CreateICmp(compare->getPredicate(), compare->isSigned() ? signed_value(compare->getOperand(0)) : value(compare->getOperand(0)), compare->isSigned() ? signed_value(compare->getOperand(1)) : value(compare->getOperand(1))); return;
        }
        if (auto* inserted = llvm::dyn_cast<llvm::InsertValueInst>(&input)) { values[&input] = b.CreateInsertValue(value(inserted->getAggregateOperand()), value(inserted->getInsertedValueOperand()), inserted->getIndices()); return; }
        if (auto* extracted = llvm::dyn_cast<llvm::ExtractValueInst>(&input)) { values[&input] = b.CreateExtractValue(value(extracted->getAggregateOperand()), extracted->getIndices()); return; }
        if (!llvm::isa<llvm::BinaryOperator>(input) && !llvm::isa<llvm::CmpInst>(input) && !llvm::isa<llvm::CastInst>(input) && !llvm::isa<llvm::UnaryOperator>(input)
            && !llvm::isa<llvm::ExtractValueInst>(input) && !llvm::isa<llvm::InsertValueInst>(input))
            fail(input, "unsupported LLVM operation");
        auto* out = input.clone();
        for (unsigned i = 0; i < input.getNumOperands(); ++i) out->setOperand(i, value(input.getOperand(i)));
        b.Insert(out); values[&input] = out;
    }
    std::unique_ptr<llvm::Module> run() {
        assert_cancellation = kernel_entry && (shader_service_requirements(kernel) & device_service_assert);
        collective_storage=native_atomic_polling(kernel);
        for (const auto& bb : kernel) for (const auto& input : bb) if (auto* stored = llvm::dyn_cast<llvm::StoreInst>(&input))
            partial_stores |= stored->getAlign().value() < 4 || stored->getValueOperand()->getType()->isIntegerTy(1) || stored->getValueOperand()->getType()->isIntegerTy(8) || stored->getValueOperand()->getType()->isIntegerTy(16);
        for(const auto& bb:kernel) for(const auto& input:bb) if(auto* call=llvm::dyn_cast<llvm::CallInst>(&input);call && call->getCalledFunction()) {
            auto* function=call->getCalledFunction(); auto name=function->hasFnAttribute("cudavm.cuda.origin")?function->getFnAttribute("cudavm.cuda.origin").getValueAsString():function->getName();
            auto operation=native_collective(name); collective_storage|=operation>=0; named_barriers|=operation>=36;
        }
        shader_local_layout local_layout(kernel);
        local_offsets = local_layout.allocations; byval_offsets = local_layout.arguments;
        local_bytes = local_layout.bytes();
        if (!kernel_entry && kernel.hasFnAttribute("cudavm.service.storage")) {
            if (local_bytes) fail(kernel, "device output helper retained private allocations");
            local_bytes = shader_local_capacity;
        }
        arena_local = kernel.hasFnAttribute("cudavm.service.storage") || ((!opencl || kernel.hasFnAttribute("cudavm.call.frames")) && local_bytes == shader_local_capacity);
        std::set<const llvm::Constant*> visited;
        std::function<void(const llvm::Value*)> collect = [&](const llvm::Value* value) {
            auto* constant = llvm::dyn_cast<llvm::Constant>(value); if (!constant || !visited.insert(constant).second) return;
            if (auto* global = llvm::dyn_cast<llvm::GlobalVariable>(constant)) {
                if (global->getAddressSpace() == 3 && !shared_offsets.contains(global)) {
                    auto bytes = dl.getTypeAllocSize(global->getValueType()).getFixedValue();
                    if (!bytes) {dynamic_shared=true;shared_offsets[global]=0;return;}
                    if (global->hasInitializer() && !llvm::isa<llvm::UndefValue>(global->getInitializer()) && !llvm::isa<llvm::PoisonValue>(global->getInitializer()))
                        fail(*global, "shared initializers require a native block prologue");
                    shared_offsets[global] = 0;
                }
                return;
            }
            for (const auto& operand : constant->operands()) collect(operand.get());
        };
        for (const auto& block : kernel) for (const auto& input : block) for (const auto& operand : input.operands()) collect(operand.get());
        for(const auto& global:source.globals()) if(shared_offsets.contains(&global)) {
            auto bytes=dl.getTypeAllocSize(global.getValueType()).getFixedValue();if(!bytes)continue;
            shared_bytes=uint32_t(llvm::alignTo(shared_bytes,global.getAlign().value_or(dl.getABITypeAlign(global.getValueType())).value()));
            if(bytes>32768 || shared_bytes>32768-bytes) fail(global,"static shared memory exceeds native profile");
            shared_offsets[&global]=shared_bytes;shared_bytes+=uint32_t(bytes);
        }
        uint64_t dynamic_alignment=16;
        for(const auto& [global,offset]:shared_offsets) if(!dl.getTypeAllocSize(global->getValueType()).getFixedValue())
            dynamic_alignment=std::max(dynamic_alignment,global->getAlign().value_or(dl.getABITypeAlign(global->getValueType())).value());
        shared_bytes=uint32_t(llvm::alignTo(shared_bytes,dynamic_alignment));
        for(auto& [global,offset]:shared_offsets) if(!dl.getTypeAllocSize(global->getValueType()).getFixedValue()) offset=shared_bytes;
        auto scratch=(collective_storage?native_collective_shared_bytes+(kernel.hasFnAttribute("cudavm.call.frames")?4096u:0u):0u)+(named_barriers?native_named_barrier_shared_bytes:0u);
        if(shared_bytes+scratch>32768) fail(kernel,"native shared storage and collective scratch exceed 32 KiB");
        for (const auto& global : source.globals()) if ((global.getAddressSpace() == 1 || global.getAddressSpace() == 4) && global.hasInitializer()) globals[&global] = uint32_t(globals.size());
        result->setTargetTriple(llvm::Triple(dxil ? "dxil-pc-shadermodel6.0-compute" : opencl ? "spirv64v1.0-unknown-unknown-opencl" : "spirv1.5-unknown-vulkan1.2-compute"));
        // CUDA offsets were computed with dl above. Physical resource pointers
        // belong to the SPIR-V environment and must use its pointer layout.
        result->setDataLayout(result->getTargetTriple().computeDataLayout());
        auto words = service_shared_words ? service_shared_words : dynamic_shared?(32768-scratch)/4:std::max(2u, (shared_bytes + 3) / 4);
        auto* storage = llvm::ArrayType::get(b.getInt32Ty(), words);
        bool service_storage = kernel.hasFnAttribute("cudavm.service.storage");
        shared = new llvm::GlobalVariable(*result, storage, false, service_storage ? llvm::GlobalValue::ExternalLinkage : llvm::GlobalValue::InternalLinkage,
            service_shared_words ? nullptr : llvm::UndefValue::get(storage), "cudavm.shared", nullptr, llvm::GlobalVariable::NotThreadLocal, 3);
        if(collective_storage) {
            auto* scratch_type=llvm::ArrayType::get(b.getInt32Ty(),2048);
            collective_data=new llvm::GlobalVariable(*result,scratch_type,false,llvm::GlobalValue::InternalLinkage,llvm::UndefValue::get(scratch_type),"cudavm.collective.data",nullptr,llvm::GlobalVariable::NotThreadLocal,3);
            auto make_state=[&](unsigned words,const char* name){auto* t=llvm::ArrayType::get(b.getInt32Ty(),words);return new llvm::GlobalVariable(*result,t,false,llvm::GlobalValue::InternalLinkage,llvm::UndefValue::get(t),name,nullptr,llvm::GlobalVariable::NotThreadLocal,3);};
            collective_pc=make_state(1024,"cudavm.collective.pc");
            collective_masks=make_state(1024,"cudavm.collective.masks");
            if(kernel.hasFnAttribute("cudavm.call.frames")) collective_frames=make_state(1024,"cudavm.collective.frames");
            collective_control=new llvm::GlobalVariable(*result,b.getInt32Ty(),false,llvm::GlobalValue::InternalLinkage,llvm::UndefValue::get(b.getInt32Ty()),"cudavm.collective.control",nullptr,llvm::GlobalVariable::NotThreadLocal,3);
            collective_control->setAlignment(llvm::Align(8));
            if(named_barriers) named_barrier_state=make_state(48,"cudavm.barrier.state");
        }
        if (local_bytes && !opencl && !arena_local) {
            auto* local_type = llvm::ArrayType::get(b.getInt32Ty(), std::max(2u,(local_bytes + 3) / 4));
            local_storage = new llvm::GlobalVariable(*result, local_type, false, llvm::GlobalValue::InternalLinkage, llvm::UndefValue::get(local_type), "cudavm.local", nullptr, llvm::GlobalVariable::NotThreadLocal, dxil ? 0 : 10);
        }
        make_resolver(); read = make_access(false); write = make_access(true);
        std::vector<llvm::Type*> signature;
        if(!kernel_entry) for(const auto& argument:kernel.args()) signature.push_back(type(argument.getType()));
        if(opencl) for (unsigned i=0;i<packet_resource_count;++i) signature.push_back(resource_pointer_type());
        entry = llvm::Function::Create(llvm::FunctionType::get(kernel_entry?b.getVoidTy():type(kernel.getReturnType()),signature,false),llvm::GlobalValue::ExternalLinkage,kernel_entry?(opencl?"main_kernel":"main"):"cudavm.native.helper."+kernel.getName().str(),*result);
        if (kernel.hasFnAttribute("cudavm.service.helper")) entry->addFnAttr(llvm::Attribute::NoInline);
        if (opencl && kernel_entry) entry->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
        else if(kernel_entry) { entry->addFnAttr("hlsl.numthreads", std::to_string(group[0]) + "," + std::to_string(group[1]) + "," + std::to_string(group[2])); entry->addFnAttr("hlsl.shader", "compute"); }
        result->getOrInsertNamedMetadata("cudavm.ir.stage")->addOperand(llvm::MDNode::get(b.getContext(), llvm::MDString::get(b.getContext(), dxil ? "cudavm.native.dxil" : "cudavm.native.spirv")));
        if (dxil && kernel_entry) result->getOrInsertNamedMetadata("dx.valver")->addOperand(llvm::MDNode::get(b.getContext(),
            {llvm::ConstantAsMetadata::get(b.getInt32(1)), llvm::ConstantAsMetadata::get(b.getInt32(8))}));
        for (const auto& block : kernel) tails[&block] = blocks[&block] = llvm::BasicBlock::Create(b.getContext(), block.getName(), entry);
        b.SetInsertPoint(blocks.at(&kernel.getEntryBlock()));
        if (local_bytes && opencl && !arena_local) opencl_local=b.CreateAlloca(llvm::ArrayType::get(b.getInt32Ty(),std::max(2u,(local_bytes+3)/4)),nullptr,"cudavm.local");
        if (local_layout.dynamic) {
            stack_top = b.CreateAlloca(b.getInt32Ty(), nullptr, "cudavm.stack.top");
            b.CreateStore(b.getInt32(local_layout.static_bytes), stack_top);
        }
        uint64_t offset = 0;
        for (unsigned i = 0; i < kernel.arg_size(); ++i) {
            if(!kernel_entry) {
                llvm::Value* argument = entry->getArg(i);
                // Planned printf buffers come from a checked caller alloca.
                if (i == 1 && kernel.getName().starts_with("__cudavm_service_printf"))
                    argument = b.CreateOr(b.CreateAnd(argument, b.getInt64(shader_local_capacity - 1)), b.getInt64(0xffffffff80000000ull));
                values[kernel.getArg(i)] = argument; continue;
            }
            auto* byval = kernel.getParamByValType(i); auto* t = byval ? byval : kernel.getArg(i)->getType();
            auto alignment = byval && kernel.getParamAlign(i) ? kernel.getParamAlign(i)->value() : dl.getABITypeAlign(t).value();
            offset = llvm::alignTo(offset, alignment);
            if (byval) {
                auto destination=0xffffffff80000000ull+byval_offsets.at(kernel.getArg(i));
                values[kernel.getArg(i)] = b.getInt64(destination);
                auto bytes=dl.getTypeAllocSize(byval).getFixedValue();
                for(uint64_t at=0;at<bytes;) {
                    auto whole=bytes-at>=4 && (offset+at)%4==0 && (destination+at)%4==0;
                    auto* scalar=whole?b.getInt32Ty():b.getInt8Ty();
                    auto* contents=load(b.getInt64(0xffffffffc0000000ull+offset+at),scalar,whole?4:1);
                    store(b.getInt64(destination+at),contents,whole?4:1,scalar);
                    at+=whole?4:1;
                }
            }
            else {
                auto* argument_type = kernel.getArg(i)->getType(); auto bits = argument_type->isPointerTy() ? 64u : argument_type->getPrimitiveSizeInBits().getFixedValue();
                if (!bits || bits > 64 || (!argument_type->isIntegerTy() && !argument_type->isPointerTy() && !argument_type->isFloatingPointTy())) fail(*kernel.getArg(i), "unsupported kernel argument type");
                auto* word = b.getInt32(uint32_t(8 + offset / 4));
                llvm::Value* argument = bits > 32 ? packet64(word) : packet(word);
                if (offset % 4) argument = b.CreateLShr(argument, llvm::ConstantInt::get(argument->getType(), (offset % 4) * 8));
                if (argument_type->isIntegerTy()) argument = canonical(b.CreateZExtOrTrunc(argument, type(argument_type)), argument_type);
                values[kernel.getArg(i)] = argument;
            }
            offset += dl.getTypeAllocSize(t).getFixedValue();
        }
        for (const auto& block : kernel) for (const auto& input : block) if (auto* phi = llvm::dyn_cast<llvm::PHINode>(&input)) {
            auto* out = llvm::PHINode::Create(type(phi->getType()), phi->getNumIncomingValues(), phi->getName(), blocks.at(&block)); values[phi] = out;
        }
        for (auto* block : llvm::ReversePostOrderTraversal<llvm::Function*>(&kernel)) {
            b.SetInsertPoint(blocks.at(block)); for (const auto& input : *block) instruction(input);
        }
        for (const auto& block : kernel) for (const auto& input : block) if (auto* phi = llvm::dyn_cast<llvm::PHINode>(&input)) {
            auto* out = llvm::cast<llvm::PHINode>(values.at(phi));
            for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
                b.SetInsertPoint(tails.at(phi->getIncomingBlock(i))->getTerminator());
                out->addIncoming(value(phi->getIncomingValue(i)), tails.at(phi->getIncomingBlock(i)));
            }
        }
        if(collective_storage) make_continuations();
        if ((kernel.hasFnAttribute("cudavm.call.frames") || shader_service_requirements(kernel)) && !collective_storage && !opencl && !dxil) {
            // Residual calls already have explicit return continuations. Use
            // the owned target's deterministic CFG dispatcher directly instead
            // of speculatively restructuring call/return or service retry CFGs.
            // Collective kernels already have an acyclic step dispatcher;
            // wrapping it in a second dispatcher makes driver compilation grow.
            entry->addFnAttr("cudavm.spirv.force.dispatch");
        }
        return std::move(result);
    }
};
}
void scalarize_native_aggregate_phis(llvm::Module& module) {
    std::vector<llvm::PHINode*> pending;
    for (auto& function : module) for (auto& block : function) for (auto& phi : block.phis())
        if (phi.getType()->isAggregateType()) pending.push_back(&phi);
    for (size_t index = 0; index < pending.size(); ++index) {
        auto* phi = pending[index]; auto* type = phi->getType();
        unsigned count = llvm::isa<llvm::StructType>(type) ? type->getStructNumElements() : unsigned(type->getArrayNumElements());
        llvm::IRBuilder<> builder(&*phi->getParent()->getFirstInsertionPt());
        llvm::Value* aggregate = llvm::PoisonValue::get(type);
        std::vector<llvm::PHINode*> fields;
        for (unsigned component = 0; component < count; ++component) {
            auto* element = type->isStructTy() ? llvm::cast<llvm::StructType>(type)->getElementType(component) : llvm::cast<llvm::ArrayType>(type)->getElementType();
            auto* field = llvm::PHINode::Create(element, phi->getNumIncomingValues(), phi->getName()+".field", phi->getIterator());
            fields.push_back(field); aggregate = builder.CreateInsertValue(aggregate, field, component);
            if (field->getType()->isAggregateType()) pending.push_back(field);
        }
        phi->replaceAllUsesWith(aggregate);
        std::map<llvm::BasicBlock*, std::vector<llvm::Value*>> edge_values;
        for (unsigned incoming = 0; incoming < phi->getNumIncomingValues(); ++incoming) {
            auto* predecessor = phi->getIncomingBlock(incoming);
            builder.SetInsertPoint(predecessor->getTerminator());
            auto [edge, inserted] = edge_values.try_emplace(predecessor);
            if (inserted) for (unsigned component = 0; component < count; ++component)
                edge->second.push_back(builder.CreateExtractValue(phi->getIncomingValue(incoming), component));
            for (unsigned component = 0; component < count; ++component) fields[component]->addIncoming(edge->second[component], predecessor);
        }
        phi->eraseFromParent();
    }
}
std::unique_ptr<llvm::Module> prepare_native_shader_module(const llvm::Module& input) {
    auto module = llvm::CloneModule(input);
    resolve_device_function_aliases(*module, true); resolve_device_data_aliases(*module, true); normalize_device_relocations(*module);
    link_shader_services(*module);
    // Device string literals and constant local-array initializers can use
    // generic-address-space globals. Give them persistent module storage with
    // the same bytes, retaining generic pointer uses through an address cast.
    std::vector<llvm::GlobalVariable*> constants;
    for (auto& global : module->globals())
        if (global.getAddressSpace() == 0 && global.isConstant() && global.hasInitializer() && !global.getName().starts_with("llvm.")) constants.push_back(&global);
    for (auto* global : constants) {
        auto* replacement = new llvm::GlobalVariable(*module, global->getValueType(), true, global->getLinkage(),
            global->getInitializer(), "", nullptr, global->getThreadLocalMode(), 4);
        replacement->copyAttributesFrom(global); replacement->takeName(global);
        global->replaceAllUsesWith(llvm::ConstantExpr::getAddrSpaceCast(replacement, global->getType())); global->eraseFromParent();
    }
    std::vector<llvm::MemIntrinsic*> memory_operations;
    for (auto& function : *module) for (auto& block : function) for (auto& instruction : block)
        if (auto* memory = llvm::dyn_cast<llvm::MemIntrinsic>(&instruction)) memory_operations.push_back(memory);
    llvm::TargetTransformInfo memory_target(module->getDataLayout());
    for (auto* memory : memory_operations) {
        if (auto* copy = llvm::dyn_cast<llvm::MemCpyInst>(memory)) llvm::expandMemCpyAsLoop(copy, memory_target);
        else if (auto* move = llvm::dyn_cast<llvm::MemMoveInst>(memory)) {
            if (!llvm::expandMemMoveAsLoop(move, memory_target)) throw std::runtime_error("cannot lower device memmove");
        } else if (auto* set = llvm::dyn_cast<llvm::MemSetInst>(memory)) llvm::expandMemSetAsLoop(set);
        else throw std::runtime_error("unsupported device memory intrinsic");
        memory->eraseFromParent();
    }
    lower_shader_half(*module);
    lower_half_math_intrinsics(*module);
    scalar_math(*module);
    optimize(*module);
    while (lower_local_calls(*module)) optimize(*module);
    lower_shader_calls(*module);
    for (auto& function : *module) if (function.hasFnAttribute("cudavm.kernel") || function.getCallingConv() == llvm::CallingConv::PTX_Kernel)
        if (shader_service_requirements(function) & device_service_output) function.addFnAttr("cudavm.service.storage");
    return module;
}
std::unique_ptr<llvm::Module> lower_native_shader(const llvm::Module& input, const std::string& symbol, shader_environment environment, const std::array<unsigned,3>& block) {
    auto trace = [&](const char* stage) { if (std::getenv("CUDAVM_TRACE_NATIVE")) std::fprintf(stderr, "native kernel=%s environment=%u stage=%s\n", symbol.c_str(), unsigned(environment), stage); };
    trace("prepare");
    auto family = environment == shader_environment::dxil ? "DXIL" : "SPIR-V";
    auto portable = prepare_native_shader_module(input);
    auto* kernel = portable->getFunction(symbol);
    if (!kernel || kernel->isDeclaration() || (!kernel->hasFnAttribute("cudavm.kernel") && kernel->getCallingConv() != llvm::CallingConv::PTX_Kernel))
        throw std::runtime_error(std::string("native ") + family + " kernel is absent: " + symbol);
    if (kernel->size() > 4096) throw std::runtime_error(std::string("native ") + family + " kernel " + symbol + ": CFG exceeds the 4096-block resource limit");
    if(environment==shader_environment::dxil) {
        llvm::DominatorTree dominators(*kernel);llvm::LoopInfo loops(dominators);
        llvm::ReversePostOrderTraversal<llvm::Function*> order(kernel);
        if(llvm::containsIrreducibleCFG<llvm::BasicBlock*>(order,loops)) {
            llvm::FunctionAnalysisManager analyses; llvm::PassBuilder builder; builder.registerFunctionAnalyses(analyses);
            // Preserve every edge update before moving irreducible entries.
            // In particular, self-loop PHIs must advance on edges that bypass
            // the new guard; moving their values to that guard freezes them.
            llvm::RegToMemPass().run(*kernel, analyses);
            analyses.clear();
            llvm::FixIrreduciblePass().run(*kernel, analyses);
            analyses.clear();
            llvm::PromotePass().run(*kernel, analyses);
        }
    }
    trace("lower");
    auto module = lowering(*portable, *kernel, environment, block).run();
    std::set<std::string> imported;
    for(;;) {
        std::string symbol;
        for(const auto& function:*module) if(function.isDeclaration() && function.hasFnAttribute("cudavm.source.helper")) {
            symbol=function.getFnAttribute("cudavm.source.helper").getValueAsString().str();break;
        }
        if(symbol.empty()) break;
        if(!imported.insert(symbol).second) throw std::runtime_error("native math helper definition is unavailable: "+symbol);
        auto* helper=portable->getFunction(symbol);
        if(!helper || helper->isDeclaration()) throw std::runtime_error("native math helper is absent: "+symbol);
        unsigned shared_words = 0;
        if (helper->hasFnAttribute("cudavm.service.storage"))
            shared_words = unsigned(llvm::cast<llvm::ArrayType>(module->getNamedGlobal("cudavm.shared")->getValueType())->getNumElements());
        auto lowered=lowering(*portable,*helper,environment,block,shared_words).run();
        if(llvm::Linker(*module).linkInModule(std::move(lowered))) throw std::runtime_error("native math helper linking failed: "+symbol);
    }
    trace("verify");
    if (auto* storage = module->getNamedGlobal("cudavm.shared"); storage && storage->hasInitializer()) storage->setLinkage(llvm::GlobalValue::InternalLinkage);
    if (llvm::any_of(*module, [](const auto& function) { return function.hasFnAttribute("cudavm.image.operations"); })) {
        llvm::LoopAnalysisManager loops; llvm::FunctionAnalysisManager functions;
        llvm::CGSCCAnalysisManager calls; llvm::ModuleAnalysisManager modules; llvm::PassBuilder builder;
        builder.registerLoopAnalyses(loops); builder.registerFunctionAnalyses(functions);
        builder.registerCGSCCAnalyses(calls); builder.registerModuleAnalyses(modules);
        builder.crossRegisterProxies(loops, functions, calls, modules);
        llvm::AlwaysInlinerPass().run(*module, modules);
        for (auto& function : *module) function.removeFnAttr("cudavm.image.operations");
    }
    // SPIR-V's aggregate instruction rewriting does not preserve aggregate
    // PHIs. Keep each field's edge value explicit before entering codegen.
    scalarize_native_aggregate_phis(*module);
    if(auto* directory=std::getenv("CUDAVM_DUMP_NATIVE_DIR")) {
        std::string text;llvm::raw_string_ostream output(text);module->print(output,nullptr);
        write_file(std::string(directory)+"/"+std::to_string(std::hash<std::string>{}(symbol))+"-"+std::to_string(unsigned(environment))+".ll",text);
    }
    // Shader switch selectors are 32-bit. Expand other selectors into ordinary
    // comparisons/branches in LLVM, with the pass updating PHI predecessors.
    // The same native CFG works on every shader adapter.
    for(auto& function:*module) if(!function.isDeclaration()) {
        bool wide=false;
        for(const auto& bb:function) if(auto* dispatch=llvm::dyn_cast<llvm::SwitchInst>(bb.getTerminator()))
            wide|=!dispatch->getCondition()->getType()->isIntegerTy(32) || function.getName().starts_with("cudavm.native.helper.");
        if(wide) {
            llvm::FunctionAnalysisManager analyses;llvm::PassBuilder builder;builder.registerFunctionAnalyses(analyses);
            llvm::LowerSwitchPass().run(function,analyses);
        }
    }
    // Imported scalar helpers are implementation details. Strip the unused
    // memory helpers and storage that each independent lowering constructed.
    for(auto& function:*module) if(function.getName().starts_with("cudavm.native.helper.") && !function.isDeclaration())
        function.setLinkage(llvm::GlobalValue::InternalLinkage);
    llvm::ModuleAnalysisManager analyses;
    llvm::PassBuilder builder; builder.registerModuleAnalyses(analyses);
    llvm::GlobalDCEPass().run(*module, analyses);
    llvm::FunctionAnalysisManager functions; builder.registerFunctionAnalyses(functions);
    llvm::LoopAnalysisManager loops; builder.registerLoopAnalyses(loops);
    llvm::CGSCCAnalysisManager calls; builder.registerCGSCCAnalyses(calls);
    builder.crossRegisterProxies(loops,functions,calls,analyses);
    for(auto& function:*module) if(!function.isDeclaration())
        llvm::InstSimplifyPass().run(function,functions);
    std::string diagnostics; llvm::raw_string_ostream out(diagnostics);
    if (llvm::verifyModule(*module, &out)) throw std::runtime_error(std::string("invalid native ") + family + " IR:\n" + diagnostics);
    return module;
}
}
