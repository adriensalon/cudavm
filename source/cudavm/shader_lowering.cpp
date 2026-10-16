#include "shader_lowering.hpp"
#include "shader_collectives.hpp"
#include "device_conversions.hpp"

#include <map>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsSPIRV.h>
#include <llvm/IR/IntrinsicsDirectX.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/Scalarizer.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/InstSimplifyPass.h>
#include <llvm/Transforms/Scalar/Reg2Mem.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/FixIrreducible.h>
namespace cudavm {
namespace {
void scalar_math(llvm::Module& module) {
    auto* divide = module.getFunction("__cudavm_native_divide");
    auto* compare = module.getFunction("__cudavm_native_compare");
    auto* square_root = module.getFunction("__cudavm_native_sqrt");
    auto* maximum = module.getFunction("__cudavm_native_max");
    auto* integral = module.getFunction("__cudavm_native_integral");
    auto numeric=[&](const std::string& name){return module.getFunction("__cudavm_native_word_"+name);};
    std::vector<llvm::Instruction*> work;
    for (auto& function : module) for (auto& bb : function) for (auto& instruction : bb) work.push_back(&instruction);
    for (auto* instruction : work) {
        llvm::IRBuilder<> builder(instruction); llvm::Value* replacement = nullptr;
        auto bits=[&](llvm::Value* value){return builder.CreateBitCast(value,value->getType()->isDoubleTy()?builder.getInt64Ty():builder.getInt32Ty());};
        if(instruction->getOpcode()==llvm::Instruction::FNeg && instruction->getType()->isDoubleTy()) replacement=builder.CreateBitCast(builder.CreateXor(bits(instruction->getOperand(0)),builder.getInt64(0x8000000000000000ull)),builder.getDoubleTy());
        if(auto* binary=llvm::dyn_cast<llvm::BinaryOperator>(instruction);binary && binary->getType()->isDoubleTy()) {
            const char* operation=binary->getOpcode()==llvm::Instruction::FAdd?"add":binary->getOpcode()==llvm::Instruction::FSub?"sub":binary->getOpcode()==llvm::Instruction::FMul?"mul":binary->getOpcode()==llvm::Instruction::FDiv?"div":nullptr;
            if(operation) if(auto* helper=numeric(std::string("f64_")+operation)) replacement=builder.CreateBitCast(builder.CreateCall(helper,{bits(binary->getOperand(0)),bits(binary->getOperand(1))}),builder.getDoubleTy());
        }
        if(auto* operation=llvm::dyn_cast<llvm::FCmpInst>(instruction);operation && operation->getOperand(0)->getType()->isDoubleTy())
            if(auto* helper=numeric("f64_compare")) replacement=builder.CreateICmpNE(builder.CreateCall(helper,{bits(operation->getOperand(0)),bits(operation->getOperand(1)),builder.getInt32(operation->getPredicate())}),builder.getInt32(0));
        if(auto* cast=llvm::dyn_cast<llvm::CastInst>(instruction)) {
            auto* source=cast->getOperand(0);
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
        if (divide && !divide->isDeclaration() && instruction->getOpcode() == llvm::Instruction::FDiv && instruction->getType()->isFloatTy())
            replacement = builder.CreateCall(divide,{instruction->getOperand(0),instruction->getOperand(1)});
        if (compare && !compare->isDeclaration()) if (auto* operation = llvm::dyn_cast<llvm::FCmpInst>(instruction); operation && operation->getOperand(0)->getType()->isFloatTy())
            replacement = builder.CreateICmpNE(builder.CreateCall(compare,{builder.CreateBitCast(operation->getOperand(0),builder.getInt32Ty()),builder.CreateBitCast(operation->getOperand(1),builder.getInt32Ty()),builder.getInt32(operation->getPredicate())}),builder.getInt32(0));
        if (auto* call = llvm::dyn_cast<llvm::CallInst>(instruction); call && call->getCalledFunction()) {
            auto* function = call->getCalledFunction();
            auto name = function->hasFnAttribute("cudavm.cuda.origin") ? function->getFnAttribute("cudavm.cuda.origin").getValueAsString() : function->getName();
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
void optimize(llvm::Module& module) {
    for (auto& f : module) if (!f.isDeclaration()) {
        f.removeFnAttr(llvm::Attribute::OptimizeNone); f.removeFnAttr(llvm::Attribute::NoInline);
        if(f.hasFnAttribute("cudavm.math") || f.getName().starts_with("_ZN13cudavm") || f.getName().starts_with("__cudavm_native_word_")) {
            f.removeFnAttr(llvm::Attribute::AlwaysInline); f.addFnAttr(llvm::Attribute::NoInline);
        } else if (!f.hasFnAttribute("cudavm.kernel") && !f.hasFnAttribute("hlsl.shader") && f.getCallingConv() != llvm::CallingConv::PTX_Kernel && f.getCallingConv() != llvm::CallingConv::SPIR_KERNEL)
            f.addFnAttr(llvm::Attribute::AlwaysInline);
    }
    llvm::LoopAnalysisManager loops; llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager calls; llvm::ModuleAnalysisManager modules;
    llvm::PassBuilder pb;
    pb.registerModuleAnalyses(modules); pb.registerCGSCCAnalyses(calls);
    pb.registerFunctionAnalyses(functions); pb.registerLoopAnalyses(loops);
    pb.crossRegisterProxies(loops, functions, calls, modules);
    llvm::ModulePassManager passes; passes.addPass(llvm::AlwaysInlinerPass());
    llvm::FunctionPassManager fp;
    fp.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG)); fp.addPass(llvm::PromotePass());
    fp.addPass(llvm::InstCombinePass()); fp.addPass(llvm::SimplifyCFGPass());
    llvm::ScalarizerPassOptions scalar; scalar.ScalarizeLoadStore = true;
    fp.addPass(llvm::ScalarizerPass(scalar));
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
    llvm::GlobalVariable* shared = nullptr;
    llvm::Constant* names[2]{};
    std::map<const llvm::Value*, llvm::Value*> values;
    std::map<const llvm::BasicBlock*, llvm::BasicBlock*> blocks;
    std::map<const llvm::GlobalVariable*, uint32_t> globals, shared_offsets;
    std::map<const llvm::AllocaInst*, uint32_t> local_offsets;
    std::map<const llvm::Argument*, uint32_t> byval_offsets;
    llvm::GlobalVariable* local_storage = nullptr;
    llvm::AllocaInst* opencl_local = nullptr;
    uint32_t local_bytes = 0;
    bool partial_stores = false;
    bool dynamic_shared = false, collective_storage = false;
    llvm::GlobalVariable* collective_data = nullptr;
    std::map<unsigned,llvm::Function*> collective_helpers;
    std::map<std::pair<unsigned,unsigned>, llvm::Function*> atomic_helpers;
    uint32_t shared_bytes = 0;
    bool opencl;
    bool dxil;
    std::array<unsigned,3> group;
    bool kernel_entry;
    lowering(llvm::Module& module, llvm::Function& f, shader_environment environment, const std::array<unsigned,3>& block) : source(module), kernel(f), dl(module.getDataLayout()),
        result(std::make_unique<llvm::Module>("cudavm.native.shader." + f.getName().str(), module.getContext())), b(module.getContext()),
        opencl(environment == shader_environment::opencl), dxil(environment == shader_environment::dxil), group(block),
        kernel_entry(f.hasFnAttribute("cudavm.kernel") || f.getCallingConv()==llvm::CallingConv::PTX_Kernel) {}
    [[noreturn]] void fail(const llvm::Value& value, const char* reason) {
        std::string text; llvm::raw_string_ostream out(text); value.print(out);
        throw std::runtime_error(std::string("native ") + (dxil ? "DXIL" : "SPIR-V") + " kernel " + kernel.getName().str() + ": " + reason + "\n" + text);
    }
    llvm::Type* type(llvm::Type* input) {
        if (input->isPointerTy()) return b.getInt64Ty();
        if(input->isFloatTy()) return b.getInt32Ty();
        if(input->isDoubleTy()) return b.getInt64Ty();
        if (input->isIntegerTy(8) || input->isIntegerTy(16)) return b.getInt32Ty();
        if (auto* structure = llvm::dyn_cast<llvm::StructType>(input)) {
            std::vector<llvm::Type*> elements; for (auto* element : structure->elements()) elements.push_back(type(element));
            return llvm::StructType::get(b.getContext(), elements, structure->isPacked());
        }
        return input;
    }
    llvm::Value* canonical(llvm::Value* output, llvm::Type* source_type) {
        if (source_type->isIntegerTy(8) || source_type->isIntegerTy(16))
            return b.CreateAnd(output, b.getInt32((1u << source_type->getIntegerBitWidth()) - 1));
        return output;
    }
    llvm::Value* signed_value(const llvm::Value* input) {
        auto* output = value(input); auto bits = input->getType()->isPointerTy() ? 64u : input->getType()->getIntegerBitWidth();
        return bits > 1 && bits < 32 ? b.CreateAShr(b.CreateShl(output, b.getInt32(32 - bits)), b.getInt32(32 - bits)) : output;
    }
    llvm::Type* resource_pointer_type() { return llvm::PointerType::get(b.getContext(), 1); }
    std::vector<llvm::Type*> arguments(std::initializer_list<llvm::Type*> base) {
        std::vector<llvm::Type*> args(base);
        if (opencl) { args.push_back(resource_pointer_type()); args.push_back(resource_pointer_type()); }
        return args;
    }
    llvm::Value* call_helper(llvm::Function* function, std::initializer_list<llvm::Value*> base) {
        auto first = base.begin();
        std::vector<llvm::Value*> args{b.CreateTrunc(*first, b.getInt32Ty()), b.CreateTrunc(b.CreateLShr(*first, b.getInt64(32)), b.getInt32Ty())};
        args.insert(args.end(), first + 1, base.end());
        if (function->hasFnAttribute("cudavm.native.local")) {
            auto* caller = b.GetInsertBlock()->getParent();
            if (caller == entry) args.push_back(b.CreateInBoundsGEP(opencl_local->getAllocatedType(),opencl_local,{b.getInt32(0),b.getInt32(0)}));
            else args.push_back(caller->getArg(unsigned(caller->arg_size()-3)));
        }
        if (opencl) {
            auto* caller = b.GetInsertBlock()->getParent();
            args.push_back(caller->getArg(unsigned(caller->arg_size() - 2)));
            args.push_back(caller->getArg(unsigned(caller->arg_size() - 1)));
        }
        return b.CreateCall(function, args);
    }
    llvm::Value* word_pointer(unsigned binding, llvm::Value* index) {
        if (dxil) {
            auto* resource = llvm::TargetExtType::get(b.getContext(), "dx.RawBuffer", {b.getInt8Ty()}, {binding ? 0u : 1u, 0});
            auto* handle = b.CreateIntrinsic(resource, llvm::Intrinsic::dx_resource_handlefrombinding,
                {b.getInt32(0), b.getInt32(0), b.getInt32(1), b.getInt32(0), llvm::ConstantPointerNull::get(b.getPtrTy())});
            return b.CreateIntrinsic(b.getPtrTy(), llvm::Intrinsic::dx_resource_getpointer, {handle, b.CreateMul(index, b.getInt32(4))});
        }
        if (opencl) {
            auto* function = b.GetInsertBlock()->getParent();
            return b.CreateGEP(b.getInt32Ty(), function->getArg(unsigned(function->arg_size() - 2 + binding)), index);
        }
        auto* buffer = llvm::TargetExtType::get(b.getContext(), "spirv.VulkanBuffer", {llvm::ArrayType::get(b.getInt32Ty(), 0)}, {12, 1});
        if (!names[binding]) names[binding] = b.CreateGlobalString(binding ? "P" : "M");
        auto* handle = b.CreateIntrinsic(buffer, llvm::Intrinsic::spv_resource_handlefrombinding,
            {b.getInt32(0), b.getInt32(binding), b.getInt32(1), b.getInt32(0), names[binding]});
        return b.CreateIntrinsic(llvm::PointerType::get(b.getContext(), 11), llvm::Intrinsic::spv_resource_getpointer, {handle, index});
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
    void make_resolver() {
        resolve = llvm::Function::Create(llvm::FunctionType::get(b.getInt32Ty(), arguments({b.getInt32Ty(), b.getInt32Ty()}), false), llvm::GlobalValue::InternalLinkage, "cudavm.resolve.address", *result);
        auto* first = llvm::BasicBlock::Create(b.getContext(), "entry", resolve);
        auto* loop = llvm::BasicBlock::Create(b.getContext(), "allocations", resolve);
        auto* test = llvm::BasicBlock::Create(b.getContext(), "range", resolve);
        auto* next = llvm::BasicBlock::Create(b.getContext(), "next", resolve);
        auto* found = llvm::BasicBlock::Create(b.getContext(), "found", resolve);
        auto* absent = llvm::BasicBlock::Create(b.getContext(), "absent", resolve);
        b.SetInsertPoint(first);
        auto* address = pair(resolve->getArg(0), resolve->getArg(1));
        auto* table = b.CreateLShr(packet(b.getInt32(6)), b.getInt32(2));
        auto* count = packet(b.getInt32(7)); b.CreateBr(loop);
        b.SetInsertPoint(loop); auto* i = b.CreatePHI(b.getInt32Ty(), 2); i->addIncoming(b.getInt32(0), first);
        b.CreateCondBr(b.CreateICmpULT(i, count), test, absent);
        b.SetInsertPoint(test); auto* row = b.CreateAdd(table, b.CreateMul(i, b.getInt32(4)));
        auto* base = packet64(row); auto* delta = b.CreateSub(address, base);
        auto* size = b.CreateZExt(packet(b.CreateAdd(row, b.getInt32(2))), b.getInt64Ty());
        b.CreateCondBr(b.CreateAnd(b.CreateICmpUGE(address, base), b.CreateICmpULT(delta, size)), found, next);
        b.SetInsertPoint(next); auto* increment = b.CreateAdd(i, b.getInt32(1)); i->addIncoming(increment, next); b.CreateBr(loop);
        b.SetInsertPoint(found); b.CreateRet(b.CreateAdd(packet(b.CreateAdd(row, b.getInt32(3))), b.CreateTrunc(delta, b.getInt32Ty())));
        b.SetInsertPoint(absent); b.CreateRet(b.getInt32(0));
    }
    llvm::Function* make_access(bool store) {
        auto base = store ? std::vector<llvm::Type*>{b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty()}
            : std::vector<llvm::Type*>{b.getInt32Ty(),b.getInt32Ty()};
        if (opencl && local_bytes) base.push_back(b.getPtrTy());
        if (opencl) {base.push_back(resource_pointer_type());base.push_back(resource_pointer_type());}
        auto* function = llvm::Function::Create(llvm::FunctionType::get(store ? b.getVoidTy() : b.getInt32Ty(),
            base, false),
            llvm::GlobalValue::InternalLinkage, store ? "cudavm.store.word" : "cudavm.load.word", *result);
        if (opencl && local_bytes) function->addFnAttr("cudavm.native.local");
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
                auto* initial = b.CreateLoad(b.getInt32Ty(), pointer()); initial->setAtomic(llvm::AtomicOrdering::Monotonic, scope); initial->setAlignment(llvm::Align(4));
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
        access(global_space, [&] { return word_pointer(0, b.CreateLShr(call_helper(resolve, {address}), b.getInt32(2))); });
        access(shared_space, [&] { return shared_pointer(b.CreateAnd(lo, b.getInt32(0x3fffffffu))); });
        if (local_space) {
          b.SetInsertPoint(local_space);
          auto* offset = b.CreateURem(b.CreateLShr(b.CreateAnd(lo,b.getInt32(0x3fffffffu)),b.getInt32(2)),b.getInt32(std::max(2u,(local_bytes+3)/4)));
          auto* ptr = opencl ? b.CreateInBoundsGEP(b.getInt32Ty(),function->getArg(store ? 4 : 2),offset)
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
    llvm::Value* gep(const llvm::GEPOperator& pointer) {
        llvm::SmallMapVector<llvm::Value*, llvm::APInt, 4> variables; llvm::APInt constant(64, 0);
        if (!pointer.collectOffset(dl, 64, variables, constant)) fail(pointer, "nonlinear pointer offset");
        auto* offset = static_cast<llvm::Value*>(llvm::ConstantInt::get(b.getInt64Ty(), constant));
        for (auto& [index, scale] : variables)
            offset = b.CreateAdd(offset, b.CreateMul(b.CreateSExtOrTrunc(value(index), b.getInt64Ty()), llvm::ConstantInt::get(b.getInt64Ty(), scale)));
        return b.CreateAdd(value(pointer.getPointerOperand()), offset);
    }
    llvm::Value* value(const llvm::Value* input) {
        if (auto found = values.find(input); found != values.end()) return found->second;
        if (auto* global = llvm::dyn_cast<llvm::GlobalVariable>(input)) {
            if (auto at = shared_offsets.find(global); at != shared_offsets.end()) return b.getInt64(0xffffffff40000000ull + at->second);
            if (auto at = globals.find(global); at != globals.end()) {
                auto* start = b.CreateAdd(b.CreateLShr(packet(b.getInt32(6)), b.getInt32(2)), b.CreateMul(packet(b.getInt32(7)), b.getInt32(4)));
                return packet64(b.CreateAdd(start, b.getInt32(at->second * 2)));
            }
            fail(*input, "unavailable global storage");
        }
        if (auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(input)) {
            if (auto* pointer = llvm::dyn_cast<llvm::GEPOperator>(expression)) return gep(*pointer);
            if (expression->getOpcode() == llvm::Instruction::AddrSpaceCast || expression->getOpcode() == llvm::Instruction::BitCast) return value(expression->getOperand(0));
            if (expression->getOpcode() == llvm::Instruction::PtrToInt) return b.CreateZExtOrTrunc(value(expression->getOperand(0)), expression->getType());
            fail(*input, "unsupported constant expression");
        }
        if (llvm::isa<llvm::ConstantPointerNull>(input)) return b.getInt64(0);
        if (auto* integer = llvm::dyn_cast<llvm::ConstantInt>(input)) return llvm::ConstantInt::get(type(integer->getType()), integer->getValue().zextOrTrunc(type(integer->getType())->getIntegerBitWidth()));
        if(auto* floating=llvm::dyn_cast<llvm::ConstantFP>(input);floating && (floating->getType()->isDoubleTy() || floating->getType()->isFloatTy()))
            return llvm::ConstantInt::get(type(floating->getType()),floating->getValueAPF().bitcastToAPInt());
        if (llvm::isa<llvm::UndefValue>(input) || llvm::isa<llvm::PoisonValue>(input)) return llvm::UndefValue::get(type(input->getType()));
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
        if(t->isDoubleTy() || t->isFloatTy()) return loaded;
        if (t->isFloatingPointTy()) return b.CreateBitCast(loaded, t);
        fail(kernel, "aggregate or vector memory access");
    }
    llvm::Value* packet_pair(llvm::Value* address, llvm::Value* lo) { return pair(lo, call_helper(read, {b.CreateAdd(address, b.getInt64(4))})); }
    llvm::Value* byte_shift(llvm::Value* address) {
        auto* lo = b.CreateTrunc(address, b.getInt32Ty());
        auto* hi = b.CreateTrunc(b.CreateLShr(address, b.getInt64(32)), b.getInt32Ty());
        auto* physical = b.CreateSelect(b.CreateICmpEQ(hi, b.getInt32(0xffffffffu)), lo, call_helper(resolve, {address}));
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
    llvm::Function* atomic_helper(unsigned operation, unsigned bits) {
        auto key = std::make_pair(operation, bits);
        if (auto found = atomic_helpers.find(key); found != atomic_helpers.end()) return found->second;
        auto saved = b.saveIP();
        auto* integer = llvm::IntegerType::get(b.getContext(), bits);
        std::vector<llvm::Type*> signature{b.getInt32Ty(),b.getInt32Ty(),integer,integer};
        if (opencl && local_bytes && bits==64) signature.push_back(b.getPtrTy());
        if (opencl) {signature.push_back(resource_pointer_type());signature.push_back(resource_pointer_type());}
        auto* function = llvm::Function::Create(llvm::FunctionType::get(integer,
            signature,false), llvm::GlobalValue::InternalLinkage,
            "cudavm.atomic." + std::to_string(operation) + "." + std::to_string(bits), *result);
        if (opencl && local_bytes && bits==64) function->addFnAttr("cudavm.native.local");
        atomic_helpers[key] = function;
        auto* first = llvm::BasicBlock::Create(b.getContext(), "entry", function); b.SetInsertPoint(first);
        auto* address = pair(function->getArg(0), function->getArg(1));
        auto scope = dxil?llvm::SyncScope::System:b.getContext().getOrInsertSyncScopeID("device");
        if (bits == 64 && opencl) {
            // Physical OpenCL pointers support complete 64-bit atomics. Avoid
            // a software lock whose progress depends on SIMD scheduling.
            auto* global_space=llvm::BasicBlock::Create(b.getContext(),"global",function);
            auto* shared_space=llvm::BasicBlock::Create(b.getContext(),"shared",function);
            b.CreateCondBr(b.CreateICmpEQ(function->getArg(1),b.getInt32(0xffffffffu)),shared_space,global_space);
            auto access=[&](llvm::BasicBlock* block,bool group_memory) {
                b.SetInsertPoint(block);
                auto pointer=[&]() -> llvm::Value* {
                    if(group_memory)return shared_pointer(b.CreateAnd(function->getArg(0),b.getInt32(0x3fffffffu)));
                    auto* offset=call_helper(resolve,{address});
                    return b.CreateGEP(b.getInt64Ty(),function->getArg(unsigned(function->arg_size()-2)),b.CreateLShr(offset,b.getInt32(3)));
                };
                if(operation==llvm::AtomicRMWInst::FAdd || operation==llvm::AtomicRMWInst::FSub) {
                    auto* initial=b.CreateExtractValue(b.CreateAtomicCmpXchg(pointer(),b.getInt64(0),b.getInt64(0),llvm::Align(8),llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::SequentiallyConsistent,scope),0);
                    auto* retry=llvm::BasicBlock::Create(b.getContext(),"add.retry",function);
                    auto* done=llvm::BasicBlock::Create(b.getContext(),"add.done",function);
                    b.CreateBr(retry);b.SetInsertPoint(retry);
                    auto* expected=b.CreatePHI(b.getInt64Ty(),2);expected->addIncoming(initial,block);
                    auto* next=numeric_call(operation==llvm::AtomicRMWInst::FAdd?"__cudavm_native_word_f64_add":"__cudavm_native_word_f64_sub",b.getInt64Ty(),{expected,function->getArg(2)});
                    auto* exchange=b.CreateAtomicCmpXchg(pointer(),expected,next,llvm::Align(8),llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::SequentiallyConsistent,scope);
                    expected->addIncoming(b.CreateExtractValue(exchange,0),retry);
                    b.CreateCondBr(b.CreateExtractValue(exchange,1),done,retry);b.SetInsertPoint(done);b.CreateRet(expected);
                } else if(operation==100) b.CreateRet(b.CreateExtractValue(b.CreateAtomicCmpXchg(pointer(),function->getArg(3),function->getArg(2),llvm::Align(8),llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::SequentiallyConsistent,scope),0));
                else b.CreateRet(b.CreateAtomicRMW(llvm::AtomicRMWInst::BinOp(operation),pointer(),function->getArg(2),llvm::Align(8),llvm::AtomicOrdering::SequentiallyConsistent,scope));
            };
            access(global_space,false);access(shared_space,true);
        } else if (bits == 32) {
            auto* global_space = llvm::BasicBlock::Create(b.getContext(), "global", function);
            auto* shared_space = llvm::BasicBlock::Create(b.getContext(), "shared", function);
            b.CreateCondBr(b.CreateICmpEQ(function->getArg(1), b.getInt32(0xffffffffu)), shared_space, global_space);
            auto access = [&](llvm::BasicBlock* block, bool group_memory) {
                b.SetInsertPoint(block);
                if(dxil && group_memory && operation!=100 && operation!=llvm::AtomicRMWInst::FAdd && operation!=llvm::AtomicRMWInst::FSub) {
                    // DXIL's legacy bitcode reader rejects LLVM's remaining
                    // TGSM atomicrmw encoding. A native CAS loop keeps the full
                    // operation atomic and returns the successful old value.
                    auto* pointer=shared_pointer(b.CreateAnd(function->getArg(0),b.getInt32(0x3fffffffu)));
                    auto* initial=b.CreateExtractValue(b.CreateAtomicCmpXchg(pointer,b.getInt32(0),b.getInt32(0),llvm::Align(4),llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::SequentiallyConsistent,scope),0);
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
                    default:fail(kernel,"unsupported native 32-bit atomic operation");
                    }
                    auto* exchange=b.CreateAtomicCmpXchg(pointer,expected,next,llvm::Align(4),llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::SequentiallyConsistent,scope);
                    expected->addIncoming(b.CreateExtractValue(exchange,0),retry);b.CreateCondBr(b.CreateExtractValue(exchange,1),done,retry);
                    b.SetInsertPoint(done);b.CreateRet(expected);return;
                }
                if(operation==llvm::AtomicRMWInst::FAdd || operation==llvm::AtomicRMWInst::FSub) {
                    auto pointer=[&](){return group_memory?shared_pointer(b.CreateAnd(function->getArg(0),b.getInt32(0x3fffffffu))):word_pointer(0,b.CreateLShr(call_helper(resolve,{address}),b.getInt32(2)));};
                    auto* retry=llvm::BasicBlock::Create(b.getContext(),"add.retry",function);
                    auto* done=llvm::BasicBlock::Create(b.getContext(),"add.done",function);
                    auto* initial=b.CreateLoad(b.getInt32Ty(),pointer());initial->setAlignment(llvm::Align(4));initial->setAtomic(llvm::AtomicOrdering::SequentiallyConsistent,scope);
                    b.CreateBr(retry);b.SetInsertPoint(retry);
                    auto* expected=b.CreatePHI(b.getInt32Ty(),2);expected->addIncoming(initial,block);
                    auto* operand=operation==llvm::AtomicRMWInst::FSub?b.CreateXor(function->getArg(2),b.getInt32(0x80000000u)):function->getArg(2);
                    auto* next=numeric_call("__cudavm_native_word_f32_add",b.getInt32Ty(),{expected,operand,b.getInt32(group_memory?0:1)});
                    auto* exchange=b.CreateAtomicCmpXchg(pointer(),expected,next,llvm::Align(4),llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::SequentiallyConsistent,scope);
                    expected->addIncoming(b.CreateExtractValue(exchange,0),retry);b.CreateCondBr(b.CreateExtractValue(exchange,1),done,retry);b.SetInsertPoint(done);b.CreateRet(expected);return;
                }
                auto* ptr = group_memory ? shared_pointer(b.CreateAnd(function->getArg(0),b.getInt32(0x3fffffffu)))
                    : word_pointer(0,b.CreateLShr(call_helper(resolve,{address}),b.getInt32(2)));
                if (operation == 100) b.CreateRet(b.CreateExtractValue(b.CreateAtomicCmpXchg(ptr,function->getArg(3),function->getArg(2),llvm::Align(4),llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::SequentiallyConsistent,scope),0));
                else b.CreateRet(b.CreateAtomicRMW(llvm::AtomicRMWInst::BinOp(operation),ptr,function->getArg(2),llvm::Align(4),llvm::AtomicOrdering::SequentiallyConsistent,scope));
            };
            access(global_space,false); access(shared_space,true);
        } else {
            // A device-scoped 32-bit lock protects each complete 64-bit update,
            // including the compare and both words. Release occurs before retry
            // lanes return to the acquisition loop; no native 64-bit atomic
            // instruction or stronger shader model is required.
            auto* retry = llvm::BasicBlock::Create(b.getContext(),"acquire",function);
            auto* critical = llvm::BasicBlock::Create(b.getContext(),"update",function);
            auto* joined = llvm::BasicBlock::Create(b.getContext(),"released",function);
            auto* done = llvm::BasicBlock::Create(b.getContext(),"done",function);
            b.CreateBr(retry); b.SetInsertPoint(retry);
            auto* lock = b.CreateAtomicCmpXchg(word_pointer(0,b.getInt32(0)),b.getInt32(0),b.getInt32(1),llvm::Align(4),llvm::AtomicOrdering::SequentiallyConsistent,llvm::AtomicOrdering::SequentiallyConsistent,scope);
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
            auto* release = b.CreateStore(b.getInt32(0),word_pointer(0,b.getInt32(0))); release->setAlignment(llvm::Align(4)); release->setAtomic(llvm::AtomicOrdering::SequentiallyConsistent,scope);
            b.CreateBr(joined); b.SetInsertPoint(joined);
            auto* acquired=b.CreatePHI(b.getInt1Ty(),2);acquired->addIncoming(b.getFalse(),retry);acquired->addIncoming(b.getTrue(),critical);
            auto* observed=b.CreatePHI(b.getInt64Ty(),2);observed->addIncoming(b.getInt64(0),retry);observed->addIncoming(previous,critical);
            b.CreateCondBr(acquired,done,retry);b.SetInsertPoint(done);b.CreateRet(observed);
        }
        b.restoreIP(saved); return function;
    }
    llvm::Value* atomic(unsigned operation, llvm::Value* address, llvm::Value* operand, llvm::Value* expected = nullptr) {
        bool real=operand->getType()->isFloatTy(); if(real) operand=b.CreateBitCast(operand,b.getInt32Ty());
        auto* t = operand->getType();
        if (!t->isIntegerTy(32) && !t->isIntegerTy(64)) fail(kernel,"native atomics require 32/64-bit integer storage");
        auto* function = atomic_helper(operation,t->getIntegerBitWidth());
        auto* out=call_helper(function,{address,operand,expected ? expected : llvm::ConstantInt::get(t,0)});
        return real?b.CreateBitCast(out,b.getFloatTy()):out;
    }
    llvm::Value* numeric_call(const std::string& name,llvm::Type* output,std::initializer_list<llvm::Value*> inputs) {
        auto* original=source.getFunction(name);if(!original || original->isDeclaration()) fail(kernel,"exact native numeric helper is unavailable");
        std::vector<llvm::Value*> args(inputs);std::vector<llvm::Type*> signature;for(auto* arg:args)signature.push_back(arg->getType());
        if(opencl) {auto* caller=b.GetInsertBlock()->getParent();signature.push_back(resource_pointer_type());signature.push_back(resource_pointer_type());args.push_back(caller->getArg(unsigned(caller->arg_size()-2)));args.push_back(caller->getArg(unsigned(caller->arg_size()-1)));}
        auto callee=result->getOrInsertFunction("cudavm.native.helper."+name,llvm::FunctionType::get(output,signature,false));
        llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr("cudavm.source.helper",name);return b.CreateCall(callee,args);
    }
    llvm::Value* thread_axis(unsigned axis) {
        auto id=dxil?llvm::Intrinsic::ID(llvm::Intrinsic::dx_thread_id_in_group):llvm::Intrinsic::ID(llvm::Intrinsic::spv_thread_id_in_group);
        return b.CreateZExtOrTrunc(b.CreateIntrinsic(opencl?b.getInt64Ty():b.getInt32Ty(),id,{b.getInt32(axis)}),b.getInt32Ty());
    }
    void memory_fence() {
        if(dxil) {
            auto callee=result->getOrInsertFunction("dx.op.barrier",b.getVoidTy(),b.getInt32Ty(),b.getInt32Ty());
            auto* function=llvm::cast<llvm::Function>(callee.getCallee());
            function->addFnAttr(llvm::Attribute::Convergent);function->addFnAttr(llvm::Attribute::NoUnwind);
            b.CreateCall(callee,{b.getInt32(80),b.getInt32(10)});
        } else b.CreateFence(llvm::AtomicOrdering::AcquireRelease,b.getContext().getOrInsertSyncScopeID("device"));
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
    void collective_participation(const llvm::CallInst& call) {
        llvm::DominatorTree dominators(kernel); llvm::PostDominatorTree postdominators(kernel); llvm::LoopInfo loops(dominators);
        std::set<const llvm::Value*> visiting;
        std::function<bool(const llvm::Value*)> uniform=[&](const llvm::Value* input) {
            if(llvm::isa<llvm::Constant>(input) || llvm::isa<llvm::Argument>(input)) return true;
            auto* instruction=llvm::dyn_cast<llvm::Instruction>(input); if(!instruction) return false;
            if(!visiting.insert(input).second) return true;
            bool result=true;
            if(llvm::isa<llvm::LoadInst>(instruction) || llvm::isa<llvm::AtomicRMWInst>(instruction) || llvm::isa<llvm::AtomicCmpXchgInst>(instruction)) result=false;
            else if(auto* operation=llvm::dyn_cast<llvm::CallInst>(instruction)) {
                auto* function=operation->getCalledFunction(); auto name=function && function->hasFnAttribute("cudavm.cuda.origin")?function->getFnAttribute("cudavm.cuda.origin").getValueAsString():function?function->getName():llvm::StringRef{};
                result=name.starts_with("llvm.nvvm.read.ptx.sreg.ntid.") || name.starts_with("llvm.nvvm.read.ptx.sreg.nctaid.") || name.starts_with("llvm.nvvm.read.ptx.sreg.ctaid.");
            } else for(const auto& operand:instruction->operands()) if(!llvm::isa<llvm::BasicBlock>(operand)) result&=uniform(operand.get());
            visiting.erase(input); return result;
        };
        for(const auto& block:kernel) if(auto* branch=llvm::dyn_cast<llvm::BranchInst>(block.getTerminator());branch && branch->isConditional() && !uniform(branch->getCondition())) {
            if(dominators.dominates(&block,call.getParent()) && !postdominators.dominates(call.getParent(),&block)) fail(call,"native workgroup collective requires uniform control flow");
            for(auto* loop=loops.getLoopFor(const_cast<llvm::BasicBlock*>(call.getParent()));loop;loop=loop->getParentLoop())
                if(loop->contains(&block) && (!loop->contains(branch->getSuccessor(0)) || !loop->contains(branch->getSuccessor(1)))) fail(call,"native collective loop requires uniform iteration counts");
        }
    }
    llvm::Value* collective(const llvm::CallInst& call,unsigned operation) {
        collective_participation(call);
        if(operation>=32 && call.arg_size()>1) if(auto* id=llvm::dyn_cast<llvm::ConstantInt>(call.getArgOperand(0));!id || !id->isZero()) fail(call,"named native block reductions are unsupported");
        if(operation==20) {
            auto* count=b.CreateSub(thread_count(),b.CreateAnd(linear_thread(),b.getInt32(~31u)));
            return b.CreateSelect(b.CreateICmpUGE(count,b.getInt32(32)),b.getInt32(0xffffffffu),b.CreateSub(b.CreateShl(b.getInt32(1),count),b.getInt32(1)));
        }
        auto saved=b.saveIP(); llvm::Function* helper=nullptr;
        if(auto found=collective_helpers.find(operation);found!=collective_helpers.end()) helper=found->second;
        else {
            helper=llvm::Function::Create(llvm::FunctionType::get(operation==10 || operation==11?b.getInt64Ty():b.getInt32Ty(),arguments({b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty()}),false),llvm::GlobalValue::InternalLinkage,"cudavm.collective."+std::to_string(operation),*result);
            helper->addFnAttr(llvm::Attribute::Convergent); collective_helpers[operation]=helper;
            auto* first=llvm::BasicBlock::Create(b.getContext(),"entry",helper); b.SetInsertPoint(first);
            auto* id=linear_thread(); auto* lane=b.CreateAnd(id,b.getInt32(31)); auto* base=b.CreateAnd(id,b.getInt32(~31u));
            auto pointer=[&](llvm::Value* index){return b.CreateInBoundsGEP(collective_data->getValueType(),collective_data,{b.getInt32(0),index});};
            b.CreateAlignedStore(helper->getArg(1),pointer(id),llvm::Align(4));
            if(operation==9 || operation==11) b.CreateAlignedStore(helper->getArg(2),pointer(b.CreateAdd(id,b.getInt32(1024))),llvm::Align(4));
            group_barrier(); llvm::Value* answer=b.getInt32(0);
            if(operation<4) {
                auto* mask=b.CreateAnd(b.CreateLShr(helper->getArg(3),b.getInt32(8)),b.getInt32(31));
                auto* minimum=b.CreateAnd(lane,mask); auto* maximum=b.CreateOr(minimum,b.CreateAnd(b.CreateAnd(helper->getArg(3),b.getInt32(31)),b.CreateNot(mask)));
                auto* offset=helper->getArg(2); llvm::Value* peer=nullptr; llvm::Value* eligible=nullptr;
                if(operation==0) peer=b.CreateOr(minimum,b.CreateAnd(offset,b.CreateNot(mask)));
                if(operation==1) peer=b.CreateSub(lane,offset);
                if(operation==2) peer=b.CreateAdd(lane,offset);
                if(operation==3) peer=b.CreateXor(lane,offset);
                eligible=operation==1?b.CreateAnd(b.CreateICmpUGE(lane,offset),b.CreateICmpUGE(peer,minimum)):b.CreateICmpULE(peer,maximum);
                auto* selected=b.CreateSelect(eligible,b.CreateAdd(base,peer),id);
                answer=b.CreateAlignedLoad(b.getInt32Ty(),pointer(selected),llvm::Align(4));
            } else if(operation==7) answer=b.getInt32(0);
            else {
                auto* loop=llvm::BasicBlock::Create(b.getContext(),"scan",helper); auto* body=llvm::BasicBlock::Create(b.getContext(),"participant",helper); auto* done=llvm::BasicBlock::Create(b.getContext(),"done",helper);
                auto* count=thread_count(); auto* start=operation>=32?b.getInt32(0):base;
                auto* end=operation>=32?count:b.CreateSelect(b.CreateICmpULT(b.CreateAdd(base,b.getInt32(32)),count),b.CreateAdd(base,b.getInt32(32)),count);
                b.CreateBr(loop); b.SetInsertPoint(loop);
                auto* index=b.CreatePHI(b.getInt32Ty(),2); index->addIncoming(start,first);
                auto initial=operation==5 || operation==33?1u:operation==13?0x7fffffffu:operation==14?0x80000000u:operation==15 || operation==17?0xffffffffu:0u;
                auto* total=b.CreatePHI(b.getInt32Ty(),2); total->addIncoming(b.getInt32(initial),first);
                b.CreateCondBr(b.CreateICmpULT(index,end),body,done); b.SetInsertPoint(body);
                auto* sample=b.CreateAlignedLoad(b.getInt32Ty(),pointer(index),llvm::Align(4));
                auto* predicate=b.CreateICmpNE(sample,b.getInt32(0));
                auto* bit=b.CreateShl(b.getInt32(1),b.CreateAnd(index,b.getInt32(31)));
                auto* participates=operation>=32?b.getTrue():b.CreateICmpNE(b.CreateAnd(helper->getArg(0),bit),b.getInt32(0));
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
                index->addIncoming(b.CreateAdd(index,b.getInt32(1)),body); total->addIncoming(next,body); b.CreateBr(loop); b.SetInsertPoint(done); answer=total;
                if(operation==10 || operation==11) {
                    auto* all=b.CreateICmpEQ(total,helper->getArg(0));
                    answer=pair(b.CreateSelect(all,total,b.getInt32(0)),b.CreateZExt(all,b.getInt32Ty()));
                }
            }
            group_barrier(); b.CreateRet(answer);
        }
        b.restoreIP(saved);
        bool reduce=operation>=12 && operation<=19;
        auto* mask=operation>=32?b.getInt32(0xffffffffu):value(call.getArgOperand(reduce?1:0));
        llvm::Value* data=operation==7?b.getInt32(0):value(call.getArgOperand(operation>=32?unsigned(call.arg_size()-1):reduce?0:1));
        auto* high=operation==9 || operation==11?b.CreateTrunc(b.CreateLShr(data,b.getInt64(32)),b.getInt32Ty()):b.getInt32(0);
        if(data->getType()->isFloatTy()) data=b.CreateBitCast(data,b.getInt32Ty()); else data=b.CreateZExtOrTrunc(data,b.getInt32Ty());
        std::vector<llvm::Value*> args{mask,data,operation<4?value(call.getArgOperand(2)):high,operation<4?value(call.getArgOperand(3)):b.getInt32(0)};
        if(opencl) {args.push_back(entry->getArg(0));args.push_back(entry->getArg(1));}
        auto* out=b.CreateCall(helper,args); out->setConvergent();
        if(call.getType()->isVoidTy()) return nullptr;
        if(operation==10 || operation==11) return b.CreateInsertValue(b.CreateInsertValue(llvm::UndefValue::get(type(call.getType())),b.CreateTrunc(out,b.getInt32Ty()),0),b.CreateICmpNE(b.CreateLShr(out,b.getInt64(32)),b.getInt64(0)),1);
        return call.getType()->isIntegerTy(1)?b.CreateICmpNE(out,b.getInt32(0)):out;
    }
    llvm::Value* call(const llvm::CallInst& instruction) {
        auto* function = instruction.getCalledFunction();
        if (!function) fail(instruction, "indirect calls or CUDA inline assembly require fallback");
        auto name = function->hasFnAttribute("cudavm.cuda.origin") ? function->getFnAttribute("cudavm.cuda.origin").getValueAsString() : function->getName();
        if(auto operation=native_collective(name);operation>=0) return collective(instruction,unsigned(operation));
        if(!function->isDeclaration() && (function->hasFnAttribute("cudavm.math") || function->getName().starts_with("_ZN13cudavm") || function->getName().starts_with("__cudavm_native_word_"))) {
            std::vector<llvm::Type*> signature; std::vector<llvm::Value*> args;
            for(const auto& argument:instruction.args()) {
                if(argument->getType()->isPointerTy() || !argument->getType()->isSingleValueType()) fail(instruction,"native math helper requires scalar arguments");
                signature.push_back(type(argument->getType())); args.push_back(value(argument.get()));
            }
            if(opencl) {signature.push_back(resource_pointer_type());signature.push_back(resource_pointer_type());args.push_back(entry->getArg(unsigned(entry->arg_size()-2)));args.push_back(entry->getArg(unsigned(entry->arg_size()-1)));}
            auto callee=result->getOrInsertFunction("cudavm.native.helper."+function->getName().str(),llvm::FunctionType::get(type(function->getReturnType()),signature,false));
            llvm::cast<llvm::Function>(callee.getCallee())->addFnAttr("cudavm.source.helper",function->getName());
            return b.CreateCall(callee,args);
        }
        if (name.starts_with("llvm.nvvm.read.ptx.sreg.")) {
            auto tail = name.drop_front(24);
            if (tail == "warpsize") return b.getInt32(32);
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
        if (name == "llvm.nvvm.barrier0" || name == "llvm.nvvm.barrier.cta.sync.aligned.all") {
            collective_participation(instruction);
            if (instruction.arg_size() && (!llvm::isa<llvm::ConstantInt>(instruction.getArgOperand(0)) || !llvm::cast<llvm::ConstantInt>(instruction.getArgOperand(0))->isZero()))
                fail(instruction, "named native CUDA barriers are unsupported");
            if (dxil) {
                auto callee = result->getOrInsertFunction("dx.op.barrier", b.getVoidTy(), b.getInt32Ty(), b.getInt32Ty());
                auto* declaration = llvm::cast<llvm::Function>(callee.getCallee());
                declaration->addFnAttr(llvm::Attribute::Convergent); declaration->addFnAttr(llvm::Attribute::NoUnwind);
                return b.CreateCall(callee, {b.getInt32(80), b.getInt32(11)});
            }
            return b.CreateIntrinsic(llvm::Intrinsic::spv_group_memory_barrier_with_group_sync, llvm::ArrayRef<llvm::Type*>{}, llvm::ArrayRef<llvm::Value*>{});
        }
        if (function->isIntrinsic()) {
            if(instruction.getType()->isDoubleTy()) fail(instruction,"native double intrinsic requires an exact portable numeric helper");
            if (name.starts_with("llvm.nvvm.")) fail(instruction, "CUDA operation requires explicit native shader lowering");
            if (function->getIntrinsicID() == llvm::Intrinsic::lifetime_start || function->getIntrinsicID() == llvm::Intrinsic::lifetime_end || function->getIntrinsicID() == llvm::Intrinsic::assume) return nullptr;
            switch (function->getIntrinsicID()) {
            case llvm::Intrinsic::fabs: case llvm::Intrinsic::copysign: {
                auto* magnitude=b.CreateBitCast(value(instruction.getArgOperand(0)),b.getInt32Ty());
                auto* bits=b.CreateAnd(magnitude,b.getInt32(0x7fffffffu));
                if(function->getIntrinsicID()==llvm::Intrinsic::copysign) {
                    auto* sign=b.CreateBitCast(value(instruction.getArgOperand(1)),b.getInt32Ty());
                    bits=b.CreateOr(bits,b.CreateAnd(sign,b.getInt32(0x80000000u)));
                }
                return bits;
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
                    if(bits==64) {
                        auto* high=b.CreateTrunc(b.CreateLShr(args[0],b.getInt64(32)),b.getInt32Ty());
                        auto* high_count=scan(high);
                        auto* primary=leading?high:low;
                        auto* first=leading?high_count:count;
                        auto* second=leading?count:high_count;
                        count=b.CreateSelect(b.CreateICmpEQ(primary,b.getInt32(0)),b.CreateAdd(second,b.getInt32(32)),first);
                    } else if(leading && bits<32) count=b.CreateSub(count,b.getInt32(32-bits));
                    count=b.CreateZExtOrTrunc(count,type(instruction.getType()));
                    auto* zero=llvm::cast<llvm::ConstantInt>(instruction.getArgOperand(1))->isOne()
                        ?llvm::PoisonValue::get(count->getType()):llvm::ConstantInt::get(count->getType(),bits);
                    return canonical(b.CreateSelect(b.CreateICmpEQ(args[0],llvm::Constant::getNullValue(args[0]->getType())),zero,count),instruction.getType());
                }
                if (!opencl && !dxil && instruction.getType()->isIntegerTy(64)
                    && (id==llvm::Intrinsic::ctpop || id==llvm::Intrinsic::bitreverse)) {
                    // The SPIR-V is also consumed by OpenGL, whose integer64
                    // extension does not supply 64-bit count/reverse overloads.
                    // Keep both halves on native 32-bit bit instructions.
                    auto* low=b.CreateTrunc(args[0],b.getInt32Ty());
                    auto* high=b.CreateTrunc(b.CreateLShr(args[0],b.getInt64(32)),b.getInt32Ty());
                    auto* low_result=b.CreateIntrinsic(b.getInt32Ty(),id,{low});
                    auto* high_result=b.CreateIntrinsic(b.getInt32Ty(),id,{high});
                    if(id==llvm::Intrinsic::ctpop) return b.CreateZExt(b.CreateAdd(low_result,high_result),b.getInt64Ty());
                    return b.CreateOr(b.CreateShl(b.CreateZExt(low_result,b.getInt64Ty()),b.getInt64(32)),b.CreateZExt(high_result,b.getInt64Ty()));
                }
                auto* out = b.CreateIntrinsic(instruction.getType()->isFloatTy()?b.getFloatTy():type(instruction.getType()),id,args);
                if(instruction.getType()->isFloatTy()) return b.CreateBitCast(out,b.getInt32Ty());
                if (instruction.getType()->isIntegerTy(8) || instruction.getType()->isIntegerTy(16)) {
                    auto bits=instruction.getType()->getIntegerBitWidth();
                    if (id==llvm::Intrinsic::ctlz) return b.CreateSub(out,b.getInt32(32-bits));
                    if (id==llvm::Intrinsic::cttz) return b.CreateSelect(b.CreateICmpEQ(args[0],b.getInt32(0)),b.getInt32(bits),out);
                    if (id==llvm::Intrinsic::bitreverse) return b.CreateLShr(out,b.getInt32(32-bits));
                }
                return canonical(out,instruction.getType());
            }
            default: fail(instruction, "LLVM intrinsic requires explicit native shader lowering");
            }
        }
        fail(instruction, "remaining device operation or recursive helper requires fallback");
    }
    void instruction(const llvm::Instruction& input) {
        if (input.getType()->isHalfTy()) fail(input, "half arithmetic is unsupported");
        if (input.getType()->isVectorTy()) fail(input, "vector values require fallback in this native profile");
        if (auto* integer = llvm::dyn_cast<llvm::IntegerType>(input.getType()); integer && integer->getBitWidth() != 1 && integer->getBitWidth() != 8
            && integer->getBitWidth() != 16 && integer->getBitWidth() != 32 && integer->getBitWidth() != 64)
            fail(input, "unsupported native integer width");
        if (llvm::isa<llvm::PHINode>(input)) return;
        if (auto* operation = llvm::dyn_cast<llvm::AtomicRMWInst>(&input)) {
            values[&input] = atomic(operation->getOperation(),value(operation->getPointerOperand()),value(operation->getValOperand())); return;
        }
        if (auto* operation = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&input)) {
            auto* expected = value(operation->getCompareOperand());
            auto* previous = atomic(100,value(operation->getPointerOperand()),value(operation->getNewValOperand()),expected);
            values[&input] = b.CreateInsertValue(b.CreateInsertValue(llvm::UndefValue::get(type(input.getType())),previous,0),b.CreateICmpEQ(previous,expected),1); return;
        }
        if (auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(&input)) { values[&input] = b.getInt64(0xffffffff80000000ull + local_offsets.at(allocation)); return; }
        if (auto* branch = llvm::dyn_cast<llvm::BranchInst>(&input)) {
            if (branch->isConditional()) b.CreateCondBr(value(branch->getCondition()), blocks.at(branch->getSuccessor(0)), blocks.at(branch->getSuccessor(1)));
            else b.CreateBr(blocks.at(branch->getSuccessor(0))); return;
        }
        if (auto* returned=llvm::dyn_cast<llvm::ReturnInst>(&input)) { if(returned->getReturnValue()) b.CreateRet(value(returned->getReturnValue()));else b.CreateRetVoid(); return; }
        if (auto* pointer = llvm::dyn_cast<llvm::GetElementPtrInst>(&input)) { values[&input] = gep(*llvm::cast<llvm::GEPOperator>(pointer)); return; }
        if (auto* cast = llvm::dyn_cast<llvm::CastInst>(&input); cast && (input.getType()->isPointerTy() || cast->getOperand(0)->getType()->isPointerTy())) {
            values[&input] = b.CreateZExtOrTrunc(value(cast->getOperand(0)), type(input.getType())); return;
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
                auto* integer=memory->getType()->isFloatTy()?b.getInt32Ty():llvm::cast<llvm::IntegerType>(type(memory->getType()));
                auto* zero=llvm::ConstantInt::get(integer,0);
                auto* previous=atomic(100,value(memory->getPointerOperand()),zero,zero);
                values[&input]=previous;return;
            }
            values[&input] = load(value(memory->getPointerOperand()), memory->getType(), unsigned(memory->getAlign().value())); return;
        }
        if (auto* memory = llvm::dyn_cast<llvm::StoreInst>(&input)) {
            if (memory->isAtomic()) {atomic(llvm::AtomicRMWInst::Xchg,value(memory->getPointerOperand()),value(memory->getValueOperand()));return;}
            store(value(memory->getPointerOperand()), value(memory->getValueOperand()), unsigned(memory->getAlign().value()), memory->getValueOperand()->getType()); return;
        }
        if (auto* operation = llvm::dyn_cast<llvm::CallInst>(&input)) { if (auto* out = call(*operation)) values[&input] = canonical(out,input.getType()); return; }
        if (auto* control = llvm::dyn_cast<llvm::SwitchInst>(&input)) {
            auto* out = b.CreateSwitch(value(control->getCondition()), blocks.at(control->getDefaultDest()), control->getNumCases());
            for (auto item : control->cases()) out->addCase(const_cast<llvm::ConstantInt*>(item.getCaseValue()), blocks.at(item.getCaseSuccessor())); return;
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
        for (const auto& bb : kernel) for (const auto& input : bb) if (auto* stored = llvm::dyn_cast<llvm::StoreInst>(&input))
            partial_stores |= stored->getAlign().value() < 4 || stored->getValueOperand()->getType()->isIntegerTy(1) || stored->getValueOperand()->getType()->isIntegerTy(8) || stored->getValueOperand()->getType()->isIntegerTy(16);
        for(const auto& bb:kernel) for(const auto& input:bb) if(auto* call=llvm::dyn_cast<llvm::CallInst>(&input);call && call->getCalledFunction()) {
            auto* function=call->getCalledFunction(); auto name=function->hasFnAttribute("cudavm.cuda.origin")?function->getFnAttribute("cudavm.cuda.origin").getValueAsString():function->getName();
            auto operation=native_collective(name); collective_storage|=operation>=0 && operation!=20;
        }
        for (const auto& bb : kernel) for (const auto& input : bb) if (auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(&input)) {
            auto* count = llvm::dyn_cast<llvm::ConstantInt>(allocation->getArraySize()); if (!count) fail(input, "dynamic invocation-private allocation is unsupported");
            local_bytes = uint32_t(llvm::alignTo(local_bytes, allocation->getAlign().value()));
            auto bytes = dl.getTypeAllocSize(allocation->getAllocatedType()).getFixedValue() * count->getZExtValue();
            if (bytes > 16384 || local_bytes > 16384 - bytes) fail(input, "invocation-private storage exceeds 16 KiB");
            local_offsets[allocation] = local_bytes; local_bytes += uint32_t(bytes);
        }
        if(kernel_entry) for(const auto& argument:kernel.args()) if(auto* byval=kernel.getParamByValType(argument.getArgNo())) {
            auto alignment=kernel.getParamAlign(argument.getArgNo()).value_or(dl.getABITypeAlign(byval)).value();
            local_bytes=uint32_t(llvm::alignTo(local_bytes,alignment));
            auto bytes=dl.getTypeAllocSize(byval).getFixedValue();
            if(bytes>16384 || local_bytes>16384-bytes) fail(argument,"invocation-private by-value storage exceeds 16 KiB");
            byval_offsets[&argument]=local_bytes;local_bytes+=uint32_t(bytes);
        }
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
        shared_bytes=uint32_t(llvm::alignTo(shared_bytes,16));
        for(auto& [global,offset]:shared_offsets) if(!dl.getTypeAllocSize(global->getValueType()).getFixedValue()) offset=shared_bytes;
        auto scratch=collective_storage?native_collective_shared_bytes:0u;
        if(shared_bytes+scratch>32768) fail(kernel,"native shared storage and collective scratch exceed 32 KiB");
        for (const auto& global : source.globals()) if ((global.getAddressSpace() == 1 || global.getAddressSpace() == 4) && global.hasInitializer()) globals[&global] = uint32_t(globals.size());
        result->setTargetTriple(llvm::Triple(dxil ? "dxil-pc-shadermodel6.0-compute" : opencl ? "spirv64v1.0-unknown-unknown-opencl" : "spirv1.5-unknown-vulkan1.2-compute"));
        // CUDA offsets were computed with dl above. Physical resource pointers
        // belong to the SPIR-V environment and must use its pointer layout.
        result->setDataLayout(result->getTargetTriple().computeDataLayout());
        auto* storage = llvm::ArrayType::get(b.getInt32Ty(), dynamic_shared?(32768-scratch)/4:std::max(2u, (shared_bytes + 3) / 4));
        shared = new llvm::GlobalVariable(*result, storage, false, llvm::GlobalValue::InternalLinkage, llvm::UndefValue::get(storage), "cudavm.shared", nullptr, llvm::GlobalVariable::NotThreadLocal, 3);
        if(collective_storage) {
            auto* scratch_type=llvm::ArrayType::get(b.getInt32Ty(),2048);
            collective_data=new llvm::GlobalVariable(*result,scratch_type,false,llvm::GlobalValue::InternalLinkage,llvm::UndefValue::get(scratch_type),"cudavm.collective.data",nullptr,llvm::GlobalVariable::NotThreadLocal,3);
        }
        if (local_bytes && !opencl) {
            auto* local_type = llvm::ArrayType::get(b.getInt32Ty(), std::max(2u,(local_bytes + 3) / 4));
            local_storage = new llvm::GlobalVariable(*result, local_type, false, llvm::GlobalValue::InternalLinkage, llvm::UndefValue::get(local_type), "cudavm.local", nullptr, llvm::GlobalVariable::NotThreadLocal, dxil ? 0 : 10);
        }
        make_resolver(); read = make_access(false); write = make_access(true);
        std::vector<llvm::Type*> signature;
        if(!kernel_entry) for(const auto& argument:kernel.args()) signature.push_back(type(argument.getType()));
        if(opencl) {signature.push_back(resource_pointer_type());signature.push_back(resource_pointer_type());}
        entry = llvm::Function::Create(llvm::FunctionType::get(kernel_entry?b.getVoidTy():type(kernel.getReturnType()),signature,false),llvm::GlobalValue::ExternalLinkage,kernel_entry?(opencl?"main_kernel":"main"):"cudavm.native.helper."+kernel.getName().str(),*result);
        if (opencl && kernel_entry) entry->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
        else if(kernel_entry) { entry->addFnAttr("hlsl.numthreads", std::to_string(group[0]) + "," + std::to_string(group[1]) + "," + std::to_string(group[2])); entry->addFnAttr("hlsl.shader", "compute"); }
        result->getOrInsertNamedMetadata("cudavm.ir.stage")->addOperand(llvm::MDNode::get(b.getContext(), llvm::MDString::get(b.getContext(), dxil ? "cudavm.native.dxil" : "cudavm.native.spirv")));
        if (dxil && kernel_entry) result->getOrInsertNamedMetadata("dx.valver")->addOperand(llvm::MDNode::get(b.getContext(),
            {llvm::ConstantAsMetadata::get(b.getInt32(1)), llvm::ConstantAsMetadata::get(b.getInt32(8))}));
        for (const auto& block : kernel) blocks[&block] = llvm::BasicBlock::Create(b.getContext(), block.getName(), entry);
        b.SetInsertPoint(blocks.at(&kernel.getEntryBlock()));
        if (local_bytes && opencl) opencl_local=b.CreateAlloca(llvm::ArrayType::get(b.getInt32Ty(),std::max(2u,(local_bytes+3)/4)),nullptr,"cudavm.local");
        uint64_t offset = 0;
        for (unsigned i = 0; i < kernel.arg_size(); ++i) {
            if(!kernel_entry) {values[kernel.getArg(i)]=entry->getArg(i);continue;}
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
                b.SetInsertPoint(blocks.at(phi->getIncomingBlock(i))->getTerminator());
                out->addIncoming(value(phi->getIncomingValue(i)), blocks.at(phi->getIncomingBlock(i)));
            }
        }
        return std::move(result);
    }
};
}
std::unique_ptr<llvm::Module> prepare_native_shader_module(const llvm::Module& input) {
    auto module = llvm::CloneModule(input);
    resolve_device_function_aliases(*module, true); resolve_device_data_aliases(*module, true); normalize_device_relocations(*module);
    scalar_math(*module);
    optimize(*module);
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
        auto lowered=lowering(*portable,*helper,environment,block).run();
        if(llvm::Linker(*module).linkInModule(std::move(lowered))) throw std::runtime_error("native math helper linking failed: "+symbol);
    }
    trace("verify");
    if(auto* directory=std::getenv("CUDAVM_DUMP_NATIVE_DIR")) {
        std::string text;llvm::raw_string_ostream output(text);module->print(output,nullptr);
        write_file(std::string(directory)+"/"+std::to_string(std::hash<std::string>{}(symbol))+"-"+std::to_string(unsigned(environment))+".ll",text);
    }
    // GLSL switch selectors are 32-bit. Expand wider selectors into ordinary
    // comparisons/branches in LLVM, with the pass updating PHI predecessors.
    // The same native CFG works on every shader adapter.
    for(auto& function:*module) if(!function.isDeclaration()) {
        bool wide=false;
        for(const auto& bb:function) if(auto* dispatch=llvm::dyn_cast<llvm::SwitchInst>(bb.getTerminator()))
            wide|=dispatch->getCondition()->getType()->isIntegerTy(64) || function.getName().starts_with("cudavm.native.helper.");
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
