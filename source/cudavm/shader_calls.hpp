#pragma once
#include "shader_collectives.hpp"
#include "shader_packet.hpp"
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/CallPromotionUtils.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

namespace cudavm {
inline constexpr uint64_t shader_function_address_base = uint64_t(packet_function_high_word) << 32;
inline bool shader_native_helper(const llvm::Function& function) {
    return function.hasFnAttribute("cudavm.math") || function.getName().starts_with("_ZN13cudavm")
        || function.getName().starts_with("__cudavm_native_");
}
struct shader_call_frame {
    llvm::Function* function;
    std::vector<llvm::BasicBlock*> blocks;
    std::map<const llvm::Value*, uint64_t> slots;
    std::set<const llvm::Instruction*> spilled;
    uint64_t bytes = 32, alignment = 16;
    explicit shader_call_frame(llvm::Function& input) : function(&input) {
        const auto& layout = input.getDataLayout();
        auto reserve = [&](const llvm::Value* value, llvm::Type* type, uint64_t align) {
            if (!type->isSized()) throw std::runtime_error("unsized shader call-frame value");
            if (align > packet_private_stack_bytes) throw std::runtime_error("shader call-frame alignment exceeds 16 KiB");
            alignment = std::max(alignment, align); bytes = llvm::alignTo(bytes, align);
            auto size = layout.getTypeAllocSize(type).getFixedValue();
            if (bytes > packet_private_stack_bytes || size > packet_private_stack_bytes - bytes)
                throw std::runtime_error("shader device call frame exceeds 16 KiB");
            slots[value] = bytes; bytes += size;
        };
        for (auto& argument : input.args()) {
            auto* byval = input.getParamByValType(argument.getArgNo());
            auto* type = byval ? byval : argument.getType();
            reserve(&argument, type, byval ? input.getParamAlign(argument.getArgNo()).value_or(layout.getABITypeAlign(type)).value()
                : layout.getABITypeAlign(type).value());
        }
        for (auto& block : input) {
            blocks.push_back(&block);
            std::map<const llvm::Instruction*, unsigned> order;
            std::vector<unsigned> calls;
            for (auto& instruction : block) {
                auto index = unsigned(order.size()); order[&instruction] = index;
                if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction); call && !call->isInlineAsm()) {
                    auto* target = call->getCalledFunction();
                    if (!target || (!target->isDeclaration() && !shader_native_helper(*target))) calls.push_back(index);
                }
            }

            for (auto& instruction : block) for (auto* user : instruction.users()) {
                auto* use = llvm::dyn_cast<llvm::Instruction>(user);
                if (!use || llvm::isa<llvm::PHINode>(use) || use->getParent() != &block || llvm::any_of(calls, [&](unsigned call) { return order.at(&instruction) < call && call < order.at(use); }))
                    spilled.insert(&instruction);
            }
            for (auto& instruction : block) if (!instruction.getType()->isVoidTy()
                && (spilled.contains(&instruction) || llvm::isa<llvm::PHINode>(instruction) || llvm::isa<llvm::CallInst>(instruction)))
                reserve(&instruction, instruction.getType(), layout.getABITypeAlign(instruction.getType()).value());
        }
        bytes = llvm::alignTo(bytes, alignment);
    }
};
// Ordinary calls are inlined first. Merge the residual graph into a kernel CFG
// with checked per-invocation frames and static return continuations. The same
// private-memory ABI retains callers across collective scheduler suspension.
struct shader_call_lowering {
    llvm::Module& module;
    llvm::Function& kernel;
    const std::map<llvm::Function*, uint64_t>& addresses;
    const std::set<llvm::Function*>& address_taken;
    llvm::IRBuilder<> builder;
    std::vector<shader_call_frame> frames;
    std::map<llvm::Function*, size_t> indices;
    std::map<llvm::BasicBlock*, llvm::BasicBlock*> blocks;
    std::map<const llvm::Value*, llvm::Value*> local_values;
    llvm::AllocaInst *current_frame = nullptr, *return_pc = nullptr;
    llvm::BasicBlock *failure = nullptr, *return_dispatch = nullptr;
    llvm::SwitchInst* returns = nullptr;
    unsigned next_return = 1;
    shader_call_lowering(llvm::Module& input, llvm::Function& entry,
        const std::map<llvm::Function*, uint64_t>& tokens, const std::set<llvm::Function*>& taken)
        : module(input), kernel(entry), addresses(tokens), address_taken(taken), builder(input.getContext()) {}
    std::vector<llvm::Function*> targets(llvm::CallInst& call) {
        if (call.isInlineAsm()) return {};
        if (auto* function = call.getCalledFunction()) {
            if (function->isDeclaration() || shader_native_helper(*function)) return {};
            return {function};
        }
        std::vector<llvm::Function*> result;
        for (auto& function : module) if (address_taken.contains(&function) && !function.isVarArg()
            && function.getFunctionType() == call.getFunctionType() && llvm::isLegalToPromote(call, &function))
            result.push_back(&function);
        return result;
    }
    llvm::Value* pointer(llvm::Value* frame, uint64_t offset) {
        return builder.CreateGEP(builder.getInt8Ty(), frame, builder.getInt64(offset));
    }
    llvm::Value* frame_offset(llvm::Value* address) {
        return builder.CreateAnd(builder.CreateTrunc(builder.CreatePtrToInt(address, builder.getInt64Ty()), builder.getInt32Ty()), builder.getInt32(0x3fffffffu));
    }
    llvm::Value* frame_pointer(llvm::Value* offset, llvm::Type* type = nullptr) {
        return builder.CreateIntToPtr(builder.CreateOr(builder.getInt64(0xffffffff80000000ull), builder.CreateZExt(builder.CreateAnd(offset, builder.getInt32(packet_private_stack_bytes * 2u - 1u)), builder.getInt64Ty())), type ? type : builder.getPtrTy());
    }
    llvm::Value* load(llvm::Type* type, llvm::Value* address, llvm::MaybeAlign alignment = {}) {
        auto align = alignment.value_or(module.getDataLayout().getABITypeAlign(type));
        if (type->isAggregateType()) {
            llvm::Value* result = llvm::PoisonValue::get(type);
            unsigned count = type->isStructTy() ? type->getStructNumElements() : unsigned(type->getArrayNumElements());
            for (unsigned i = 0; i < count; ++i) {
                auto* element = type->isStructTy() ? type->getStructElementType(i) : type->getArrayElementType();
                auto offset = type->isStructTy() ? module.getDataLayout().getStructLayout(llvm::cast<llvm::StructType>(type))->getElementOffset(i)
                    : module.getDataLayout().getTypeAllocSize(element).getFixedValue() * i;
                result = builder.CreateInsertValue(result, load(element, pointer(address, offset), llvm::commonAlignment(align, offset)), i);
            }
            return result;
        }
        return builder.CreateAlignedLoad(type, address, align);
    }
    void store(llvm::Value* value, llvm::Value* address, llvm::MaybeAlign alignment = {}) {
        auto* type = value->getType();
        auto align = alignment.value_or(module.getDataLayout().getABITypeAlign(type));
        if (type->isAggregateType()) {
            unsigned count = type->isStructTy() ? type->getStructNumElements() : unsigned(type->getArrayNumElements());
            for (unsigned i = 0; i < count; ++i) {
                auto* element = type->isStructTy() ? type->getStructElementType(i) : type->getArrayElementType();
                auto offset = type->isStructTy() ? module.getDataLayout().getStructLayout(llvm::cast<llvm::StructType>(type))->getElementOffset(i)
                    : module.getDataLayout().getTypeAllocSize(element).getFixedValue() * i;
                store(builder.CreateExtractValue(value, i), pointer(address, offset), llvm::commonAlignment(align, offset));
            }
        } else builder.CreateAlignedStore(value, address, align);
    }
    llvm::Value* value(shader_call_frame& frame, llvm::Value* base, llvm::Value* input) {
        if (auto found = local_values.find(input); found != local_values.end()) return found->second;
        auto found = frame.slots.find(input);
        if (found == frame.slots.end()) return input;
        auto* address = pointer(base, found->second);
        if (auto* argument = llvm::dyn_cast<llvm::Argument>(input); argument && frame.function->getParamByValType(argument->getArgNo())) return address;
        return load(input->getType(), address);
    }
    void arguments(shader_call_frame& target, llvm::Value* base, const std::vector<llvm::Value*>& inputs) {
        for (auto& argument : target.function->args()) {
            auto* destination = pointer(base, target.slots.at(&argument)); auto* input = inputs.at(argument.getArgNo());
            if (auto* byval = target.function->getParamByValType(argument.getArgNo())) {
                auto bytes = module.getDataLayout().getTypeAllocSize(byval).getFixedValue();
                for (uint64_t i = 0; i < bytes; ++i) store(load(builder.getInt8Ty(), pointer(input, i)), pointer(destination, i));
            } else store(input, destination);
        }
    }
    llvm::Value* allocate(shader_call_frame& frame, llvm::BasicBlock* initialize) {
        auto* storage = builder.CreateAlloca(builder.getInt8Ty(), builder.getInt64(frame.bytes), "call.frame");
        storage->setAlignment(llvm::Align(frame.alignment));
        auto valid = module.getOrInsertFunction("cudavm.shader.stack.valid", builder.getInt1Ty());
        builder.CreateCondBr(builder.CreateCall(valid), initialize, failure); builder.SetInsertPoint(initialize);
        return frame_pointer(frame_offset(storage));
    }
    llvm::BasicBlock* edge(shader_call_frame& frame, llvm::Value* base, llvm::BasicBlock* from, llvm::BasicBlock* to) {
        if (to->phis().empty()) return blocks.at(to);
        auto* saved = builder.GetInsertBlock();
        auto* result = llvm::BasicBlock::Create(module.getContext(), "call.edge", &kernel); builder.SetInsertPoint(result);
        std::vector<std::pair<llvm::PHINode*, llvm::Value*>> incoming;
        // Read all predecessors before writing: loop PHIs can exchange slots.
        for (auto& phi : to->phis()) incoming.push_back({&phi, value(frame, base, phi.getIncomingValueForBlock(from))});
        for (auto [phi, input] : incoming) store(input, pointer(base, frame.slots.at(phi)));
        builder.CreateBr(blocks.at(to)); builder.SetInsertPoint(saved); return result;
    }
    void run() {
        kernel.addFnAttr("cudavm.call.frames");
        indices[&kernel] = 0; frames.emplace_back(kernel);
        for (size_t i = 0; i < frames.size(); ++i) {
            auto source_blocks = frames[i].blocks;
            for (auto* block : source_blocks) for (auto& instruction : *block)
                if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) for (auto* target : targets(*call))
                    if (!indices.contains(target)) {
                        if (target->isVarArg()) throw std::runtime_error("variadic shader device calls require lowering");
                        indices[target] = frames.size(); frames.emplace_back(*target);
                    }
        }
        auto* original_entry = &kernel.getEntryBlock();
        auto* entry = llvm::BasicBlock::Create(module.getContext(), "call.entry", &kernel, original_entry);
        failure = llvm::BasicBlock::Create(module.getContext(), "call.exhausted", &kernel);
        builder.SetInsertPoint(failure); builder.CreateRetVoid();
        return_dispatch = llvm::BasicBlock::Create(module.getContext(), "call.return", &kernel);
        builder.SetInsertPoint(entry);
        current_frame = builder.CreateAlloca(builder.getInt32Ty(), nullptr, "call.current");
        return_pc = builder.CreateAlloca(builder.getInt32Ty(), nullptr, "call.return.pc");
        auto* root = llvm::BasicBlock::Create(module.getContext(), "call.root", &kernel);
        builder.CreateBr(root); builder.SetInsertPoint(root);
        auto* initialize = llvm::BasicBlock::Create(module.getContext(), "call.root.ready", &kernel);
        auto* root_frame = allocate(frames[0], initialize);
        store(builder.getInt32(0), pointer(root_frame, 24));
        std::vector<llvm::Value*> inputs; for (auto& argument : kernel.args()) inputs.push_back(&argument);
        arguments(frames[0], root_frame, inputs); builder.CreateStore(frame_offset(root_frame), current_frame);
        for (auto& frame : frames) for (auto* block : frame.blocks)
            blocks[block] = llvm::BasicBlock::Create(module.getContext(), frame.function->getName()+".call."+block->getName(), &kernel);
        builder.CreateBr(blocks.at(original_entry));
        builder.SetInsertPoint(return_dispatch);
        returns = builder.CreateSwitch(builder.CreateLoad(builder.getInt32Ty(), return_pc), failure);
        for (auto& frame : frames) for (auto* block : frame.blocks) {
            local_values.clear();
            builder.SetInsertPoint(blocks.at(block));
            llvm::Value* base = frame_pointer(builder.CreateLoad(builder.getInt32Ty(), current_frame));
            for (auto& instruction : *block) {
                if (llvm::isa<llvm::PHINode>(instruction)) continue;
                if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
                    auto callees = targets(*call);
                    if (!callees.empty()) {
                        auto* continuation = llvm::BasicBlock::Create(module.getContext(), "call.resume", &kernel);
                        auto id = next_return++; returns->addCase(builder.getInt32(id), continuation);
                        std::vector<llvm::Value*> args;
                        for (auto& argument : call->args()) args.push_back(value(frame, base, argument.get()));
                        auto* destination = call->getType()->isVoidTy() ? static_cast<llvm::Value*>(llvm::ConstantPointerNull::get(builder.getPtrTy())) : pointer(base, frame.slots.at(call));
                        auto* saved_stack = builder.CreateStackSave();
                        auto* before = builder.GetInsertBlock();
                        llvm::SwitchInst* dispatch = nullptr;
                        if (!call->getCalledFunction()) dispatch = builder.CreateSwitch(builder.CreatePtrToInt(value(frame, base, call->getCalledOperand()), builder.getInt64Ty()), failure);
                        for (auto* callee : callees) {
                            auto* enter = llvm::BasicBlock::Create(module.getContext(), "call.enter", &kernel);
                            builder.SetInsertPoint(before);
                            if (dispatch) dispatch->addCase(builder.getInt64(addresses.at(callee)), enter);
                            else builder.CreateBr(enter);
                            builder.SetInsertPoint(enter);
                            auto* ready = llvm::BasicBlock::Create(module.getContext(), "call.ready", &kernel);
                            auto& target = frames.at(indices.at(callee));
                            auto* next = allocate(target, ready);
                            store(frame_offset(base), pointer(next, 0)); store(frame_offset(saved_stack), pointer(next, 8));
                            store(frame_offset(destination), pointer(next, 16)); store(builder.getInt32(id), pointer(next, 24));
                            arguments(target, next, args); builder.CreateStore(frame_offset(next), current_frame);
                            builder.CreateBr(blocks.at(target.blocks.front()));
                        }
                        local_values.clear();
                        builder.SetInsertPoint(continuation); base = frame_pointer(builder.CreateLoad(builder.getInt32Ty(), current_frame));
                        continue;
                    }
                }
                if (auto* returned = llvm::dyn_cast<llvm::ReturnInst>(&instruction)) {
                    if (frame.function == &kernel) builder.CreateRetVoid();
                    else {
                        auto* parent = load(builder.getInt32Ty(), pointer(base, 0));
                        auto* saved_stack = frame_pointer(load(builder.getInt32Ty(), pointer(base, 8)));
                        auto* destination = frame_pointer(load(builder.getInt32Ty(), pointer(base, 16)));
                        auto* id = load(builder.getInt32Ty(), pointer(base, 24));
                        if (auto* result = returned->getReturnValue()) store(value(frame, base, result), destination);
                        builder.CreateStore(parent, current_frame); builder.CreateStore(id, return_pc);
                        builder.CreateStackRestore(saved_stack); builder.CreateBr(return_dispatch);
                    }
                    continue;
                }
                if (auto* branch = llvm::dyn_cast<llvm::BranchInst>(&instruction)) {
                    auto* yes = edge(frame, base, block, branch->getSuccessor(0));
                    if (branch->isUnconditional()) builder.CreateBr(yes);
                    else {
                        auto* no = edge(frame, base, block, branch->getSuccessor(1));
                        builder.CreateCondBr(value(frame, base, branch->getCondition()), yes, no);
                    }
                    continue;
                }
                if (auto* choice = llvm::dyn_cast<llvm::SwitchInst>(&instruction)) {
                    auto* condition = value(frame, base, choice->getCondition());
                    auto* fallback = edge(frame, base, block, choice->getDefaultDest());
                    std::vector<std::pair<llvm::ConstantInt*, llvm::BasicBlock*>> cases;
                    for (auto item : choice->cases()) cases.push_back({item.getCaseValue(), edge(frame, base, block, item.getCaseSuccessor())});
                    auto* next = builder.CreateSwitch(condition, fallback, unsigned(cases.size()));
                    for (auto [tag, target] : cases) next->addCase(tag, target);
                    continue;
                }
                auto* copy = instruction.clone();
                for (unsigned i = 0; i < copy->getNumOperands(); ++i) copy->setOperand(i, value(frame, base, instruction.getOperand(i)));
                if (auto* call = llvm::dyn_cast<llvm::CallInst>(copy); call && call->getCalledFunction()) {
                    auto* function = call->getCalledFunction();
                    auto name = function->hasFnAttribute("cudavm.cuda.origin") ? function->getFnAttribute("cudavm.cuda.origin").getValueAsString() : function->getName();
                    if (native_collective(name) >= 0) {
                        llvm::SmallVector<llvm::OperandBundleDef> bundles; call->getOperandBundlesAsDefs(bundles);
                        bundles.emplace_back("cudavm.call.frame", std::vector<llvm::Value*>{base});
                        copy = llvm::CallInst::Create(call, bundles); delete call;
                    }
                }
                builder.Insert(copy);
                if (!copy->getType()->isVoidTy()) {
                    local_values[&instruction] = copy;
                    if (frame.spilled.contains(&instruction)) store(copy, pointer(base, frame.slots.at(&instruction)));
                }
            }
        }
        for (auto* block : frames[0].blocks) block->dropAllReferences();
        for (auto* block : frames[0].blocks) block->eraseFromParent();
    }
};
inline void lower_shader_calls(llvm::Module& module) {
    std::map<llvm::Function*, uint64_t> addresses;
    std::set<llvm::Function*> address_taken;
    std::vector<llvm::Function*> kernels;
    for (auto& function : module) if (!function.isDeclaration()) {
        if (function.hasFnAttribute("cudavm.kernel") || function.getCallingConv() == llvm::CallingConv::PTX_Kernel) kernels.push_back(&function);
        else {
            addresses[&function] = shader_function_address_base + addresses.size() + 1;
            if (function.hasAddressTaken(nullptr, false, true, true)) address_taken.insert(&function);
        }
    }
    for (auto* kernel : kernels) {
        bool needed = false;
        for (auto& instruction : llvm::instructions(kernel)) if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction); call && !call->isInlineAsm()) {
            auto* target = call->getCalledFunction(); needed |= !target || (!target->isDeclaration() && !shader_native_helper(*target));
        }
        if (needed) shader_call_lowering(module, *kernel, addresses, address_taken).run();
    }
    // Keep direct calls symbolic; all observable function-pointer bytes use the
    // same module-local tokens, including global initializers and device stores.
    llvm::removeFromUsedLists(module, [&](llvm::Constant* value) {
        return addresses.contains(llvm::dyn_cast<llvm::Function>(value->stripPointerCasts()));
    });
    for (auto [function, token] : addresses) {
        auto* address = llvm::ConstantExpr::getIntToPtr(llvm::ConstantInt::get(llvm::Type::getInt64Ty(module.getContext()), token), function->getType());
        function->replaceUsesWithIf(address, [](llvm::Use& use) {
            auto* call = llvm::dyn_cast<llvm::CallBase>(use.getUser()); return !call || !call->isCallee(&use);
        });
    }
}
}
