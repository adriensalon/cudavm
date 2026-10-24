#pragma once
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <stdexcept>
#include <set>

namespace cudavm {
// The existing transport relocates one complete 64-bit address. Normalize
// symbolic integer arithmetic to the same target-plus-byte-offset expression.
struct address_initializer {
    const llvm::DataLayout& layout;
    std::string symbol;
    struct address { llvm::GlobalVariable* target; llvm::APInt offset; unsigned space; };
    [[noreturn]] void fail(const char* reason) const {
        throw std::runtime_error("module address initializer for " + symbol + ": " + reason);
    }
    llvm::APInt add(llvm::APInt left, const llvm::APInt& right) const {
        bool overflow = false;
        auto result = left.sadd_ov(right, overflow);
        if (overflow) fail("byte offset overflow");
        return result;
    }
    address parse(llvm::Constant* value, std::set<llvm::GlobalAlias*>& visited) const {
        if (auto* target = llvm::dyn_cast<llvm::GlobalVariable>(value))
            return {target, llvm::APInt(64, 0), target->getAddressSpace()};
        if (auto* alias = llvm::dyn_cast<llvm::GlobalAlias>(value)) {
            if (!visited.insert(alias).second) fail("cyclic address aliases");
            auto result = parse(alias->getAliasee(), visited);
            visited.erase(alias);
            return result;
        }
        auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(value);
        if (!expression) fail("one module storage target is required");
        auto opcode = expression->getOpcode();
        if (opcode == llvm::Instruction::Add || opcode == llvm::Instruction::Sub) {
            llvm::ConstantInt* number = llvm::dyn_cast<llvm::ConstantInt>(expression->getOperand(1));
            llvm::Constant* base = expression->getOperand(0);
            if (!number && opcode == llvm::Instruction::Add) {
                number = llvm::dyn_cast<llvm::ConstantInt>(expression->getOperand(0));
                base = expression->getOperand(1);
            }
            if (!number || !number->getType()->isIntegerTy(64)) fail("only a single address plus/minus a 64-bit constant is supported");
            auto result = parse(base, visited);
            auto offset = number->getValue();
            if (opcode == llvm::Instruction::Sub) {
                if (offset.isMinSignedValue()) fail("byte offset overflow");
                offset = -offset;
            }
            result.offset = add(result.offset, offset);
            return result;
        }
        if (auto* gep = llvm::dyn_cast<llvm::GEPOperator>(expression)) {
            llvm::APInt offset(64, 0);
            if (!gep->accumulateConstantOffset(layout, offset)) fail("nonconstant byte offset");
            auto result = parse(expression->getOperand(0), visited);
            result.offset = add(result.offset, offset);
            return result;
        }
        if (opcode == llvm::Instruction::BitCast || opcode == llvm::Instruction::AddrSpaceCast
                || opcode == llvm::Instruction::PtrToInt || opcode == llvm::Instruction::IntToPtr) {
            if (opcode == llvm::Instruction::PtrToInt && (!expression->getType()->isIntegerTy(64)
                    || layout.getPointerTypeSizeInBits(expression->getOperand(0)->getType()) != 64
                    || layout.isNonIntegralPointerType(expression->getOperand(0)->getType())))
                fail("address representation must remain an integral 64-bit pointer");
            if (opcode == llvm::Instruction::IntToPtr && !expression->getOperand(0)->getType()->isIntegerTy(64))
                fail("address representation must remain 64 bits");
            auto result = parse(expression->getOperand(0), visited);
            if (expression->getType()->isPointerTy()) {
                auto space = expression->getType()->getPointerAddressSpace();
                if (opcode == llvm::Instruction::IntToPtr && space != result.space) fail("integer-to-pointer address space changes are unsupported");
                result.space = space;
            }
            return result;
        }
        fail("unsupported address arithmetic");
    }
    llvm::Constant* normalize(llvm::Constant* value) const {
        auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(value);
        auto storage = [&](auto&& visit, llvm::Constant* input, std::set<llvm::Constant*>& seen) -> bool {
            if (!seen.insert(input).second) return false;
            if (llvm::isa<llvm::GlobalVariable>(input)) return true;
            if (llvm::isa<llvm::GlobalAlias>(input) || llvm::isa<llvm::ConstantExpr>(input))
                for (const auto& operand : input->operands())
                    if (auto* constant = llvm::dyn_cast<llvm::Constant>(operand); constant && visit(visit, constant, seen)) return true;
            return false;
        };
        std::set<llvm::Constant*> seen;
        if (expression && (value->getType()->isIntegerTy(64) || value->getType()->isPointerTy()) && storage(storage, value, seen)) {
            std::set<llvm::GlobalAlias*> visited;
            auto address = parse(value, visited);
            if (address.target->hasInitializer()) {
                auto size = layout.getTypeAllocSize(address.target->getValueType());
                if (size.isScalable() || address.offset.isNegative() || address.offset.getZExtValue() > size.getFixedValue())
                    fail("relocated address is outside its target");
            } else if (address.offset.isNegative()) fail("relocated address has a negative byte offset");
            auto& context = value->getContext();
            llvm::Constant* pointer = address.target;
            if (!address.offset.isZero()) pointer = llvm::ConstantExpr::getGetElementPtr(llvm::Type::getInt8Ty(context), pointer,
                llvm::ConstantInt::get(context, address.offset));
            pointer = llvm::ConstantExpr::getPointerCast(pointer, llvm::PointerType::get(context, address.space));
            return value->getType()->isPointerTy() ? pointer : llvm::ConstantExpr::getPtrToInt(pointer, value->getType());
        }
        if (!llvm::isa<llvm::ConstantStruct>(value) && !llvm::isa<llvm::ConstantArray>(value)) return value;
        llvm::SmallVector<llvm::Constant*> fields;
        bool changed = false;
        for (const auto& operand : value->operands()) {
            auto* original = llvm::cast<llvm::Constant>(operand);
            auto* normalized = normalize(original);
            fields.push_back(normalized);
            changed |= normalized != original;
        }
        if (!changed) return value;
        if (auto* type = llvm::dyn_cast<llvm::StructType>(value->getType())) return llvm::ConstantStruct::get(type, fields);
        return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(value->getType()), fields);
    }
};
}
