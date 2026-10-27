#pragma once
#include "shader_packet.hpp"
#include <cstdint>
#include <map>
#include <stdexcept>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

namespace cudavm {
inline constexpr uint32_t shader_local_capacity = packet_private_stack_bytes;
// Descriptor accounting and lowering must reserve the same static prefix.
// Non-entry allocas allocate on every execution, even with a constant count.
struct shader_local_layout {
    std::map<const llvm::AllocaInst*, uint32_t> allocations;
    std::map<const llvm::Argument*, uint32_t> arguments;
    uint32_t static_bytes = 0;
    bool dynamic = false;
    uint32_t bytes() const { return dynamic ? shader_local_capacity : static_bytes; }
    explicit shader_local_layout(const llvm::Function& function) {
        const auto& layout = function.getDataLayout();
        auto reserve = [&](uint64_t size, uint64_t count, uint64_t alignment) {
            if (alignment > shader_local_capacity)
                throw std::runtime_error("invocation-private alignment exceeds 16 KiB");
            auto offset = llvm::alignTo(uint64_t(static_bytes), alignment);
            if (offset > shader_local_capacity || (size && count > (shader_local_capacity - offset) / size))
                throw std::runtime_error("invocation-private storage exceeds 16 KiB");
            static_bytes = uint32_t(offset + size * count);
            return uint32_t(offset);
        };
        for (const auto& block : function) for (const auto& instruction : block)
            if (auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(&instruction)) {
                if (!allocation->isStaticAlloca() || allocation->getMetadata("cudavm.ptx.dynamic")) { dynamic = true; continue; }
                auto* count = llvm::cast<llvm::ConstantInt>(allocation->getArraySize());
                allocations[allocation] = reserve(layout.getTypeAllocSize(allocation->getAllocatedType()).getFixedValue(),
                    count->getValue().getLimitedValue(), allocation->getAlign().value());
            }
        if (function.hasFnAttribute("cudavm.kernel") || function.getCallingConv() == llvm::CallingConv::PTX_Kernel)
            for (const auto& argument : function.args()) if (auto* byval = function.getParamByValType(argument.getArgNo()))
                arguments[&argument] = reserve(layout.getTypeAllocSize(byval).getFixedValue(), 1,
                    function.getParamAlign(argument.getArgNo()).value_or(layout.getABITypeAlign(byval)).value());
    }
};
}
