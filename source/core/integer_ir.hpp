#pragma once
#include <string_view>
#include <llvm/IR/IRBuilder.h>

namespace cudavm {
inline bool byte_permutation_mode(std::string_view mode) {
    return mode.empty() || mode == "f4e" || mode == "b4e" || mode == "rc8" || mode == "ecl" || mode == "ecr" || mode == "rc16";
}
// Raw PTX and NVVM use sign replication in the generic selector's high bit.
// CUDA's __byte_perm wrapper masks that bit before entering this lowering.
inline llvm::Value* byte_permutation_ir(llvm::IRBuilder<>& b, llvm::Value* low, llvm::Value* high,
        llvm::Value* control, std::string_view mode = {}) {
    llvm::Value* value = b.getInt32(0);
    auto* selector = b.CreateAnd(control, b.getInt32(3));
    for (unsigned lane = 0; lane < 4; ++lane) {
        llvm::Value* index = nullptr;
        if (mode.empty()) index = b.CreateAnd(b.CreateLShr(control, lane * 4), b.getInt32(15));
        else if (mode == "f4e") index = b.CreateAdd(selector, b.getInt32(lane));
        else if (mode == "b4e") index = b.CreateAnd(b.CreateAdd(selector, b.getInt32(8 - lane)), b.getInt32(7));
        else if (mode == "rc8") index = selector;
        else if (mode == "rc16") index = b.CreateOr(b.CreateShl(b.CreateAnd(selector, b.getInt32(1)), 1), b.getInt32(lane & 1));
        else if (mode == "ecl") index = b.CreateSelect(b.CreateICmpUGT(selector, b.getInt32(lane)), selector, b.getInt32(lane));
        else index = b.CreateSelect(b.CreateICmpULT(selector, b.getInt32(lane)), selector, b.getInt32(lane));
        auto* source = b.CreateSelect(b.CreateICmpNE(b.CreateAnd(index, b.getInt32(4)), b.getInt32(0)), high, low);
        auto* byte = b.CreateAnd(b.CreateLShr(source, b.CreateShl(b.CreateAnd(index, b.getInt32(3)), 3)), b.getInt32(255));
        if (mode.empty()) byte = b.CreateSelect(b.CreateICmpNE(b.CreateAnd(index, b.getInt32(8)), b.getInt32(0)),
            b.CreateAnd(b.CreateNeg(b.CreateLShr(byte, 7)), b.getInt32(255)), byte);
        value = b.CreateOr(value, b.CreateShl(byte, lane * 8));
    }
    return value;
}
}
