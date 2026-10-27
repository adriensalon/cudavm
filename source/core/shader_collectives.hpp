#pragma once
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Instructions.h>
namespace cudavm {
inline int native_collective(llvm::StringRef name) {
    const char* shuffle[]{"idx","up","down","bfly"};
    for (unsigned i=0;i<4;++i) if (name.starts_with(std::string("llvm.nvvm.shfl.sync.")+shuffle[i]+".")) return int(i);
    if (name=="llvm.nvvm.vote.ballot.sync") return 4;
    if (name=="llvm.nvvm.vote.all.sync") return 5;
    if (name=="llvm.nvvm.vote.any.sync") return 6;
    if (name=="llvm.nvvm.bar.warp.sync") return 7;
    if (name=="llvm.nvvm.match.any.sync.i32") return 8;
    if (name=="llvm.nvvm.match.any.sync.i64") return 9;
    if (name=="llvm.nvvm.match.all.sync.i32p") return 10;
    if (name=="llvm.nvvm.match.all.sync.i64p") return 11;
    const char* reductions[]{"add","min","max","umin","umax","and","or","xor"};
    for(unsigned i=0;i<8;++i) if(name==std::string("llvm.nvvm.redux.sync.")+reductions[i]) return int(12+i);
    if (name=="llvm.nvvm.activemask") return 20;
    if (name=="llvm.nvvm.barrier0" || name.starts_with("llvm.nvvm.barrier.cta.sync.")) return 36;
    if (name.starts_with("llvm.nvvm.barrier.cta.arrive.")) return 37;
    const char* votes[]{"popc","and","or"};
    for(unsigned i=0;i<3;++i) {
        if(name==std::string("llvm.nvvm.barrier0.")+votes[i]) return int(32+i);
        if(name.starts_with(std::string("llvm.nvvm.barrier.cta.red.")+votes[i]+".")) return int(38+i);
    }
    return -1;
}
// Two words of payload and a program counter/mask per possible CUDA thread,
// plus the uniform continuation-loop control word (rounded to eight bytes).
inline constexpr unsigned native_collective_shared_bytes = 16392;
// Arrival counts, expected counts and generations for the 16 CTA barriers.
inline constexpr unsigned native_named_barrier_state_words = 112;
inline constexpr unsigned native_named_barrier_shared_bytes = native_named_barrier_state_words * 4;
inline bool native_atomic_polling(const llvm::Function& function) {
    for(const auto& block:function) for(const auto& input:block)
        if(auto* load=llvm::dyn_cast<llvm::LoadInst>(&input);load && load->isAtomic() && !input.getMetadata("cudavm.nonblocking")) return true;
    // CAS/RMW polling also needs independent lane progress. Limit the extra
    // scheduler storage to cyclic CFG regions rather than every atomic kernel.
    for(auto region=llvm::scc_begin(&function);!region.isAtEnd();++region)
        if(region.hasCycle()) for(auto* block:*region) for(const auto& input:*block)
        if(llvm::isa<llvm::AtomicCmpXchgInst,llvm::AtomicRMWInst>(input) && !input.getMetadata("cudavm.nonblocking")) return true;
    return false;
}
}
