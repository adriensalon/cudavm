#pragma once
#include <llvm/ADT/StringRef.h>
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
    if (name=="llvm.nvvm.barrier0" || name=="llvm.nvvm.barrier.cta.sync.aligned.all") return 35;
    const char* votes[]{"popc","and","or"};
    for(unsigned i=0;i<3;++i) if(name==std::string("llvm.nvvm.barrier0.")+votes[i] || name.starts_with(std::string("llvm.nvvm.barrier.cta.red.")+votes[i]+".")) return int(32+i);
    return -1;
}
// Two words of payload and a program counter/mask per possible CUDA thread,
// plus the uniform continuation-loop control word (rounded to eight bytes).
inline constexpr unsigned native_collective_shared_bytes = 16392;
}
