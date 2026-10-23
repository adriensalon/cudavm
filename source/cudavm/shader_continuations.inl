// Members of native shader lowering. CUDA lanes retain their own continuation;
// every native invocation, including retired CUDA lanes, reaches the scheduler's
// workgroup barriers. No hardware subgroup ordering or width is assumed.
llvm::Value* collective_pointer(llvm::GlobalVariable* storage,llvm::Value* index) {
    if(!storage->getValueType()->isArrayTy()) return storage;
    return b.CreateInBoundsGEP(storage->getValueType(),storage,{b.getInt32(0),index});
}
llvm::LoadInst* collective_load(llvm::Value* pointer) {
    auto* loaded=b.CreateAlignedLoad(b.getInt32Ty(),pointer,llvm::Align(4));
    loaded->setVolatile(true); return loaded;
}
void collective_store(llvm::Value* contents,llvm::Value* pointer) {
    b.CreateAlignedStore(contents,pointer,llvm::Align(4))->setVolatile(true);
}
llvm::Function* make_named_arrival() {
    auto* function=llvm::Function::Create(llvm::FunctionType::get(b.getInt32Ty(),{b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty()},false),llvm::GlobalValue::InternalLinkage,"cudavm.barrier.arrive",*result);
    function->addFnAttr(llvm::Attribute::NoInline);
    auto* first=llvm::BasicBlock::Create(b.getContext(),"entry",function);
    auto* retry=llvm::BasicBlock::Create(b.getContext(),"retry",function);
    auto* done=llvm::BasicBlock::Create(b.getContext(),"done",function);
    b.SetInsertPoint(first);
    auto* id=function->getArg(0);
    auto counts=[&](){return collective_pointer(named_barrier_state,id);};
    auto* expected=collective_pointer(named_barrier_state,b.CreateAdd(id,b.getInt32(16)));
    auto* generation=collective_load(collective_pointer(named_barrier_state,b.CreateAdd(id,b.getInt32(32))));
    auto scope=dxil?llvm::SyncScope::System:b.getContext().getOrInsertSyncScopeID("workgroup");
    b.CreateAtomicCmpXchg(expected,b.getInt32(0),function->getArg(1),llvm::Align(4),llvm::AtomicOrdering::Monotonic,llvm::AtomicOrdering::Monotonic,scope);
    auto* scan=llvm::BasicBlock::Create(b.getContext(),"warp.scan",function);
    auto* sample=llvm::BasicBlock::Create(b.getContext(),"warp.sample",function);
    auto* scanned=llvm::BasicBlock::Create(b.getContext(),"warp.scanned",function);
    auto* base=b.CreateAnd(function->getArg(2),b.getInt32(~31u));
    auto* warp_end=b.CreateAdd(base,b.getInt32(32));
    auto* end=b.CreateSelect(b.CreateICmpULT(function->getArg(3),warp_end),function->getArg(3),warp_end);
    b.CreateBr(scan);b.SetInsertPoint(scan);
    auto* index=b.CreatePHI(b.getInt32Ty(),2);index->addIncoming(base,first);
    auto* live=b.CreatePHI(b.getInt32Ty(),2);live->addIncoming(b.getInt32(0),first);
    auto* prior=b.CreatePHI(b.getInt1Ty(),2);prior->addIncoming(b.getFalse(),first);
    b.CreateCondBr(b.CreateICmpULT(index,end),sample,scanned);b.SetInsertPoint(sample);
    auto* present=b.CreateICmpNE(collective_load(collective_pointer(collective_pc,index)),b.getInt32(0));
    live->addIncoming(b.CreateAdd(live,b.CreateZExt(present,b.getInt32Ty())),sample);
    prior->addIncoming(b.CreateOr(prior,b.CreateAnd(present,b.CreateICmpULT(index,function->getArg(2)))),sample);
    index->addIncoming(b.CreateAdd(index,b.getInt32(1)),sample);b.CreateBr(scan);b.SetInsertPoint(scanned);
    // Counted CTA barriers credit whole CUDA warps. Credit exited/absent lanes
    // once, via the first live lane, while implicit counts track live threads.
    auto* credit=b.CreateSelect(b.CreateAnd(b.CreateICmpNE(function->getArg(1),b.getInt32(0)),b.CreateNot(prior)),b.CreateSub(b.getInt32(33),live),b.getInt32(1));
    auto* initial=b.CreateAtomicCmpXchg(counts(),b.getInt32(0),b.getInt32(0),llvm::Align(4),llvm::AtomicOrdering::Monotonic,llvm::AtomicOrdering::Monotonic,scope);
    auto* initial_count=b.CreateExtractValue(initial,0);
    b.CreateBr(retry);b.SetInsertPoint(retry);
    auto* previous=b.CreatePHI(b.getInt32Ty(),2);previous->addIncoming(initial_count,scanned);
    auto* exchange=b.CreateAtomicCmpXchg(counts(),previous,b.CreateAdd(previous,credit),llvm::Align(4),llvm::AtomicOrdering::Monotonic,llvm::AtomicOrdering::Monotonic,scope);
    previous->addIncoming(b.CreateExtractValue(exchange,0),retry);
    b.CreateCondBr(b.CreateExtractValue(exchange,1),done,retry);
    b.SetInsertPoint(done);b.CreateRet(generation);
    return function;
}
llvm::Function* make_named_advance() {
    auto* function=llvm::Function::Create(llvm::FunctionType::get(b.getVoidTy(),{b.getInt32Ty()},false),llvm::GlobalValue::InternalLinkage,"cudavm.barrier.advance",*result);
    function->addFnAttr(llvm::Attribute::NoInline);
    auto* first=llvm::BasicBlock::Create(b.getContext(),"entry",function);
    auto* head=llvm::BasicBlock::Create(b.getContext(),"scan",function);
    auto* check=llvm::BasicBlock::Create(b.getContext(),"check",function);
    auto* release=llvm::BasicBlock::Create(b.getContext(),"release",function);
    auto* next=llvm::BasicBlock::Create(b.getContext(),"next",function);
    auto* done=llvm::BasicBlock::Create(b.getContext(),"done",function);
    b.SetInsertPoint(first);b.CreateBr(head);b.SetInsertPoint(head);
    auto* id=b.CreatePHI(b.getInt32Ty(),2);id->addIncoming(b.getInt32(0),first);
    b.CreateCondBr(b.CreateICmpULT(id,b.getInt32(16)),check,done);b.SetInsertPoint(check);
    // The uniform scheduler barriers bracket the leader's phase transitions.
    // No arrival or waiter observes a partly reset generation. Recreate native
    // pointers within each block for the target's pointer-SSA legalization.
    auto* arrived=collective_load(collective_pointer(named_barrier_state,id));
    auto* expected=collective_load(collective_pointer(named_barrier_state,b.CreateAdd(id,b.getInt32(16))));
    auto* required=b.CreateSelect(b.CreateICmpEQ(expected,b.getInt32(0)),function->getArg(0),expected);
    b.CreateCondBr(b.CreateAnd(b.CreateICmpNE(arrived,b.getInt32(0)),b.CreateICmpUGE(arrived,required)),release,next);
    b.SetInsertPoint(release);
    collective_store(b.getInt32(0),collective_pointer(named_barrier_state,id));
    collective_store(b.getInt32(0),collective_pointer(named_barrier_state,b.CreateAdd(id,b.getInt32(16))));
    auto* phase=collective_pointer(named_barrier_state,b.CreateAdd(id,b.getInt32(32)));
    collective_store(b.CreateAdd(collective_load(phase),b.getInt32(1)),phase);
    b.CreateBr(next);b.SetInsertPoint(next);id->addIncoming(b.CreateAdd(id,b.getInt32(1)),next);b.CreateBr(head);
    b.SetInsertPoint(done);b.CreateRetVoid();return function;
}
llvm::Function* make_named_initialize() {
    auto* function=llvm::Function::Create(llvm::FunctionType::get(b.getVoidTy(),{b.getInt32Ty(),b.getInt32Ty()},false),llvm::GlobalValue::InternalLinkage,"cudavm.barrier.initialize",*result);
    function->addFnAttr(llvm::Attribute::NoInline);
    auto* first=llvm::BasicBlock::Create(b.getContext(),"entry",function);
    auto* head=llvm::BasicBlock::Create(b.getContext(),"scan",function);
    auto* body=llvm::BasicBlock::Create(b.getContext(),"clear",function);
    auto* done=llvm::BasicBlock::Create(b.getContext(),"done",function);
    b.SetInsertPoint(first);b.CreateBr(head);b.SetInsertPoint(head);
    auto* index=b.CreatePHI(b.getInt32Ty(),2);index->addIncoming(function->getArg(0),first);
    b.CreateCondBr(b.CreateICmpULT(index,b.getInt32(48)),body,done);b.SetInsertPoint(body);
    collective_store(b.getInt32(0),collective_pointer(named_barrier_state,index));
    index->addIncoming(b.CreateAdd(index,function->getArg(1)),body);b.CreateBr(head);
    b.SetInsertPoint(done);b.CreateRetVoid();return function;
}
llvm::Function* make_named_warp_poll(const std::vector<unsigned>& states) {
    auto* function=llvm::Function::Create(llvm::FunctionType::get(b.getInt1Ty(),{b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty()},false),llvm::GlobalValue::InternalLinkage,"cudavm.barrier.warp.ready",*result);
    auto* first=llvm::BasicBlock::Create(b.getContext(),"entry",function);
    auto* head=llvm::BasicBlock::Create(b.getContext(),"scan",function);
    auto* body=llvm::BasicBlock::Create(b.getContext(),"peer",function);
    auto* check=llvm::BasicBlock::Create(b.getContext(),"barrier",function);
    auto* next=llvm::BasicBlock::Create(b.getContext(),"next",function);
    auto* done=llvm::BasicBlock::Create(b.getContext(),"done",function);
    b.SetInsertPoint(first);
    auto* base=b.CreateAnd(function->getArg(0),b.getInt32(~31u));auto* warp_end=b.CreateAdd(base,b.getInt32(32));
    auto* end=b.CreateSelect(b.CreateICmpULT(function->getArg(1),warp_end),function->getArg(1),warp_end);
    b.CreateBr(head);b.SetInsertPoint(head);
    auto* index=b.CreatePHI(b.getInt32Ty(),2);index->addIncoming(base,first);
    auto* ready=b.CreatePHI(b.getInt1Ty(),2);ready->addIncoming(b.getTrue(),first);
    b.CreateCondBr(b.CreateICmpULT(index,end),body,done);b.SetInsertPoint(body);
    auto* pc=collective_load(collective_pointer(collective_pc,index));
    auto* retired=b.CreateICmpEQ(pc,b.getInt32(0));
    llvm::Value* waiting=b.getFalse();for(auto state:states)waiting=b.CreateOr(waiting,b.CreateICmpEQ(pc,b.getInt32(state)));
    b.CreateCondBr(waiting,check,next);b.SetInsertPoint(check);
    auto* same=b.CreateICmpEQ(collective_load(collective_pointer(collective_masks,index)),function->getArg(2));
    b.CreateBr(next);b.SetInsertPoint(next);
    auto* agrees=b.CreatePHI(b.getInt1Ty(),2);agrees->addIncoming(retired,body);agrees->addIncoming(same,check);
    ready->addIncoming(b.CreateAnd(ready,agrees),next);index->addIncoming(b.CreateAdd(index,b.getInt32(1)),next);b.CreateBr(head);
    b.SetInsertPoint(done);b.CreateRet(ready);return function;
}
llvm::Function* make_collective_poll() {
    auto* poll=llvm::Function::Create(llvm::FunctionType::get(b.getInt64Ty(),arguments({b.getInt32Ty(),b.getInt32Ty(),b.getInt32Ty()}),false),llvm::GlobalValue::InternalLinkage,"cudavm.collective.poll",*result);
    auto* first=llvm::BasicBlock::Create(b.getContext(),"entry",poll);
    auto* loop=llvm::BasicBlock::Create(b.getContext(),"scan",poll);
    auto* body=llvm::BasicBlock::Create(b.getContext(),"peer",poll);
    auto* mask_test=llvm::BasicBlock::Create(b.getContext(),"mask",poll);
    auto* advance=llvm::BasicBlock::Create(b.getContext(),"advance",poll);
    auto* done=llvm::BasicBlock::Create(b.getContext(),"done",poll);
    b.SetInsertPoint(first);
    auto* id=linear_thread(); auto* base=b.CreateAnd(id,b.getInt32(~31u));
    auto* block=b.CreateICmpUGE(poll->getArg(2),b.getInt32(32));
    auto* active=b.CreateICmpEQ(poll->getArg(2),b.getInt32(20));
    auto* start=b.CreateSelect(block,b.getInt32(0),base);
    auto* count=thread_count(); auto* warp_end=b.CreateAdd(base,b.getInt32(32));
    auto* end=b.CreateSelect(block,count,b.CreateSelect(b.CreateICmpULT(count,warp_end),count,warp_end));
    b.CreateBr(loop); b.SetInsertPoint(loop);
    auto* index=b.CreatePHI(b.getInt32Ty(),2); index->addIncoming(start,first);
    auto* ready=b.CreatePHI(b.getInt1Ty(),2); ready->addIncoming(b.getTrue(),first);
    auto* members=b.CreatePHI(b.getInt32Ty(),2); members->addIncoming(b.getInt32(0),first);
    b.CreateCondBr(b.CreateICmpULT(index,end),body,done); b.SetInsertPoint(body);
    auto* pc=collective_load(collective_pointer(collective_pc,index));
    auto* live=b.CreateICmpNE(pc,b.getInt32(0));
    auto* same=b.CreateICmpEQ(pc,poll->getArg(1));
    auto* bit=b.CreateShl(b.getInt32(1),b.CreateAnd(index,b.getInt32(31)));
    auto* required=b.CreateAnd(live,b.CreateOr(block,b.CreateICmpNE(b.CreateAnd(poll->getArg(0),bit),b.getInt32(0))));
    // A peer outside this rendezvous may be publishing its next mask. Read its
    // mask only when the immutable PC snapshot says it is already waiting here.
    b.CreateCondBr(b.CreateAnd(same,b.CreateNot(b.CreateOr(active,block))),mask_test,advance);
    b.SetInsertPoint(mask_test);
    auto* same_mask=b.CreateICmpEQ(collective_load(collective_pointer(collective_masks,index)),poll->getArg(0));
    b.CreateBr(advance); b.SetInsertPoint(advance);
    auto* agrees=b.CreatePHI(b.getInt1Ty(),2); agrees->addIncoming(same,body);agrees->addIncoming(same_mask,mask_test);
    ready->addIncoming(b.CreateAnd(ready,b.CreateOr(b.CreateNot(required),agrees)),advance);
    auto* included=b.CreateSelect(active,same,required);
    members->addIncoming(b.CreateOr(members,b.CreateSelect(included,bit,b.getInt32(0))),advance);
    index->addIncoming(b.CreateAdd(index,b.getInt32(1)),advance);b.CreateBr(loop);
    b.SetInsertPoint(done);b.CreateRet(pair(members,b.CreateZExt(b.CreateOr(active,ready),b.getInt32Ty())));
    return poll;
}
void make_continuations() {
    if(!kernel_entry) fail(kernel,"collective helper must be inlined into its CUDA kernel");
    // Fold lowered cmpxchg tuples before spilling live values. Otherwise a
    // failed CAS can leave an aggregate load/store across scheduler steps,
    // which the shader targets cannot represent as invocation-private state.
    llvm::FunctionAnalysisManager analyses;llvm::PassBuilder builder;builder.registerFunctionAnalyses(analyses);
    llvm::InstSimplifyPass().run(*entry,analyses);
    analyses.clear();
    struct rendezvous {llvm::CallInst* call;llvm::BasicBlock* wait;unsigned operation;llvm::AllocaInst* generation=nullptr;llvm::BasicBlock* warp_wait=nullptr;llvm::BasicBlock* arrival=nullptr;};
    std::vector<rendezvous> sites;
    for(auto& bb:*entry) for(auto& input:bb) if(auto* call=llvm::dyn_cast<llvm::CallInst>(&input))
        if(auto* f=call->getCalledFunction();f && f->hasFnAttribute("cudavm.collective"))
            sites.push_back({call,nullptr,unsigned(std::stoul(f->getFnAttribute("cudavm.collective").getValueAsString().str()))});
    auto saved=b.saveIP();
    auto* named_arrival=named_barriers?make_named_arrival():nullptr;
    auto* named_advance=named_barriers?make_named_advance():nullptr;
    auto* named_initialize=named_barriers?make_named_initialize():nullptr;
    b.restoreIP(saved);
    for(auto& site:sites) {
        auto* arrival=site.call->getParent();
        site.wait=arrival->splitBasicBlock(site.call,"collective.wait");
        site.wait->splitBasicBlock(site.call->getNextNode(),"collective.resume");
        if(site.operation>=36) {
            site.warp_wait=site.wait;
            site.arrival=site.warp_wait->splitBasicBlock(site.call,"barrier.arrival");
            site.wait=site.arrival->splitBasicBlock(site.call,"barrier.wait");
            b.SetInsertPoint(&*entry->getEntryBlock().getFirstInsertionPt());
            site.generation=b.CreateAlloca(b.getInt32Ty(),nullptr,"barrier.generation");
            b.SetInsertPoint(site.arrival->getTerminator());
            b.CreateStore(b.CreateCall(named_arrival,{site.call->getArgOperand(0),site.call->getArgOperand(1),linear_thread(),thread_count()}),site.generation);
        }
        b.SetInsertPoint(arrival->getTerminator()); auto* id=linear_thread();
        collective_store(site.call->getArgOperand(0),collective_pointer(collective_masks,id));
        collective_store(site.call->getArgOperand(1),collective_pointer(collective_data,id));
        if(site.operation==9 || site.operation==11)
            collective_store(site.call->getArgOperand(2),collective_pointer(collective_data,b.CreateAdd(id,b.getInt32(1024))));
    }
    // Values live across a yield belong to the invocation, including PHI edge
    // updates. Demote before changing edges so loops and critical edges retain
    // their source meaning. Native allocas are hoisted into the new prologue.
    llvm::RegToMemPass().run(*entry,analyses);
    std::vector<llvm::BasicBlock*> quanta;
    std::map<llvm::BasicBlock*,unsigned> states;
    for(auto& bb:*entry) {quanta.push_back(&bb);states[&bb]=unsigned(quanta.size());}
    auto* initial=&entry->getEntryBlock();
    auto* prologue=llvm::BasicBlock::Create(b.getContext(),"continuation.entry",entry,initial);
    auto* head=llvm::BasicBlock::Create(b.getContext(),"continuation.publish",entry);
    auto* leader=llvm::BasicBlock::Create(b.getContext(),"continuation.leader",entry);
    auto* scan=llvm::BasicBlock::Create(b.getContext(),"continuation.scan",entry);
    auto* sample=llvm::BasicBlock::Create(b.getContext(),"continuation.sample",entry);
    auto* scanned=llvm::BasicBlock::Create(b.getContext(),"continuation.scanned",entry);
    auto* dispatch=llvm::BasicBlock::Create(b.getContext(),"continuation.dispatch",entry);
    auto* select=llvm::BasicBlock::Create(b.getContext(),"continuation.select",entry);
    auto* latch=llvm::BasicBlock::Create(b.getContext(),"continuation.yield",entry);
    auto* exit=llvm::BasicBlock::Create(b.getContext(),"continuation.exit",entry);
    b.SetInsertPoint(prologue);
    if (local_storage) {
        // Anchor user private storage in the entry as well as the step helper.
        // SPIR-V optimizers may otherwise give it a new lifetime on each call.
        b.CreateStore(b.getInt32(0), b.CreateInBoundsGEP(local_storage->getValueType(), local_storage,
            {b.getInt32(0), b.getInt32(1)}))->setVolatile(true);
    }
    std::vector<llvm::AllocaInst*> slots;
    for(auto* bb:quanta) for(auto& input:*bb) if(auto* slot=llvm::dyn_cast<llvm::AllocaInst>(&input)) slots.push_back(slot);
    for(auto* slot:slots) {slot->removeFromParent();b.Insert(slot);}
    auto* local_pc=b.CreateAlloca(b.getInt32Ty(),nullptr,"continuation.pc");
    b.CreateStore(b.getInt32(states.at(initial)),local_pc);
    if(named_barriers) b.CreateCall(named_initialize,{linear_thread(),thread_count()});
    b.CreateBr(head);
    b.SetInsertPoint(head);auto* id=linear_thread();auto* count=thread_count();
    auto* pc=b.CreateLoad(b.getInt32Ty(),local_pc);
    collective_store(pc,collective_pointer(collective_pc,id));group_barrier();
    b.CreateCondBr(b.CreateICmpEQ(id,b.getInt32(0)),leader,dispatch);
    b.SetInsertPoint(leader);b.CreateBr(scan);b.SetInsertPoint(scan);
    auto* index=b.CreatePHI(b.getInt32Ty(),2);index->addIncoming(b.getInt32(0),leader);
    auto* any=b.CreatePHI(b.getInt32Ty(),2);any->addIncoming(b.getInt32(0),leader);
    auto* live_count=b.CreatePHI(b.getInt32Ty(),2);live_count->addIncoming(b.getInt32(0),leader);
    b.CreateCondBr(b.CreateICmpULT(index,count),sample,scanned);
    b.SetInsertPoint(sample);
    auto* peer_pc=collective_load(collective_pointer(collective_pc,index));
    any->addIncoming(b.CreateOr(any,peer_pc),sample);
    live_count->addIncoming(b.CreateAdd(live_count,b.CreateZExt(b.CreateICmpNE(peer_pc,b.getInt32(0)),b.getInt32Ty())),sample);
    index->addIncoming(b.CreateAdd(index,b.getInt32(1)),sample);b.CreateBr(scan);
    b.SetInsertPoint(scanned);collective_store(any,collective_pointer(collective_control,b.getInt32(0)));
    if(named_barriers) b.CreateCall(named_advance,{live_count});
    b.CreateBr(dispatch);
    b.SetInsertPoint(dispatch);group_barrier();
    b.CreateCondBr(b.CreateICmpNE(collective_load(collective_pointer(collective_control,b.getInt32(0))),b.getInt32(0)),select,exit);
    b.SetInsertPoint(select);auto* choice=b.CreateSwitch(pc,latch,unsigned(quanta.size()));
    for(auto* bb:quanta) choice->addCase(b.getInt32(states.at(bb)),bb);
    b.SetInsertPoint(latch);group_barrier();b.CreateBr(head);
    b.SetInsertPoint(exit);b.CreateRetVoid();
    // Execute at most one original CFG block per round, including polling loops.
    // Retired lanes stay in the scheduler with PC zero and never execute again.
    for(auto* bb:quanta) {
        auto* terminator=bb->getTerminator();b.SetInsertPoint(terminator);
        llvm::Value* next=b.getInt32(0);
        if(auto* branch=llvm::dyn_cast<llvm::BranchInst>(terminator)) {
            next=b.getInt32(states.at(branch->getSuccessor(0)));
            if(branch->isConditional())next=b.CreateSelect(branch->getCondition(),next,b.getInt32(states.at(branch->getSuccessor(1))));
        } else if(auto* control=llvm::dyn_cast<llvm::SwitchInst>(terminator)) {
            next=b.getInt32(states.at(control->getDefaultDest()));
            for(auto item:control->cases())next=b.CreateSelect(b.CreateICmpEQ(control->getCondition(),item.getCaseValue()),b.getInt32(states.at(item.getCaseSuccessor())),next);
        } else if(!llvm::isa<llvm::ReturnInst>(terminator) && !llvm::isa<llvm::UnreachableInst>(terminator)) fail(kernel,"unsupported continuation terminator");
        b.CreateStore(next,local_pc);b.CreateBr(latch);terminator->eraseFromParent();
    }
    auto* poll=make_collective_poll();
    std::vector<unsigned> named_wait_states;
    for(auto& site:sites) if(site.operation>=36)named_wait_states.push_back(states.at(site.warp_wait));
    auto* named_warp_poll=named_barriers?make_named_warp_poll(named_wait_states):nullptr;
    for(auto& site:sites) {
        if(site.operation>=36) {
            b.SetInsertPoint(site.warp_wait->getTerminator());
            auto* barrier_id=collective_load(collective_pointer(collective_masks,linear_thread()));
            auto* ready=b.CreateCall(named_warp_poll,{linear_thread(),thread_count(),barrier_id});
            b.CreateStore(b.CreateSelect(ready,b.getInt32(states.at(site.arrival)),b.getInt32(states.at(site.warp_wait))),local_pc);
        }
        if(site.operation==37) continue; // Warp arrival does not wait for other warps.
        auto* execute=site.wait->splitBasicBlock(site.call,"collective.execute");
        auto* branch=site.wait->getTerminator();b.SetInsertPoint(branch);
        if(site.operation>=36) {
            auto* barrier_id=collective_load(collective_pointer(collective_masks,linear_thread()));
            llvm::Value* ready=nullptr;
            {
                auto* phase=collective_pointer(named_barrier_state,b.CreateAdd(barrier_id,b.getInt32(32)));
                ready=b.CreateICmpNE(collective_load(phase),b.CreateLoad(b.getInt32Ty(),site.generation));
            }
            b.CreateCondBr(ready,execute,latch);
            branch->eraseFromParent();continue;
        }
        auto* mask=collective_load(collective_pointer(collective_masks,linear_thread()));
        std::vector<llvm::Value*> args{mask,b.getInt32(states.at(site.wait)),b.getInt32(site.operation)};
        if(opencl) for(unsigned i=0;i<packet_resource_count;++i) args.push_back(entry->getArg(i));
        auto* status=b.CreateCall(poll,args);
        auto* members=b.CreateTrunc(status,b.getInt32Ty());
        b.CreateCondBr(b.CreateICmpNE(b.CreateLShr(status,b.getInt64(32)),b.getInt64(0)),execute,latch);branch->eraseFromParent();
        // Retired and absent lanes do not contribute to votes, matching or
        // reductions, even when their bits remain set in the caller's mask.
        site.call->setArgOperand(0,members);
    }
    // Keep native barriers outside the per-invocation dispatcher. Target CFG
    // legalization may itself dispatch complex functions a block at a time;
    // doing that to a function containing barriers can strand its other lanes.
    // A finite step helper contains no native barriers and always returns to
    // the uniform outer loop, even when its own CFG needs target fallback.
    slots.push_back(local_pc);
    std::vector<llvm::Type*> signature;
    if(opencl) {
        for(auto* slot:slots) signature.push_back(slot->getType());
        for(unsigned i=0;i<packet_resource_count;++i) signature.push_back(resource_pointer_type());
    }
    auto* step=llvm::Function::Create(llvm::FunctionType::get(b.getVoidTy(),signature,false),llvm::GlobalValue::InternalLinkage,"cudavm.continuation.step",*result);
    step->addFnAttr(llvm::Attribute::NoInline);step->addFnAttr(llvm::Attribute::Convergent);
    auto* step_exit=llvm::BasicBlock::Create(b.getContext(),"step.exit",step);
    b.SetInsertPoint(step_exit);b.CreateRetVoid();
    std::set<llvm::BasicBlock*> outer{prologue,head,leader,scan,sample,scanned,dispatch,latch,exit};
    std::vector<llvm::BasicBlock*> work{select};
    for(auto& bb:*entry) if(&bb!=select && !outer.contains(&bb))work.push_back(&bb);
    b.SetInsertPoint(choice);choice->setCondition(b.CreateLoad(b.getInt32Ty(),local_pc));
    for(auto* bb:work) {
        auto* terminator=bb->getTerminator();
        for(unsigned i=0;i<terminator->getNumSuccessors();++i)
            if(terminator->getSuccessor(i)==latch)terminator->setSuccessor(i,step_exit);
        bb->removeFromParent();bb->insertInto(step,step_exit);
    }
    if(opencl) {
        for(auto& bb:*step) for(auto& input:bb) for(unsigned i=0;i<input.getNumOperands();++i) {
            auto* operand=input.getOperand(i);
            for(unsigned slot=0;slot<slots.size();++slot) if(operand==slots[slot])input.setOperand(i,step->getArg(slot));
            for(unsigned resource=0;resource<packet_resource_count;++resource)
                if(operand==entry->getArg(resource))input.setOperand(i,step->getArg(unsigned(slots.size()+resource)));
        }
    } else for(auto* slot:slots) {
        auto* storage=new llvm::GlobalVariable(*result,slot->getAllocatedType(),false,llvm::GlobalValue::InternalLinkage,llvm::UndefValue::get(slot->getAllocatedType()),"cudavm.continuation.value",nullptr,llvm::GlobalVariable::NotThreadLocal,dxil?0:10);
        storage->setAlignment(slot->getAlign());
        while(!slot->use_empty()) {
            auto& use=*slot->use_begin();use.getUser()->setOperand(use.getOperandNo(),storage);
        }
        // SPIRV-Tools localizes Private variables referenced by just one
        // function, even when that function is called repeatedly. An observable
        // entry initialization anchors each slot's invocation-wide lifetime.
        b.SetInsertPoint(&*prologue->getFirstInsertionPt());
        b.CreateStore(llvm::Constant::getNullValue(slot->getAllocatedType()),storage)->setVolatile(true);
        slot->eraseFromParent();
    }
    auto* invoke=llvm::BasicBlock::Create(b.getContext(),"continuation.step",entry,latch);
    auto* conditional=llvm::cast<llvm::BranchInst>(dispatch->getTerminator());conditional->setSuccessor(0,invoke);
    b.SetInsertPoint(invoke);
    std::vector<llvm::Value*> arguments;
    if(opencl) {for(auto* slot:slots)arguments.push_back(slot);for(unsigned i=0;i<packet_resource_count;++i) arguments.push_back(entry->getArg(i));}
    b.CreateCall(step,arguments);b.CreateBr(latch);
}
