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
    struct rendezvous {llvm::CallInst* call;llvm::BasicBlock* wait;unsigned operation;};
    std::vector<rendezvous> sites;
    for(auto& bb:*entry) for(auto& input:bb) if(auto* call=llvm::dyn_cast<llvm::CallInst>(&input))
        if(auto* f=call->getCalledFunction();f && f->hasFnAttribute("cudavm.collective"))
            sites.push_back({call,nullptr,unsigned(std::stoul(f->getFnAttribute("cudavm.collective").getValueAsString().str()))});
    for(auto& site:sites) {
        auto* arrival=site.call->getParent();
        site.wait=arrival->splitBasicBlock(site.call,"collective.wait");
        site.wait->splitBasicBlock(site.call->getNextNode(),"collective.resume");
        b.SetInsertPoint(arrival->getTerminator()); auto* id=linear_thread();
        collective_store(site.call->getArgOperand(0),collective_pointer(collective_masks,id));
        collective_store(site.call->getArgOperand(1),collective_pointer(collective_data,id));
        if(site.operation==9 || site.operation==11)
            collective_store(site.call->getArgOperand(2),collective_pointer(collective_data,b.CreateAdd(id,b.getInt32(1024))));
    }
    // Values live across a yield belong to the invocation, including PHI edge
    // updates. Demote before changing edges so loops and critical edges retain
    // their source meaning. Native allocas are hoisted into the new prologue.
    llvm::FunctionAnalysisManager analyses;llvm::PassBuilder builder;builder.registerFunctionAnalyses(analyses);
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
    std::vector<llvm::AllocaInst*> slots;
    for(auto* bb:quanta) for(auto& input:*bb) if(auto* slot=llvm::dyn_cast<llvm::AllocaInst>(&input)) slots.push_back(slot);
    for(auto* slot:slots) {slot->removeFromParent();b.Insert(slot);}
    auto* local_pc=b.CreateAlloca(b.getInt32Ty(),nullptr,"continuation.pc");
    b.CreateStore(b.getInt32(states.at(initial)),local_pc);b.CreateBr(head);
    b.SetInsertPoint(head);auto* id=linear_thread();auto* count=thread_count();
    auto* pc=b.CreateLoad(b.getInt32Ty(),local_pc);
    collective_store(pc,collective_pointer(collective_pc,id));group_barrier();
    b.CreateCondBr(b.CreateICmpEQ(id,b.getInt32(0)),leader,dispatch);
    b.SetInsertPoint(leader);b.CreateBr(scan);b.SetInsertPoint(scan);
    auto* index=b.CreatePHI(b.getInt32Ty(),2);index->addIncoming(b.getInt32(0),leader);
    auto* any=b.CreatePHI(b.getInt32Ty(),2);any->addIncoming(b.getInt32(0),leader);
    b.CreateCondBr(b.CreateICmpULT(index,count),sample,scanned);
    b.SetInsertPoint(sample);
    any->addIncoming(b.CreateOr(any,collective_load(collective_pointer(collective_pc,index))),sample);
    index->addIncoming(b.CreateAdd(index,b.getInt32(1)),sample);b.CreateBr(scan);
    b.SetInsertPoint(scanned);collective_store(any,collective_pointer(collective_control,b.getInt32(0)));b.CreateBr(dispatch);
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
    for(auto& site:sites) {
        auto* execute=site.wait->splitBasicBlock(site.call,"collective.execute");
        auto* branch=site.wait->getTerminator();b.SetInsertPoint(branch);
        auto* mask=collective_load(collective_pointer(collective_masks,linear_thread()));
        std::vector<llvm::Value*> args{mask,b.getInt32(states.at(site.wait)),b.getInt32(site.operation)};
        if(opencl) {args.push_back(entry->getArg(0));args.push_back(entry->getArg(1));}
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
        signature.push_back(resource_pointer_type());signature.push_back(resource_pointer_type());
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
            if(operand==entry->getArg(0))input.setOperand(i,step->getArg(unsigned(slots.size())));
            if(operand==entry->getArg(1))input.setOperand(i,step->getArg(unsigned(slots.size()+1)));
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
    if(opencl) {for(auto* slot:slots)arguments.push_back(slot);arguments.push_back(entry->getArg(0));arguments.push_back(entry->getArg(1));}
    b.CreateCall(step,arguments);b.CreateBr(latch);
}
