// Translate only the checked PTX forms used by the pinned half headers. The
// CUDA module keeps its native instructions; this runs on the shader clone.
void lower_shader_half(llvm::Module& module) {
    std::vector<llvm::CallInst*> calls;
    for(auto& function:module)for(auto& block:function)for(auto& instruction:block)
        if(auto* call=llvm::dyn_cast<llvm::CallInst>(&instruction);call && call->isInlineAsm())calls.push_back(call);
    for(auto* call:calls) {
        auto* assembly=llvm::cast<llvm::InlineAsm>(call->getCalledOperand());
        if(assembly->hasSideEffects())continue;
        std::string text;
        for(char c:assembly->getAsmString())if(c!=' ' && c!='\n' && c!='\r' && c!='\t')text+=c;
        for(size_t at=0;(at=text.find("$$",at))!=std::string::npos;++at)text.erase(at,1);
        if(text.starts_with('{') && text.ends_with('}'))text=text.substr(1,text.size()-2);
        llvm::IRBuilder<> b(call);llvm::Value* result=nullptr;
        auto word=[&](unsigned i){
            if(!call->getArgOperand(i)->getType()->isIntegerTy())throw std::runtime_error("half PTX requires integer storage operands");
            return b.CreateZExtOrTrunc(call->getArgOperand(i),b.getInt32Ty());
        };
        auto helper=[&](const char* name,std::vector<llvm::Value*> args)->llvm::Value* {
            auto* function=module.getFunction(std::string("__cudavm_native_word_")+name);
            if(!function || function->isDeclaration())throw std::runtime_error("missing half numeric helper: "+std::string(name));
            if(function->arg_size()!=args.size())throw std::runtime_error("invalid half numeric helper signature: "+std::string(name));
            for(unsigned i=0;i<args.size();++i)if(args[i]->getType()!=function->getArg(i)->getType())
                throw std::runtime_error("invalid half PTX operand type: "+text);
            return b.CreateCall(function,args);
        };
        auto from_float=[&](llvm::Value* a,unsigned mode){return helper("f32_to_f16",{b.CreateBitCast(a,b.getInt32Ty()),b.getInt32(mode)});};
        auto to_float=[&](llvm::Value* a){return b.CreateBitCast(helper("f16_to_f32",{a}),b.getFloatTy());};
        auto pack=[&](llvm::Value* lo,llvm::Value* hi){return b.CreateOr(lo,b.CreateShl(hi,b.getInt32(16)));};
        auto lane=[&](unsigned i,unsigned shift){return b.CreateAnd(b.CreateLShr(word(i),b.getInt32(shift)),b.getInt32(65535));};
        for(auto [mode,spelling]:{std::pair{0u,"rn"},{1u,"rz"},{2u,"rm"},{3u,"rp"}}) {
            if(text==std::string("cvt.")+spelling+".f16.f32$0,$1;" && call->arg_size()==1)
                result=from_float(call->getArgOperand(0),mode);
            if(text==std::string("cvt.")+spelling+".f16.f64$0,$1;" && call->arg_size()==1)
                result=helper("f64_to_f16",{b.CreateBitCast(call->getArgOperand(0),b.getInt64Ty()),b.getInt32(mode)});
            for(unsigned width:{8u,16u,32u,64u})for(bool unsign:{false,true}) {
                auto integer=std::string(unsign?"u":"s")+std::to_string(width);
                if(text==std::string("cvt.")+spelling+".f16."+integer+"$0,$1;" && call->arg_size()==1) {
                    auto* input=b.CreateZExtOrTrunc(call->getArgOperand(0),b.getIntNTy(width));
                    auto* wide=unsign?b.CreateZExtOrTrunc(input,b.getInt64Ty()):b.CreateSExtOrTrunc(input,b.getInt64Ty());
                    auto* negative=unsign?b.getFalse():b.CreateICmpSLT(wide,b.getInt64(0));
                    result=helper("f16_round",{b.CreateSelect(negative,b.CreateNeg(wide),wide),b.getInt32(0),b.CreateSelect(negative,b.getInt32(0x8000),b.getInt32(0)),b.getInt32(mode)});
                }
                if(text==std::string("cvt.")+spelling+"i."+integer+".f16$0,$1;" && call->arg_size()==1)
                    result=helper("f16_to_integer",{word(0),b.getInt32(mode),b.getInt32(width),b.getInt32(unsign)});
            }
        }
        if(text=="cvt.f32.f16$0,$1;" && call->arg_size()==1)result=to_float(word(0));
        if(text==".reg.f16low;cvt.rn.f16.f32low,$1;mov.b32$0,{low,low};" && call->arg_size()==1) {
            auto* half=from_float(call->getArgOperand(0),0);result=pack(half,half);
        }
        if((text==".reg.f16low,high;cvt.rn.f16.f32low,$1;cvt.rn.f16.f32high,$2;mov.b32$0,{low,high};" || text=="cvt.rn.f16x2.f32$0,$2,$1;") && call->arg_size()==2)
            result=pack(from_float(call->getArgOperand(0),0),from_float(call->getArgOperand(1),0));
        if(text=="mov.b32$0,{$1,$2};" && call->arg_size()==2)result=pack(word(0),word(1));
        if(text=="mov.b32$0,{$1,$1};" && call->arg_size()==1)result=pack(word(0),word(0));
        if(text==".reg.f16low,high;mov.b32{low,high},$1;mov.b16$0,low;" && call->arg_size()==1)result=lane(0,0);
        if(text==".reg.f16low,high;mov.b32{low,high},$1;mov.b16$0,high;" && call->arg_size()==1)result=lane(0,16);
        if(text==".reg.f16low,high;mov.b32{low,high},$1;cvt.f32.f16$0,low;" && call->arg_size()==1)result=to_float(lane(0,0));
        if(text==".reg.f16low,high;mov.b32{low,high},$1;cvt.f32.f16$0,high;" && call->arg_size()==1)result=to_float(lane(0,16));
        for(bool packed:{false,true}) {
            std::string format=packed?"f16x2":"f16";
            for(auto operation:{"add","sub","mul","fma"})for(auto modifier:{"",".rn",".sat",".rn.sat"}) {
                bool fused=std::string_view(operation)=="fma";
                if(text!=std::string(operation)+modifier+"."+format+(fused?"$0,$1,$2,$3;":"$0,$1,$2;") || call->arg_size()!=(fused?3u:2u))continue;
                auto compute=[&](unsigned shift) {
                    auto* a=lane(0,shift);auto* v=lane(1,shift);
                    if(std::string_view(operation)=="sub")v=b.CreateXor(v,b.getInt32(0x8000));
                    auto* value=fused?helper("f16_fma",{a,v,lane(2,shift)}):helper(std::string_view(operation)=="mul"?"f16_mul":"f16_add",{a,v});
                    if(std::string_view(modifier).find("sat")!=std::string_view::npos) {
                        auto* magnitude=b.CreateAnd(value,b.getInt32(0x7fff));
                        auto* zero=b.CreateOr(b.CreateICmpNE(b.CreateAnd(value,b.getInt32(0x8000)),b.getInt32(0)),b.CreateICmpUGT(magnitude,b.getInt32(0x7c00)));
                        value=b.CreateSelect(zero,b.getInt32(0),b.CreateSelect(b.CreateICmpUGT(value,b.getInt32(0x3c00)),b.getInt32(0x3c00),value));
                    }
                    return value;
                };
                result=packed?pack(compute(0),compute(16)):compute(0);
            }
            for(auto operation:{"neg","abs"})if(text==std::string(operation)+"."+format+"$0,$1;" && call->arg_size()==1)
                result=std::string_view(operation)=="neg"?b.CreateXor(word(0),b.getInt32(packed?0x80008000u:0x8000u)):b.CreateAnd(word(0),b.getInt32(packed?0x7fff7fffu:0x7fffu));
        }
        for(auto [spelling,predicate]:{std::pair{"eq",llvm::CmpInst::FCMP_OEQ},{"ne",llvm::CmpInst::FCMP_ONE},{"lt",llvm::CmpInst::FCMP_OLT},{"le",llvm::CmpInst::FCMP_OLE},{"gt",llvm::CmpInst::FCMP_OGT},{"ge",llvm::CmpInst::FCMP_OGE},{"equ",llvm::CmpInst::FCMP_UEQ},{"neu",llvm::CmpInst::FCMP_UNE},{"ltu",llvm::CmpInst::FCMP_ULT},{"leu",llvm::CmpInst::FCMP_ULE},{"gtu",llvm::CmpInst::FCMP_UGT},{"geu",llvm::CmpInst::FCMP_UGE}}) {
            if(call->arg_size()!=2)continue;
            auto compare=[&](unsigned shift){return b.CreateFCmp(predicate,to_float(lane(0,shift)),to_float(lane(1,shift)));};
            if(text==std::string(".reg.pred__$temp3;setp.")+spelling+".f16__$temp3,$1,$2;selp.u16$0,1,0,__$temp3;")result=b.CreateZExt(compare(0),b.getInt32Ty());
            for(bool mask:{false,true})if(text==std::string("set.")+spelling+(mask?".u32.f16x2":".f16x2.f16x2")+"$0,$1,$2;")
                result=pack(b.CreateSelect(compare(0),b.getInt32(mask?65535:0x3c00),b.getInt32(0)),b.CreateSelect(compare(16),b.getInt32(mask?65535:0x3c00),b.getInt32(0)));
        }
        if(!result)continue;
        if(result->getType()!=call->getType()) {
            if(!result->getType()->isIntegerTy() || !call->getType()->isIntegerTy())throw std::runtime_error("invalid half PTX result type: "+text);
            result=b.CreateZExtOrTrunc(result,call->getType());
        }
        call->replaceAllUsesWith(result);call->eraseFromParent();
    }
}
