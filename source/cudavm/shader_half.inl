// Translate only the checked PTX forms used by the pinned half headers. The
// CUDA module keeps its native instructions; this runs on the shader clone.
void lower_shader_half(llvm::Module& module) {
    // The headers refine an approximate FP32 reciprocal for half division.
    // Select the exact shared half quotient directly, avoiding an expansion of
    // that correction sequence through strict FP32 division and FMA helpers.
    for(bool pair:{false,true}) {
        std::string name=pair?"_ZL7__h2div7__half2S_":"_ZL6__hdiv6__halfS_";
        for(auto& function:module) {
            if(function.isDeclaration() || !function.hasFnAttribute("cudavm.cuda.source") || (function.getName()!=name && !function.getName().starts_with(name+".")))continue;
            auto* helper=module.getFunction("__cudavm_native_word_f16_div");
            if(!helper || helper->isDeclaration())throw std::runtime_error("missing half division helper");
            auto* result_type=function.getReturnType();
            bool aggregate=!pair && result_type->isStructTy() && result_type->getStructNumElements()==1 && result_type->getStructElementType(0)->isIntegerTy(16);
            if((pair && (!result_type->isVoidTy() || function.arg_size()!=3 || !function.getArg(0)->getType()->isPointerTy())) ||
                (!pair && (function.arg_size()!=2 || (!aggregate && !result_type->isIntegerTy(16)))))
                throw std::runtime_error("unsupported pinned half division result ABI");
            function.deleteBody();llvm::IRBuilder<> b(llvm::BasicBlock::Create(module.getContext(),"entry",&function));
            auto input=[&](unsigned i)->llvm::Value* {
                llvm::Value* value=function.getArg(i+(pair?1:0));auto* storage=pair?b.getInt32Ty():b.getInt16Ty();
                if(value->getType()->isPointerTy())value=b.CreateAlignedLoad(storage,value,llvm::Align(pair?4:2));
                if(value->getType()!=storage)throw std::runtime_error("unsupported pinned half division argument ABI");
                return b.CreateZExtOrTrunc(value,b.getInt32Ty());
            };
            auto* a=input(0);auto* c=input(1);
            auto lane=[&](unsigned shift) {
                return b.CreateCall(helper,{b.CreateAnd(b.CreateLShr(a,shift),b.getInt32(65535)),b.CreateAnd(b.CreateLShr(c,shift),b.getInt32(65535))});
            };
            if(pair) {
                b.CreateAlignedStore(b.CreateOr(lane(0),b.CreateShl(lane(16),16)),function.getArg(0),llvm::Align(4));b.CreateRetVoid();
            } else {
                auto* bits=b.CreateTrunc(lane(0),b.getInt16Ty());
                b.CreateRet(aggregate?b.CreateInsertValue(llvm::UndefValue::get(result_type),bits,0):bits);
            }
        }
    }
    // Extrema return one of their binary16 inputs. Avoid a widening/rounding
    // round trip, including the pre-SM80 header's FP32 assembly fallback.
    for(auto [name,maximum,propagate]:{std::tuple{"_ZL6__hmin6__halfS_",0u,0u},
            std::tuple{"_ZL6__hmax6__halfS_",1u,0u},std::tuple{"_ZL10__hmin_nan6__halfS_",0u,1u},
            std::tuple{"_ZL10__hmax_nan6__halfS_",1u,1u}}) {
        for(auto& candidate:module) {
            auto* function=&candidate;
            if(function->isDeclaration() || !function->hasFnAttribute("cudavm.cuda.source") || (function->getName()!=name && !function->getName().starts_with(std::string(name)+".")))continue;
            auto* helper=module.getFunction("__cudavm_native_word_f16_minmax");
            if(!helper || helper->isDeclaration() || function->arg_size()!=2)throw std::runtime_error("invalid pinned half extrema helper");
            auto* result_type=function->getReturnType();
            bool aggregate=result_type->isStructTy() && result_type->getStructNumElements()==1 && result_type->getStructElementType(0)->isIntegerTy(16);
            if(!aggregate && !result_type->isIntegerTy(16))throw std::runtime_error("unsupported pinned half extrema result ABI");
            function->deleteBody();llvm::IRBuilder<> b(llvm::BasicBlock::Create(module.getContext(),"entry",function));
            auto input=[&](unsigned i)->llvm::Value* {
                llvm::Value* value=function->getArg(i);
                if(value->getType()->isPointerTy())value=b.CreateAlignedLoad(b.getInt16Ty(),value,llvm::Align(2));
                if(!value->getType()->isIntegerTy(16))throw std::runtime_error("unsupported pinned half extrema argument ABI");
                return b.CreateZExt(value,b.getInt32Ty());
            };
            auto* bits=b.CreateTrunc(b.CreateCall(helper,{input(0),input(1),b.getInt32(maximum),b.getInt32(propagate)}),b.getInt16Ty());
            b.CreateRet(aggregate?b.CreateInsertValue(llvm::UndefValue::get(result_type),bits,0):bits);
        }
    }
    // Lower each public half math overload at its checked CUDA ABI boundary.
    // The shared source owns its approximation and final half rounding; NVIDIA
    // correction terms tailored to native approximations are unnecessary here.
    for(auto [scalar,packed,numeric]:{
            std::tuple{"hsin","h2sin","half_sin"},std::tuple{"hcos","h2cos","half_cos"},
            std::tuple{"hexp","h2exp","exp"},std::tuple{"hexp2","h2exp2","exp2"},std::tuple{"hexp10","h2exp10","exp10"},
            std::tuple{"hlog","h2log","half_log"},std::tuple{"hlog2","h2log2","half_log2"},std::tuple{"hlog10","h2log10","half_log10"},
            std::tuple{"hrcp","h2rcp","half_rcp"},std::tuple{"hsqrt","h2sqrt","half_sqrt"},std::tuple{"hrsqrt","h2rsqrt","half_rsqrt"},
            std::tuple{"htanh","h2tanh","half_tanh"},std::tuple{"htanh_approx","h2tanh_approx","half_tanh"}}) {
        for(bool pair:{false,true}) {
            std::string spelling=pair?packed:scalar;
            std::string name="_ZL"+std::to_string(spelling.size())+spelling+(pair?"7__half2":"6__half");
            for(auto& function:module) {
                if(function.isDeclaration() || !function.hasFnAttribute("cudavm.cuda.source") || (function.getName()!=name && !function.getName().starts_with(name+".")))continue;
                auto require=[&](const std::string& helper) {
                    auto* value=module.getFunction("__cudavm_native_"+helper);
                    if(!value || value->isDeclaration())throw std::runtime_error("missing half math helper: "+helper);
                    return value;
                };
                auto* result_type=function.getReturnType();
                bool aggregate=!pair && result_type->isStructTy() && result_type->getStructNumElements()==1 && result_type->getStructElementType(0)->isIntegerTy(16);
                if((pair && (!result_type->isVoidTy() || function.arg_size()!=2 || !function.getArg(0)->getType()->isPointerTy())) ||
                    (!pair && (function.arg_size()!=1 || (!aggregate && !result_type->isIntegerTy(16)))))
                    throw std::runtime_error("unsupported pinned half math result ABI: "+name);
                function.deleteBody();llvm::IRBuilder<> b(llvm::BasicBlock::Create(module.getContext(),"entry",&function));
                llvm::Value* input=function.getArg(pair?1:0);auto* storage=pair?b.getInt32Ty():b.getInt16Ty();
                if(input->getType()->isPointerTy())input=b.CreateAlignedLoad(storage,input,llvm::Align(pair?4:2));
                if(input->getType()!=storage)throw std::runtime_error("unsupported pinned half math argument ABI: "+name);
                auto lane=[&](unsigned shift) {
                    auto* bits=b.CreateAnd(b.CreateLShr(b.CreateZExtOrTrunc(input,b.getInt32Ty()),shift),b.getInt32(65535));
                    auto* x=b.CreateBitCast(b.CreateCall(require("word_f16_to_f32"),{bits}),b.getFloatTy());
                    auto* y=b.CreateCall(require(numeric),{x});
                    return b.CreateCall(require("word_f32_to_f16"),{b.CreateBitCast(y,b.getInt32Ty()),b.getInt32(0)});
                };
                if(pair) {
                    auto* bits=b.CreateOr(lane(0),b.CreateShl(lane(16),16));
                    b.CreateAlignedStore(bits,function.getArg(0),llvm::Align(4));b.CreateRetVoid();
                } else {
                    auto* bits=b.CreateTrunc(lane(0),b.getInt16Ty());
                    b.CreateRet(aggregate?b.CreateInsertValue(llvm::UndefValue::get(result_type),bits,0):bits);
                }
            }
        }
    }
    // The pinned header's private trig kernels accept widened half inputs.
    // Select the shared binary16-range source before expanding their polynomial
    // CFG, retaining the public header's lane packing and correction terms.
    for(auto [name,numeric]:{std::pair{"_ZL18__float_simpl_sinff","__cudavm_native_half_sin"},
            std::pair{"_ZL18__float_simpl_cosff","__cudavm_native_half_cos"}}) {
        for(auto& candidate:module) {
            auto* function=&candidate;
            if(function->isDeclaration() || !function->hasFnAttribute("cudavm.cuda.source") || (function->getName()!=name && !function->getName().starts_with(std::string(name)+".")))continue;
            auto* helper=module.getFunction(numeric);
            if(!helper || helper->isDeclaration() || helper->getFunctionType()!=function->getFunctionType())
                throw std::runtime_error("missing pinned half trig helper: "+std::string(numeric));
            function->deleteBody();llvm::IRBuilder<> b(llvm::BasicBlock::Create(module.getContext(),"entry",function));
            b.CreateRet(b.CreateCall(helper,{function->getArg(0)}));
        }
    }
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
            for(auto operation:{"add","sub","mul","fma"})for(auto modifier:{"",".rn",".sat",".rn.sat",".rn.relu"}) {
                bool fused=std::string_view(operation)=="fma";
                if(std::string_view(modifier)==".rn.relu" && !fused)continue;
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
                    if(std::string_view(modifier)==".rn.relu")
                        value=b.CreateSelect(b.CreateICmpUGT(b.CreateAnd(value,b.getInt32(0x7fff)),b.getInt32(0x7c00)),b.getInt32(0x7fff),b.CreateSelect(b.CreateICmpNE(b.CreateAnd(value,b.getInt32(0x8000)),b.getInt32(0)),b.getInt32(0),value));
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
        if(!result) {
            // The pinned math headers contain scoped straight-line PTX with
            // correction terms. Decode it with the external PTX importer so
            // both input paths retain exactly the same instruction semantics.
            if(text.find("f16")!=std::string::npos || text=="rcp.approx.ftz.f32$0,$1;" || text=="max.f32$0,$1,$2;" || text=="min.f32$0,$1,$2;")
                result=import_numeric_inline_ptx(*call);
            else continue;
        }
        if(result->getType()!=call->getType()) {
            if(!result->getType()->isIntegerTy() || !call->getType()->isIntegerTy())throw std::runtime_error("invalid half PTX result type: "+text);
            result=b.CreateZExtOrTrunc(result,call->getType());
        }
        call->replaceAllUsesWith(result);call->eraseFromParent();
    }
}
