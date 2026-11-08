// The pinned fallbacks convert every source type through these scalar routines.
// Keep one numerical helper instead of inlining many scalar/pair/quad copies
// into the entry point, where dispatcher repair grows the shader excessively.
void preserve_shader_low_precision_conversions(llvm::Module& module) {
    for (const std::string name : {
        "_ZL22__nv_cvt_double_to_fp8d17__nv_saturation_t25__nv_fp8_interpretation_t",
        "_ZL22__nv_cvt_double_to_fp6d25__nv_fp6_interpretation_t13cudaRoundMode",
        "_ZL22__nv_cvt_double_to_fp4d25__nv_fp4_interpretation_t13cudaRoundMode"}) for (auto& function : module) {
        if (function.isDeclaration() || !function.hasFnAttribute("cudavm.cuda.source") ||
            (function.getName() != name && !function.getName().starts_with(name + "."))) continue;
        if (!function.getReturnType()->isIntegerTy(8) || function.arg_size() != 3 ||
            !function.getArg(0)->getType()->isDoubleTy() || !function.getArg(1)->getType()->isIntegerTy(32) ||
            !function.getArg(2)->getType()->isIntegerTy(32))
            throw std::runtime_error("unsupported pinned low precision conversion ABI");
        function.addFnAttr("cudavm.math", "low_precision_conversion");
    }
}

// Public wrappers expose constant format/mode arguments after inlining. Clone
// those cases before service emission so valid modes do not retain impossible
// assertion paths. Dynamic or unsupported modes keep the original routine.
bool specialize_shader_low_precision_conversions(llvm::Module& module) {
    std::vector<llvm::CallInst*> calls;
    for (auto& function : module) for (auto& block : function) for (auto& instruction : block)
        if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
            auto* target = call->getCalledFunction();
            if (target && target->hasFnAttribute("cudavm.math") && !target->hasFnAttribute("cudavm.constant.conversion") &&
                (target->getName().starts_with("_ZL22__nv_cvt_double_to_fp6") || target->getName().starts_with("_ZL22__nv_cvt_double_to_fp4")) &&
                target->arg_size() == 3) calls.push_back(call);
        }
    bool changed = false;
    for (auto* call : calls) {
        auto* format = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
        auto* rounding = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
        if (!format || !rounding || format->getZExtValue() > 1 || rounding->getZExtValue() > 1) continue;
        auto* original = call->getCalledFunction();
        if (original->getName().starts_with("_ZL22__nv_cvt_double_to_fp4") && !format->isZero()) continue;
        const auto name = original->getName().str() + ".constant." + std::to_string(format->getZExtValue()) + "." + std::to_string(rounding->getZExtValue());
        auto* clone = module.getFunction(name);
        if (!clone) {
            clone = llvm::Function::Create(original->getFunctionType(), llvm::GlobalValue::InternalLinkage, name, module);
            clone->copyAttributesFrom(original);
            llvm::ValueToValueMapTy values;
            values[original->getArg(0)] = clone->getArg(0);
            values[original->getArg(1)] = format;
            values[original->getArg(2)] = rounding;
            llvm::SmallVector<llvm::ReturnInst*> returns;
            llvm::CloneFunctionInto(clone, original, values, llvm::CloneFunctionChangeType::LocalChangesOnly, returns);
        }
        clone->addFnAttr("cudavm.constant.conversion");
        call->setCalledFunction(clone);
        changed = true;
    }
    return changed;
}
