// Surface storage is already resolved into the arena. Keep masked writes in a
// small helper without the general pointer helper's address-space dispatch.
void store_surface_word(llvm::Value* address, llvm::Value* value, llvm::Value* mask, bool partial) {
    auto saved = b.saveIP();
    auto name = partial ? "cudavm.surface.store.partial" : "cudavm.surface.store.word";
    auto* helper = result->getFunction(name);
    if (!helper) {
        helper = llvm::Function::Create(llvm::FunctionType::get(b.getVoidTy(), arguments({b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty()}), false),
            llvm::GlobalValue::InternalLinkage, name, *result);
        auto* first = llvm::BasicBlock::Create(b.getContext(), "entry", helper);
        auto* active = llvm::BasicBlock::Create(b.getContext(), "active", helper);
        auto* done = llvm::BasicBlock::Create(b.getContext(), "done", helper);
        b.SetInsertPoint(first); b.CreateCondBr(b.CreateICmpEQ(helper->getArg(1), b.getInt32(0)), active, done);
        b.SetInsertPoint(active);
        auto* pointer = word_pointer(0, b.CreateLShr(helper->getArg(0), b.getInt32(2)));
        if (!partial) { b.CreateAlignedStore(helper->getArg(2), pointer, llvm::Align(4), true); b.CreateBr(done); }
        else {
            auto scope = dxil ? llvm::SyncScope::System : b.getContext().getOrInsertSyncScopeID("device");
            auto* initial = b.CreateExtractValue(b.CreateAtomicCmpXchg(pointer, b.getInt32(0), b.getInt32(0), llvm::Align(4),
                llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering::Monotonic, scope), 0);
            auto* retry = llvm::BasicBlock::Create(b.getContext(), "retry", helper); b.CreateBr(retry); b.SetInsertPoint(retry);
            auto* previous = b.CreatePHI(b.getInt32Ty(), 2); previous->addIncoming(initial, active);
            auto* merged = b.CreateOr(b.CreateAnd(previous, b.CreateNot(helper->getArg(3))), b.CreateAnd(helper->getArg(2), helper->getArg(3)));
            auto* exchange = b.CreateAtomicCmpXchg(pointer, previous, merged, llvm::Align(4), llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering::Monotonic, scope);
            previous->addIncoming(b.CreateExtractValue(exchange, 0), retry);
            b.CreateCondBr(b.CreateExtractValue(exchange, 1), done, retry);
        }
        b.SetInsertPoint(done); b.CreateRetVoid();
    }
    b.restoreIP(saved); call_helper(helper, {address, value, mask});
}
// Match the pinned headers' unformatted 2D transfers, including packed vectors.
bool surface_operation(const llvm::CallInst& instruction, llvm::InlineAsm& assembly,
                       const std::string& text, llvm::Value*& output) {
    unsigned components = 0, bits = 0; bool writing = false, clamp = false, trap = false;
    std::string constraints;
    for (unsigned count : {1u, 2u, 4u}) for (unsigned width : {8u, 16u, 32u, 64u}) {
        if (count * width > 128) continue;
        std::string suffix = count == 1 ? "" : "v" + std::to_string(count) + ".";
        suffix += "b" + std::to_string(width) + ".";
        std::string destinations, sources;
        for (unsigned i = 0; i < count; ++i) {
            if (i) { destinations += ","; sources += ","; }
            destinations += "$" + std::to_string(i); sources += "$" + std::to_string(i + 3);
        }
        for (auto mode : {"zero", "clamp", "trap"}) {
            auto load_text = "suld.b.2d." + suffix + mode + "{" + destinations + "},[$" + std::to_string(count) + ",{$" +
                std::to_string(count + 1) + ",$" + std::to_string(count + 2) + "}];";
            auto store_text = "sust.b.2d." + suffix + mode + "[$0,{$1,$2}],{" + sources + "};";
            if (text != load_text && text != store_text) continue;
            components = count; bits = width; writing = text == store_text;
            clamp = std::string_view(mode) == "clamp"; trap = std::string_view(mode) == "trap";
            auto reg = width < 32 ? "h" : width == 32 ? "r" : "l";
            if (writing) { constraints = "l,r,r"; for (unsigned i = 0; i < count; ++i) constraints += std::string(",") + reg; }
            else { for (unsigned i = 0; i < count; ++i) constraints += std::string(i ? ",=" : "=") + reg; constraints += ",l,r,r"; }
        }
    }
    if (!components) return false;
    auto valid_type = [&](llvm::Type* t) {
        return t->isIntegerTy(bits) || (bits == 8 && t->isIntegerTy(16)) || (bits == 32 && t->isFloatTy());
    };
    if (instruction.arg_size() != (writing ? components + 3 : 3) ||
        !instruction.getArgOperand(0)->getType()->isIntegerTy(64) ||
        !instruction.getArgOperand(1)->getType()->isIntegerTy(32) || !instruction.getArgOperand(2)->getType()->isIntegerTy(32) ||
        assembly.getConstraintString() != constraints) fail(instruction, "invalid surface instruction signature");
    if (writing) {
        if (!instruction.getType()->isVoidTy()) fail(instruction, "invalid surface store result");
        for (unsigned i = 0; i < components; ++i) if (!valid_type(instruction.getArgOperand(3 + i)->getType())) fail(instruction, "invalid surface store component");
    } else if (components == 1) {
        if (!valid_type(instruction.getType())) fail(instruction, "invalid surface load result");
    } else {
        auto* tuple = llvm::dyn_cast<llvm::StructType>(instruction.getType());
        if (!tuple || tuple->getNumElements() != components || !llvm::all_of(tuple->elements(), valid_type)) fail(instruction, "invalid surface load components");
    }
    entry->addFnAttr("cudavm.image.operations");
    if (dxil) {
        read->addFnAttr(llvm::Attribute::AlwaysInline);
        resolve->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    auto* object = b.CreateZExt(call_helper(resolve, {value(instruction.getArgOperand(0))}), b.getInt64Ty());
    auto* base = pair(image_word(object), image_word(object, 4));
    auto* physical = b.CreateZExt(call_helper(resolve, {base}), b.getInt64Ty());
    auto* width = image_word(object, 8);
    auto* height = image_word(object, 12);
    auto* texel_bytes = image_word(object, 52);
    auto* pitch = image_word(object, 64);
    auto* row = b.CreateMul(width, texel_bytes);
    auto* transfer = b.getInt32(components * bits / 8);
    llvm::Value *x = value(instruction.getArgOperand(1)), *y = value(instruction.getArgOperand(2));
    auto* fits = b.CreateICmpUGE(row, transfer);
    llvm::Value* valid = b.CreateAnd(fits, b.CreateAnd(b.CreateICmpULE(x, b.CreateSub(row, transfer)), b.CreateICmpULT(y, height)));
    if (clamp) {
        auto clamp_index = [&](llvm::Value* index, llvm::Value* maximum) {
            return b.CreateSelect(b.CreateICmpSLT(index, b.getInt32(0)), b.getInt32(0), b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, index, maximum));
        };
        x = clamp_index(x, b.CreateSub(row, transfer)); y = clamp_index(y, b.CreateSub(height, b.getInt32(1))); valid = fits;
    }
    if (trap) {
        auto scope = dxil ? llvm::SyncScope::System : b.getContext().getOrInsertSyncScopeID("device");
        b.CreateAtomicRMW(llvm::AtomicRMWInst::Or, word_pointer(0, b.getInt32(2)), b.CreateZExt(b.CreateNot(valid), b.getInt32Ty()),
            llvm::Align(4), llvm::AtomicOrdering::Monotonic, scope);
    }
    auto* offset = b.CreateAdd(b.CreateMul(b.CreateZExt(y, b.getInt64Ty()), b.CreateZExt(pitch, b.getInt64Ty())), b.CreateZExt(x, b.getInt64Ty()));
    llvm::Value* tuple = writing ? nullptr : llvm::UndefValue::get(type(instruction.getType()));
    for (unsigned i = 0; i < components; ++i) {
        auto* component_offset = b.CreateAdd(offset, b.getInt64(i * bits / 8));
        auto* address = b.CreateSelect(valid, b.CreateAdd(physical, component_offset), b.getInt64(0xffffffff00000000ull));
        if (writing) {
            auto* component = value(instruction.getArgOperand(3 + i));
            if (bits >= 32) {
                store_surface_word(address, b.CreateZExtOrTrunc(component, b.getInt32Ty()), b.getInt32(UINT32_MAX), false);
                if (bits == 64) store_surface_word(b.CreateAdd(address, b.getInt64(4)), b.CreateTrunc(b.CreateLShr(component, b.getInt64(32)), b.getInt32Ty()), b.getInt32(UINT32_MAX), false);
            }
            else {
                auto* shift = b.CreateMul(b.CreateAnd(b.CreateTrunc(b.CreateAdd(physical, component_offset), b.getInt32Ty()), b.getInt32(3)), b.getInt32(8));
                auto* mask = b.CreateShl(b.getInt32((1u << bits) - 1), shift);
                store_surface_word(address, b.CreateShl(component, shift), mask, true);
            }
        } else {
            auto* source = b.CreateSelect(valid, b.CreateAdd(physical, component_offset), b.getInt64(0));
            auto* word = image_word(source);
            llvm::Value* component = word;
            if (bits == 64) component = pair(word, image_word(source, 4));
            else if (bits < 32) {
                auto* shift = b.CreateMul(b.CreateAnd(b.CreateTrunc(source, b.getInt32Ty()), b.getInt32(3)), b.getInt32(8));
                component = b.CreateAnd(b.CreateLShr(word, shift), b.getInt32((1u << bits) - 1));
            }
            component = b.CreateSelect(valid, component, llvm::ConstantInt::get(component->getType(), 0));
            tuple = components == 1 ? component : b.CreateInsertValue(tuple, component, i);
        }
    }
    output = tuple; return true;
}
