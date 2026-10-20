// Linear filtering shares the array backing and descriptor with point sampling.
// Keep its CFG inside a helper: source-block PHI edges retain their identities,
// and point sampling returns before performing extra tap reads or arithmetic.
llvm::Value* filter_texture_component(llvm::Value* object, llvm::Value* x, llvm::Value* y, llvm::Value* point, unsigned component) {
    auto saved = b.saveIP();
    auto name = "cudavm.filter.texture." + std::to_string(component);
    auto* helper = result->getFunction(name);
    if (!helper) {
        std::vector<llvm::Type*> signature{b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty(), point->getType()};
        if (opencl && local_bytes) signature.push_back(b.getPtrTy());
        if (opencl) { signature.push_back(resource_pointer_type()); signature.push_back(resource_pointer_type()); }
        helper = llvm::Function::Create(llvm::FunctionType::get(point->getType(), signature, false),
            llvm::GlobalValue::InternalLinkage, name, *result);
        if (opencl && local_bytes) helper->addFnAttr("cudavm.native.local");
        helper->addFnAttr(llvm::Attribute::AlwaysInline);
        auto* first = llvm::BasicBlock::Create(b.getContext(), "entry", helper);
        auto* nearest = llvm::BasicBlock::Create(b.getContext(), "point", helper);
        auto* linear = llvm::BasicBlock::Create(b.getContext(), "linear", helper);
        b.SetInsertPoint(first);
        auto* descriptor = pair(helper->getArg(0), helper->getArg(1));
        auto* flags = load(b.CreateAdd(descriptor, b.getInt64(20)), b.getInt32Ty());
        b.CreateCondBr(b.CreateICmpNE(b.CreateAnd(flags, b.getInt32(2)), b.getInt32(0)), linear, nearest);
        b.SetInsertPoint(nearest); b.CreateRet(helper->getArg(4)); b.SetInsertPoint(linear);
        auto* normalized = b.CreateICmpNE(b.CreateAnd(flags, b.getInt32(1)), b.getInt32(0));
        auto* base = load(descriptor, b.getInt64Ty());
        auto* width = load(b.CreateAdd(descriptor, b.getInt64(8)), b.getInt32Ty());
        auto* height = load(b.CreateAdd(descriptor, b.getInt64(12)), b.getInt32Ty());
        auto* channels = load(b.CreateAdd(descriptor, b.getInt64(48)), b.getInt32Ty());
        auto* texel_bytes = load(b.CreateAdd(descriptor, b.getInt64(52)), b.getInt32Ty());
        auto* present = b.CreateICmpUGT(channels, b.getInt32(component));
        auto* mode_x = load(b.CreateAdd(descriptor, b.getInt64(24)), b.getInt32Ty());
        auto* mode_y = load(b.CreateAdd(descriptor, b.getInt64(28)), b.getInt32Ty());
        // Quantize texel coordinates to 1/256 before subtracting the half-texel
        // offset. Significand products avoid overflow, FP-to-int poison and
        // premature FP rounding at coefficient halfways. Ties round upward.
        auto axis = [&](llvm::Value* bits, llvm::Value* extent, llvm::Value* mode) {
            auto* exponent = b.CreateAnd(b.CreateLShr(bits, b.getInt32(23)), b.getInt32(255));
            auto* significand = b.CreateOr(b.CreateAnd(bits, b.getInt32(0x7fffff)), b.getInt32(0x800000));
            auto* shift = b.CreateSub(b.getInt32(150), exponent);
            auto* mask = b.CreateSub(b.CreateShl(b.getInt32(1), b.CreateAnd(shift, b.getInt32(31))), b.getInt32(1));
            auto* fraction = b.CreateSelect(b.CreateICmpULT(exponent, b.getInt32(127)), significand, b.CreateAnd(significand, mask));
            fraction = b.CreateSelect(b.CreateICmpUGE(exponent, b.getInt32(150)), b.getInt32(0), fraction);
            auto* repeated = b.CreateAnd(normalized, b.CreateOr(b.CreateICmpEQ(mode, b.getInt32(0)), b.CreateICmpEQ(mode, b.getInt32(2))));
            auto* magnitude = b.CreateSelect(repeated, fraction, significand);
            magnitude = b.CreateSelect(b.CreateICmpEQ(exponent, b.getInt32(0)), b.getInt32(0), magnitude);
            auto* scale = b.CreateZExt(b.CreateSelect(normalized, extent, b.getInt32(1)), b.getInt64Ty());
            auto* product = b.CreateShl(b.CreateMul(b.CreateZExt(magnitude, b.getInt64Ty()), scale), b.getInt64(8));
            auto* safe_shift = b.CreateZExt(b.CreateAnd(shift, b.getInt32(63)), b.getInt64Ty());
            auto* whole = b.CreateSelect(b.CreateICmpULT(shift, b.getInt32(64)), b.CreateLShr(product, safe_shift), b.getInt64(0));
            auto* remainder = b.CreateSub(product, b.CreateShl(whole, safe_shift));
            auto* halfway = b.CreateShl(b.getInt64(1), b.CreateZExt(b.CreateAnd(b.CreateSub(shift, b.getInt32(1)), b.getInt32(63)), b.getInt64Ty()));
            auto* negative = b.CreateICmpSLT(bits, b.getInt32(0));
            auto* increment = b.CreateSelect(negative, b.CreateICmpUGT(remainder, halfway), b.CreateICmpUGE(remainder, halfway));
            increment = b.CreateAnd(increment, b.CreateAnd(b.CreateICmpUGT(shift, b.getInt32(0)), b.CreateICmpULT(shift, b.getInt32(64))));
            auto* right = b.CreateAdd(whole, b.CreateZExt(increment, b.getInt64Ty()));
            auto* left = b.CreateShl(product, b.CreateZExt(b.CreateAnd(b.CreateSub(exponent, b.getInt32(150)), b.getInt32(63)), b.getInt64Ty()));
            auto* fixed = b.CreateSelect(b.CreateICmpULE(exponent, b.getInt32(150)), right, left);
            auto* bound = b.CreateShl(b.CreateAdd(b.CreateZExt(extent, b.getInt64Ty()), b.getInt64(1)), b.getInt64(8));
            fixed = b.CreateSelect(b.CreateAnd(b.CreateNot(repeated), b.CreateICmpUGE(exponent, b.getInt32(158))), bound,
                b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, fixed, bound));
            auto* odd = b.CreateAnd(b.CreateICmpUGE(exponent, b.getInt32(127)), b.CreateICmpULE(exponent, b.getInt32(150)));
            odd = b.CreateAnd(odd, b.CreateICmpNE(b.CreateAnd(b.CreateLShr(significand, b.CreateAnd(shift, b.getInt32(31))), b.getInt32(1)), b.getInt32(0)));
            auto* mirror_period = b.CreateAnd(repeated, b.CreateAnd(odd, b.CreateICmpEQ(mode, b.getInt32(2))));
            fixed = b.CreateAdd(fixed, b.CreateSelect(mirror_period, b.CreateShl(b.CreateZExt(extent, b.getInt64Ty()), b.getInt64(8)), b.getInt64(0)));
            fixed = b.CreateSub(b.CreateSelect(negative, b.CreateNeg(fixed), fixed), b.getInt64(128));
            auto* weight = b.CreateFMul(b.CreateUIToFP(b.CreateTrunc(b.CreateAnd(fixed, b.getInt64(255)), b.getInt32Ty()), b.getFloatTy()), llvm::ConstantFP::get(b.getFloatTy(), 1.0 / 256.0));
            return std::array<llvm::Value*, 2>{b.CreateAShr(fixed, b.getInt64(8)), weight};
        };
        auto horizontal = axis(helper->getArg(2), width, mode_x), vertical = axis(helper->getArg(3), height, mode_y);
        auto address = [&](llvm::Value* index, llvm::Value* extent, llvm::Value* mode) {
            auto* size = b.CreateZExt(extent, b.getInt64Ty());
            auto* mirror = b.CreateICmpEQ(mode, b.getInt32(2));
            auto* period = b.CreateSelect(mirror, b.CreateMul(size, b.getInt64(2)), size);
            auto* repeated = index;
            for (unsigned step = 0; step < 2; ++step) repeated = b.CreateSelect(b.CreateICmpSLT(repeated, b.getInt64(0)), b.CreateAdd(repeated, period), repeated);
            repeated = b.CreateSelect(b.CreateICmpUGE(repeated, period), b.CreateSub(repeated, period), repeated);
            repeated = b.CreateSelect(b.CreateAnd(mirror, b.CreateICmpUGE(repeated, size)), b.CreateSub(b.CreateSub(period, b.getInt64(1)), repeated), repeated);
            auto* maximum = b.CreateSub(size, b.getInt64(1));
            auto* clamped = b.CreateSelect(b.CreateICmpSLT(index, b.getInt64(0)), b.getInt64(0), b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, index, maximum));
            auto* use_repeat = b.CreateOr(b.CreateICmpEQ(mode, b.getInt32(0)), mirror);
            auto* valid = b.CreateOr(b.CreateICmpNE(mode, b.getInt32(3)), b.CreateICmpULT(index, size));
            return std::array<llvm::Value*, 2>{b.CreateSelect(use_repeat, repeated, clamped), valid};
        };
        std::array<llvm::Value*, 4> samples{}, valid{};
        for (unsigned row = 0; row < 2; ++row) for (unsigned column = 0; column < 2; ++column) {
            auto xx = address(b.CreateAdd(horizontal[0], b.getInt64(column)), width, mode_x);
            auto yy = address(b.CreateAdd(vertical[0], b.getInt64(row)), height, mode_y);
            unsigned tap = 2 * row + column; valid[tap] = b.CreateAnd(xx[1], yy[1]);
            auto* offset = b.CreateAdd(b.CreateMul(b.CreateAdd(b.CreateMul(yy[0], b.CreateZExt(width, b.getInt64Ty())), xx[0]), b.CreateZExt(texel_bytes, b.getInt64Ty())), b.getInt64(4 * component));
            auto* sample = load(b.CreateSelect(b.CreateAnd(valid[tap], present), b.CreateAdd(base, offset), b.getInt64(0xffffffff00000000ull)), b.getInt32Ty());
            samples[tap] = b.CreateSelect(present, sample, b.getInt32(component == 3 ? 0x3f800000 : 0));
        }
        auto flush = [&](llvm::Value* real) {
            auto* bits = b.CreateBitCast(real, b.getInt32Ty());
            return b.CreateBitCast(b.CreateSelect(b.CreateICmpEQ(b.CreateAnd(bits, b.getInt32(0x7f800000)), b.getInt32(0)), b.CreateAnd(bits, b.getInt32(0x80000000)), bits), b.getFloatTy());
        };
        auto interpolate = [&](llvm::Value* a, llvm::Value* c, llvm::Value* weight) {
            a = flush(a); c = flush(c);
            auto* magnitude_a = b.CreateAnd(b.CreateBitCast(a,b.getInt32Ty()),b.getInt32(0x7fffffff));
            auto* magnitude_c = b.CreateAnd(b.CreateBitCast(c,b.getInt32Ty()),b.getInt32(0x7fffffff));
            auto* rescale = b.CreateICmpULT(b.CreateBinaryIntrinsic(llvm::Intrinsic::umax,magnitude_a,magnitude_c), b.getInt32(0x6c800000));
            // Keep normal texels near FLT_MIN from losing their weighted
            // products on shader hardware that flushes intermediate subnormals.
            auto* scale = b.CreateSelect(rescale, llvm::ConstantFP::get(b.getFloatTy(), 4294967296.0), llvm::ConstantFP::get(b.getFloatTy(), 1.0));
            auto* unscale = b.CreateSelect(rescale, llvm::ConstantFP::get(b.getFloatTy(), 1.0 / 4294967296.0), llvm::ConstantFP::get(b.getFloatTy(), 1.0));
            auto* blended = b.CreateIntrinsic(b.getFloatTy(), dxil ? llvm::Intrinsic::fmuladd : llvm::Intrinsic::fma, {b.CreateFMul(a,scale),
                b.CreateFSub(llvm::ConstantFP::get(b.getFloatTy(), 1.0), weight), b.CreateFMul(b.CreateFMul(c,scale), weight)});
            auto* bits = b.CreateBitCast(flush(b.CreateFMul(blended, unscale)), b.getInt32Ty());
            auto* a_bits = b.CreateBitCast(a,b.getInt32Ty()); auto* c_bits = b.CreateBitCast(c,b.getInt32Ty());
            auto* a_inf = b.CreateICmpEQ(magnitude_a,b.getInt32(0x7f800000));
            auto* c_inf = b.CreateICmpEQ(magnitude_c,b.getInt32(0x7f800000));
            auto* opposite = b.CreateAnd(b.CreateAnd(a_inf,c_inf), b.CreateICmpSLT(b.CreateXor(a_bits,c_bits),b.getInt32(0)));
            auto* nan = b.CreateOr(opposite, b.CreateOr(b.CreateICmpUGT(magnitude_a,b.getInt32(0x7f800000)), b.CreateICmpUGT(magnitude_c,b.getInt32(0x7f800000))));
            // Explicit bit selection also protects GLSL implementations whose
            // interpolation optimizations assume finite inputs.
            bits = b.CreateSelect(b.CreateOr(a_inf,c_inf), b.CreateSelect(a_inf,a_bits,c_bits), bits);
            bits = b.CreateSelect(nan,b.getInt32(0x7fc00000),bits);
            bits = b.CreateSelect(b.CreateFCmpOEQ(weight,llvm::ConstantFP::get(b.getFloatTy(),0.0)),a_bits,bits);
            return b.CreateBitCast(bits,b.getFloatTy());
        };
        auto* border = load(b.CreateAdd(descriptor, b.getInt64(32 + 4 * component)), b.getInt32Ty());
        std::array<llvm::Value*, 4> values;
        for (unsigned tap = 0; tap < 4; ++tap) values[tap] = b.CreateBitCast(b.CreateSelect(valid[tap], samples[tap], border), b.getFloatTy());
        auto* top = interpolate(values[0], values[1], horizontal[1]);
        auto* bottom = interpolate(values[2], values[3], horizontal[1]);
        b.CreateRet(b.CreateBitCast(interpolate(top, bottom, vertical[1]), b.getInt32Ty()));
    }
    b.restoreIP(saved);
    return call_helper(helper, {object, x, y, point});
}
llvm::Value* filter_texture(llvm::Value* object, llvm::Value* x, llvm::Value* y, llvm::Value* point, const llvm::CallInst& instruction) {
    std::array<bool,4> needed{};
    for (auto* user : instruction.users()) {
        auto* extracted = llvm::dyn_cast<llvm::ExtractValueInst>(user);
        if (!extracted || extracted->getNumIndices() != 1) { needed.fill(true); break; }
        needed[extracted->getIndices()[0]] = true;
    }
    llvm::Value* tuple = point;
    for (unsigned component = 0; component < 4; ++component) if (needed[component]) {
        auto* filtered = filter_texture_component(object, x, y, b.CreateExtractValue(point,component), component);
        tuple = b.CreateInsertValue(tuple, filtered, component);
    }
    return tuple;
}
