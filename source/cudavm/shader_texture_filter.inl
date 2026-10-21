// Image descriptors and their complete array backing each occupy one arena
// record. Resolve those records once, then access their physical byte offsets.
llvm::Value* image_word(llvm::Value* physical, unsigned offset = 0) {
    auto* index = b.CreateLShr(b.CreateAdd(b.CreateTrunc(physical, b.getInt32Ty()), b.getInt32(offset)), b.getInt32(2));
    return b.CreateAlignedLoad(b.getInt32Ty(), word_pointer(0, index), llvm::Align(4), true);
}
llvm::Value* normalize_texture_integer(llvm::Value* integer, llvm::Value* bits, llvm::Value* signed_format) {
    // n/(2^k-1) has the repeating binary fraction 0.(n's k bits). Retain
    // leading zeros, a full binary32 significand and its rounding bit. Nonzero
    // nonendpoint quotients cannot be exactly halfway between binary32 values.
    auto* width = b.CreateSub(bits, b.CreateZExt(signed_format, b.getInt32Ty()));
    auto* negative = b.CreateAnd(signed_format, b.CreateICmpSLT(integer, b.getInt32(0)));
    auto* magnitude = b.CreateSelect(negative, b.CreateNeg(integer), integer);
    auto* maximum = b.CreateSub(b.CreateShl(b.getInt32(1), width), b.getInt32(1));
    magnitude = b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, magnitude, maximum);
    auto* leading = dxil ? b.CreateSelect(b.CreateICmpEQ(magnitude, b.getInt32(0)), b.getInt32(32),
        b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::dx_firstbituhigh, {magnitude}))
        : b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::ctlz, {magnitude, b.getFalse()});
    auto* zeros = b.CreateSub(b.CreateAdd(width, leading), b.getInt32(32));
    // Align the leading one before repeating, so 32 bits retain the complete
    // significand and rounding bit even for the smallest 16-bit input.
    llvm::Value* pattern = b.CreateShl(b.CreateShl(magnitude, zeros), b.CreateSub(b.getInt32(32), width));
    for (unsigned step = 0; step < 3; ++step) {
        auto* distance = b.CreateShl(width, b.getInt32(step));
        auto* repeated = b.CreateOr(pattern, b.CreateLShr(pattern, b.CreateAnd(distance, b.getInt32(31))));
        pattern = b.CreateSelect(b.CreateICmpULT(distance, b.getInt32(32)), repeated, pattern);
    }
    auto* significand = b.CreateLShr(pattern, b.getInt32(8));
    auto* round = b.CreateAnd(b.CreateLShr(pattern, b.getInt32(7)), b.getInt32(1));
    auto* encoded = b.CreateAdd(b.CreateShl(b.CreateSub(b.getInt32(126), zeros), b.getInt32(23)), b.CreateSub(b.CreateAdd(significand, round), b.getInt32(0x800000)));
    encoded = b.CreateSelect(b.CreateICmpEQ(magnitude, b.getInt32(0)), b.getInt32(0), encoded);
    return b.CreateOr(encoded, b.CreateSelect(negative, b.getInt32(0x80000000u), b.getInt32(0)));
}
// Packed integer components are naturally aligned within their transport word.
// Return register bits shared by point sampling and each linear-filter tap.
llvm::Value* texture_component(llvm::Value* object, llvm::Value* texel, unsigned component, bool floating) {
    auto saved = b.saveIP();
    auto name = floating ? "cudavm.texture.component.float" : "cudavm.texture.component.integer";
    auto* helper = result->getFunction(name);
    if (!helper) {
        std::vector<llvm::Type*> signature(5, b.getInt32Ty());
        if (opencl && local_bytes) signature.push_back(b.getPtrTy());
        if (opencl) { signature.push_back(resource_pointer_type()); signature.push_back(resource_pointer_type()); }
        helper = llvm::Function::Create(llvm::FunctionType::get(b.getInt32Ty(), signature, false), llvm::GlobalValue::InternalLinkage, name, *result);
        if (opencl && local_bytes) helper->addFnAttr("cudavm.native.local");
        helper->addFnAttr(llvm::Attribute::AlwaysInline);
        b.SetInsertPoint(llvm::BasicBlock::Create(b.getContext(), "entry", helper));
        auto* descriptor = pair(helper->getArg(0), helper->getArg(1));
        auto* component_index = helper->getArg(4);
        auto* channels = image_word(descriptor, 48);
        auto* bytes = image_word(descriptor, 56);
        auto* kind = image_word(descriptor, 60);
        auto* flags = image_word(descriptor, 20);
        auto* present = b.CreateICmpULT(component_index, channels);
        auto* texel_address = pair(helper->getArg(2), helper->getArg(3));
        auto* valid = b.CreateICmpNE(texel_address, b.getInt64(0xffffffff00000000ull));
        auto* address = b.CreateSelect(present, b.CreateAdd(texel_address, b.CreateZExt(b.CreateMul(component_index, bytes), b.getInt64Ty())), b.getInt64(0xffffffff00000000ull));
        auto* bits = b.CreateMul(bytes, b.getInt32(8));
        auto* shift = b.CreateSub(b.getInt32(32), bits);
        auto* byte_offset = b.CreateAnd(b.CreateTrunc(address, b.getInt32Ty()), b.getInt32(3));
        auto* word = b.CreateLShr(image_word(address), b.CreateMul(byte_offset, b.getInt32(8)));
        auto* border = image_word(b.CreateAdd(descriptor, b.CreateZExt(b.CreateMul(component_index, b.getInt32(4)), b.getInt64Ty())), 32);
        auto* value = b.CreateAnd(b.CreateSelect(valid, word, border), b.CreateLShr(b.getInt32(UINT32_MAX), shift));
        auto* signed_format = b.CreateICmpEQ(kind, b.getInt32(2));
        auto* integer = b.CreateSelect(signed_format, b.CreateAShr(b.CreateShl(value, shift), shift), value);
        auto* float_alpha = b.CreateAnd(b.CreateICmpEQ(kind, b.getInt32(0)), b.CreateICmpEQ(component_index, b.getInt32(3)));
        float_alpha = b.CreateAnd(float_alpha, b.CreateICmpNE(image_word(descriptor, 16), b.getInt32(3)));
        auto* missing = b.CreateSelect(valid,
            b.CreateSelect(float_alpha, b.getInt32(floating ? 0x3f800000 : 1), b.getInt32(0)), border);
        if (!floating) b.CreateRet(b.CreateSelect(present, integer, missing));
        else {
            auto* normalized = llvm::BasicBlock::Create(b.getContext(), "normalized", helper);
            auto* element = llvm::BasicBlock::Create(b.getContext(), "element", helper);
            b.CreateCondBr(b.CreateICmpNE(b.CreateAnd(flags, b.getInt32(4)), b.getInt32(0)), normalized, element);
            b.SetInsertPoint(element); b.CreateRet(b.CreateSelect(present, integer, missing));
            b.SetInsertPoint(normalized);
            auto* encoded = normalize_texture_integer(integer, bits, signed_format);
            b.CreateRet(b.CreateSelect(present, encoded, missing));
        }
    }
    b.restoreIP(saved);
    return call_helper(helper, {object, b.CreateTrunc(texel, b.getInt32Ty()), b.CreateTrunc(b.CreateLShr(texel, b.getInt64(32)), b.getInt32Ty()), b.getInt32(component)});
}
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
        auto* flags = image_word(descriptor, 20);
        b.CreateCondBr(b.CreateICmpNE(b.CreateAnd(flags, b.getInt32(2)), b.getInt32(0)), linear, nearest);
        b.SetInsertPoint(nearest); b.CreateRet(helper->getArg(4)); b.SetInsertPoint(linear);
        auto* normalized = b.CreateICmpNE(b.CreateAnd(flags, b.getInt32(1)), b.getInt32(0));
        auto* base = b.CreateZExt(call_helper(resolve, {pair(image_word(descriptor), image_word(descriptor, 4))}), b.getInt64Ty());
        auto* width = image_word(descriptor, 8);
        auto* height = image_word(descriptor, 12);
        auto* texel_bytes = image_word(descriptor, 52);
        auto* pitch = image_word(descriptor, 64);
        auto* mode_x = image_word(descriptor, 24);
        auto* mode_y = image_word(descriptor, 28);
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
            auto* fraction_bits = b.CreateTrunc(b.CreateAnd(fixed, b.getInt64(255)), b.getInt32Ty());
            auto* weight = b.CreateFMul(b.CreateUIToFP(fraction_bits, b.getFloatTy()), llvm::ConstantFP::get(b.getFloatTy(), 1.0 / 256.0));
            return std::array<llvm::Value*, 3>{b.CreateAShr(fixed, b.getInt64(8)), weight, fraction_bits};
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
        auto samples = [&](bool floating) {
            std::array<llvm::Value*,4> values{};
            for (unsigned row = 0; row < 2; ++row) for (unsigned column = 0; column < 2; ++column) {
                auto xx = address(b.CreateAdd(horizontal[0], b.getInt64(column)), width, mode_x);
                auto yy = address(b.CreateAdd(vertical[0], b.getInt64(row)), height, mode_y);
                auto* valid = b.CreateAnd(xx[1], yy[1]);
                auto* offset = b.CreateAdd(b.CreateMul(yy[0], b.CreateZExt(pitch, b.getInt64Ty())), b.CreateMul(xx[0], b.CreateZExt(texel_bytes, b.getInt64Ty())));
                values[2*row+column] = texture_component(descriptor, b.CreateSelect(valid, b.CreateAdd(base, offset), b.getInt64(0xffffffff00000000ull)), component, floating);
            }
            return values;
        };
        auto* integer_linear = llvm::BasicBlock::Create(b.getContext(), "integer.linear", helper);
        auto* float_linear = llvm::BasicBlock::Create(b.getContext(), "float.linear", helper);
        b.CreateCondBr(b.CreateICmpNE(b.CreateAnd(flags, b.getInt32(4)), b.getInt32(0)), integer_linear, float_linear);
        b.SetInsertPoint(integer_linear);
        auto integers = samples(false);
        auto* signed_format = b.CreateICmpEQ(image_word(descriptor, 60), b.getInt32(2));
        auto* byte_format = b.CreateICmpEQ(image_word(descriptor, 56), b.getInt32(1));
        auto integer_blend = [&](llvm::Value* a, llvm::Value* c, llvm::Value* weight) {
            return b.CreateAdd(b.CreateMul(a, b.CreateSub(b.getInt32(256), weight)), b.CreateMul(c, weight));
        };
        auto* numerator = integer_blend(integer_blend(integers[0], integers[1], horizontal[2]),
            integer_blend(integers[2], integers[3], horizontal[2]), vertical[2]);
        // A single signed/unsigned 16-bit result is rounded after the full
        // bilinear sum, with ties toward positive infinity, then normalized.
        // For signed bytes, 32767/127 = 258 + 1/127. Retain the negative
        // endpoint through interpolation and floor the residual exactly.
        auto* quotient = b.CreateSDiv(numerator, b.getInt32(127));
        auto* remainder = b.CreateSub(numerator, b.CreateMul(quotient, b.getInt32(127)));
        auto* floor_quotient = b.CreateSub(quotient, b.CreateZExt(b.CreateAnd(b.CreateICmpSLT(numerator, b.getInt32(0)), b.CreateICmpNE(remainder, b.getInt32(0))), b.getInt32Ty()));
        auto* signed_bytes = b.CreateAdd(b.CreateMul(b.CreateSExt(numerator, b.getInt64Ty()), b.getInt64(258)), b.CreateSExt(floor_quotient, b.getInt64Ty()));
        auto* signed_byte_result = b.CreateTrunc(b.CreateAShr(b.CreateAdd(signed_bytes, b.getInt64(32768)), b.getInt64(16)), b.getInt32Ty());
        auto* unsigned_byte_result = b.CreateLShr(b.CreateAdd(b.CreateMul(numerator, b.getInt32(257)), b.getInt32(32768)), b.getInt32(16));
        // Every 16-bit bilinear sum and its rounding bias fit in the original
        // signed/unsigned 32-bit word. Only signed-byte rescaling needs 64 bits.
        auto* biased = b.CreateAdd(numerator, b.getInt32(32768));
        auto* halfword_result = b.CreateSelect(signed_format, b.CreateAShr(biased, b.getInt32(16)), b.CreateLShr(biased, b.getInt32(16)));
        auto* rounded = b.CreateSelect(byte_format, b.CreateSelect(signed_format, signed_byte_result, unsigned_byte_result), halfword_result);
        b.CreateRet(normalize_texture_integer(rounded, b.getInt32(16), signed_format));
        b.SetInsertPoint(float_linear);
        auto floating_samples = samples(true);
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
        std::array<llvm::Value*, 4> values;
        for (unsigned tap = 0; tap < 4; ++tap) values[tap] = b.CreateBitCast(floating_samples[tap], b.getFloatTy());
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
