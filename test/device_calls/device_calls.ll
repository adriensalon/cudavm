; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
define internal i64 @mix_words(i64 %a, i64 %b, i32 %shift) noinline {
 %n = and i32 %shift, 63
 %negative = sub i32 0, %n
 %other = and i32 %negative, 63
 %n64 = zext i32 %n to i64
 %other64 = zext i32 %other to i64
 %left = shl i64 %a, %n64
 %right = lshr i64 %a, %other64
 %rotate = or i64 %left, %right
 %result = add i64 %rotate, %b
 ret i64 %result
}
define internal i32 @fold_words(i32 %a, i32 %b) noinline {
 %scaled = mul i32 %b, 33
 %value = xor i32 %a, %scaled
 %high = icmp slt i32 %value, 0
 br i1 %high, label %negative, label %positive
negative:
 %sub = sub i32 %value, 324508639
 ret i32 %sub
positive:
 %add = add i32 %value, -1985229329
 ret i32 %add
}
define internal i64 @nested_words(i64 %a, i64 %b, i32 %rounds) noinline {
entry:
 br label %loop
loop:
 %round = phi i32 [0, %entry], [%next_round, %body]
 %state = phi i64 [%a, %entry], [%next_state, %body]
 %more = icmp ult i32 %round, %rounds
 br i1 %more, label %body, label %done
body:
 %wide_round = zext i32 %round to i64
 %next_b = add i64 %b, %wide_round
 %shift = mul i32 %round, 7
 %mixed = call i64 @mix_words(i64 %state, i64 %next_b, i32 %shift)
 %low = trunc i64 %mixed to i32
 %high64 = lshr i64 %mixed, 32
 %high = trunc i64 %high64 to i32
 %folded = call i32 @fold_words(i32 %low, i32 %high)
 %wide = zext i32 %folded to i64
 %upper = shl i64 %wide, 32
 %replicated = or i64 %wide, %upper
 %next_state = xor i64 %mixed, %replicated
 %next_round = add i32 %round, 1
 br label %loop
done:
 ret i64 %state
}
define void @call_results(i32* %out, i64* %in, i32 %count, i32 %rounds) {
entry:
 %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %base = mul i32 %block, %threads
 %i = add i32 %base, %thread
 %valid = icmp ult i32 %i, %count
 br i1 %valid, label %body, label %done
body:
 %offset = zext i32 %i to i64
 %input_offset = mul i64 %offset, 2
 %ap = getelementptr i64, i64* %in, i64 %input_offset
 %bp = getelementptr i64, i64* %ap, i64 1
 %a = load i64, i64* %ap, align 8
 %b = load i64, i64* %bp, align 8
 %mixed = call i64 @mix_words(i64 %a, i64 %b, i32 %i)
 %mixed_low = trunc i64 %mixed to i32
 %mixed_high64 = lshr i64 %mixed, 32
 %mixed_high = trunc i64 %mixed_high64 to i32
 %folded = call i32 @fold_words(i32 %mixed_low, i32 %mixed_high)
 %extra = and i32 %i, 3
 %steps = add i32 %rounds, %extra
 %nested = call i64 @nested_words(i64 %a, i64 %b, i32 %steps)
 %parity = and i32 %i, 1
 %odd = icmp ne i32 %parity, 0
 br i1 %odd, label %odd_arm, label %even_arm
odd_arm:
 %a_low = trunc i64 %a to i32
 %b_low = trunc i64 %b to i32
 %odd_result = call i32 @fold_words(i32 %a_low, i32 %b_low)
 br label %write
even_arm:
 %a_high64 = lshr i64 %a, 32
 %b_high64 = lshr i64 %b, 32
 %a_high = trunc i64 %a_high64 to i32
 %b_high = trunc i64 %b_high64 to i32
 %even_result = call i32 @fold_words(i32 %b_high, i32 %a_high)
 br label %write
write:
 %branch = phi i32 [%odd_result, %odd_arm], [%even_result, %even_arm]
 %nested_low = trunc i64 %nested to i32
 %nested_high64 = lshr i64 %nested, 32
 %nested_high = trunc i64 %nested_high64 to i32
 %output_base = mul i64 %offset, 6
 %off0 = add i64 %output_base, 0
 %p0 = getelementptr i32, i32* %out, i64 %off0
 store i32 %mixed_low, i32* %p0, align 4
 %off1 = add i64 %output_base, 1
 %p1 = getelementptr i32, i32* %out, i64 %off1
 store i32 %mixed_high, i32* %p1, align 4
 %off2 = add i64 %output_base, 2
 %p2 = getelementptr i32, i32* %out, i64 %off2
 store i32 %folded, i32* %p2, align 4
 %off3 = add i64 %output_base, 3
 %p3 = getelementptr i32, i32* %out, i64 %off3
 store i32 %nested_low, i32* %p3, align 4
 %off4 = add i64 %output_base, 4
 %p4 = getelementptr i32, i32* %out, i64 %off4
 store i32 %nested_high, i32* %p4, align 4
 %off5 = add i64 %output_base, 5
 %p5 = getelementptr i32, i32* %out, i64 %off5
 store i32 %branch, i32* %p5, align 4
 br label %done
done:
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i64*, i32, i32)* @call_results, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
