; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
%record = type { i32, i16, i8, i8, i64, [3 x i32], i32 }
%vector = type { i32, i32, i32, i32 }
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define void @parameter_layout(i32* %out, i8 %marker, %record* byval align 16 %record,
    i16 %bias, %vector* byval align 16 %vector, i64 %salt, i32 %count) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %i = add i32 %base, %lane
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %tagptr = getelementptr %record, %record* %record, i32 0, i32 0
  %smallptr = getelementptr %record, %record* %record, i32 0, i32 1
  %lowptr = getelementptr %record, %record* %record, i32 0, i32 2
  %highptr = getelementptr %record, %record* %record, i32 0, i32 3
  %wideptr = getelementptr %record, %record* %record, i32 0, i32 4
  %v0ptr = getelementptr %record, %record* %record, i32 0, i32 5, i32 0
  %v1ptr = getelementptr %record, %record* %record, i32 0, i32 5, i32 1
  %v2ptr = getelementptr %record, %record* %record, i32 0, i32 5, i32 2
  %tailptr = getelementptr %record, %record* %record, i32 0, i32 6
  %xptr = getelementptr %vector, %vector* %vector, i32 0, i32 0
  %yptr = getelementptr %vector, %vector* %vector, i32 0, i32 1
  %zptr = getelementptr %vector, %vector* %vector, i32 0, i32 2
  %wptr = getelementptr %vector, %vector* %vector, i32 0, i32 3
  %tag = load i32, i32* %tagptr, align 4
  %small16 = load i16, i16* %smallptr, align 2
  %low8 = load i8, i8* %lowptr, align 1
  %high8 = load i8, i8* %highptr, align 1
  %wide = load i64, i64* %wideptr, align 8
  %v0 = load i32, i32* %v0ptr, align 4
  %v1 = load i32, i32* %v1ptr, align 4
  %v2 = load i32, i32* %v2ptr, align 4
  %tail = load i32, i32* %tailptr, align 4
  %x = load i32, i32* %xptr, align 4
  %y = load i32, i32* %yptr, align 4
  %z = load i32, i32* %zptr, align 4
  %w = load i32, i32* %wptr, align 4
  %marker32 = zext i8 %marker to i32
  %small = zext i16 %small16 to i32
  %bias32 = zext i16 %bias to i32
  %biased = add i32 %small, %bias32
  %low = zext i8 %low8 to i32
  %high = zext i8 %high8 to i32
  %tagged = xor i32 %tag, %i
  %wide_low = trunc i64 %wide to i32
  %wide_shift = lshr i64 %wide, 32
  %wide_high = trunc i64 %wide_shift to i32
  %a = add i32 %v0, %i
  %b = xor i32 %v1, %i
  %c = sub i32 %v2, %i
  %i64 = zext i32 %i to i64
  %partial = add i64 %wide, %salt
  %sum = add i64 %partial, %i64
  %sum_low = trunc i64 %sum to i32
  %sum_shift = lshr i64 %sum, 32
  %sum_high = trunc i64 %sum_shift to i32
  %offset = mul i64 %i64, 17
  %p0 = getelementptr i32, i32* %out, i64 %offset
  %p1 = getelementptr i32, i32* %p0, i64 1
  %p2 = getelementptr i32, i32* %p0, i64 2
  %p3 = getelementptr i32, i32* %p0, i64 3
  %p4 = getelementptr i32, i32* %p0, i64 4
  %p5 = getelementptr i32, i32* %p0, i64 5
  %p6 = getelementptr i32, i32* %p0, i64 6
  %p7 = getelementptr i32, i32* %p0, i64 7
  %p8 = getelementptr i32, i32* %p0, i64 8
  %p9 = getelementptr i32, i32* %p0, i64 9
  %p10 = getelementptr i32, i32* %p0, i64 10
  %p11 = getelementptr i32, i32* %p0, i64 11
  %p12 = getelementptr i32, i32* %p0, i64 12
  %p13 = getelementptr i32, i32* %p0, i64 13
  %p14 = getelementptr i32, i32* %p0, i64 14
  %p15 = getelementptr i32, i32* %p0, i64 15
  %p16 = getelementptr i32, i32* %p0, i64 16
  store i32 %marker32, i32* %p0, align 4
  store i32 %biased, i32* %p1, align 4
  store i32 %low, i32* %p2, align 4
  store i32 %high, i32* %p3, align 4
  store i32 %tagged, i32* %p4, align 4
  store i32 %wide_low, i32* %p5, align 4
  store i32 %wide_high, i32* %p6, align 4
  store i32 %a, i32* %p7, align 4
  store i32 %b, i32* %p8, align 4
  store i32 %c, i32* %p9, align 4
  store i32 %tail, i32* %p10, align 4
  store i32 %x, i32* %p11, align 4
  store i32 %y, i32* %p12, align 4
  store i32 %z, i32* %p13, align 4
  store i32 %w, i32* %p14, align 4
  store i32 %sum_low, i32* %p15, align 4
  store i32 %sum_high, i32* %p16, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i8, %record*, i16, %vector*, i64, i32)* @parameter_layout, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
