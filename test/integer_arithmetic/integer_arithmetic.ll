; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.mulhi.i(i32, i32)
declare i32 @llvm.nvvm.mulhi.ui(i32, i32)
declare i64 @llvm.nvvm.mulhi.ll(i64, i64)
declare i64 @llvm.nvvm.mulhi.ull(i64, i64)

define void @integer_values(i64* %out, i64* %in, i32 %count) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %thread
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %offset = zext i32 %i to i64
  %source_offset = mul i64 %offset, 2
  %pa = getelementptr i64, i64* %in, i64 %source_offset
  %pb = getelementptr i64, i64* %pa, i64 1
  %a = load i64, i64* %pa, align 8
  %b = load i64, i64* %pb, align 8
  %x = trunc i64 %a to i32
  %y = trunc i64 %b to i32
  %zero = icmp eq i32 %y, 0
  %minimum = icmp eq i32 %x, -2147483648
  %negative_one = icmp eq i32 %y, -1
  %overflow = and i1 %minimum, %negative_one
  %invalid_division = or i1 %zero, %overflow
  %divisor = select i1 %invalid_division, i32 1, i32 %y
  %uq = udiv i32 %x, %divisor
  %ur = urem i32 %x, %divisor
  %sq = sdiv i32 %x, %divisor
  %sr = srem i32 %x, %divisor
  %sh = call i32 @llvm.nvvm.mulhi.i(i32 %x, i32 %y)
  %uh = call i32 @llvm.nvvm.mulhi.ui(i32 %x, i32 %y)
  %r0 = zext i32 %uq to i64
  %r1 = zext i32 %ur to i64
  %r2 = zext i32 %sq to i64
  %r3 = zext i32 %sr to i64
  %r4 = zext i32 %sh to i64
  %r5 = zext i32 %uh to i64
  %ux = zext i32 %x to i64
  %uy = zext i32 %y to i64
  %sx = sext i32 %x to i64
  %sy = sext i32 %y to i64
  %r6 = mul i64 %ux, %uy
  %r7 = mul i64 %sx, %sy
  %r8 = add i64 %a, %b
  %r9 = sub i64 %a, %b
  %r10 = call i64 @llvm.nvvm.mulhi.ull(i64 %a, i64 %b)
  %r11 = call i64 @llvm.nvvm.mulhi.ll(i64 %a, i64 %b)
  %target_offset = mul i64 %offset, 12
  %p0 = getelementptr i64, i64* %out, i64 %target_offset
  %p1 = getelementptr i64, i64* %p0, i64 1
  %p2 = getelementptr i64, i64* %p0, i64 2
  %p3 = getelementptr i64, i64* %p0, i64 3
  %p4 = getelementptr i64, i64* %p0, i64 4
  %p5 = getelementptr i64, i64* %p0, i64 5
  %p6 = getelementptr i64, i64* %p0, i64 6
  %p7 = getelementptr i64, i64* %p0, i64 7
  %p8 = getelementptr i64, i64* %p0, i64 8
  %p9 = getelementptr i64, i64* %p0, i64 9
  %p10 = getelementptr i64, i64* %p0, i64 10
  %p11 = getelementptr i64, i64* %p0, i64 11
  store i64 %r0, i64* %p0, align 8
  store i64 %r1, i64* %p1, align 8
  store i64 %r2, i64* %p2, align 8
  store i64 %r3, i64* %p3, align 8
  store i64 %r4, i64* %p4, align 8
  store i64 %r5, i64* %p5, align 8
  store i64 %r6, i64* %p6, align 8
  store i64 %r7, i64* %p7, align 8
  store i64 %r8, i64* %p8, align 8
  store i64 %r9, i64* %p9, align 8
  store i64 %r10, i64* %p10, align 8
  store i64 %r11, i64* %p11, align 8
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i64*, i64*, i32)* @integer_values, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
