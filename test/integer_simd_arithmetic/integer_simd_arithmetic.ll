; Independently authored NVVM IR 2.0, LLVM 7 typed pointers.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
define internal i32 @simd_0_16(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <2 x i16>
 %bv = bitcast i32 %b to <2 x i16>
 %x = sext <2 x i16> %av to <2 x i32>
 %y = sext <2 x i16> %bv to <2 x i32>
 %value = add <2 x i32> %x, zeroinitializer
 %negative = icmp slt <2 x i32> %value, zeroinitializer
 %negated = sub <2 x i32> zeroinitializer, %value
 %absolute = select <2 x i1> %negative, <2 x i32> %negated, <2 x i32> %value
 %small = trunc <2 x i32> %absolute to <2 x i16>
 %word = bitcast <2 x i16> %small to i32
 ret i32 %word
}
define internal i32 @simd_0_8(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <4 x i8>
 %bv = bitcast i32 %b to <4 x i8>
 %x = sext <4 x i8> %av to <4 x i32>
 %y = sext <4 x i8> %bv to <4 x i32>
 %value = add <4 x i32> %x, zeroinitializer
 %negative = icmp slt <4 x i32> %value, zeroinitializer
 %negated = sub <4 x i32> zeroinitializer, %value
 %absolute = select <4 x i1> %negative, <4 x i32> %negated, <4 x i32> %value
 %small = trunc <4 x i32> %absolute to <4 x i8>
 %word = bitcast <4 x i8> %small to i32
 ret i32 %word
}
define internal i32 @simd_1_16(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <2 x i16>
 %bv = bitcast i32 %b to <2 x i16>
 %x = zext <2 x i16> %av to <2 x i32>
 %y = zext <2 x i16> %bv to <2 x i32>
 %value = add <2 x i32> %x, %y
 %small = trunc <2 x i32> %value to <2 x i16>
 %word = bitcast <2 x i16> %small to i32
 ret i32 %word
}
define internal i32 @simd_1_8(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <4 x i8>
 %bv = bitcast i32 %b to <4 x i8>
 %x = zext <4 x i8> %av to <4 x i32>
 %y = zext <4 x i8> %bv to <4 x i32>
 %value = add <4 x i32> %x, %y
 %small = trunc <4 x i32> %value to <4 x i8>
 %word = bitcast <4 x i8> %small to i32
 ret i32 %word
}
define internal i32 @simd_2_16(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <2 x i16>
 %bv = bitcast i32 %b to <2 x i16>
 %x = zext <2 x i16> %av to <2 x i32>
 %y = zext <2 x i16> %bv to <2 x i32>
 %value = sub <2 x i32> %x, %y
 %small = trunc <2 x i32> %value to <2 x i16>
 %word = bitcast <2 x i16> %small to i32
 ret i32 %word
}
define internal i32 @simd_2_8(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <4 x i8>
 %bv = bitcast i32 %b to <4 x i8>
 %x = zext <4 x i8> %av to <4 x i32>
 %y = zext <4 x i8> %bv to <4 x i32>
 %value = sub <4 x i32> %x, %y
 %small = trunc <4 x i32> %value to <4 x i8>
 %word = bitcast <4 x i8> %small to i32
 ret i32 %word
}
define internal i32 @simd_3_16(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <2 x i16>
 %bv = bitcast i32 %b to <2 x i16>
 %x = zext <2 x i16> %av to <2 x i32>
 %y = zext <2 x i16> %bv to <2 x i32>
 %value = sub <2 x i32> zeroinitializer, %x
 %small = trunc <2 x i32> %value to <2 x i16>
 %word = bitcast <2 x i16> %small to i32
 ret i32 %word
}
define internal i32 @simd_3_8(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <4 x i8>
 %bv = bitcast i32 %b to <4 x i8>
 %x = zext <4 x i8> %av to <4 x i32>
 %y = zext <4 x i8> %bv to <4 x i32>
 %value = sub <4 x i32> zeroinitializer, %x
 %small = trunc <4 x i32> %value to <4 x i8>
 %word = bitcast <4 x i8> %small to i32
 ret i32 %word
}
define internal i32 @simd_4_16(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <2 x i16>
 %bv = bitcast i32 %b to <2 x i16>
 %x = sext <2 x i16> %av to <2 x i32>
 %y = sext <2 x i16> %bv to <2 x i32>
 %value = sub <2 x i32> %x, %y
 %negative = icmp slt <2 x i32> %value, zeroinitializer
 %negated = sub <2 x i32> zeroinitializer, %value
 %absolute = select <2 x i1> %negative, <2 x i32> %negated, <2 x i32> %value
 %small = trunc <2 x i32> %absolute to <2 x i16>
 %word = bitcast <2 x i16> %small to i32
 ret i32 %word
}
define internal i32 @simd_4_8(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <4 x i8>
 %bv = bitcast i32 %b to <4 x i8>
 %x = sext <4 x i8> %av to <4 x i32>
 %y = sext <4 x i8> %bv to <4 x i32>
 %value = sub <4 x i32> %x, %y
 %negative = icmp slt <4 x i32> %value, zeroinitializer
 %negated = sub <4 x i32> zeroinitializer, %value
 %absolute = select <4 x i1> %negative, <4 x i32> %negated, <4 x i32> %value
 %small = trunc <4 x i32> %absolute to <4 x i8>
 %word = bitcast <4 x i8> %small to i32
 ret i32 %word
}
define internal i32 @simd_5_16(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <2 x i16>
 %bv = bitcast i32 %b to <2 x i16>
 %x = zext <2 x i16> %av to <2 x i32>
 %y = zext <2 x i16> %bv to <2 x i32>
 %value = sub <2 x i32> %x, %y
 %negative = icmp slt <2 x i32> %value, zeroinitializer
 %negated = sub <2 x i32> zeroinitializer, %value
 %absolute = select <2 x i1> %negative, <2 x i32> %negated, <2 x i32> %value
 %small = trunc <2 x i32> %absolute to <2 x i16>
 %word = bitcast <2 x i16> %small to i32
 ret i32 %word
}
define internal i32 @simd_5_8(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <4 x i8>
 %bv = bitcast i32 %b to <4 x i8>
 %x = zext <4 x i8> %av to <4 x i32>
 %y = zext <4 x i8> %bv to <4 x i32>
 %value = sub <4 x i32> %x, %y
 %negative = icmp slt <4 x i32> %value, zeroinitializer
 %negated = sub <4 x i32> zeroinitializer, %value
 %absolute = select <4 x i1> %negative, <4 x i32> %negated, <4 x i32> %value
 %small = trunc <4 x i32> %absolute to <4 x i8>
 %word = bitcast <4 x i8> %small to i32
 ret i32 %word
}
define internal i32 @simd_6_16(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <2 x i16>
 %bv = bitcast i32 %b to <2 x i16>
 %x = sext <2 x i16> %av to <2 x i32>
 %y = sext <2 x i16> %bv to <2 x i32>
 %value = sub <2 x i32> %x, %y
 %negative = icmp slt <2 x i32> %value, zeroinitializer
 %negated = sub <2 x i32> zeroinitializer, %value
 %absolute = select <2 x i1> %negative, <2 x i32> %negated, <2 x i32> %value
 %lane0 = extractelement <2 x i32> %absolute, i32 0
 %lane1 = extractelement <2 x i32> %absolute, i32 1
 %sum1 = add i32 %lane0, %lane1
 ret i32 %sum1
}
define internal i32 @simd_6_8(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <4 x i8>
 %bv = bitcast i32 %b to <4 x i8>
 %x = sext <4 x i8> %av to <4 x i32>
 %y = sext <4 x i8> %bv to <4 x i32>
 %value = sub <4 x i32> %x, %y
 %negative = icmp slt <4 x i32> %value, zeroinitializer
 %negated = sub <4 x i32> zeroinitializer, %value
 %absolute = select <4 x i1> %negative, <4 x i32> %negated, <4 x i32> %value
 %lane0 = extractelement <4 x i32> %absolute, i32 0
 %lane1 = extractelement <4 x i32> %absolute, i32 1
 %sum1 = add i32 %lane0, %lane1
 %lane2 = extractelement <4 x i32> %absolute, i32 2
 %sum2 = add i32 %sum1, %lane2
 %lane3 = extractelement <4 x i32> %absolute, i32 3
 %sum3 = add i32 %sum2, %lane3
 ret i32 %sum3
}
define internal i32 @simd_7_16(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <2 x i16>
 %bv = bitcast i32 %b to <2 x i16>
 %x = zext <2 x i16> %av to <2 x i32>
 %y = zext <2 x i16> %bv to <2 x i32>
 %value = sub <2 x i32> %x, %y
 %negative = icmp slt <2 x i32> %value, zeroinitializer
 %negated = sub <2 x i32> zeroinitializer, %value
 %absolute = select <2 x i1> %negative, <2 x i32> %negated, <2 x i32> %value
 %lane0 = extractelement <2 x i32> %absolute, i32 0
 %lane1 = extractelement <2 x i32> %absolute, i32 1
 %sum1 = add i32 %lane0, %lane1
 ret i32 %sum1
}
define internal i32 @simd_7_8(i32 %a, i32 %b) {
 %av = bitcast i32 %a to <4 x i8>
 %bv = bitcast i32 %b to <4 x i8>
 %x = zext <4 x i8> %av to <4 x i32>
 %y = zext <4 x i8> %bv to <4 x i32>
 %value = sub <4 x i32> %x, %y
 %negative = icmp slt <4 x i32> %value, zeroinitializer
 %negated = sub <4 x i32> zeroinitializer, %value
 %absolute = select <4 x i1> %negative, <4 x i32> %negated, <4 x i32> %value
 %lane0 = extractelement <4 x i32> %absolute, i32 0
 %lane1 = extractelement <4 x i32> %absolute, i32 1
 %sum1 = add i32 %lane0, %lane1
 %lane2 = extractelement <4 x i32> %absolute, i32 2
 %sum2 = add i32 %sum1, %lane2
 %lane3 = extractelement <4 x i32> %absolute, i32 3
 %sum3 = add i32 %sum2, %lane3
 ret i32 %sum3
}
define void @simd_arithmetic(i32* %out, i32* %in, i32 %count) {
entry:
 %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
 %nx = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
 %base = mul i32 %by, %nx
 %block = add i32 %base, %bx
 %start = mul i32 %block, %ntid
 %i = add i32 %start, %tid
 %valid = icmp ult i32 %i, %count
 br i1 %valid, label %write, label %done
write:
 %offset = zext i32 %i to i64
 %input_offset = mul i64 %offset, 3
 %pa = getelementptr i32, i32* %in, i64 %input_offset
 %pb = getelementptr i32, i32* %pa, i64 1
 %pc = getelementptr i32, i32* %pa, i64 2
 %a = load i32, i32* %pa, align 4
 %b = load i32, i32* %pb, align 4
 %c = load i32, i32* %pc, align 4
 %output_offset = mul i64 %offset, 20
 %output = getelementptr i32, i32* %out, i64 %output_offset
 %r0 = call i32 @simd_0_16(i32 %a, i32 %b)
 %r1 = call i32 @simd_0_8(i32 %a, i32 %b)
 %r2 = call i32 @simd_1_16(i32 %a, i32 %b)
 %r3 = call i32 @simd_1_8(i32 %a, i32 %b)
 %r4 = call i32 @simd_2_16(i32 %a, i32 %b)
 %r5 = call i32 @simd_2_8(i32 %a, i32 %b)
 %r6 = call i32 @simd_3_16(i32 %a, i32 %b)
 %r7 = call i32 @simd_3_8(i32 %a, i32 %b)
 %r8 = call i32 @simd_4_16(i32 %a, i32 %b)
 %r9 = call i32 @simd_4_8(i32 %a, i32 %b)
 %r10 = call i32 @simd_5_16(i32 %a, i32 %b)
 %r11 = call i32 @simd_5_8(i32 %a, i32 %b)
 %r12 = call i32 @simd_6_16(i32 %a, i32 %b)
 %r13 = call i32 @simd_6_8(i32 %a, i32 %b)
 %r14 = call i32 @simd_7_16(i32 %a, i32 %b)
 %r15 = call i32 @simd_7_8(i32 %a, i32 %b)
 %r16 = add i32 %r12, %c
 %r17 = add i32 %r13, %c
 %r18 = add i32 %r14, %c
 %r19 = add i32 %r15, %c
 %p0 = getelementptr i32, i32* %output, i64 0
 store i32 %r0, i32* %p0, align 4
 %p1 = getelementptr i32, i32* %output, i64 1
 store i32 %r1, i32* %p1, align 4
 %p2 = getelementptr i32, i32* %output, i64 2
 store i32 %r2, i32* %p2, align 4
 %p3 = getelementptr i32, i32* %output, i64 3
 store i32 %r3, i32* %p3, align 4
 %p4 = getelementptr i32, i32* %output, i64 4
 store i32 %r4, i32* %p4, align 4
 %p5 = getelementptr i32, i32* %output, i64 5
 store i32 %r5, i32* %p5, align 4
 %p6 = getelementptr i32, i32* %output, i64 6
 store i32 %r6, i32* %p6, align 4
 %p7 = getelementptr i32, i32* %output, i64 7
 store i32 %r7, i32* %p7, align 4
 %p8 = getelementptr i32, i32* %output, i64 8
 store i32 %r8, i32* %p8, align 4
 %p9 = getelementptr i32, i32* %output, i64 9
 store i32 %r9, i32* %p9, align 4
 %p10 = getelementptr i32, i32* %output, i64 10
 store i32 %r10, i32* %p10, align 4
 %p11 = getelementptr i32, i32* %output, i64 11
 store i32 %r11, i32* %p11, align 4
 %p12 = getelementptr i32, i32* %output, i64 12
 store i32 %r12, i32* %p12, align 4
 %p13 = getelementptr i32, i32* %output, i64 13
 store i32 %r13, i32* %p13, align 4
 %p14 = getelementptr i32, i32* %output, i64 14
 store i32 %r14, i32* %p14, align 4
 %p15 = getelementptr i32, i32* %output, i64 15
 store i32 %r15, i32* %p15, align 4
 %p16 = getelementptr i32, i32* %output, i64 16
 store i32 %r16, i32* %p16, align 4
 %p17 = getelementptr i32, i32* %output, i64 17
 store i32 %r17, i32* %p17, align 4
 %p18 = getelementptr i32, i32* %output, i64 18
 store i32 %r18, i32* %p18, align 4
 %p19 = getelementptr i32, i32* %output, i64 19
 store i32 %r19, i32* %p19, align 4
 br label %done
done:
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32)* @simd_arithmetic, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
