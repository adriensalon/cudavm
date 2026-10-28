; Independently authored NVVM IR 2.0, LLVM 7 typed pointers.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
define internal <2 x i1> @compare_0_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp eq <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_0_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp eq <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_1_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp sge <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_1_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp sge <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_2_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp uge <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_2_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp uge <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_3_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp sgt <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_3_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp sgt <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_4_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp ugt <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_4_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp ugt <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_5_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp sle <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_5_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp sle <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_6_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp ule <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_6_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp ule <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_7_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp slt <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_7_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp slt <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_8_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp ult <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_8_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp ult <4 x i8> %x, %y
 ret <4 x i1> %test
}
define internal <2 x i1> @compare_9_16(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <2 x i16>
 %y = bitcast i32 %b to <2 x i16>
 %test = icmp ne <2 x i16> %x, %y
 ret <2 x i1> %test
}
define internal <4 x i1> @compare_9_8(i32 %a, i32 %b) {
 %x = bitcast i32 %a to <4 x i8>
 %y = bitcast i32 %b to <4 x i8>
 %test = icmp ne <4 x i8> %x, %y
 ret <4 x i1> %test
}
define void @simd_comparison(i32* %out, i32* %in, i32 %count) {
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
 %output_offset = mul i64 %offset, 48
 %output = getelementptr i32, i32* %out, i64 %output_offset
 %t0_16 = call <2 x i1> @compare_0_16(i32 %a, i32 %b)
 %lanes0 = zext <2 x i1> %t0_16 to <2 x i16>
 %r0 = bitcast <2 x i16> %lanes0 to i32
 %lanes20 = sext <2 x i1> %t0_16 to <2 x i16>
 %r20 = bitcast <2 x i16> %lanes20 to i32
 %t0_8 = call <4 x i1> @compare_0_8(i32 %a, i32 %b)
 %lanes1 = zext <4 x i1> %t0_8 to <4 x i8>
 %r1 = bitcast <4 x i8> %lanes1 to i32
 %lanes21 = sext <4 x i1> %t0_8 to <4 x i8>
 %r21 = bitcast <4 x i8> %lanes21 to i32
 %t1_16 = call <2 x i1> @compare_1_16(i32 %a, i32 %b)
 %lanes2 = zext <2 x i1> %t1_16 to <2 x i16>
 %r2 = bitcast <2 x i16> %lanes2 to i32
 %lanes22 = sext <2 x i1> %t1_16 to <2 x i16>
 %r22 = bitcast <2 x i16> %lanes22 to i32
 %t1_8 = call <4 x i1> @compare_1_8(i32 %a, i32 %b)
 %lanes3 = zext <4 x i1> %t1_8 to <4 x i8>
 %r3 = bitcast <4 x i8> %lanes3 to i32
 %lanes23 = sext <4 x i1> %t1_8 to <4 x i8>
 %r23 = bitcast <4 x i8> %lanes23 to i32
 %t2_16 = call <2 x i1> @compare_2_16(i32 %a, i32 %b)
 %lanes4 = zext <2 x i1> %t2_16 to <2 x i16>
 %r4 = bitcast <2 x i16> %lanes4 to i32
 %lanes24 = sext <2 x i1> %t2_16 to <2 x i16>
 %r24 = bitcast <2 x i16> %lanes24 to i32
 %t2_8 = call <4 x i1> @compare_2_8(i32 %a, i32 %b)
 %lanes5 = zext <4 x i1> %t2_8 to <4 x i8>
 %r5 = bitcast <4 x i8> %lanes5 to i32
 %lanes25 = sext <4 x i1> %t2_8 to <4 x i8>
 %r25 = bitcast <4 x i8> %lanes25 to i32
 %t3_16 = call <2 x i1> @compare_3_16(i32 %a, i32 %b)
 %lanes6 = zext <2 x i1> %t3_16 to <2 x i16>
 %r6 = bitcast <2 x i16> %lanes6 to i32
 %lanes26 = sext <2 x i1> %t3_16 to <2 x i16>
 %r26 = bitcast <2 x i16> %lanes26 to i32
 %t3_8 = call <4 x i1> @compare_3_8(i32 %a, i32 %b)
 %lanes7 = zext <4 x i1> %t3_8 to <4 x i8>
 %r7 = bitcast <4 x i8> %lanes7 to i32
 %lanes27 = sext <4 x i1> %t3_8 to <4 x i8>
 %r27 = bitcast <4 x i8> %lanes27 to i32
 %t4_16 = call <2 x i1> @compare_4_16(i32 %a, i32 %b)
 %lanes8 = zext <2 x i1> %t4_16 to <2 x i16>
 %r8 = bitcast <2 x i16> %lanes8 to i32
 %lanes28 = sext <2 x i1> %t4_16 to <2 x i16>
 %r28 = bitcast <2 x i16> %lanes28 to i32
 %t4_8 = call <4 x i1> @compare_4_8(i32 %a, i32 %b)
 %lanes9 = zext <4 x i1> %t4_8 to <4 x i8>
 %r9 = bitcast <4 x i8> %lanes9 to i32
 %lanes29 = sext <4 x i1> %t4_8 to <4 x i8>
 %r29 = bitcast <4 x i8> %lanes29 to i32
 %t5_16 = call <2 x i1> @compare_5_16(i32 %a, i32 %b)
 %lanes10 = zext <2 x i1> %t5_16 to <2 x i16>
 %r10 = bitcast <2 x i16> %lanes10 to i32
 %lanes30 = sext <2 x i1> %t5_16 to <2 x i16>
 %r30 = bitcast <2 x i16> %lanes30 to i32
 %t5_8 = call <4 x i1> @compare_5_8(i32 %a, i32 %b)
 %lanes11 = zext <4 x i1> %t5_8 to <4 x i8>
 %r11 = bitcast <4 x i8> %lanes11 to i32
 %lanes31 = sext <4 x i1> %t5_8 to <4 x i8>
 %r31 = bitcast <4 x i8> %lanes31 to i32
 %t6_16 = call <2 x i1> @compare_6_16(i32 %a, i32 %b)
 %lanes12 = zext <2 x i1> %t6_16 to <2 x i16>
 %r12 = bitcast <2 x i16> %lanes12 to i32
 %lanes32 = sext <2 x i1> %t6_16 to <2 x i16>
 %r32 = bitcast <2 x i16> %lanes32 to i32
 %t6_8 = call <4 x i1> @compare_6_8(i32 %a, i32 %b)
 %lanes13 = zext <4 x i1> %t6_8 to <4 x i8>
 %r13 = bitcast <4 x i8> %lanes13 to i32
 %lanes33 = sext <4 x i1> %t6_8 to <4 x i8>
 %r33 = bitcast <4 x i8> %lanes33 to i32
 %t7_16 = call <2 x i1> @compare_7_16(i32 %a, i32 %b)
 %lanes14 = zext <2 x i1> %t7_16 to <2 x i16>
 %r14 = bitcast <2 x i16> %lanes14 to i32
 %lanes34 = sext <2 x i1> %t7_16 to <2 x i16>
 %r34 = bitcast <2 x i16> %lanes34 to i32
 %t7_8 = call <4 x i1> @compare_7_8(i32 %a, i32 %b)
 %lanes15 = zext <4 x i1> %t7_8 to <4 x i8>
 %r15 = bitcast <4 x i8> %lanes15 to i32
 %lanes35 = sext <4 x i1> %t7_8 to <4 x i8>
 %r35 = bitcast <4 x i8> %lanes35 to i32
 %t8_16 = call <2 x i1> @compare_8_16(i32 %a, i32 %b)
 %lanes16 = zext <2 x i1> %t8_16 to <2 x i16>
 %r16 = bitcast <2 x i16> %lanes16 to i32
 %lanes36 = sext <2 x i1> %t8_16 to <2 x i16>
 %r36 = bitcast <2 x i16> %lanes36 to i32
 %t8_8 = call <4 x i1> @compare_8_8(i32 %a, i32 %b)
 %lanes17 = zext <4 x i1> %t8_8 to <4 x i8>
 %r17 = bitcast <4 x i8> %lanes17 to i32
 %lanes37 = sext <4 x i1> %t8_8 to <4 x i8>
 %r37 = bitcast <4 x i8> %lanes37 to i32
 %t9_16 = call <2 x i1> @compare_9_16(i32 %a, i32 %b)
 %lanes18 = zext <2 x i1> %t9_16 to <2 x i16>
 %r18 = bitcast <2 x i16> %lanes18 to i32
 %lanes38 = sext <2 x i1> %t9_16 to <2 x i16>
 %r38 = bitcast <2 x i16> %lanes38 to i32
 %t9_8 = call <4 x i1> @compare_9_8(i32 %a, i32 %b)
 %lanes19 = zext <4 x i1> %t9_8 to <4 x i8>
 %r19 = bitcast <4 x i8> %lanes19 to i32
 %lanes39 = sext <4 x i1> %t9_8 to <4 x i8>
 %r39 = bitcast <4 x i8> %lanes39 to i32
 %bit40_0 = extractelement <2 x i1> %t0_16, i32 0
 %lane40_0 = zext i1 %bit40_0 to i32
 %sum40_0 = add i32 %c, %lane40_0
 %bit40_1 = extractelement <2 x i1> %t0_16, i32 1
 %lane40_1 = zext i1 %bit40_1 to i32
 %sum40_1 = add i32 %sum40_0, %lane40_1
 %r40 = add i32 %sum40_1, 0
 %bit41_0 = extractelement <4 x i1> %t0_8, i32 0
 %lane41_0 = zext i1 %bit41_0 to i32
 %sum41_0 = add i32 %c, %lane41_0
 %bit41_1 = extractelement <4 x i1> %t0_8, i32 1
 %lane41_1 = zext i1 %bit41_1 to i32
 %sum41_1 = add i32 %sum41_0, %lane41_1
 %bit41_2 = extractelement <4 x i1> %t0_8, i32 2
 %lane41_2 = zext i1 %bit41_2 to i32
 %sum41_2 = add i32 %sum41_1, %lane41_2
 %bit41_3 = extractelement <4 x i1> %t0_8, i32 3
 %lane41_3 = zext i1 %bit41_3 to i32
 %sum41_3 = add i32 %sum41_2, %lane41_3
 %r41 = add i32 %sum41_3, 0
 %bit42_0 = extractelement <2 x i1> %t9_16, i32 0
 %lane42_0 = zext i1 %bit42_0 to i32
 %sum42_0 = add i32 %c, %lane42_0
 %bit42_1 = extractelement <2 x i1> %t9_16, i32 1
 %lane42_1 = zext i1 %bit42_1 to i32
 %sum42_1 = add i32 %sum42_0, %lane42_1
 %r42 = add i32 %sum42_1, 0
 %bit43_0 = extractelement <4 x i1> %t9_8, i32 0
 %lane43_0 = zext i1 %bit43_0 to i32
 %sum43_0 = add i32 %c, %lane43_0
 %bit43_1 = extractelement <4 x i1> %t9_8, i32 1
 %lane43_1 = zext i1 %bit43_1 to i32
 %sum43_1 = add i32 %sum43_0, %lane43_1
 %bit43_2 = extractelement <4 x i1> %t9_8, i32 2
 %lane43_2 = zext i1 %bit43_2 to i32
 %sum43_2 = add i32 %sum43_1, %lane43_2
 %bit43_3 = extractelement <4 x i1> %t9_8, i32 3
 %lane43_3 = zext i1 %bit43_3 to i32
 %sum43_3 = add i32 %sum43_2, %lane43_3
 %r43 = add i32 %sum43_3, 0
 %bit44_0 = extractelement <2 x i1> %t7_16, i32 0
 %lane44_0 = zext i1 %bit44_0 to i32
 %sum44_0 = add i32 %c, %lane44_0
 %bit44_1 = extractelement <2 x i1> %t7_16, i32 1
 %lane44_1 = zext i1 %bit44_1 to i32
 %sum44_1 = add i32 %sum44_0, %lane44_1
 %r44 = add i32 %sum44_1, 0
 %bit45_0 = extractelement <4 x i1> %t7_8, i32 0
 %lane45_0 = zext i1 %bit45_0 to i32
 %sum45_0 = add i32 %c, %lane45_0
 %bit45_1 = extractelement <4 x i1> %t7_8, i32 1
 %lane45_1 = zext i1 %bit45_1 to i32
 %sum45_1 = add i32 %sum45_0, %lane45_1
 %bit45_2 = extractelement <4 x i1> %t7_8, i32 2
 %lane45_2 = zext i1 %bit45_2 to i32
 %sum45_2 = add i32 %sum45_1, %lane45_2
 %bit45_3 = extractelement <4 x i1> %t7_8, i32 3
 %lane45_3 = zext i1 %bit45_3 to i32
 %sum45_3 = add i32 %sum45_2, %lane45_3
 %r45 = add i32 %sum45_3, 0
 %bit46_0 = extractelement <2 x i1> %t2_16, i32 0
 %lane46_0 = zext i1 %bit46_0 to i32
 %sum46_0 = add i32 %c, %lane46_0
 %bit46_1 = extractelement <2 x i1> %t2_16, i32 1
 %lane46_1 = zext i1 %bit46_1 to i32
 %sum46_1 = add i32 %sum46_0, %lane46_1
 %r46 = add i32 %sum46_1, 0
 %bit47_0 = extractelement <4 x i1> %t2_8, i32 0
 %lane47_0 = zext i1 %bit47_0 to i32
 %sum47_0 = add i32 %c, %lane47_0
 %bit47_1 = extractelement <4 x i1> %t2_8, i32 1
 %lane47_1 = zext i1 %bit47_1 to i32
 %sum47_1 = add i32 %sum47_0, %lane47_1
 %bit47_2 = extractelement <4 x i1> %t2_8, i32 2
 %lane47_2 = zext i1 %bit47_2 to i32
 %sum47_2 = add i32 %sum47_1, %lane47_2
 %bit47_3 = extractelement <4 x i1> %t2_8, i32 3
 %lane47_3 = zext i1 %bit47_3 to i32
 %sum47_3 = add i32 %sum47_2, %lane47_3
 %r47 = add i32 %sum47_3, 0
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
 %p20 = getelementptr i32, i32* %output, i64 20
 store i32 %r20, i32* %p20, align 4
 %p21 = getelementptr i32, i32* %output, i64 21
 store i32 %r21, i32* %p21, align 4
 %p22 = getelementptr i32, i32* %output, i64 22
 store i32 %r22, i32* %p22, align 4
 %p23 = getelementptr i32, i32* %output, i64 23
 store i32 %r23, i32* %p23, align 4
 %p24 = getelementptr i32, i32* %output, i64 24
 store i32 %r24, i32* %p24, align 4
 %p25 = getelementptr i32, i32* %output, i64 25
 store i32 %r25, i32* %p25, align 4
 %p26 = getelementptr i32, i32* %output, i64 26
 store i32 %r26, i32* %p26, align 4
 %p27 = getelementptr i32, i32* %output, i64 27
 store i32 %r27, i32* %p27, align 4
 %p28 = getelementptr i32, i32* %output, i64 28
 store i32 %r28, i32* %p28, align 4
 %p29 = getelementptr i32, i32* %output, i64 29
 store i32 %r29, i32* %p29, align 4
 %p30 = getelementptr i32, i32* %output, i64 30
 store i32 %r30, i32* %p30, align 4
 %p31 = getelementptr i32, i32* %output, i64 31
 store i32 %r31, i32* %p31, align 4
 %p32 = getelementptr i32, i32* %output, i64 32
 store i32 %r32, i32* %p32, align 4
 %p33 = getelementptr i32, i32* %output, i64 33
 store i32 %r33, i32* %p33, align 4
 %p34 = getelementptr i32, i32* %output, i64 34
 store i32 %r34, i32* %p34, align 4
 %p35 = getelementptr i32, i32* %output, i64 35
 store i32 %r35, i32* %p35, align 4
 %p36 = getelementptr i32, i32* %output, i64 36
 store i32 %r36, i32* %p36, align 4
 %p37 = getelementptr i32, i32* %output, i64 37
 store i32 %r37, i32* %p37, align 4
 %p38 = getelementptr i32, i32* %output, i64 38
 store i32 %r38, i32* %p38, align 4
 %p39 = getelementptr i32, i32* %output, i64 39
 store i32 %r39, i32* %p39, align 4
 %p40 = getelementptr i32, i32* %output, i64 40
 store i32 %r40, i32* %p40, align 4
 %p41 = getelementptr i32, i32* %output, i64 41
 store i32 %r41, i32* %p41, align 4
 %p42 = getelementptr i32, i32* %output, i64 42
 store i32 %r42, i32* %p42, align 4
 %p43 = getelementptr i32, i32* %output, i64 43
 store i32 %r43, i32* %p43, align 4
 %p44 = getelementptr i32, i32* %output, i64 44
 store i32 %r44, i32* %p44, align 4
 %p45 = getelementptr i32, i32* %output, i64 45
 store i32 %r45, i32* %p45, align 4
 %p46 = getelementptr i32, i32* %output, i64 46
 store i32 %r46, i32* %p46, align 4
 %p47 = getelementptr i32, i32* %output, i64 47
 store i32 %r47, i32* %p47, align 4
 br label %done
done:
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32)* @simd_comparison, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
