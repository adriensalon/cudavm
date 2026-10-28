; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.fabs.f32(float)
declare float @llvm.copysign.f32(float, float)

define void @compare_float(i32* %out, float* %in, i32 %count) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %i = add i32 %base, %lane
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %index = zext i32 %i to i64
  %input_offset = mul i64 %index, 2
  %source = getelementptr float, float* %in, i64 %input_offset
  %source1 = getelementptr float, float* %source, i64 1
  %a = load float, float* %source, align 4
  %b = load float, float* %source1, align 4
  %pred0 = fcmp oeq float %a, %b
  %v0 = zext i1 %pred0 to i32
  %pred1 = fcmp une float %a, %b
  %v1 = zext i1 %pred1 to i32
  %pred2 = fcmp olt float %a, %b
  %v2 = zext i1 %pred2 to i32
  %pred3 = fcmp ole float %a, %b
  %v3 = zext i1 %pred3 to i32
  %pred4 = fcmp ogt float %a, %b
  %v4 = zext i1 %pred4 to i32
  %pred5 = fcmp oge float %a, %b
  %v5 = zext i1 %pred5 to i32
  %pred6 = fcmp ult float %a, %b
  %v6 = zext i1 %pred6 to i32
  %pred7 = fcmp ule float %a, %b
  %v7 = zext i1 %pred7 to i32
  %pred8 = fcmp ugt float %a, %b
  %v8 = zext i1 %pred8 to i32
  %pred9 = fcmp uge float %a, %b
  %v9 = zext i1 %pred9 to i32
  %a_bits = bitcast float %a to i32
  %b_bits = bitcast float %b to i32
  %ax = and i32 %a_bits, 2147483647
  %bx = and i32 %b_bits, 2147483647
  %nan = icmp ugt i32 %ax, 2139095040
  %infinite = icmp eq i32 %ax, 2139095040
  %finite = icmp ult i32 %ax, 2139095040
  %v10 = zext i1 %nan to i32
  %v11 = zext i1 %infinite to i32
  %v12 = zext i1 %finite to i32
  %v13 = lshr i32 %a_bits, 31
  %minimum = call float @llvm.minnum.f32(float %a, float %b)
  %maximum = call float @llvm.maxnum.f32(float %a, float %b)
  %absolute = call float @llvm.fabs.f32(float %a)
  %copy_sign = call float @llvm.copysign.f32(float %a, float %b)
  %v14 = bitcast float %minimum to i32
  %v15 = bitcast float %maximum to i32
  %v16 = bitcast float %absolute to i32
  %v17 = bitcast float %copy_sign to i32
  %magnitudes = or i32 %ax, %bx
  %zero_pair = icmp eq i32 %magnitudes, 0
  %min_zero = or i32 %a_bits, %b_bits
  %max_zero = and i32 %a_bits, %b_bits
  %v18 = select i1 %zero_pair, i32 %min_zero, i32 %v14
  %v19 = select i1 %zero_pair, i32 %max_zero, i32 %v15
  %above = icmp uge i32 %ax, 8388608
  %nonzero_normal = and i1 %above, %finite
  %zero = icmp eq i32 %ax, 0
  %normal = or i1 %nonzero_normal, %zero
  %below = icmp ult i32 %ax, 8388608
  %nonzero = icmp ne i32 %ax, 0
  %subnormal = and i1 %below, %nonzero
  %v20 = zext i1 %normal to i32
  %v21 = zext i1 %subnormal to i32
  %output_offset = mul i64 %index, 24
  %p0 = getelementptr i32, i32* %out, i64 %output_offset
  store i32 %v0, i32* %p0, align 4
  %p1 = getelementptr i32, i32* %p0, i64 1
  store i32 %v1, i32* %p1, align 4
  %p2 = getelementptr i32, i32* %p0, i64 2
  store i32 %v2, i32* %p2, align 4
  %p3 = getelementptr i32, i32* %p0, i64 3
  store i32 %v3, i32* %p3, align 4
  %p4 = getelementptr i32, i32* %p0, i64 4
  store i32 %v4, i32* %p4, align 4
  %p5 = getelementptr i32, i32* %p0, i64 5
  store i32 %v5, i32* %p5, align 4
  %p6 = getelementptr i32, i32* %p0, i64 6
  store i32 %v6, i32* %p6, align 4
  %p7 = getelementptr i32, i32* %p0, i64 7
  store i32 %v7, i32* %p7, align 4
  %p8 = getelementptr i32, i32* %p0, i64 8
  store i32 %v8, i32* %p8, align 4
  %p9 = getelementptr i32, i32* %p0, i64 9
  store i32 %v9, i32* %p9, align 4
  %p10 = getelementptr i32, i32* %p0, i64 10
  store i32 %v10, i32* %p10, align 4
  %p11 = getelementptr i32, i32* %p0, i64 11
  store i32 %v11, i32* %p11, align 4
  %p12 = getelementptr i32, i32* %p0, i64 12
  store i32 %v12, i32* %p12, align 4
  %p13 = getelementptr i32, i32* %p0, i64 13
  store i32 %v13, i32* %p13, align 4
  %p14 = getelementptr i32, i32* %p0, i64 14
  store i32 %v14, i32* %p14, align 4
  %p15 = getelementptr i32, i32* %p0, i64 15
  store i32 %v15, i32* %p15, align 4
  %p16 = getelementptr i32, i32* %p0, i64 16
  store i32 %v16, i32* %p16, align 4
  %p17 = getelementptr i32, i32* %p0, i64 17
  store i32 %v17, i32* %p17, align 4
  %p18 = getelementptr i32, i32* %p0, i64 18
  store i32 %v18, i32* %p18, align 4
  %p19 = getelementptr i32, i32* %p0, i64 19
  store i32 %v19, i32* %p19, align 4
  %p20 = getelementptr i32, i32* %p0, i64 20
  store i32 %v20, i32* %p20, align 4
  %p21 = getelementptr i32, i32* %p0, i64 21
  store i32 %v21, i32* %p21, align 4
  %p22 = getelementptr i32, i32* %p0, i64 22
  store i32 %v14, i32* %p22, align 4
  %p23 = getelementptr i32, i32* %p0, i64 23
  store i32 %v15, i32* %p23, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, float*, i32)* @compare_float, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
