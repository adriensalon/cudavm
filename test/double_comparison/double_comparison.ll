; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare double @llvm.minnum.f64(double, double)
declare double @llvm.maxnum.f64(double, double)
declare double @llvm.fabs.f64(double)
declare double @llvm.copysign.f64(double, double)

define void @compare_double(i32* %out, i32* %in, i32 %count) {
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
  %input_offset = mul i64 %index, 6
  %source = getelementptr i32, i32* %in, i64 %input_offset
  %source_b = getelementptr i32, i32* %source, i64 2
  %source_c = getelementptr i32, i32* %source, i64 4
  %a_pointer = bitcast i32* %source to double*
  %b_pointer = bitcast i32* %source_b to double*
  %c_pointer = bitcast i32* %source_c to float*
  %a = load double, double* %a_pointer, align 8
  %b = load double, double* %b_pointer, align 8
  %small_c = load float, float* %c_pointer, align 4
  %c = fpext float %small_c to double
  %pred0 = fcmp oeq double %a, %b
  %v0 = zext i1 %pred0 to i32
  %pred1 = fcmp une double %a, %b
  %v1 = zext i1 %pred1 to i32
  %pred2 = fcmp olt double %a, %b
  %v2 = zext i1 %pred2 to i32
  %pred3 = fcmp ole double %a, %b
  %v3 = zext i1 %pred3 to i32
  %pred4 = fcmp ogt double %a, %b
  %v4 = zext i1 %pred4 to i32
  %pred5 = fcmp oge double %a, %b
  %v5 = zext i1 %pred5 to i32
  %pred6 = fcmp ult double %a, %b
  %v6 = zext i1 %pred6 to i32
  %pred7 = fcmp ule double %a, %b
  %v7 = zext i1 %pred7 to i32
  %pred8 = fcmp ugt double %a, %b
  %v8 = zext i1 %pred8 to i32
  %pred9 = fcmp uge double %a, %b
  %v9 = zext i1 %pred9 to i32
  %a_bits = bitcast double %a to i64
  %b_bits = bitcast double %b to i64
  %ax = and i64 %a_bits, 9223372036854775807
  %bx = and i64 %b_bits, 9223372036854775807
  %nan = icmp ugt i64 %ax, 9218868437227405312
  %infinite = icmp eq i64 %ax, 9218868437227405312
  %finite = icmp ult i64 %ax, 9218868437227405312
  %v10 = zext i1 %nan to i32
  %v11 = zext i1 %infinite to i32
  %v12 = zext i1 %finite to i32
  %sign_wide = lshr i64 %a_bits, 63
  %v13 = trunc i64 %sign_wide to i32
  %minimum = call double @llvm.minnum.f64(double %a, double %b)
  %maximum = call double @llvm.maxnum.f64(double %a, double %b)
  %absolute = call double @llvm.fabs.f64(double %a)
  %copy_sign = call double @llvm.copysign.f64(double %a, double %b)
  %v14 = bitcast double %minimum to i64
  %v16 = bitcast double %maximum to i64
  %v18 = bitcast double %absolute to i64
  %v20 = bitcast double %copy_sign to i64
  %magnitudes = or i64 %ax, %bx
  %zero_pair = icmp eq i64 %magnitudes, 0
  %min_zero = or i64 %a_bits, %b_bits
  %max_zero = and i64 %a_bits, %b_bits
  %v22 = select i1 %zero_pair, i64 %min_zero, i64 %v14
  %v24 = select i1 %zero_pair, i64 %max_zero, i64 %v16
  %above = icmp uge i64 %ax, 4503599627370496
  %nonzero_normal = and i1 %above, %finite
  %zero = icmp eq i64 %ax, 0
  %normal = or i1 %nonzero_normal, %zero
  %below = icmp ult i64 %ax, 4503599627370496
  %nonzero = icmp ne i64 %ax, 0
  %subnormal = and i1 %below, %nonzero
  %v30 = zext i1 %normal to i32
  %v31 = zext i1 %subnormal to i32
  %mixed_min1 = call double @llvm.minnum.f64(double %a, double %c)
  %mixed_min2 = call double @llvm.minnum.f64(double %c, double %a)
  %mixed_max1 = call double @llvm.maxnum.f64(double %a, double %c)
  %mixed_max2 = call double @llvm.maxnum.f64(double %c, double %a)
  %v32 = bitcast double %mixed_min1 to i64
  %v34 = bitcast double %mixed_min2 to i64
  %v36 = bitcast double %mixed_max1 to i64
  %v38 = bitcast double %mixed_max2 to i64
  %v40 = select i1 %nan, i64 %a_bits, i64 %ax
  %b_sign = and i64 %b_bits, 9223372036854775808
  %v42 = or i64 %ax, %b_sign
  %output_offset = mul i64 %index, 44
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
  %wide_p14 = bitcast i32* %p14 to i64*
  store i64 %v14, i64* %wide_p14, align 8
  %p16 = getelementptr i32, i32* %p0, i64 16
  %wide_p16 = bitcast i32* %p16 to i64*
  store i64 %v16, i64* %wide_p16, align 8
  %p18 = getelementptr i32, i32* %p0, i64 18
  %wide_p18 = bitcast i32* %p18 to i64*
  store i64 %v18, i64* %wide_p18, align 8
  %p20 = getelementptr i32, i32* %p0, i64 20
  %wide_p20 = bitcast i32* %p20 to i64*
  store i64 %v20, i64* %wide_p20, align 8
  %p22 = getelementptr i32, i32* %p0, i64 22
  %wide_p22 = bitcast i32* %p22 to i64*
  store i64 %v22, i64* %wide_p22, align 8
  %p24 = getelementptr i32, i32* %p0, i64 24
  %wide_p24 = bitcast i32* %p24 to i64*
  store i64 %v24, i64* %wide_p24, align 8
  %p26 = getelementptr i32, i32* %p0, i64 26
  %wide_p26 = bitcast i32* %p26 to i64*
  store i64 %v14, i64* %wide_p26, align 8
  %p28 = getelementptr i32, i32* %p0, i64 28
  %wide_p28 = bitcast i32* %p28 to i64*
  store i64 %v16, i64* %wide_p28, align 8
  %p30 = getelementptr i32, i32* %p0, i64 30
  store i32 %v30, i32* %p30, align 4
  %p31 = getelementptr i32, i32* %p0, i64 31
  store i32 %v31, i32* %p31, align 4
  %p32 = getelementptr i32, i32* %p0, i64 32
  %wide_p32 = bitcast i32* %p32 to i64*
  store i64 %v32, i64* %wide_p32, align 8
  %p34 = getelementptr i32, i32* %p0, i64 34
  %wide_p34 = bitcast i32* %p34 to i64*
  store i64 %v34, i64* %wide_p34, align 8
  %p36 = getelementptr i32, i32* %p0, i64 36
  %wide_p36 = bitcast i32* %p36 to i64*
  store i64 %v36, i64* %wide_p36, align 8
  %p38 = getelementptr i32, i32* %p0, i64 38
  %wide_p38 = bitcast i32* %p38 to i64*
  store i64 %v38, i64* %wide_p38, align 8
  %p40 = getelementptr i32, i32* %p0, i64 40
  %wide_p40 = bitcast i32* %p40 to i64*
  store i64 %v40, i64* %wide_p40, align 8
  %p42 = getelementptr i32, i32* %p0, i64 42
  %wide_p42 = bitcast i32* %p42 to i64*
  store i64 %v42, i64* %wide_p42, align 8
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32)* @compare_double, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
