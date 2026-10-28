; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define void @numeric_bits(i32* %out, i32* %in, i32 %count) {
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
  %input_offset = mul i64 %index, 4
  %source = getelementptr i32, i32* %in, i64 %input_offset
  %source1 = getelementptr i32, i32* %source, i64 1
  %source2 = getelementptr i32, i32* %source, i64 2
  %float0 = bitcast i32* %source to float*
  %float1 = bitcast i32* %source1 to float*
  %double_source = bitcast i32* %source2 to double*
  %wide_source = bitcast i32* %source2 to i64*
  %fa = load float, float* %float0, align 4
  %fb = load float, float* %float1, align 4
  %a_bits = bitcast float %fa to i32
  %b_bits = bitcast float %fb to i32
  %a = load i32, i32* %source, align 4
  %b = load i32, i32* %source1, align 4
  %a_float = bitcast i32 %a to float
  %b_float = bitcast i32 %b to float
  %wide = load i64, i64* %wide_source, align 8
  %wide_double = bitcast i64 %wide to double
  %real = load double, double* %double_source, align 8
  %real_bits = bitcast double %real to i64
  %low = trunc i64 %real_bits to i32
  %shifted = lshr i64 %real_bits, 32
  %high = trunc i64 %shifted to i32
  %a_wide = zext i32 %a to i64
  %a_high = shl i64 %a_wide, 32
  %b_wide = zext i32 %b to i64
  %constructed = or i64 %a_high, %b_wide
  %constructed_double = bitcast i64 %constructed to double
  %output_offset = mul i64 %index, 12
  %p0 = getelementptr i32, i32* %out, i64 %output_offset
  %p1 = getelementptr i32, i32* %p0, i64 1
  %p2 = getelementptr i32, i32* %p0, i64 2
  %p3 = getelementptr i32, i32* %p0, i64 3
  %p4 = getelementptr i32, i32* %p0, i64 4
  %p6 = getelementptr i32, i32* %p0, i64 6
  %p8 = getelementptr i32, i32* %p0, i64 8
  %p9 = getelementptr i32, i32* %p0, i64 9
  %p10 = getelementptr i32, i32* %p0, i64 10
  %float_out2 = bitcast i32* %p2 to float*
  %float_out3 = bitcast i32* %p3 to float*
  %double_out4 = bitcast i32* %p4 to double*
  %wide_out6 = bitcast i32* %p6 to i64*
  %double_out10 = bitcast i32* %p10 to double*
  store i32 %a_bits, i32* %p0, align 4
  store i32 %b_bits, i32* %p1, align 4
  store float %a_float, float* %float_out2, align 4
  store float %b_float, float* %float_out3, align 4
  store double %wide_double, double* %double_out4, align 8
  store i64 %real_bits, i64* %wide_out6, align 8
  store i32 %low, i32* %p8, align 4
  store i32 %high, i32* %p9, align 4
  store double %constructed_double, double* %double_out10, align 8
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32)* @numeric_bits, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
