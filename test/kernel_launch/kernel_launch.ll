; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define void @cubin_scalar(i32* %out, i32* %in, i32 %add) {
  %source = getelementptr i32, i32* %in, i64 1
  %destination = getelementptr i32, i32* %out, i64 1
  %value = load i32, i32* %source, align 4
  %result = add i32 %value, %add
  store i32 %result, i32* %destination, align 4
  ret void
}
define void @cubin_branch(i32* %out, i32* %in, i32 %threshold) {
  %value = load i32, i32* %in, align 4
  %small = icmp ult i32 %value, %threshold
  br i1 %small, label %multiply, label %increment
multiply:
  %product = mul i32 %value, 3
  br label %done
increment:
  %sum = add i32 %value, 5
  br label %done
done:
  %result = phi i32 [ %product, %multiply ], [ %sum, %increment ]
  store i32 %result, i32* %out, align 4
  ret void
}
define void @cubin_wide(i32* %out, i32 %left, i32 %right, i64 %add) {
  %a = zext i32 %left to i64
  %b = zext i32 %right to i64
  %product = mul i64 %a, %b
  %sum = add i64 %product, %add
  %low = trunc i64 %sum to i32
  %shifted = lshr i64 %sum, 32
  %high = trunc i64 %shifted to i32
  %next = getelementptr i32, i32* %out, i64 1
  store i32 %low, i32* %out, align 4
  store i32 %high, i32* %next, align 4
  ret void
}
define void @cubin_indexed(i32* %out, i32* %in, i32 %count) {
  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %tz = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %nx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %ny = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %nz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %bz = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %gx = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %gy = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %gz = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %tzny = mul i32 %tz, %ny
  %tzy = add i32 %tzny, %ty
  %tzynx = mul i32 %tzy, %nx
  %lane = add i32 %tzynx, %tx
  %bzgy = mul i32 %bz, %gy
  %bzy = add i32 %bzgy, %by
  %bzygx = mul i32 %bzy, %gx
  %block = add i32 %bzygx, %bx
  %xy = mul i32 %nx, %ny
  %volume = mul i32 %xy, %nz
  %base = mul i32 %block, %volume
  %index = add i32 %base, %lane
  %valid = icmp ult i32 %index, %count
  br i1 %valid, label %write, label %done
write:
  %offset = zext i32 %index to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %destination = getelementptr i32, i32* %out, i64 %offset
  %value = load i32, i32* %source, align 4
  %gridxy = mul i32 %gx, %gy
  %grid = mul i32 %gridxy, %gz
  %partial = add i32 %value, %index
  %result = add i32 %partial, %grid
  store i32 %result, i32* %destination, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0, !1, !2, !3}
!0 = !{void (i32*, i32*, i32)* @cubin_scalar, !"kernel", i32 1}
!1 = !{void (i32*, i32*, i32)* @cubin_branch, !"kernel", i32 1}
!2 = !{void (i32*, i32, i32, i64)* @cubin_wide, !"kernel", i32 1}
!3 = !{void (i32*, i32*, i32)* @cubin_indexed, !"kernel", i32 1}
!nvvmir.version = !{!4}
!4 = !{i32 2, i32 0, i32 3, i32 1}
