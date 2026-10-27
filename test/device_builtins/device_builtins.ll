; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
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
declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
declare i32 @llvm.nvvm.read.ptx.sreg.warpsize()

define void @record_builtins(i32* %out, i32 %count) {
entry:
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
  %local = add i32 %tzynx, %tx
  %bzgy = mul i32 %bz, %gy
  %bzy = add i32 %bzgy, %by
  %bzygx = mul i32 %bzy, %gx
  %block = add i32 %bzygx, %bx
  %xy = mul i32 %nx, %ny
  %volume = mul i32 %xy, %nz
  %base = mul i32 %block, %volume
  %i = add i32 %base, %local
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %warp = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %wide = zext i32 %i to i64
  %offset = mul i64 %wide, 16
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
  store i32 %tx, i32* %p0, align 4
  store i32 %ty, i32* %p1, align 4
  store i32 %tz, i32* %p2, align 4
  store i32 %nx, i32* %p3, align 4
  store i32 %ny, i32* %p4, align 4
  store i32 %nz, i32* %p5, align 4
  store i32 %bx, i32* %p6, align 4
  store i32 %by, i32* %p7, align 4
  store i32 %bz, i32* %p8, align 4
  store i32 %gx, i32* %p9, align 4
  store i32 %gy, i32* %p10, align 4
  store i32 %gz, i32* %p11, align 4
  store i32 %lane, i32* %p12, align 4
  store i32 %warp, i32* %p13, align 4
  store i32 %local, i32* %p14, align 4
  store i32 %block, i32* %p15, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32)* @record_builtins, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
