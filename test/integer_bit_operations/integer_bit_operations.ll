; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctpop.i64(i64)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)

define void @count_reverse_bits(i32* %out, i64* %in, i32 %count) {
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
  %source = getelementptr i64, i64* %in, i64 %offset
  %wide = load i64, i64* %source, align 8
  %narrow = trunc i64 %wide to i32
  %p32 = call i32 @llvm.ctpop.i32(i32 %narrow)
  %l32 = call i32 @llvm.ctlz.i32(i32 %narrow, i1 false)
  %r32 = call i32 @llvm.bitreverse.i32(i32 %narrow)
  %trailing32 = call i32 @llvm.cttz.i32(i32 %narrow, i1 false)
  %first32 = add i32 %trailing32, 1
  %zero32 = icmp eq i32 %narrow, 0
  %f32 = select i1 %zero32, i32 0, i32 %first32
  %population64 = call i64 @llvm.ctpop.i64(i64 %wide)
  %leading64 = call i64 @llvm.ctlz.i64(i64 %wide, i1 false)
  %reverse64 = call i64 @llvm.bitreverse.i64(i64 %wide)
  %trailing64 = call i64 @llvm.cttz.i64(i64 %wide, i1 false)
  %first64 = add i64 %trailing64, 1
  %zero64 = icmp eq i64 %wide, 0
  %ffs64 = select i1 %zero64, i64 0, i64 %first64
  %p64 = trunc i64 %population64 to i32
  %l64 = trunc i64 %leading64 to i32
  %lo = trunc i64 %reverse64 to i32
  %shifted = lshr i64 %reverse64, 32
  %hi = trunc i64 %shifted to i32
  %f64 = trunc i64 %ffs64 to i32
  %destination_offset = mul i64 %offset, 9
  %p0 = getelementptr i32, i32* %out, i64 %destination_offset
  %p1 = getelementptr i32, i32* %p0, i64 1
  %p2 = getelementptr i32, i32* %p0, i64 2
  %p3 = getelementptr i32, i32* %p0, i64 3
  %p4 = getelementptr i32, i32* %p0, i64 4
  %p5 = getelementptr i32, i32* %p0, i64 5
  %p6 = getelementptr i32, i32* %p0, i64 6
  %p7 = getelementptr i32, i32* %p0, i64 7
  %p8 = getelementptr i32, i32* %p0, i64 8
  store i32 %p32, i32* %p0, align 4
  store i32 %l32, i32* %p1, align 4
  store i32 %r32, i32* %p2, align 4
  store i32 %f32, i32* %p3, align 4
  store i32 %p64, i32* %p4, align 4
  store i32 %l64, i32* %p5, align 4
  store i32 %lo, i32* %p6, align 4
  store i32 %hi, i32* %p7, align 4
  store i32 %f64, i32* %p8, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i64*, i32)* @count_reverse_bits, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
