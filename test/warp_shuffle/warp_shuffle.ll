; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32) convergent
declare i32 @llvm.nvvm.shfl.sync.up.i32(i32, i32, i32, i32) convergent
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32) convergent
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) convergent

define void @shuffle_modes(i32* %out, i32* %in, i32 %width, i32 %distance) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %index = add i32 %base, %thread
  %offset = zext i32 %index to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %value = load i32, i32* %source, align 4
  %segment = sub i32 32, %width
  %up_control = shl i32 %segment, 8
  %control = or i32 %up_control, 31
  %indexed = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %value, i32 %distance, i32 %control)
  %up = call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %value, i32 %distance, i32 %up_control)
  %down = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %value, i32 %distance, i32 %control)
  %xor = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %value, i32 %distance, i32 %control)
  %output_offset = mul i64 %offset, 4
  %first = getelementptr i32, i32* %out, i64 %output_offset
  %second = getelementptr i32, i32* %first, i64 1
  %third = getelementptr i32, i32* %first, i64 2
  %fourth = getelementptr i32, i32* %first, i64 3
  store i32 %indexed, i32* %first, align 4
  store i32 %up, i32* %second, align 4
  store i32 %down, i32* %third, align 4
  store i32 %xor, i32* %fourth, align 4
  ret void
}

define void @shuffle_scan(i32* %out, i32* %in) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %index = add i32 %base, %thread
  %offset = zext i32 %index to i64
  %lane = and i32 %thread, 31
  %warp_base = and i32 %thread, -32
  %remaining = sub i32 %threads, %warp_base
  %partial = icmp ult i32 %remaining, 32
  %live = select i1 %partial, i32 %remaining, i32 32
  %last = sub i32 %live, 1
  %source = getelementptr i32, i32* %in, i64 %offset
  %initial = load i32, i32* %source, align 4
  br label %loop
loop:
  %step = phi i32 [ 1, %entry ], [ %next, %loop ]
  %value = phi i32 [ %initial, %entry ], [ %sum, %loop ]
  %other = call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %value, i32 %step, i32 0)
  %include = icmp uge i32 %lane, %step
  %addend = select i1 %include, i32 %other, i32 0
  %sum = add i32 %value, %addend
  %next = shl i32 %step, 1
  %more = icmp ult i32 %next, 32
  br i1 %more, label %loop, label %finish
finish:
  %total = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %sum, i32 %last, i32 31)
  %output_offset = mul i64 %offset, 2
  %first = getelementptr i32, i32* %out, i64 %output_offset
  %second = getelementptr i32, i32* %first, i64 1
  store i32 %sum, i32* %first, align 4
  store i32 %total, i32* %second, align 4
  ret void
}
!nvvm.annotations = !{!0, !1}
!0 = !{void (i32*, i32*, i32, i32)* @shuffle_modes, !"kernel", i32 1}
!1 = !{void (i32*, i32*)* @shuffle_scan, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 1}
