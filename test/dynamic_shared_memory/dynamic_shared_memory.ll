; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@dynamic_values = external addrspace(3) global [0 x i32], align 16
@mixed_fixed = internal addrspace(3) global [3 x i32] undef, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() convergent

define void @dynamic_reduce(i32* %out, i32* %in, i32 %count) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %offset = zext i32 %index to i64
  %valid = icmp ult i32 %index, %count
  br i1 %valid, label %read, label %initialize
read:
  %source = getelementptr i32, i32* %in, i64 %offset
  %input = load i32, i32* %source, align 4
  br label %initialize
initialize:
  %value = phi i32 [ 0, %entry ], [ %input, %read ]
  %own = getelementptr [0 x i32], [0 x i32] addrspace(3)* @dynamic_values, i32 0, i32 %lane
  store i32 %value, i32 addrspace(3)* %own, align 4
  call void @llvm.nvvm.barrier0()
  %first_step = lshr i32 %lanes, 1
  br label %loop
loop:
  %step = phi i32 [ %first_step, %initialize ], [ %next, %rendezvous ]
  %more = icmp ne i32 %step, 0
  br i1 %more, label %choose, label %finish
choose:
  %active = icmp ult i32 %lane, %step
  br i1 %active, label %combine, label %rendezvous
combine:
  %peer_index = add i32 %lane, %step
  %peer = getelementptr [0 x i32], [0 x i32] addrspace(3)* @dynamic_values, i32 0, i32 %peer_index
  %left = load i32, i32 addrspace(3)* %own, align 4
  %right = load i32, i32 addrspace(3)* %peer, align 4
  %sum = add i32 %left, %right
  store i32 %sum, i32 addrspace(3)* %own, align 4
  br label %rendezvous
rendezvous:
  call void @llvm.nvvm.barrier0()
  %next = lshr i32 %step, 1
  br label %loop
finish:
  %leader = icmp eq i32 %lane, 0
  br i1 %leader, label %write, label %done
write:
  %result = load i32, i32 addrspace(3)* %own, align 4
  %block_offset = zext i32 %block to i64
  %destination = getelementptr i32, i32* %out, i64 %block_offset
  store i32 %result, i32* %destination, align 4
  br label %done
done:
  ret void
}

define void @mixed_exchange(i32* %out, i32* %in, i32 %count) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %offset = zext i32 %index to i64
  %fixed = getelementptr [3 x i32], [3 x i32] addrspace(3)* @mixed_fixed, i32 0, i32 2
  %leader = icmp eq i32 %lane, 0
  br i1 %leader, label %set_bias, label %choose
set_bias:
  %bias_value = add i32 %block, 7
  store i32 %bias_value, i32 addrspace(3)* %fixed, align 4
  br label %choose
choose:
  %valid = icmp ult i32 %index, %count
  br i1 %valid, label %read, label %initialize
read:
  %source = getelementptr i32, i32* %in, i64 %offset
  %input = load i32, i32* %source, align 4
  br label %initialize
initialize:
  %value = phi i32 [ 0, %choose ], [ %input, %read ]
  %own = getelementptr [0 x i32], [0 x i32] addrspace(3)* @dynamic_values, i32 0, i32 %lane
  store i32 %value, i32 addrspace(3)* %own, align 4
  call void @llvm.nvvm.barrier0()
  %reverse = sub i32 %lanes, %lane
  %peer_index = sub i32 %reverse, 1
  %peer = getelementptr [0 x i32], [0 x i32] addrspace(3)* @dynamic_values, i32 0, i32 %peer_index
  %reversed = load i32, i32 addrspace(3)* %peer, align 4
  %bias = load i32, i32 addrspace(3)* %fixed, align 4
  %result = add i32 %reversed, %bias
  br i1 %valid, label %write, label %done
write:
  %destination = getelementptr i32, i32* %out, i64 %offset
  store i32 %result, i32* %destination, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0, !1}
!0 = !{void (i32*, i32*, i32)* @dynamic_reduce, !"kernel", i32 1}
!1 = !{void (i32*, i32*, i32)* @mixed_exchange, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 1}
