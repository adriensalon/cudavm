; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@exchange_values = internal addrspace(3) global [129 x i32] undef, align 16
@transpose_values = internal addrspace(3) global [80 x i32] undef, align 4
@reduce_values = internal addrspace(3) global [128 x i32] undef, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() convergent

define void @shared_exchange(i32* %out, i32* %in, i32 %count) {
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
  %initial = phi i32 [ 0, %entry ], [ %input, %read ]
  %own_index = add i32 %lane, 1
  %own = getelementptr [129 x i32], [129 x i32] addrspace(3)* @exchange_values, i32 0, i32 %own_index
  %steps = lshr i32 %lane, 4
  br label %loop
loop:
  %value = phi i32 [ %initial, %initialize ], [ %changed, %body ]
  %step = phi i32 [ 0, %initialize ], [ %next, %body ]
  %more = icmp ult i32 %step, %steps
  br i1 %more, label %body, label %ready
body:
  %product = mul i32 %value, 3
  %changed = add i32 %product, %step
  store volatile i32 %changed, i32 addrspace(3)* %own, align 4
  %next = add i32 %step, 1
  br label %loop
ready:
  store i32 %value, i32 addrspace(3)* %own, align 4
  call void @llvm.nvvm.barrier0()
  %peer_index = sub i32 %lanes, %lane
  %peer = getelementptr [129 x i32], [129 x i32] addrspace(3)* @exchange_values, i32 0, i32 %peer_index
  %reversed = load i32, i32 addrspace(3)* %peer, align 4
  call void @llvm.nvvm.barrier0()
  %scaled = mul i32 %reversed, 3
  %updated = add i32 %scaled, %lane
  store i32 %updated, i32 addrspace(3)* %own, align 4
  call void @llvm.nvvm.barrier0()
  %result = load i32, i32 addrspace(3)* %peer, align 4
  br i1 %valid, label %write, label %done
write:
  %destination = getelementptr i32, i32* %out, i64 %offset
  store i32 %result, i32* %destination, align 4
  br label %done
done:
  ret void
}

define void @shared_transpose(i32* %out, i32* %in, i32 %tiles) {
  %x = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %y = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, 64
  %row = mul i32 %y, 8
  %column = add i32 %row, %x
  %index = add i32 %base, %column
  %offset = zext i32 %index to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %value = load i32, i32* %source, align 4
  %shared_row = mul i32 %y, 10
  %shared_column = add i32 %shared_row, %x
  %own_index = add i32 %shared_column, 1
  %own = getelementptr [80 x i32], [80 x i32] addrspace(3)* @transpose_values, i32 0, i32 %own_index
  store i32 %value, i32 addrspace(3)* %own, align 4
  call void @llvm.nvvm.barrier0()
  %valid = icmp ult i32 %block, %tiles
  br i1 %valid, label %write, label %done
write:
  %transposed_row = mul i32 %x, 10
  %transposed_column = add i32 %transposed_row, %y
  %peer_index = add i32 %transposed_column, 1
  %peer = getelementptr [80 x i32], [80 x i32] addrspace(3)* @transpose_values, i32 0, i32 %peer_index
  %result = load i32, i32 addrspace(3)* %peer, align 4
  %destination = getelementptr i32, i32* %out, i64 %offset
  store i32 %result, i32* %destination, align 4
  br label %done
done:
  ret void
}

define void @shared_reduce(i32* %out, i32* %in, i32 %count) {
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
  %own = getelementptr [128 x i32], [128 x i32] addrspace(3)* @reduce_values, i32 0, i32 %lane
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
  %peer = getelementptr [128 x i32], [128 x i32] addrspace(3)* @reduce_values, i32 0, i32 %peer_index
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
!nvvm.annotations = !{!0, !1, !2}
!0 = !{void (i32*, i32*, i32)* @shared_exchange, !"kernel", i32 1}
!1 = !{void (i32*, i32*, i32)* @shared_transpose, !"kernel", i32 1}
!2 = !{void (i32*, i32*, i32)* @shared_reduce, !"kernel", i32 1}
!nvvmir.version = !{!3}
!3 = !{i32 2, i32 0, i32 3, i32 1}
