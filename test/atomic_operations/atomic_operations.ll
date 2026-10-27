; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@dynamic_values = external addrspace(3) global [0 x i32], align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare void @llvm.nvvm.barrier0() convergent

define void @global_add(i32* %out, i32* %state, i32 %count) {
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %valid = icmp ult i32 %index, %count
  %delta = zext i1 %valid to i32
  %bucket = and i32 %index, 1
  %bucket_offset = zext i32 %bucket to i64
  %global = addrspacecast i32* %state to i32 addrspace(1)*
  %counter = getelementptr i32, i32 addrspace(1)* %global, i64 %bucket_offset
  %old = atomicrmw add i32 addrspace(1)* %counter, i32 %delta syncscope("device") monotonic
  %offset = zext i32 %index to i64
  %destination = getelementptr i32, i32* %out, i64 %offset
  store i32 %old, i32* %destination, align 4
  %shadow = getelementptr i32, i32 addrspace(1)* %counter, i64 2
  %discard = atomicrmw add i32 addrspace(1)* %shadow, i32 %delta syncscope("device") monotonic
  ret void
}

define void @global_cas(i32* %out, i32* %state, i32 %replacement) {
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %global = addrspacecast i32* %state to i32 addrspace(1)*
  %result = cmpxchg i32 addrspace(1)* %global, i32 0, i32 %replacement syncscope("device") monotonic monotonic
  %old = extractvalue { i32, i1 } %result, 0
  %offset = zext i32 %index to i64
  %destination = getelementptr i32, i32* %out, i64 %offset
  store i32 %old, i32* %destination, align 4
  %shadow = getelementptr i32, i32 addrspace(1)* %global, i64 1
  %discard = cmpxchg i32 addrspace(1)* %shadow, i32 0, i32 %replacement syncscope("device") monotonic monotonic
  ret void
}

define void @shared_cas(i32* %out, i32 %replacement, i32 %unused) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %grid = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %counter = getelementptr [0 x i32], [0 x i32] addrspace(3)* @dynamic_values, i32 0, i32 0
  %shadow = getelementptr i32, i32 addrspace(3)* %counter, i32 1
  %initialize = icmp ult i32 %lane, 2
  br i1 %initialize, label %clear, label %ready
clear:
  %own = getelementptr i32, i32 addrspace(3)* %counter, i32 %lane
  store i32 0, i32 addrspace(3)* %own, align 4
  br label %ready
ready:
  call void @llvm.nvvm.barrier0()
  %result = cmpxchg i32 addrspace(3)* %counter, i32 0, i32 %replacement syncscope("block") monotonic monotonic
  %old = extractvalue { i32, i1 } %result, 0
  %discard = cmpxchg i32 addrspace(3)* %shadow, i32 0, i32 %replacement syncscope("block") monotonic monotonic
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %offset = zext i32 %index to i64
  %destination = getelementptr i32, i32* %out, i64 %offset
  store i32 %old, i32* %destination, align 4
  call void @llvm.nvvm.barrier0()
  %leader = icmp eq i32 %lane, 0
  br i1 %leader, label %snapshot, label %done
snapshot:
  %final = load i32, i32 addrspace(3)* %shadow, align 4
  %total = mul i32 %grid, %lanes
  %snapshot_index = add i32 %total, %block
  %snapshot_offset = zext i32 %snapshot_index to i64
  %snapshot_destination = getelementptr i32, i32* %out, i64 %snapshot_offset
  store i32 %final, i32* %snapshot_destination, align 4
  br label %done
done:
  ret void
}

define void @shared_histogram(i32* %out, i32* %in, i32 %count) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %grid = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %offset = zext i32 %index to i64
  %own = getelementptr [0 x i32], [0 x i32] addrspace(3)* @dynamic_values, i32 0, i32 %lane
  %own_shadow = getelementptr i32, i32 addrspace(3)* %own, i32 4
  %initialize = icmp ult i32 %lane, 4
  br i1 %initialize, label %clear, label %ready
clear:
  store i32 0, i32 addrspace(3)* %own, align 4
  store i32 0, i32 addrspace(3)* %own_shadow, align 4
  br label %ready
ready:
  call void @llvm.nvvm.barrier0()
  %valid = icmp ult i32 %index, %count
  %delta = zext i1 %valid to i32
  br i1 %valid, label %read, label %add
read:
  %source = getelementptr i32, i32* %in, i64 %offset
  %input = load i32, i32* %source, align 4
  br label %add
add:
  %value = phi i32 [ 0, %ready ], [ %input, %read ]
  %bucket = and i32 %value, 3
  %counter = getelementptr [0 x i32], [0 x i32] addrspace(3)* @dynamic_values, i32 0, i32 %bucket
  %shadow = getelementptr i32, i32 addrspace(3)* %counter, i32 4
  %old = atomicrmw add i32 addrspace(3)* %counter, i32 %delta syncscope("block") monotonic
  %discard = atomicrmw add i32 addrspace(3)* %shadow, i32 %delta syncscope("block") monotonic
  %prefix = mul i32 %grid, 8
  %ticket_index = add i32 %prefix, %index
  %ticket_offset = zext i32 %ticket_index to i64
  %destination = getelementptr i32, i32* %out, i64 %ticket_offset
  store i32 %old, i32* %destination, align 4
  call void @llvm.nvvm.barrier0()
  br i1 %initialize, label %snapshot, label %done
snapshot:
  %final = load i32, i32 addrspace(3)* %own, align 4
  %final_shadow = load i32, i32 addrspace(3)* %own_shadow, align 4
  %block_prefix = mul i32 %block, 8
  %snapshot_index = add i32 %block_prefix, %lane
  %snapshot_offset = zext i32 %snapshot_index to i64
  %snapshot_destination = getelementptr i32, i32* %out, i64 %snapshot_offset
  %shadow_destination = getelementptr i32, i32* %snapshot_destination, i64 4
  store i32 %final, i32* %snapshot_destination, align 4
  store i32 %final_shadow, i32* %shadow_destination, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0, !1, !2, !3}
!0 = !{void (i32*, i32*, i32)* @global_add, !"kernel", i32 1}
!1 = !{void (i32*, i32*, i32)* @global_cas, !"kernel", i32 1}
!2 = !{void (i32*, i32, i32)* @shared_cas, !"kernel", i32 1}
!3 = !{void (i32*, i32*, i32)* @shared_histogram, !"kernel", i32 1}
!nvvmir.version = !{!4}
!4 = !{i32 2, i32 0, i32 3, i32 1}
