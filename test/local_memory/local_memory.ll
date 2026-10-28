; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define void @local_arrays(i32* %out, i32* %in, i32 %count, i32 %rounds) {
entry:
  %values = alloca [19 x i32], align 16
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %i = add i32 %base, %lane
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %prepare, label %done
prepare:
  %offset = zext i32 %i to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %seed = load i32, i32* %source, align 4
  %left_guard = xor i32 %i, 324508639
  %right_guard = xor i32 %seed, 3970458656
  %left = getelementptr [19 x i32], [19 x i32]* %values, i32 0, i32 0
  %right = getelementptr [19 x i32], [19 x i32]* %values, i32 0, i32 18
  store volatile i32 %left_guard, i32* %left, align 4
  store volatile i32 %right_guard, i32* %right, align 4
  %salt = mul i32 %i, 17
  br label %initialize
initialize:
  %slot = phi i32 [ 0, %prepare ], [ %next_slot, %initialize ]
  %product = mul i32 %slot, 66051
  %added = add i32 %seed, %product
  %key = add i32 %salt, %slot
  %initial = xor i32 %added, %key
  %index = add i32 %slot, 1
  %own = getelementptr [19 x i32], [19 x i32]* %values, i32 0, i32 %index
  store volatile i32 %initial, i32* %own, align 4
  %next_slot = add i32 %slot, 1
  %more_slots = icmp ult i32 %next_slot, 17
  br i1 %more_slots, label %initialize, label %ready
ready:
  %extra = and i32 %i, 3
  %steps = add i32 %rounds, %extra
  br label %loop
loop:
  %round = phi i32 [ 0, %ready ], [ %next_round, %update ]
  %more = icmp ult i32 %round, %steps
  br i1 %more, label %update, label %observe
update:
  %scaled = mul i32 %round, 5
  %position = add i32 %seed, %scaled
  %selected_slot = urem i32 %position, 17
  %selected_index = add i32 %selected_slot, 1
  %selected = getelementptr [19 x i32], [19 x i32]* %values, i32 0, i32 %selected_index
  %shifted = add i32 %selected_slot, 7
  %peer_slot = urem i32 %shifted, 17
  %peer_index = add i32 %peer_slot, 1
  %peer = getelementptr [19 x i32], [19 x i32]* %values, i32 0, i32 %peer_index
  %old = load volatile i32, i32* %selected, align 4
  %neighbor = load volatile i32, i32* %peer, align 4
  %rhs = add i32 %neighbor, %round
  %new = xor i32 %old, %rhs
  store volatile i32 %new, i32* %selected, align 4
  %next_round = add i32 %round, 1
  br label %loop
observe:
  br label %checksum
checksum:
  %read_slot = phi i32 [ 0, %observe ], [ %next_read, %checksum ]
  %sum = phi i32 [ 0, %observe ], [ %new_sum, %checksum ]
  %hash = phi i32 [ 0, %observe ], [ %new_hash, %checksum ]
  %read_index = add i32 %read_slot, 1
  %read_pointer = getelementptr [19 x i32], [19 x i32]* %values, i32 0, i32 %read_index
  %value = load volatile i32, i32* %read_pointer, align 4
  %new_sum = add i32 %sum, %value
  %hash_product = mul i32 %hash, 33
  %new_hash = xor i32 %hash_product, %value
  %next_read = add i32 %read_slot, 1
  %more_reads = icmp ult i32 %next_read, 17
  br i1 %more_reads, label %checksum, label %write
write:
  %choice = urem i32 %seed, 17
  %choice_index = add i32 %choice, 1
  %choice_pointer = getelementptr [19 x i32], [19 x i32]* %values, i32 0, i32 %choice_index
  %chosen = load volatile i32, i32* %choice_pointer, align 4
  %left_read = load volatile i32, i32* %left, align 4
  %right_read = load volatile i32, i32* %right, align 4
  %target_offset = mul i64 %offset, 5
  %p0 = getelementptr i32, i32* %out, i64 %target_offset
  %p1 = getelementptr i32, i32* %p0, i64 1
  %p2 = getelementptr i32, i32* %p0, i64 2
  %p3 = getelementptr i32, i32* %p0, i64 3
  %p4 = getelementptr i32, i32* %p0, i64 4
  store i32 %new_sum, i32* %p0, align 4
  store i32 %new_hash, i32* %p1, align 4
  store i32 %chosen, i32* %p2, align 4
  store i32 %left_read, i32* %p3, align 4
  store i32 %right_read, i32* %p4, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32, i32)* @local_arrays, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
