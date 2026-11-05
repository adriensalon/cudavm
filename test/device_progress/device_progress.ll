; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@values = internal addrspace(3) global [64 x i32] undef, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() convergent
define void @progress_rounds(i32* %out, i32* %in, i32* %tickets, i32* %completed, i32 %count, i32 %seed) {
entry:
 %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %base = mul i32 %block, %lanes
 %id = add i32 %base, %lane
 %offset = zext i32 %id to i64
 %source = getelementptr i32, i32* %in, i64 %offset
 %input = load i32, i32* %source, align 4
 %initial = xor i32 %input, %seed
 %salt = add i32 %block, %seed
 %remainder = and i32 %salt, 3
 %rounds = add i32 %remainder, 1
 %own = getelementptr [64 x i32], [64 x i32] addrspace(3)* @values, i32 0, i32 %lane
 %neighbor = add i32 %lane, 1
 %wrap = icmp eq i32 %neighbor, %lanes
 %peer = select i1 %wrap, i32 0, i32 %neighbor
 %peer_address = getelementptr [64 x i32], [64 x i32] addrspace(3)* @values, i32 0, i32 %peer
 br label %round_loop
round_loop:
 %round = phi i32 [0, %entry], [%next_round, %exchange]
 %value = phi i32 [%initial, %entry], [%next_value, %exchange]
 %more = icmp ult i32 %round, %rounds
 br i1 %more, label %lane_start, label %finish
lane_start:
 %vary = add i32 %lane, %round
 %steps = and i32 %vary, 3
 br label %lane_loop
lane_loop:
 %step = phi i32 [0, %lane_start], [%next_step, %lane_body]
 %working = phi i32 [%value, %lane_start], [%next_working, %lane_body]
 %again = icmp ult i32 %step, %steps
 br i1 %again, label %lane_body, label %exchange
lane_body:
 %scaled = mul i32 %round, 13
 %iteration = add i32 %scaled, %step
 %key = add i32 %iteration, %seed
 %changed = xor i32 %working, %key
 %product = mul i32 %changed, 33
 %next_working = add i32 %product, %lane
 %next_step = add i32 %step, 1
 br label %lane_loop
exchange:
 store i32 %working, i32 addrspace(3)* %own, align 4
 call void @llvm.nvvm.barrier0()
 %other = load i32, i32 addrspace(3)* %peer_address, align 4
 %key_scale = mul i32 %round, 17
 %key_sum = add i32 %key_scale, %seed
 %peer_value = xor i32 %other, %key_sum
 %next_value = add i32 %working, %peer_value
 call void @llvm.nvvm.barrier0()
 %next_round = add i32 %round, 1
 br label %round_loop
finish:
 %valid = icmp ult i32 %id, %count
 br i1 %valid, label %write, label %ticket_test
write:
 %destination = getelementptr i32, i32* %out, i64 %offset
 store i32 %value, i32* %destination, align 4
 br label %ticket_test
ticket_test:
 %leader = icmp eq i32 %lane, 0
 br i1 %leader, label %ticket, label %done
ticket:
 %counter = addrspacecast i32* %completed to i32 addrspace(1)*
 %old = atomicrmw add i32 addrspace(1)* %counter, i32 1 syncscope("device") monotonic
 %block_offset = zext i32 %block to i64
 %ticket_pointer = getelementptr i32, i32* %tickets, i64 %block_offset
 store i32 %old, i32* %ticket_pointer, align 4
 br label %done
done:
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32*, i32*, i32, i32)* @progress_rounds, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
