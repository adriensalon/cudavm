; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@values = internal addrspace(3) global [96 x i32] undef, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() convergent
declare i32 @llvm.nvvm.barrier0.popc(i32) convergent
declare i32 @llvm.nvvm.barrier0.and(i32) convergent
declare i32 @llvm.nvvm.barrier0.or(i32) convergent

define void @block_rounds(i32* %out, i32* %in, i32* %scratch,
                         i32 %count, i32 %rounds, i32 %pattern) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %i = add i32 %base, %lane
  %offset = zext i32 %i to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %input = load i32, i32* %source, align 4
  %own = getelementptr [96 x i32], [96 x i32] addrspace(3)* @values, i32 0, i32 %lane
  %global = getelementptr i32, i32* %scratch, i64 %offset
  br label %loop
loop:
  %round = phi i32 [ 0, %entry ], [ %next, %body ]
  %votes = phi i32 [ 0, %entry ], [ %new_votes, %body ]
  %every = phi i32 [ 0, %entry ], [ %new_every, %body ]
  %any = phi i32 [ 0, %entry ], [ %new_any, %body ]
  %shared_sum = phi i32 [ 0, %entry ], [ %new_shared_sum, %body ]
  %global_sum = phi i32 [ 0, %entry ], [ %new_global_sum, %body ]
  %more = icmp ult i32 %round, %rounds
  br i1 %more, label %body, label %finish
body:
  %scaled = mul i32 %round, 17
  %salt = add i32 %scaled, %i
  %value = xor i32 %input, %salt
  store i32 %value, i32 addrspace(3)* %own, align 4
  %global_value = add i32 %value, 74565
  store i32 %global_value, i32* %global, align 4
  %moving = urem i32 %round, %lanes
  %single = icmp eq i32 %lane, %moving
  %changed = add i32 %input, %round
  %bits = and i32 %changed, 3
  %mixed = icmp ne i32 %bits, 0
  %none = icmp eq i32 %pattern, 0
  %all = icmp eq i32 %pattern, 1
  %one = icmp eq i32 %pattern, 2
  %selected = select i1 %one, i1 %single, i1 %mixed
  %nonempty = or i1 %all, %selected
  %truth = select i1 %none, i1 false, i1 %nonempty
  %predicate = select i1 %truth, i32 -2147483648, i32 0
  %n = call i32 @llvm.nvvm.barrier0.popc(i32 %predicate)
  %a = call i32 @llvm.nvvm.barrier0.and(i32 %predicate)
  %o = call i32 @llvm.nvvm.barrier0.or(i32 %predicate)
  %a_bool = icmp ne i32 %a, 0
  %o_bool = icmp ne i32 %o, 0
  %a_word = zext i1 %a_bool to i32
  %o_word = zext i1 %o_bool to i32
  %new_votes = add i32 %votes, %n
  %new_every = add i32 %every, %a_word
  %new_any = add i32 %any, %o_word
  %shifted = add i32 %lane, %round
  %incremented = add i32 %shifted, 1
  %peer = urem i32 %incremented, %lanes
  %shared_peer = getelementptr [96 x i32], [96 x i32] addrspace(3)* @values, i32 0, i32 %peer
  %shared_read = load i32, i32 addrspace(3)* %shared_peer, align 4
  %new_shared_sum = add i32 %shared_sum, %shared_read
  %peer_index = add i32 %base, %peer
  %peer_offset = zext i32 %peer_index to i64
  %global_peer = getelementptr i32, i32* %scratch, i64 %peer_offset
  %global_read = load i32, i32* %global_peer, align 4
  %new_global_sum = add i32 %global_sum, %global_read
  call void @llvm.nvvm.barrier0()
  %next = add i32 %round, 1
  br label %loop
finish:
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %target_offset = mul i64 %offset, 5
  %p0 = getelementptr i32, i32* %out, i64 %target_offset
  %p1 = getelementptr i32, i32* %p0, i64 1
  %p2 = getelementptr i32, i32* %p0, i64 2
  %p3 = getelementptr i32, i32* %p0, i64 3
  %p4 = getelementptr i32, i32* %p0, i64 4
  store i32 %votes, i32* %p0, align 4
  store i32 %every, i32* %p1, align 4
  store i32 %any, i32* %p2, align 4
  store i32 %shared_sum, i32* %p3, align 4
  store i32 %global_sum, i32* %p4, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32*, i32, i32, i32)* @block_rounds, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
