; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
; CTA barriers are expressed with scoped inline PTX and memory clobbers.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@values = internal addrspace(3) global [128 x i32] undef, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
define void @named_whole(i32* %out, i32* %in, i32* %scratch, i32 %rounds, i32 %id, i32 %pattern) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %lane
  %offset = zext i32 %i to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %input = load i32, i32* %source, align 4
  %own = getelementptr [128 x i32], [128 x i32] addrspace(3)* @values, i32 0, i32 %lane
  %global = getelementptr i32, i32* %scratch, i64 %offset
  %group_base = and i32 %lane, -64
  %first = icmp eq i32 %group_base, 0
  %group_id = select i1 %first, i32 1, i32 15
  br label %loop
loop:
  %round = phi i32 [ 0, %entry ], [ %next, %latch ]
  %s0 = phi i32 [ 0, %entry ], [ %new0, %latch ]
  %s1 = phi i32 [ 0, %entry ], [ %new1, %latch ]
  %s2 = phi i32 [ 0, %entry ], [ %new2, %latch ]
  %s3 = phi i32 [ 0, %entry ], [ %new3, %latch ]
  %s4 = phi i32 [ 0, %entry ], [ %new4, %latch ]
  %more = icmp ult i32 %round, %rounds
  br i1 %more, label %body, label %done
body:
  %scaled = mul i32 %round, 17
  %salt = add i32 %scaled, %i
  %value = xor i32 %input, %salt
  %global_value = add i32 %value, 74565
  %shift = add i32 %lane, %round
  %increment = add i32 %shift, 1
  store i32 %value, i32 addrspace(3)* %own, align 4
  store i32 %global_value, i32* %global, align 4
  call void asm sideeffect "barrier.sync $0;", "r,~{memory}"(i32 %id)
  %peer = urem i32 %increment, %threads
  %whole_shared = getelementptr [128 x i32], [128 x i32] addrspace(3)* @values, i32 0, i32 %peer
  %whole_value = load i32, i32 addrspace(3)* %whole_shared, align 4
  %whole_index = add i32 %base, %peer
  %whole_wide = zext i32 %whole_index to i64
  %whole_global = getelementptr i32, i32* %scratch, i64 %whole_wide
  %whole_gvalue = load i32, i32* %whole_global, align 4
  %whole_s = add i32 %s0, %whole_value
  %whole_g = add i32 %s1, %whole_gvalue
  call void asm sideeffect "barrier.sync $0;", "r,~{memory}"(i32 %id)
  br label %latch
latch:
  %new0 = add i32 %whole_s, 0
  %new1 = add i32 %whole_g, 0
  %new2 = add i32 %s2, 0
  %new3 = add i32 %s3, 0
  %new4 = add i32 %s4, 0
  %next = add i32 %round, 1
  br label %loop
done:
  %out_offset = mul i64 %offset, 5
  %index0 = add i64 %out_offset, 0
  %out0 = getelementptr i32, i32* %out, i64 %index0
  store i32 %s0, i32* %out0, align 4
  %index1 = add i64 %out_offset, 1
  %out1 = getelementptr i32, i32* %out, i64 %index1
  store i32 %s1, i32* %out1, align 4
  %index2 = add i64 %out_offset, 2
  %out2 = getelementptr i32, i32* %out, i64 %index2
  store i32 %s2, i32* %out2, align 4
  %index3 = add i64 %out_offset, 3
  %out3 = getelementptr i32, i32* %out, i64 %index3
  store i32 %s3, i32* %out3, align 4
  %index4 = add i64 %out_offset, 4
  %out4 = getelementptr i32, i32* %out, i64 %index4
  store i32 %s4, i32* %out4, align 4
  ret void
}
define void @named_groups(i32* %out, i32* %in, i32* %scratch, i32 %rounds, i32 %id, i32 %pattern) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %lane
  %offset = zext i32 %i to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %input = load i32, i32* %source, align 4
  %own = getelementptr [128 x i32], [128 x i32] addrspace(3)* @values, i32 0, i32 %lane
  %global = getelementptr i32, i32* %scratch, i64 %offset
  %group_base = and i32 %lane, -64
  %first = icmp eq i32 %group_base, 0
  %group_id = select i1 %first, i32 1, i32 15
  br label %loop
loop:
  %round = phi i32 [ 0, %entry ], [ %next, %latch ]
  %s0 = phi i32 [ 0, %entry ], [ %new0, %latch ]
  %s1 = phi i32 [ 0, %entry ], [ %new1, %latch ]
  %s2 = phi i32 [ 0, %entry ], [ %new2, %latch ]
  %s3 = phi i32 [ 0, %entry ], [ %new3, %latch ]
  %s4 = phi i32 [ 0, %entry ], [ %new4, %latch ]
  %more = icmp ult i32 %round, %rounds
  br i1 %more, label %body, label %done
body:
  %scaled = mul i32 %round, 17
  %salt = add i32 %scaled, %i
  %value = xor i32 %input, %salt
  %global_value = add i32 %value, 74565
  %shift = add i32 %lane, %round
  %increment = add i32 %shift, 1
  store i32 %value, i32 addrspace(3)* %own, align 4
  store i32 %global_value, i32* %global, align 4
  %rank = and i32 %lane, 63
  %moving = and i32 %round, 63
  %single = icmp eq i32 %rank, %moving
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
  %r2 = call i32 asm sideeffect "{ .reg .pred p; setp.ne.u32 p, $3, 0; barrier.red.popc.u32 $0, $1, $2, p; }", "=r,r,r,r,~{memory}"(i32 %group_id, i32 64, i32 %predicate)
  %r3 = call i32 asm sideeffect "{ .reg .pred p, q; setp.ne.u32 p, $3, 0; barrier.red.and.pred q, $1, $2, p; selp.u32 $0, 1, 0, q; }", "=r,r,r,r,~{memory}"(i32 %group_id, i32 64, i32 %predicate)
  %r4 = call i32 asm sideeffect "{ .reg .pred p, q; setp.ne.u32 p, $3, 0; barrier.red.or.pred q, $1, $2, p; selp.u32 $0, 1, 0, q; }", "=r,r,r,r,~{memory}"(i32 %group_id, i32 64, i32 %predicate)
  %peer_rank = and i32 %increment, 63
  %peer = add i32 %group_base, %peer_rank
  %group_shared = getelementptr [128 x i32], [128 x i32] addrspace(3)* @values, i32 0, i32 %peer
  %group_value = load i32, i32 addrspace(3)* %group_shared, align 4
  %group_index = add i32 %base, %peer
  %group_wide = zext i32 %group_index to i64
  %group_global = getelementptr i32, i32* %scratch, i64 %group_wide
  %group_gvalue = load i32, i32* %group_global, align 4
  %group_s = add i32 %s0, %group_value
  %group_g = add i32 %s1, %group_gvalue
  call void asm sideeffect "barrier.sync $0, $1;", "r,r,~{memory}"(i32 %group_id, i32 64)
  br label %latch
latch:
  %new0 = add i32 %group_s, 0
  %new1 = add i32 %group_g, 0
  %new2 = add i32 %s2, %r2
  %new3 = add i32 %s3, %r3
  %new4 = add i32 %s4, %r4
  %next = add i32 %round, 1
  br label %loop
done:
  %out_offset = mul i64 %offset, 5
  %index0 = add i64 %out_offset, 0
  %out0 = getelementptr i32, i32* %out, i64 %index0
  store i32 %s0, i32* %out0, align 4
  %index1 = add i64 %out_offset, 1
  %out1 = getelementptr i32, i32* %out, i64 %index1
  store i32 %s1, i32* %out1, align 4
  %index2 = add i64 %out_offset, 2
  %out2 = getelementptr i32, i32* %out, i64 %index2
  store i32 %s2, i32* %out2, align 4
  %index3 = add i64 %out_offset, 3
  %out3 = getelementptr i32, i32* %out, i64 %index3
  store i32 %s3, i32* %out3, align 4
  %index4 = add i64 %out_offset, 4
  %out4 = getelementptr i32, i32* %out, i64 %index4
  store i32 %s4, i32* %out4, align 4
  ret void
}
define void @named_handoff(i32* %out, i32* %in, i32* %scratch, i32 %rounds, i32 %id, i32 %pattern) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %lane
  %offset = zext i32 %i to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %input = load i32, i32* %source, align 4
  %own = getelementptr [128 x i32], [128 x i32] addrspace(3)* @values, i32 0, i32 %lane
  %global = getelementptr i32, i32* %scratch, i64 %offset
  %group_base = and i32 %lane, -64
  %first = icmp eq i32 %group_base, 0
  %group_id = select i1 %first, i32 1, i32 15
  br label %loop
loop:
  %round = phi i32 [ 0, %entry ], [ %next, %latch ]
  %s0 = phi i32 [ 0, %entry ], [ %new0, %latch ]
  %s1 = phi i32 [ 0, %entry ], [ %new1, %latch ]
  %s2 = phi i32 [ 0, %entry ], [ %new2, %latch ]
  %s3 = phi i32 [ 0, %entry ], [ %new3, %latch ]
  %s4 = phi i32 [ 0, %entry ], [ %new4, %latch ]
  %more = icmp ult i32 %round, %rounds
  br i1 %more, label %body, label %done
body:
  %scaled = mul i32 %round, 17
  %salt = add i32 %scaled, %i
  %value = xor i32 %input, %salt
  %global_value = add i32 %value, 74565
  %shift = add i32 %lane, %round
  %increment = add i32 %shift, 1
  %independent = icmp uge i32 %lane, 64
  br i1 %independent, label %separate, label %pair
pair:
  %producer = icmp ult i32 %lane, 32
  br i1 %producer, label %produce, label %consume
produce:
  store i32 %value, i32 addrspace(3)* %own, align 4
  store i32 %global_value, i32* %global, align 4
  call void asm sideeffect "barrier.arrive $0, $1;", "r,r,~{memory}"(i32 1, i32 64)
  call void asm sideeffect "barrier.sync $0, $1;", "r,r,~{memory}"(i32 2, i32 64)
  call void asm sideeffect "barrier.sync $0, $1;", "r,r,~{memory}"(i32 3, i32 64)
  br label %latch
consume:
  call void asm sideeffect "barrier.sync $0, $1;", "r,r,~{memory}"(i32 2, i32 64)
  call void asm sideeffect "barrier.sync $0, $1;", "r,r,~{memory}"(i32 1, i32 64)
  %consumer_peer = and i32 %increment, 31
  %consumer_shared = getelementptr [128 x i32], [128 x i32] addrspace(3)* @values, i32 0, i32 %consumer_peer
  %consumer_value = load i32, i32 addrspace(3)* %consumer_shared, align 4
  %consumer_index = add i32 %base, %consumer_peer
  %consumer_wide = zext i32 %consumer_index to i64
  %consumer_global = getelementptr i32, i32* %scratch, i64 %consumer_wide
  %consumer_gvalue = load i32, i32* %consumer_global, align 4
  %consumer_s = add i32 %s0, %consumer_value
  %consumer_g = add i32 %s1, %consumer_gvalue
  call void asm sideeffect "barrier.arrive $0, $1;", "r,r,~{memory}"(i32 3, i32 64)
  br label %latch
separate:
  store i32 %value, i32 addrspace(3)* %own, align 4
  store i32 %global_value, i32* %global, align 4
  call void asm sideeffect "barrier.sync $0, $1;", "r,r,~{memory}"(i32 15, i32 64)
  %separate_rank = and i32 %increment, 63
  %separate_peer = add i32 %separate_rank, 64
  %separate_shared = getelementptr [128 x i32], [128 x i32] addrspace(3)* @values, i32 0, i32 %separate_peer
  %separate_value = load i32, i32 addrspace(3)* %separate_shared, align 4
  %separate_index = add i32 %base, %separate_peer
  %separate_wide = zext i32 %separate_index to i64
  %separate_global = getelementptr i32, i32* %scratch, i64 %separate_wide
  %separate_gvalue = load i32, i32* %separate_global, align 4
  %separate_s = add i32 %s0, %separate_value
  %separate_g = add i32 %s1, %separate_gvalue
  call void asm sideeffect "barrier.sync $0, $1;", "r,r,~{memory}"(i32 15, i32 64)
  br label %latch
latch:
  %new0 = phi i32 [ %s0, %produce ], [ %consumer_s, %consume ], [ %separate_s, %separate ]
  %new1 = phi i32 [ %s1, %produce ], [ %consumer_g, %consume ], [ %separate_g, %separate ]
  %new2 = add i32 %s2, 0
  %new3 = add i32 %s3, 0
  %new4 = add i32 %s4, 0
  %next = add i32 %round, 1
  br label %loop
done:
  %out_offset = mul i64 %offset, 5
  %index0 = add i64 %out_offset, 0
  %out0 = getelementptr i32, i32* %out, i64 %index0
  store i32 %s0, i32* %out0, align 4
  %index1 = add i64 %out_offset, 1
  %out1 = getelementptr i32, i32* %out, i64 %index1
  store i32 %s1, i32* %out1, align 4
  %index2 = add i64 %out_offset, 2
  %out2 = getelementptr i32, i32* %out, i64 %index2
  store i32 %s2, i32* %out2, align 4
  %index3 = add i64 %out_offset, 3
  %out3 = getelementptr i32, i32* %out, i64 %index3
  store i32 %s3, i32* %out3, align 4
  %index4 = add i64 %out_offset, 4
  %out4 = getelementptr i32, i32* %out, i64 %index4
  store i32 %s4, i32* %out4, align 4
  ret void
}
!nvvm.annotations = !{!0, !1, !2}
!0 = !{void (i32*, i32*, i32*, i32, i32, i32)* @named_whole, !"kernel", i32 1}
!1 = !{void (i32*, i32*, i32*, i32, i32, i32)* @named_groups, !"kernel", i32 1}
!2 = !{void (i32*, i32*, i32*, i32, i32, i32)* @named_handoff, !"kernel", i32 1}
!nvvmir.version = !{!3}
!3 = !{i32 2, i32 0, i32 3, i32 1}
