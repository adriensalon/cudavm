; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@values = internal addrspace(3) global [96 x i32] undef, align 4
declare void @llvm.nvvm.bar.warp.sync(i32) convergent
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
define void @warp_publish(i32* %out, i32* %in, i32* %scratch, i32 %mask, i32 %rounds, i32 %mode) {
entry:
 %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %ty = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
 %tz = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
 %nx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %ny = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
 %nz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
 %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %zbase = mul i32 %tz, %ny
 %yz = add i32 %zbase, %ty
 %yzbase = mul i32 %yz, %nx
 %thread = add i32 %yzbase, %tx
 %xy = mul i32 %nx, %ny
 %threads = mul i32 %xy, %nz
 %base = mul i32 %block, %threads
 %i = add i32 %base, %thread
 %lane = and i32 %thread, 31
 %warp_base = and i32 %thread, -32
 %remaining = sub i32 %threads, %warp_base
 %partial = icmp ult i32 %remaining, 32
 %live = select i1 %partial, i32 %remaining, i32 32
 %shift = sub i32 32, %live
 %live_mask = lshr i32 -1, %shift
 %live_members = and i32 %mask, %live_mask
 %split = icmp eq i32 %mode, 1
 %parity = and i32 %lane, 1
 %odd = icmp ne i32 %parity, 0
 %group = select i1 %odd, i32 -1431655766, i32 1431655765
 %group_members = and i32 %live_members, %group
 %members = select i1 %split, i32 %group_members, i32 %live_members
 %bit = shl i32 1, %lane
 %own_bit = and i32 %members, %bit
 %active = icmp ne i32 %own_bit, 0
 br i1 %active, label %peer_init, label %done
peer_init:
 %first_peer = add i32 %lane, 1
 br label %peer_loop
peer_loop:
 %candidate = phi i32 [%first_peer, %peer_init], [%next_peer, %peer_loop]
 %peer_lane = and i32 %candidate, 31
 %peer_bit = shl i32 1, %peer_lane
 %peer_member = and i32 %members, %peer_bit
 %found = icmp ne i32 %peer_member, 0
 %next_peer = add i32 %peer_lane, 1
 br i1 %found, label %initialize, label %peer_loop
initialize:
 %peer = add i32 %warp_base, %peer_lane
 %peer_i = add i32 %base, %peer
 %offset = zext i32 %i to i64
 %peer_offset = zext i32 %peer_i to i64
 %source = getelementptr i32, i32* %in, i64 %offset
 %global_own = getelementptr i32, i32* %scratch, i64 %offset
 %global_peer = getelementptr i32, i32* %scratch, i64 %peer_offset
 %shared_own = getelementptr [96 x i32], [96 x i32] addrspace(3)* @values, i32 0, i32 %thread
 %shared_peer = getelementptr [96 x i32], [96 x i32] addrspace(3)* @values, i32 0, i32 %peer
 %warp = lshr i32 %thread, 5
 %warp_extra = urem i32 %warp, 3
 %extra = add i32 %warp_extra, %parity
 %nonzero = icmp ne i32 %rounds, 0
 %different = and i1 %split, %nonzero
 %longer = add i32 %rounds, %extra
 %limit = select i1 %different, i32 %longer, i32 %rounds
 %divergent = icmp eq i32 %mode, 2
 %odd_arm = and i1 %divergent, %odd
 %ticks = urem i32 %lane, 3
 %n = select i1 %divergent, i32 %ticks, i32 0
 %n1 = add i32 %n, 1
 %product = mul i32 %n, %n1
 %delta = lshr i32 %product, 1
 %output_base = mul i64 %offset, 16
 br label %round_test
round_test:
 %round = phi i32 [0, %initialize], [%next_round, %read_peer]
 %more = icmp ult i32 %round, %limit
 br i1 %more, label %publish, label %done
publish:
 %input = load i32, i32* %source, align 4
 %scaled = mul i32 %round, 66051
 %salt = add i32 %scaled, %i
 %value = xor i32 %input, %salt
 %global_value = add i32 %value, -1640531527
 store volatile i32 %value, i32 addrspace(3)* %shared_own, align 4
 store i32 %global_value, i32* %global_own, align 4
 br i1 %odd_arm, label %tick_loop, label %even_arm
tick_loop:
 %tick = phi i32 [0, %publish], [%next_tick, %tick_body]
 %tick_more = icmp ult i32 %tick, %ticks
 br i1 %tick_more, label %tick_body, label %odd_wait
tick_body:
 %tick_delta = add i32 %tick, 1
 %old_shared = load volatile i32, i32 addrspace(3)* %shared_own, align 4
 %new_shared = add i32 %old_shared, %tick_delta
 store volatile i32 %new_shared, i32 addrspace(3)* %shared_own, align 4
 %old_global = load i32, i32* %global_own, align 4
 %new_global = add i32 %old_global, %tick_delta
 store i32 %new_global, i32* %global_own, align 4
 %next_tick = add i32 %tick, 1
 br label %tick_loop
odd_wait:
 call void @llvm.nvvm.bar.warp.sync(i32 %members)
 br label %read_peer
even_arm:
 %even_old_shared = load volatile i32, i32 addrspace(3)* %shared_own, align 4
 %even_shared = add i32 %even_old_shared, %delta
 store volatile i32 %even_shared, i32 addrspace(3)* %shared_own, align 4
 %even_old_global = load i32, i32* %global_own, align 4
 %even_global = add i32 %even_old_global, %delta
 store i32 %even_global, i32* %global_own, align 4
 call void @llvm.nvvm.bar.warp.sync(i32 %members)
 br label %read_peer
read_peer:
 %shared_read = load volatile i32, i32 addrspace(3)* %shared_peer, align 4
 %global_read = load i32, i32* %global_peer, align 4
 %wide_round = zext i32 %round to i64
 %round_offset = mul i64 %wide_round, 2
 %output_offset = add i64 %output_base, %round_offset
 %p0 = getelementptr i32, i32* %out, i64 %output_offset
 %p1 = getelementptr i32, i32* %p0, i64 1
 store i32 %shared_read, i32* %p0, align 4
 store i32 %global_read, i32* %p1, align 4
 call void @llvm.nvvm.bar.warp.sync(i32 %members)
 %next_round = add i32 %round, 1
 br label %round_test
done:
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32*, i32, i32, i32)* @warp_publish, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
