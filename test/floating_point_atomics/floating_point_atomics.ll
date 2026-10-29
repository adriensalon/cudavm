; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@floating_counters = internal addrspace(3) global [2 x float] undef, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() convergent
declare float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)*, float)
declare double @llvm.nvvm.atomic.load.add.f64.p1f64(double addrspace(1)*, double)
declare float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)*, float)

define void @floating_global_add(i32* %out, float* %state, i32 %count, float %delta) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %active = icmp ult i32 %index, %count
  br i1 %active, label %add, label %done
add:
  %counter = addrspacecast float* %state to float addrspace(1)*
  %old = call float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* %counter, float %delta)
  %bits = bitcast float %old to i32
  %offset = zext i32 %index to i64
  %destination = getelementptr i32, i32* %out, i64 %offset
  store i32 %bits, i32* %destination, align 4
  %shadow = getelementptr float, float addrspace(1)* %counter, i64 1
  %discard = call float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* %shadow, float %delta)
  br label %done
done:
  ret void
}

define void @floating_global_double_add(i64* %out, double* %state, i32 %count, double %delta) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %active = icmp ult i32 %index, %count
  br i1 %active, label %add, label %done
add:
  %counter = addrspacecast double* %state to double addrspace(1)*
  %old = call double @llvm.nvvm.atomic.load.add.f64.p1f64(double addrspace(1)* %counter, double %delta)
  %bits = bitcast double %old to i64
  %offset = zext i32 %index to i64
  %destination = getelementptr i64, i64* %out, i64 %offset
  store i64 %bits, i64* %destination, align 8
  %shadow = getelementptr double, double addrspace(1)* %counter, i64 1
  %discard = call double @llvm.nvvm.atomic.load.add.f64.p1f64(double addrspace(1)* %shadow, double %delta)
  br label %done
done:
  ret void
}

define void @floating_shared_add(i32* %out, i32* %totals, i32 %count, float %seed, float %delta) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %counter = getelementptr [2 x float], [2 x float] addrspace(3)* @floating_counters, i32 0, i32 0
  %shadow = getelementptr float, float addrspace(3)* %counter, i32 1
  %leader = icmp eq i32 %lane, 0
  br i1 %leader, label %initialize, label %ready
initialize:
  store float %seed, float addrspace(3)* %counter, align 4
  store float %seed, float addrspace(3)* %shadow, align 4
  br label %ready
ready:
  call void @llvm.nvvm.barrier0()
  %base = mul i32 %block, %lanes
  %index = add i32 %base, %lane
  %active = icmp ult i32 %index, %count
  br i1 %active, label %add, label %complete
add:
  %old = call float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* %counter, float %delta)
  %discard = call float @llvm.nvvm.atomic.load.add.f32.p3f32(float addrspace(3)* %shadow, float %delta)
  %bits = bitcast float %old to i32
  %offset = zext i32 %index to i64
  %destination = getelementptr i32, i32* %out, i64 %offset
  store i32 %bits, i32* %destination, align 4
  br label %complete
complete:
  call void @llvm.nvvm.barrier0()
  br i1 %leader, label %snapshot, label %done
snapshot:
  %first = load float, float addrspace(3)* %counter, align 4
  %second = load float, float addrspace(3)* %shadow, align 4
  %first_bits = bitcast float %first to i32
  %second_bits = bitcast float %second to i32
  %snapshot_index = mul i32 %block, 2
  %snapshot_offset = zext i32 %snapshot_index to i64
  %first_destination = getelementptr i32, i32* %totals, i64 %snapshot_offset
  %second_destination = getelementptr i32, i32* %first_destination, i64 1
  store i32 %first_bits, i32* %first_destination, align 4
  store i32 %second_bits, i32* %second_destination, align 4
  br label %done
done:
  ret void
}

define void @floating_exchange(i32* %out, float* %state, i32 %replacement) {
  %bits = bitcast float* %state to i32*
  %global = addrspacecast i32* %bits to i32 addrspace(1)*
  %old = atomicrmw xchg i32 addrspace(1)* %global, i32 %replacement syncscope("device") monotonic
  store i32 %old, i32* %out, align 4
  ret void
}
!nvvm.annotations = !{!0, !1, !2, !3}
!0 = !{void (i32*, float*, i32, float)* @floating_global_add, !"kernel", i32 1}
!1 = !{void (i64*, double*, i32, double)* @floating_global_double_add, !"kernel", i32 1}
!2 = !{void (i32*, i32*, i32, float, float)* @floating_shared_add, !"kernel", i32 1}
!3 = !{void (i32*, float*, i32)* @floating_exchange, !"kernel", i32 1}
!nvvmir.version = !{!4}
!4 = !{i32 2, i32 0, i32 3, i32 1}
