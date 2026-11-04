; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@values = external addrspace(3) global [0 x i32], align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare void @llvm.nvvm.barrier0() convergent

define void @kernel_node_write(i32* %output, i32 %count, i32 %seed) {
entry:
  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %tz = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %nx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %ny = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %nz = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %bz = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %gx = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %gy = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %ly = mul i32 %tz, %ny
  %ly1 = add i32 %ly, %ty
  %lx = mul i32 %ly1, %nx
  %lane = add i32 %lx, %tx
  %nxy = mul i32 %nx, %ny
  %threads = mul i32 %nxy, %nz
  %by0 = mul i32 %bz, %gy
  %by1 = add i32 %by0, %by
  %bx0 = mul i32 %by1, %gx
  %block = add i32 %bx0, %bx
  %base = mul i32 %block, %threads
  %i = add i32 %base, %lane
  %v0 = mul i32 %tx, 3
  %v1 = add i32 %seed, %v0
  %v2 = mul i32 %ty, 5
  %v3 = add i32 %v1, %v2
  %v4 = mul i32 %tz, 7
  %v5 = add i32 %v3, %v4
  %v6 = mul i32 %bx, 11
  %v7 = add i32 %v5, %v6
  %v8 = mul i32 %by, 13
  %v9 = add i32 %v7, %v8
  %v10 = mul i32 %bz, 17
  %v11 = add i32 %v9, %v10
  %v12 = mul i32 %nx, 19
  %v13 = add i32 %v11, %v12
  %v14 = mul i32 %ny, 23
  %v15 = add i32 %v13, %v14
  %v16 = mul i32 %nz, 29
  %value = add i32 %v15, %v16
  %own = getelementptr [0 x i32], [0 x i32] addrspace(3)* @values, i32 0, i32 %lane
  store i32 %value, i32 addrspace(3)* %own, align 4
  call void @llvm.nvvm.barrier0()
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %last = sub i32 %threads, 1
  %peer_index = sub i32 %last, %lane
  %peer = getelementptr [0 x i32], [0 x i32] addrspace(3)* @values, i32 0, i32 %peer_index
  %result = load i32, i32 addrspace(3)* %peer, align 4
  %index = zext i32 %i to i64
  %destination = getelementptr i32, i32* %output, i64 %index
  store i32 %result, i32* %destination, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32, i32)* @kernel_node_write, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
