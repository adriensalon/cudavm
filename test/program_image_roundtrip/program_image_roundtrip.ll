; Independently authored NVVM IR 2.0 equivalent of the module boundary kernels.
target datalayout = "e-p:64:64-i1:8-i8:8-i16:16-i32:32-i64:64-i128:128-f32:32-f64:64-v16:16-v32:32-v64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@module_seed = addrspace(1) global i32 13, align 4
@module_bias = addrspace(4) constant i32 29, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
define void @module_alpha(i32* %out, i32* %in, i32 %count) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %index = add i32 %base, %thread
  %active = icmp ult i32 %index, %count
  br i1 %active, label %work, label %done
work:
  %wide = zext i32 %index to i64
  %source = getelementptr i32, i32* %in, i64 %wide
  %input = load i32, i32* %source, align 4
  %seed = load i32, i32 addrspace(1)* @module_seed, align 4
  %product = mul i32 %input, 3
  %sum = add i32 %product, %seed
  %result = add i32 %sum, %index
  %destination = getelementptr i32, i32* %out, i64 %wide
  store i32 %result, i32* %destination, align 4
  br label %done
done:
  ret void
}
define void @module_beta(i32* %out, i32* %in, i32 %count) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %index = add i32 %base, %thread
  %active = icmp ult i32 %index, %count
  br i1 %active, label %work, label %done
work:
  %wide = zext i32 %index to i64
  %source = getelementptr i32, i32* %in, i64 %wide
  %input = load i32, i32* %source, align 4
  %seed = load i32, i32 addrspace(1)* @module_seed, align 4
  %bias = load i32, i32 addrspace(4)* @module_bias, align 4
  %bits = xor i32 %input, %bias
  %result = add i32 %bits, %seed
  %destination = getelementptr i32, i32* %out, i64 %wide
  store i32 %result, i32* %destination, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0, !1}
!0 = !{void (i32*, i32*, i32)* @module_alpha, !"kernel", i32 1}
!1 = !{void (i32*, i32*, i32)* @module_beta, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 1}
