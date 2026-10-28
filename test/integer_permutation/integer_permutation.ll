; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.prmt(i32, i32, i32)

define void @permute_bytes(i32* %out, i32* %in, i32 %count) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %thread
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %offset = zext i32 %i to i64
  %source_offset = mul i64 %offset, 3
  %p0 = getelementptr i32, i32* %in, i64 %source_offset
  %p1 = getelementptr i32, i32* %p0, i64 1
  %p2 = getelementptr i32, i32* %p0, i64 2
  %low = load i32, i32* %p0, align 4
  %high = load i32, i32* %p1, align 4
  %selector = load i32, i32* %p2, align 4
  %masked = and i32 %selector, 30583
  %bytes = call i32 @llvm.nvvm.prmt(i32 %low, i32 %high, i32 %masked)
  %raw = call i32 @llvm.nvvm.prmt(i32 %low, i32 %high, i32 %selector)
  %target_offset = mul i64 %offset, 2
  %target0 = getelementptr i32, i32* %out, i64 %target_offset
  %target1 = getelementptr i32, i32* %target0, i64 1
  store i32 %bytes, i32* %target0, align 4
  store i32 %raw, i32* %target1, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32)* @permute_bytes, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
