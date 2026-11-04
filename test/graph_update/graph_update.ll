; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define void @update_write(i32* %output, i32 %count, i32 %value) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %thread
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %index = zext i32 %i to i64
  %pointer = getelementptr i32, i32* %output, i64 %index
  %rank = mul i32 %i, 7
  %result = add i32 %value, %rank
  store i32 %result, i32* %pointer, align 4
  br label %done
done:
  ret void
}
define void @update_twist(i32* %output, i32 %count, i32 %value) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %thread
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %index = zext i32 %i to i64
  %pointer = getelementptr i32, i32* %output, i64 %index
  %product = mul i32 %i, 13
  %rank = add i32 %product, -1515890086
  %result = xor i32 %value, %rank
  store i32 %result, i32* %pointer, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0, !1}
!0 = !{void (i32*, i32, i32)* @update_write, !"kernel", i32 1}
!1 = !{void (i32*, i32, i32)* @update_twist, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 1}
