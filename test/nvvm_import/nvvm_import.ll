; Independently authored opaque-pointer NVVM IR 2.0 boundary fixture.
target datalayout = "e-p:64:64-i1:8-i8:8-i16:16-i32:32-i64:64-i128:128-f32:32-f64:64-v16:16-v32:32-v64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
define void @nvvm_import_control(ptr addrspace(1) %out) {
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %value = add i32 %lane, 37
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{ptr @nvvm_import_control, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
