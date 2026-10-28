; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define void @mixed_width_access(i8* %out, i8* %in, i32 %count) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %thread
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %wide_index = zext i32 %i to i64
  %offset = mul i64 %wide_index, 16
  %source = getelementptr i8, i8* %in, i64 %offset
  %target = getelementptr i8, i8* %out, i64 %offset
  %x8 = load i8, i8* %source, align 1
  %y8 = xor i8 %x8, -91
  %target1 = getelementptr i8, i8* %target, i64 1
  store i8 %y8, i8* %target1, align 1
  %source2 = getelementptr i8, i8* %source, i64 2
  %target2 = getelementptr i8, i8* %target, i64 2
  %source16 = bitcast i8* %source2 to i16*
  %target16 = bitcast i8* %target2 to i16*
  %x16 = load i16, i16* %source16, align 2
  %y16 = xor i16 %x16, 23205
  store i16 %y16, i16* %target16, align 2
  %source4 = getelementptr i8, i8* %source, i64 4
  %target4 = getelementptr i8, i8* %target, i64 4
  %source32 = bitcast i8* %source4 to i32*
  %target32 = bitcast i8* %target4 to i32*
  %x32 = load i32, i32* %source32, align 4
  %y32 = add i32 %x32, 16909061
  store i32 %y32, i32* %target32, align 4
  %source8 = getelementptr i8, i8* %source, i64 8
  %target8 = getelementptr i8, i8* %target, i64 8
  %source64 = bitcast i8* %source8 to i64*
  %target64 = bitcast i8* %target8 to i64*
  %x64 = load i64, i64* %source64, align 8
  %y64 = xor i64 %x64, 81985529216486895
  store i64 %y64, i64* %target64, align 8
  br label %done
done:
  ret void
}

define void @neighbor_byte_access(i8* %out, i8* %in, i32 %count) {
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
  %source = getelementptr i8, i8* %in, i64 %offset
  %target = getelementptr i8, i8* %out, i64 %offset
  %value = load i8, i8* %source, align 1
  %result = xor i8 %value, 90
  store i8 %result, i8* %target, align 1
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0, !1}
!0 = !{void (i8*, i8*, i32)* @mixed_width_access, !"kernel", i32 1}
!1 = !{void (i8*, i8*, i32)* @neighbor_byte_access, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 1}
