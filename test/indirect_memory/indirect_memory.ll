; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define void @follow_pointers(i32* %out, i32*** %roots, i32* %selection, i32 %count) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %thread
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %tables, label %done
tables:
  %offset = zext i32 %i to i64
  %root_shift = lshr i32 %i, 2
  %root_index = and i32 %root_shift, 1
  %root_offset = zext i32 %root_index to i64
  %root_slot = getelementptr i32**, i32*** %roots, i64 %root_offset
  %table = load i32**, i32*** %root_slot, align 8
  %selection_slot = getelementptr i32, i32* %selection, i64 %offset
  %selector = load i32, i32* %selection_slot, align 4
  %entry_index = and i32 %selector, 3
  %entry_offset = zext i32 %entry_index to i64
  %leaf_slot = getelementptr i32*, i32** %table, i64 %entry_offset
  %data = load i32*, i32** %leaf_slot, align 8
  %present = icmp ne i32* %data, null
  br i1 %present, label %write, label %done
write:
  %element_shift = lshr i32 %selector, 2
  %element = and i32 %element_shift, 7
  %element_offset = zext i32 %element to i64
  %source = getelementptr i32, i32* %data, i64 %element_offset
  %value = load i32, i32* %source, align 4
  %addend = mul i32 %i, 257
  %result = add i32 %value, %addend
  %address = ptrtoint i32* %data to i64
  %low = trunc i64 %address to i32
  %shifted = lshr i64 %address, 32
  %high = trunc i64 %shifted to i32
  %destination_offset = mul i64 %offset, 3
  %p0 = getelementptr i32, i32* %out, i64 %destination_offset
  %p1 = getelementptr i32, i32* %p0, i64 1
  %p2 = getelementptr i32, i32* %p0, i64 2
  store i32 %result, i32* %p0, align 4
  store i32 %low, i32* %p1, align 4
  store i32 %high, i32* %p2, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32***, i32*, i32)* @follow_pointers, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
