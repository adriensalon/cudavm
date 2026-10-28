; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@mutable_words = addrspace(1) global [19 x i32] [i32 324508639, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 3970458656], align 4
@constant_words = addrspace(4) global [7 x i32] [i32 0, i32 1, i32 2147483648, i32 4294967295, i32 305419896, i32 2882400001, i32 17], align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define void @read_symbols(i32* %out, i32 %count) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %i = add i32 %base, %lane
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %write, label %done
write:
  %remainder = urem i32 %i, 17
  %slot = add i32 %remainder, 1
  %constant_slot = urem i32 %i, 7
  %global = getelementptr [19 x i32], [19 x i32] addrspace(1)* @mutable_words, i32 0, i32 %slot
  %constant = getelementptr [7 x i32], [7 x i32] addrspace(4)* @constant_words, i32 0, i32 %constant_slot
  %value = load i32, i32 addrspace(1)* %global, align 4
  %other = load i32, i32 addrspace(4)* %constant, align 4
  %mixed = xor i32 %value, %other
  %result = add i32 %mixed, %i
  %generic = addrspacecast i32 addrspace(1)* %global to i32*
  %constant_generic = addrspacecast i32 addrspace(4)* %constant to i32*
  %address = ptrtoint i32* %generic to i64
  %constant_address = ptrtoint i32* %constant_generic to i64
  %low = trunc i64 %address to i32
  %shift = lshr i64 %address, 32
  %high = trunc i64 %shift to i32
  %constant_low = trunc i64 %constant_address to i32
  %constant_shift = lshr i64 %constant_address, 32
  %constant_high = trunc i64 %constant_shift to i32
  %index = zext i32 %i to i64
  %offset = mul i64 %index, 7
  %p0 = getelementptr i32, i32* %out, i64 %offset
  %p1 = getelementptr i32, i32* %p0, i64 1
  %p2 = getelementptr i32, i32* %p0, i64 2
  %p3 = getelementptr i32, i32* %p0, i64 3
  %p4 = getelementptr i32, i32* %p0, i64 4
  %p5 = getelementptr i32, i32* %p0, i64 5
  %p6 = getelementptr i32, i32* %p0, i64 6
  store i32 %value, i32* %p0, align 4
  store i32 %other, i32* %p1, align 4
  store i32 %result, i32* %p2, align 4
  store i32 %low, i32* %p3, align 4
  store i32 %high, i32* %p4, align 4
  store i32 %constant_low, i32* %p5, align 4
  store i32 %constant_high, i32* %p6, align 4
  br label %done
done:
  ret void
}
define void @update_symbols(i32 %salt) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %lanes
  %i = add i32 %base, %lane
  %valid = icmp ult i32 %i, 17
  br i1 %valid, label %update, label %done
update:
  %slot = add i32 %i, 1
  %constant_slot = urem i32 %i, 7
  %global = getelementptr [19 x i32], [19 x i32] addrspace(1)* @mutable_words, i32 0, i32 %slot
  %constant = getelementptr [7 x i32], [7 x i32] addrspace(4)* @constant_words, i32 0, i32 %constant_slot
  %value = load i32, i32 addrspace(1)* %global, align 4
  %other = load i32, i32 addrspace(4)* %constant, align 4
  %operand = xor i32 %salt, %other
  %result = add i32 %value, %operand
  store i32 %result, i32 addrspace(1)* %global, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0, !1}
!0 = !{void (i32*, i32)* @read_symbols, !"kernel", i32 1}
!1 = !{void (i32)* @update_symbols, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 1}
