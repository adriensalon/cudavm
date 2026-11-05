; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
define internal i32 @record_checksum(i64 %value, i32 %tag, i32 %salt) {
 %low = trunc i64 %value to i32
 %upper = lshr i64 %value, 32
 %high = trunc i64 %upper to i32
 %words = xor i32 %low, %high
 %tagged = xor i32 %words, %tag
 %result = xor i32 %tagged, %salt
 ret i32 %result
}
define void @object_results(i32* %out, i32* %in, i32 %count) {
entry:
 %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %base = mul i32 %block, %threads
 %i = add i32 %base, %thread
 %valid = icmp ult i32 %i, %count
 br i1 %valid, label %body, label %done
body:
 %index = zext i32 %i to i64
 %source = getelementptr i32, i32* %in, i64 %index
 %seed = load i32, i32* %source, align 4
 %seed64 = zext i32 %seed to i64
 %initial_high = shl i64 %seed64, 32
 %low32 = xor i32 %seed, -1985229329
 %initial_low = zext i32 %low32 to i64
 %initial = or i64 %initial_high, %initial_low
 %initial_tag = add i32 %seed, 17
 %initial_salt = xor i32 %seed, 324508639
 %delta_a = add i32 %seed, 5
 %da64 = zext i32 %delta_a to i64
 %da_high = shl i64 %da64, 32
 %da = or i64 %da_high, %da64
 %original_value = add i64 %initial, %da
 %original_tag = xor i32 %initial_tag, %delta_a
 %da_salt = mul i32 %delta_a, 7
 %original_salt = add i32 %initial_salt, %da_salt
 %copied_value = xor i64 %initial, 4294967297
 %copied_tag = add i32 %initial_tag, 3
 %copied_salt = xor i32 %initial_salt, -1
 %delta_b = xor i32 %seed, 270544960
 %db64 = zext i32 %delta_b to i64
 %db_high = shl i64 %db64, 32
 %db = or i64 %db_high, %db64
 %copy_value = add i64 %copied_value, %db
 %copy_tag = xor i32 %copied_tag, %delta_b
 %db_salt = mul i32 %delta_b, 7
 %copy_salt = add i32 %copied_salt, %db_salt
 %construction = mul i32 %seed, 33
 %copied = xor i32 %construction, -1073624322
 %copy_check = call i32 @record_checksum(i64 %copy_value, i32 %copy_tag, i32 %copy_salt)
 %copy_step = mul i32 %copied, 33
 %copy_destroyed = xor i32 %copy_step, %copy_check
 %original_check = call i32 @record_checksum(i64 %original_value, i32 %original_tag, i32 %original_salt)
 %original_step = mul i32 %copy_destroyed, 33
 %journal = xor i32 %original_step, %original_check
 %row = mul i64 %index, 15
 %destination = getelementptr i32, i32* %out, i64 %row
 store i32 %journal, i32* %destination, align 4
 %a0 = trunc i64 %original_value to i32
 %a_upper = lshr i64 %original_value, 32
 %a1 = trunc i64 %a_upper to i32
 %b0 = trunc i64 %copy_value to i32
 %b_upper = lshr i64 %copy_value, 32
 %b1 = trunc i64 %b_upper to i32
 %o1 = getelementptr i32, i32* %destination, i32 1
 %o2 = getelementptr i32, i32* %destination, i32 2
 %o3 = getelementptr i32, i32* %destination, i32 3
 %o4 = getelementptr i32, i32* %destination, i32 4
 %o5 = getelementptr i32, i32* %destination, i32 5
 %o6 = getelementptr i32, i32* %destination, i32 6
 %o7 = getelementptr i32, i32* %destination, i32 7
 %o8 = getelementptr i32, i32* %destination, i32 8
 %o9 = getelementptr i32, i32* %destination, i32 9
 %o10 = getelementptr i32, i32* %destination, i32 10
 %o11 = getelementptr i32, i32* %destination, i32 11
 %o12 = getelementptr i32, i32* %destination, i32 12
 %o13 = getelementptr i32, i32* %destination, i32 13
 %o14 = getelementptr i32, i32* %destination, i32 14
 store i32 %a0, i32* %o1, align 4
 store i32 %a1, i32* %o2, align 4
 store i32 %original_tag, i32* %o3, align 4
 store i32 %original_salt, i32* %o4, align 4
 store i32 %b0, i32* %o5, align 4
 store i32 %b1, i32* %o6, align 4
 store i32 %copy_tag, i32* %o7, align 4
 store i32 %copy_salt, i32* %o8, align 4
 store i32 32, i32* %o9, align 4
 store i32 16, i32* %o10, align 4
 store i32 0, i32* %o11, align 4
 store i32 8, i32* %o12, align 4
 store i32 12, i32* %o13, align 4
 store i32 16, i32* %o14, align 4
 br label %done
done:
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32)* @object_results, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
