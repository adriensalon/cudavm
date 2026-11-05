; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
define internal i32 @frame_leaf(i32 %seed) noinline {
 %frame = alloca [3 x i32], align 16
 %p0 = getelementptr [3 x i32], [3 x i32]* %frame, i32 0, i32 0
 %p1 = getelementptr [3 x i32], [3 x i32]* %frame, i32 0, i32 1
 %p2 = getelementptr [3 x i32], [3 x i32]* %frame, i32 0, i32 2
 %b = xor i32 %seed, -1985229329
 %c = add i32 %seed, 124076833
 store volatile i32 %seed, i32* %p0, align 4
 store volatile i32 %b, i32* %p1, align 4
 store volatile i32 %c, i32* %p2, align 4
 %a0 = load volatile i32, i32* %p0, align 4
 %a1 = load volatile i32, i32* %p1, align 4
 %a2 = load volatile i32, i32* %p2, align 4
 %scaled = mul i32 %a0, 3
 %added = add i32 %scaled, %a1
 %result = xor i32 %added, %a2
 ret i32 %result
}
define internal i32 @frame_middle(i32 %seed, i32 %rounds) noinline {
entry:
 %frame = alloca [5 x i32], align 16
 br label %initialize
initialize:
 %j = phi i32 [0, %entry], [%next_j, %initialize]
 %scaled = mul i32 %j, 16909060
 %initial = add i32 %seed, %scaled
 %p = getelementptr [5 x i32], [5 x i32]* %frame, i32 0, i32 %j
 store volatile i32 %initial, i32* %p, align 4
 %next_j = add i32 %j, 1
 %init_more = icmp ult i32 %next_j, 5
 br i1 %init_more, label %initialize, label %test
test:
 %r = phi i32 [0, %initialize], [%next_r, %body]
 %more = icmp ult i32 %r, %rounds
 br i1 %more, label %body, label %checksum_start
body:
 %rs = mul i32 %r, 7
 %arg = add i32 %seed, %rs
 %returned = call i32 @frame_leaf(i32 %arg)
 %v = add i32 %returned, %r
 %index0 = add i32 %seed, %r
 %index = urem i32 %index0, 5
 %q = getelementptr [5 x i32], [5 x i32]* %frame, i32 0, i32 %index
 %old = load volatile i32, i32* %q, align 4
 %updated = xor i32 %old, %v
 store volatile i32 %updated, i32* %q, align 4
 %next_r = add i32 %r, 1
 br label %test
checksum_start:
 br label %checksum
checksum:
 %k = phi i32 [0, %checksum_start], [%next_k, %checksum]
 %hash = phi i32 [0, %checksum_start], [%next_hash, %checksum]
 %hp = getelementptr [5 x i32], [5 x i32]* %frame, i32 0, i32 %k
 %hv = load volatile i32, i32* %hp, align 4
 %hm = mul i32 %hash, 33
 %next_hash = add i32 %hm, %hv
 %next_k = add i32 %k, 1
 %hash_more = icmp ult i32 %next_k, 5
 br i1 %hash_more, label %checksum, label %done
done:
 ret i32 %next_hash
}
define void @frame_results(i32* %out, i32* %in, i32 %count, i32 %rounds) {
entry:
 %caller = alloca [3 x i32], align 16
 %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %base = mul i32 %block, %threads
 %i = add i32 %base, %thread
 %valid = icmp ult i32 %i, %count
 br i1 %valid, label %body, label %done
body:
 %offset = zext i32 %i to i64
 %input = getelementptr i32, i32* %in, i64 %offset
 %seed = load i32, i32* %input, align 4
 %p0 = getelementptr [3 x i32], [3 x i32]* %caller, i32 0, i32 0
 %p1 = getelementptr [3 x i32], [3 x i32]* %caller, i32 0, i32 1
 %p2 = getelementptr [3 x i32], [3 x i32]* %caller, i32 0, i32 2
 %x = add i32 %seed, 17
 %y = xor i32 %seed, 324508639
 %z = add i32 %i, -324508640
 store volatile i32 %x, i32* %p0, align 4
 store volatile i32 %y, i32* %p1, align 4
 store volatile i32 %z, i32* %p2, align 4
 %extra = urem i32 %i, 3
 %first_rounds = add i32 %rounds, %extra
 %first = call i32 @frame_middle(i32 %seed, i32 %first_rounds)
 %second_seed = xor i32 %seed, %first
 %second_rounds = add i32 %rounds, 2
 %second = call i32 @frame_middle(i32 %second_seed, i32 %second_rounds)
 %a = load volatile i32, i32* %p0, align 4
 %b = load volatile i32, i32* %p1, align 4
 %c = load volatile i32, i32* %p2, align 4
 %row = mul i64 %offset, 5
 %output = getelementptr i32, i32* %out, i64 %row
 %o1 = getelementptr i32, i32* %output, i32 1
 %o2 = getelementptr i32, i32* %output, i32 2
 %o3 = getelementptr i32, i32* %output, i32 3
 %o4 = getelementptr i32, i32* %output, i32 4
 store i32 %first, i32* %output, align 4
 store i32 %second, i32* %o1, align 4
 store i32 %a, i32* %o2, align 4
 store i32 %b, i32* %o3, align 4
 store i32 %c, i32* %o4, align 4
 br label %done
done:
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32, i32)* @frame_results, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
