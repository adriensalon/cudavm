; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.match.any.sync.i32(i32, i32) convergent
declare {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32, i32) convergent
declare i32 @llvm.nvvm.match.any.sync.i64(i32, i64) convergent
declare {i32, i1} @llvm.nvvm.match.all.sync.i64p(i32, i64) convergent
define void @masked_match(i32* %out, i64* %in, i32 %mask) {
entry:
 %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %base = mul i32 %block, %threads
 %i = add i32 %base, %thread
 %lane = and i32 %thread, 31
 %bit = shl i32 1, %lane
 %member_bit = and i32 %mask, %bit
 %member = icmp ne i32 %member_bit, 0
 br i1 %member, label %initialize, label %done
initialize:
 %offset = zext i32 %i to i64
 %source = getelementptr i64, i64* %in, i64 %offset
 %value = load i64, i64* %source, align 8
 %low = trunc i64 %value to i32
 %output_base = mul i64 %offset, 18
 %any0 = call i32 @llvm.nvvm.match.any.sync.i32(i32 %mask, i32 %low)
 %pair0 = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 %mask, i32 %low)
 %all0 = extractvalue {i32, i1} %pair0, 0
 %pred0 = extractvalue {i32, i1} %pair0, 1
 %word0 = zext i1 %pred0 to i32
 %off0 = add i64 %output_base, 0
 %dst0 = getelementptr i32, i32* %out, i64 %off0
 store i32 %any0, i32* %dst0, align 4
 %off1 = add i64 %output_base, 1
 %dst1 = getelementptr i32, i32* %out, i64 %off1
 store i32 %all0, i32* %dst1, align 4
 %off2 = add i64 %output_base, 2
 %dst2 = getelementptr i32, i32* %out, i64 %off2
 store i32 %word0, i32* %dst2, align 4
 %any1 = call i32 @llvm.nvvm.match.any.sync.i32(i32 %mask, i32 %low)
 %pair1 = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 %mask, i32 %low)
 %all1 = extractvalue {i32, i1} %pair1, 0
 %pred1 = extractvalue {i32, i1} %pair1, 1
 %word1 = zext i1 %pred1 to i32
 %off3 = add i64 %output_base, 3
 %dst3 = getelementptr i32, i32* %out, i64 %off3
 store i32 %any1, i32* %dst3, align 4
 %off4 = add i64 %output_base, 4
 %dst4 = getelementptr i32, i32* %out, i64 %off4
 store i32 %all1, i32* %dst4, align 4
 %off5 = add i64 %output_base, 5
 %dst5 = getelementptr i32, i32* %out, i64 %off5
 store i32 %word1, i32* %dst5, align 4
 %any2 = call i32 @llvm.nvvm.match.any.sync.i64(i32 %mask, i64 %value)
 %pair2 = call {i32, i1} @llvm.nvvm.match.all.sync.i64p(i32 %mask, i64 %value)
 %all2 = extractvalue {i32, i1} %pair2, 0
 %pred2 = extractvalue {i32, i1} %pair2, 1
 %word2 = zext i1 %pred2 to i32
 %off6 = add i64 %output_base, 6
 %dst6 = getelementptr i32, i32* %out, i64 %off6
 store i32 %any2, i32* %dst6, align 4
 %off7 = add i64 %output_base, 7
 %dst7 = getelementptr i32, i32* %out, i64 %off7
 store i32 %all2, i32* %dst7, align 4
 %off8 = add i64 %output_base, 8
 %dst8 = getelementptr i32, i32* %out, i64 %off8
 store i32 %word2, i32* %dst8, align 4
 %any3 = call i32 @llvm.nvvm.match.any.sync.i64(i32 %mask, i64 %value)
 %pair3 = call {i32, i1} @llvm.nvvm.match.all.sync.i64p(i32 %mask, i64 %value)
 %all3 = extractvalue {i32, i1} %pair3, 0
 %pred3 = extractvalue {i32, i1} %pair3, 1
 %word3 = zext i1 %pred3 to i32
 %off9 = add i64 %output_base, 9
 %dst9 = getelementptr i32, i32* %out, i64 %off9
 store i32 %any3, i32* %dst9, align 4
 %off10 = add i64 %output_base, 10
 %dst10 = getelementptr i32, i32* %out, i64 %off10
 store i32 %all3, i32* %dst10, align 4
 %off11 = add i64 %output_base, 11
 %dst11 = getelementptr i32, i32* %out, i64 %off11
 store i32 %word3, i32* %dst11, align 4
 %any4 = call i32 @llvm.nvvm.match.any.sync.i32(i32 %mask, i32 %low)
 %pair4 = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 %mask, i32 %low)
 %all4 = extractvalue {i32, i1} %pair4, 0
 %pred4 = extractvalue {i32, i1} %pair4, 1
 %word4 = zext i1 %pred4 to i32
 %off12 = add i64 %output_base, 12
 %dst12 = getelementptr i32, i32* %out, i64 %off12
 store i32 %any4, i32* %dst12, align 4
 %off13 = add i64 %output_base, 13
 %dst13 = getelementptr i32, i32* %out, i64 %off13
 store i32 %all4, i32* %dst13, align 4
 %off14 = add i64 %output_base, 14
 %dst14 = getelementptr i32, i32* %out, i64 %off14
 store i32 %word4, i32* %dst14, align 4
 %any5 = call i32 @llvm.nvvm.match.any.sync.i64(i32 %mask, i64 %value)
 %pair5 = call {i32, i1} @llvm.nvvm.match.all.sync.i64p(i32 %mask, i64 %value)
 %all5 = extractvalue {i32, i1} %pair5, 0
 %pred5 = extractvalue {i32, i1} %pair5, 1
 %word5 = zext i1 %pred5 to i32
 %off15 = add i64 %output_base, 15
 %dst15 = getelementptr i32, i32* %out, i64 %off15
 store i32 %any5, i32* %dst15, align 4
 %off16 = add i64 %output_base, 16
 %dst16 = getelementptr i32, i32* %out, i64 %off16
 store i32 %all5, i32* %dst16, align 4
 %off17 = add i64 %output_base, 17
 %dst17 = getelementptr i32, i32* %out, i64 %off17
 store i32 %word5, i32* %dst17, align 4
 br label %done
done:
 ret void
}
define void @retiring_match(i32* %out, i64* %in, i32 %mask, i32 %keep) {
entry:
 %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %base = mul i32 %block, %threads
 %i = add i32 %base, %thread
 %lane = and i32 %thread, 31
 %bit = shl i32 1, %lane
 %member_bit = and i32 %mask, %bit
 %member = icmp ne i32 %member_bit, 0
 %kept = icmp ult i32 %lane, %keep
 %active = and i1 %member, %kept
 br i1 %active, label %initialize, label %done
initialize:
 %offset = zext i32 %i to i64
 %source = getelementptr i64, i64* %in, i64 %offset
 %value = load i64, i64* %source, align 8
 %low = trunc i64 %value to i32
 %output_base = mul i64 %offset, 24
 br label %loop
loop:
 %phase = phi i32 [0, %initialize], [%next, %compare]
 %retire = icmp ult i32 %lane, %phase
 br i1 %retire, label %done, label %compare
compare:
 %wide_phase = zext i32 %phase to i64
 %phase_offset = mul i64 %wide_phase, 6
 %start = add i64 %output_base, %phase_offset
 %any0 = call i32 @llvm.nvvm.match.any.sync.i32(i32 %mask, i32 %low)
 %pair0 = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 %mask, i32 %low)
 %all0 = extractvalue {i32, i1} %pair0, 0
 %pred0 = extractvalue {i32, i1} %pair0, 1
 %word0 = zext i1 %pred0 to i32
 %off0 = add i64 %start, 0
 %dst0 = getelementptr i32, i32* %out, i64 %off0
 store i32 %any0, i32* %dst0, align 4
 %off1 = add i64 %start, 1
 %dst1 = getelementptr i32, i32* %out, i64 %off1
 store i32 %all0, i32* %dst1, align 4
 %off2 = add i64 %start, 2
 %dst2 = getelementptr i32, i32* %out, i64 %off2
 store i32 %word0, i32* %dst2, align 4
 %any1 = call i32 @llvm.nvvm.match.any.sync.i64(i32 %mask, i64 %value)
 %pair1 = call {i32, i1} @llvm.nvvm.match.all.sync.i64p(i32 %mask, i64 %value)
 %all1 = extractvalue {i32, i1} %pair1, 0
 %pred1 = extractvalue {i32, i1} %pair1, 1
 %word1 = zext i1 %pred1 to i32
 %off3 = add i64 %start, 3
 %dst3 = getelementptr i32, i32* %out, i64 %off3
 store i32 %any1, i32* %dst3, align 4
 %off4 = add i64 %start, 4
 %dst4 = getelementptr i32, i32* %out, i64 %off4
 store i32 %all1, i32* %dst4, align 4
 %off5 = add i64 %start, 5
 %dst5 = getelementptr i32, i32* %out, i64 %off5
 store i32 %word1, i32* %dst5, align 4
 %next = add i32 %phase, 1
 %more = icmp ult i32 %next, 4
 br i1 %more, label %loop, label %done
done:
 ret void
}
!nvvm.annotations = !{!0, !1}
!0 = !{void (i32*, i64*, i32)* @masked_match, !"kernel", i32 1}
!1 = !{void (i32*, i64*, i32, i32)* @retiring_match, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 1}
