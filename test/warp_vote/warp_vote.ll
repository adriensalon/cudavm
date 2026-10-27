; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1) convergent
declare i1 @llvm.nvvm.vote.all.sync(i32, i1) convergent
declare i1 @llvm.nvvm.vote.any.sync(i32, i1) convergent

define void @masked_vote(i32* %out, i32* %in, i32 %mask) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %index = add i32 %base, %thread
  %lane = and i32 %thread, 31
  %bit = shl i32 1, %lane
  %member_bit = and i32 %mask, %bit
  %member = icmp ne i32 %member_bit, 0
  br i1 %member, label %initialize, label %done
initialize:
  %offset = zext i32 %index to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %value = load i32, i32* %source, align 4
  %output_base = mul i64 %offset, 12
  br label %loop
loop:
  %phase = phi i32 [ 0, %initialize ], [ %next, %loop ]
  %high = icmp eq i32 %phase, 3
  %shift = select i1 %high, i32 31, i32 %phase
  %predicate_bit = shl i32 1, %shift
  %predicate_value = and i32 %value, %predicate_bit
  %predicate = icmp ne i32 %predicate_value, 0
  %ballot = call i32 @llvm.nvvm.vote.ballot.sync(i32 %mask, i1 %predicate)
  %all = call i1 @llvm.nvvm.vote.all.sync(i32 %mask, i1 %predicate)
  %any = call i1 @llvm.nvvm.vote.any.sync(i32 %mask, i1 %predicate)
  %all_word = zext i1 %all to i32
  %any_word = zext i1 %any to i32
  %phase_wide = zext i32 %phase to i64
  %phase_offset = mul i64 %phase_wide, 3
  %output_offset = add i64 %output_base, %phase_offset
  %first = getelementptr i32, i32* %out, i64 %output_offset
  %second = getelementptr i32, i32* %first, i64 1
  %third = getelementptr i32, i32* %first, i64 2
  store i32 %ballot, i32* %first, align 4
  store i32 %all_word, i32* %second, align 4
  store i32 %any_word, i32* %third, align 4
  %next = add i32 %phase, 1
  %more = icmp ult i32 %next, 4
  br i1 %more, label %loop, label %done
done:
  ret void
}

define void @retiring_vote(i32* %out, i32 %keep) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %index = add i32 %base, %thread
  %lane = and i32 %thread, 31
  %member = icmp ult i32 %lane, %keep
  br i1 %member, label %initialize, label %done
initialize:
  %remainder = urem i32 %lane, 3
  %offset = zext i32 %index to i64
  %output_base = mul i64 %offset, 12
  br label %loop
loop:
  %phase = phi i32 [ 0, %initialize ], [ %next, %vote ]
  %retire = icmp ult i32 %lane, %phase
  br i1 %retire, label %done, label %vote
vote:
  %predicate = icmp eq i32 %remainder, %phase
  %ballot = call i32 @llvm.nvvm.vote.ballot.sync(i32 -1, i1 %predicate)
  %all = call i1 @llvm.nvvm.vote.all.sync(i32 -1, i1 %predicate)
  %any = call i1 @llvm.nvvm.vote.any.sync(i32 -1, i1 %predicate)
  %all_word = zext i1 %all to i32
  %any_word = zext i1 %any to i32
  %phase_wide = zext i32 %phase to i64
  %phase_offset = mul i64 %phase_wide, 3
  %output_offset = add i64 %output_base, %phase_offset
  %first = getelementptr i32, i32* %out, i64 %output_offset
  %second = getelementptr i32, i32* %first, i64 1
  %third = getelementptr i32, i32* %first, i64 2
  store i32 %ballot, i32* %first, align 4
  store i32 %all_word, i32* %second, align 4
  store i32 %any_word, i32* %third, align 4
  %next = add i32 %phase, 1
  %more = icmp ult i32 %next, 4
  br i1 %more, label %loop, label %done
done:
  ret void
}
!nvvm.annotations = !{!0, !1}
!0 = !{void (i32*, i32*, i32)* @masked_vote, !"kernel", i32 1}
!1 = !{void (i32*, i32)* @retiring_vote, !"kernel", i32 1}
!nvvmir.version = !{!2}
!2 = !{i32 2, i32 0, i32 3, i32 1}
