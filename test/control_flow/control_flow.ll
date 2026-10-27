; Independently authored NVVM IR 2.0 in the LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define void @branch_loop(i32* %out, i32* %in, i32 %count, i32 %seed) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %base = mul i32 %block, %threads
  %i = add i32 %base, %thread
  %valid = icmp ult i32 %i, %count
  br i1 %valid, label %load, label %done
load:
  %offset = zext i32 %i to i64
  %source = getelementptr i32, i32* %in, i64 %offset
  %value = load i32, i32* %source, align 4
  %nonzero = icmp ne i32 %value, 0
  br i1 %nonzero, label %initialize, label %done
initialize:
  %initial = xor i32 %value, %seed
  %limit = and i32 %value, 7
  %shifted = lshr i32 %value, 5
  %stop = and i32 %shifted, 7
  br label %loop
loop:
  %step = phi i32 [ 0, %initialize ], [ %next_step, %increment ]
  %state = phi i32 [ %initial, %initialize ], [ %next_state, %increment ]
  %within = icmp ult i32 %step, %limit
  br i1 %within, label %check_break, label %finish
check_break:
  %stop_now = icmp eq i32 %step, %stop
  br i1 %stop_now, label %finish, label %body
body:
  %parity = and i32 %step, 1
  %even = icmp eq i32 %parity, 0
  br i1 %even, label %multiply, label %halve
multiply:
  %product = mul i32 %state, 3
  %sum = add i32 %product, %step
  br label %merge
halve:
  %half = lshr i32 %state, 1
  %addend = add i32 %value, %step
  %mixed = xor i32 %half, %addend
  br label %merge
merge:
  %changed = phi i32 [ %sum, %multiply ], [ %mixed, %halve ]
  %low = and i32 %changed, 3
  %continue = icmp eq i32 %low, 0
  br i1 %continue, label %adjust, label %scramble
adjust:
  %adjusted = add i32 %changed, 17
  br label %increment
scramble:
  %scrambled = xor i32 %changed, -1640531527
  br label %increment
increment:
  %next_state = phi i32 [ %adjusted, %adjust ], [ %scrambled, %scramble ]
  %next_step = add i32 %step, 1
  br label %loop
finish:
  %shift_selector = lshr i32 %value, 3
  %selector = and i32 %shift_selector, 3
  switch i32 %selector, label %invert [ i32 0, label %add_seed
    i32 1, label %xor_seed
    i32 2, label %rotate ]
add_seed:
  %added = add i32 %state, %seed
  br label %write
xor_seed:
  %xored = xor i32 %state, %seed
  br label %write
rotate:
  %left = shl i32 %state, 3
  %right = lshr i32 %state, 29
  %rotated = or i32 %left, %right
  br label %write
invert:
  %inverted = xor i32 %state, -1
  br label %write
write:
  %result = phi i32 [ %added, %add_seed ], [ %xored, %xor_seed ],
    [ %rotated, %rotate ], [ %inverted, %invert ]
  %destination_offset = mul i64 %offset, 2
  %destination = getelementptr i32, i32* %out, i64 %destination_offset
  %iterations = getelementptr i32, i32* %destination, i64 1
  store i32 %result, i32* %destination, align 4
  store i32 %step, i32* %iterations, align 4
  br label %done
done:
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32, i32)* @branch_loop, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
