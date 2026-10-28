; Independently authored NVVM IR 2.0, LLVM 7 typed pointers.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
declare i32 @llvm.nvvm.idp4a.s.s(i32,i32,i32)
declare i32 @llvm.nvvm.idp2a.s.s(i32,i32,i1,i32)
declare i32 @llvm.nvvm.idp4a.s.u(i32,i32,i32)
declare i32 @llvm.nvvm.idp2a.s.u(i32,i32,i1,i32)
declare i32 @llvm.nvvm.idp4a.u.s(i32,i32,i32)
declare i32 @llvm.nvvm.idp2a.u.s(i32,i32,i1,i32)
declare i32 @llvm.nvvm.idp4a.u.u(i32,i32,i32)
declare i32 @llvm.nvvm.idp2a.u.u(i32,i32,i1,i32)
define internal i32 @packed_16_0(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=sext <2 x i16> %av to <2 x i32>
 %y=sext <2 x i16> %bv to <2 x i32>
 %sum=add <2 x i32> %x,%y
 %small=icmp slt <2 x i32> %sum,<i32 -32768, i32 -32768>
 %lower=select <2 x i1> %small,<2 x i32> <i32 -32768, i32 -32768>,<2 x i32> %sum
 %large=icmp sgt <2 x i32> %lower,<i32 32767, i32 32767>
 %clamped=select <2 x i1> %large,<2 x i32> <i32 32767, i32 32767>,<2 x i32> %lower
 %narrow=trunc <2 x i32> %clamped to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_1(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=zext <2 x i16> %av to <2 x i32>
 %y=zext <2 x i16> %bv to <2 x i32>
 %sum=add <2 x i32> %x,%y
 %small=icmp slt <2 x i32> %sum,<i32 0, i32 0>
 %lower=select <2 x i1> %small,<2 x i32> <i32 0, i32 0>,<2 x i32> %sum
 %large=icmp sgt <2 x i32> %lower,<i32 65535, i32 65535>
 %clamped=select <2 x i1> %large,<2 x i32> <i32 65535, i32 65535>,<2 x i32> %lower
 %narrow=trunc <2 x i32> %clamped to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_2(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=sext <2 x i16> %av to <2 x i32>
 %y=sext <2 x i16> %bv to <2 x i32>
 %sum=sub <2 x i32> %x,%y
 %small=icmp slt <2 x i32> %sum,<i32 -32768, i32 -32768>
 %lower=select <2 x i1> %small,<2 x i32> <i32 -32768, i32 -32768>,<2 x i32> %sum
 %large=icmp sgt <2 x i32> %lower,<i32 32767, i32 32767>
 %clamped=select <2 x i1> %large,<2 x i32> <i32 32767, i32 32767>,<2 x i32> %lower
 %narrow=trunc <2 x i32> %clamped to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_3(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=zext <2 x i16> %av to <2 x i32>
 %y=zext <2 x i16> %bv to <2 x i32>
 %sum=sub <2 x i32> %x,%y
 %small=icmp slt <2 x i32> %sum,<i32 0, i32 0>
 %lower=select <2 x i1> %small,<2 x i32> <i32 0, i32 0>,<2 x i32> %sum
 %large=icmp sgt <2 x i32> %lower,<i32 65535, i32 65535>
 %clamped=select <2 x i1> %large,<2 x i32> <i32 65535, i32 65535>,<2 x i32> %lower
 %narrow=trunc <2 x i32> %clamped to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_4(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=sext <2 x i16> %av to <2 x i32>
 %y=sext <2 x i16> %bv to <2 x i32>
 %sum=add <2 x i32> %x,%y
 %positive=icmp sge <2 x i32> %sum,zeroinitializer
 %bias=zext <2 x i1> %positive to <2 x i32>
 %biased=add <2 x i32> %sum,%bias
 %half=ashr <2 x i32> %biased,<i32 1, i32 1>
 %narrow=trunc <2 x i32> %half to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_5(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=zext <2 x i16> %av to <2 x i32>
 %y=zext <2 x i16> %bv to <2 x i32>
 %sum=add <2 x i32> %x,%y
 %biased=add <2 x i32> %sum,<i32 1, i32 1>
 %half=lshr <2 x i32> %biased,<i32 1, i32 1>
 %narrow=trunc <2 x i32> %half to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_6(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=zext <2 x i16> %av to <2 x i32>
 %y=zext <2 x i16> %bv to <2 x i32>
 %sum=add <2 x i32> %x,%y
 %half=lshr <2 x i32> %sum,<i32 1, i32 1>
 %narrow=trunc <2 x i32> %half to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_7(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=sext <2 x i16> %av to <2 x i32>
 %y=sext <2 x i16> %bv to <2 x i32>
 %choose=icmp slt <2 x i32> %x,%y
 %sum=select <2 x i1> %choose,<2 x i32> %x,<2 x i32> %y
 %narrow=trunc <2 x i32> %sum to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_8(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=zext <2 x i16> %av to <2 x i32>
 %y=zext <2 x i16> %bv to <2 x i32>
 %choose=icmp slt <2 x i32> %x,%y
 %sum=select <2 x i1> %choose,<2 x i32> %x,<2 x i32> %y
 %narrow=trunc <2 x i32> %sum to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_9(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=sext <2 x i16> %av to <2 x i32>
 %y=sext <2 x i16> %bv to <2 x i32>
 %choose=icmp sgt <2 x i32> %x,%y
 %sum=select <2 x i1> %choose,<2 x i32> %x,<2 x i32> %y
 %narrow=trunc <2 x i32> %sum to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_10(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=zext <2 x i16> %av to <2 x i32>
 %y=zext <2 x i16> %bv to <2 x i32>
 %choose=icmp sgt <2 x i32> %x,%y
 %sum=select <2 x i1> %choose,<2 x i32> %x,<2 x i32> %y
 %narrow=trunc <2 x i32> %sum to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_11(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=sext <2 x i16> %av to <2 x i32>
 %y=sext <2 x i16> %bv to <2 x i32>
 %negative=icmp slt <2 x i32> %x,zeroinitializer
 %neg=sub <2 x i32> zeroinitializer,%x
 %sum=select <2 x i1> %negative,<2 x i32> %neg,<2 x i32> %x
 %small=icmp slt <2 x i32> %sum,<i32 -32768, i32 -32768>
 %lower=select <2 x i1> %small,<2 x i32> <i32 -32768, i32 -32768>,<2 x i32> %sum
 %large=icmp sgt <2 x i32> %lower,<i32 32767, i32 32767>
 %clamped=select <2 x i1> %large,<2 x i32> <i32 32767, i32 32767>,<2 x i32> %lower
 %narrow=trunc <2 x i32> %clamped to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_16_12(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <2 x i16>
 %bv=bitcast i32 %b to <2 x i16>
 %x=sext <2 x i16> %av to <2 x i32>
 %y=sext <2 x i16> %bv to <2 x i32>
 %sum=sub <2 x i32> zeroinitializer,%x
 %small=icmp slt <2 x i32> %sum,<i32 -32768, i32 -32768>
 %lower=select <2 x i1> %small,<2 x i32> <i32 -32768, i32 -32768>,<2 x i32> %sum
 %large=icmp sgt <2 x i32> %lower,<i32 32767, i32 32767>
 %clamped=select <2 x i1> %large,<2 x i32> <i32 32767, i32 32767>,<2 x i32> %lower
 %narrow=trunc <2 x i32> %clamped to <2 x i16>
 %word=bitcast <2 x i16> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_0(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=sext <4 x i8> %av to <4 x i32>
 %y=sext <4 x i8> %bv to <4 x i32>
 %sum=add <4 x i32> %x,%y
 %small=icmp slt <4 x i32> %sum,<i32 -128, i32 -128, i32 -128, i32 -128>
 %lower=select <4 x i1> %small,<4 x i32> <i32 -128, i32 -128, i32 -128, i32 -128>,<4 x i32> %sum
 %large=icmp sgt <4 x i32> %lower,<i32 127, i32 127, i32 127, i32 127>
 %clamped=select <4 x i1> %large,<4 x i32> <i32 127, i32 127, i32 127, i32 127>,<4 x i32> %lower
 %narrow=trunc <4 x i32> %clamped to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_1(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=zext <4 x i8> %av to <4 x i32>
 %y=zext <4 x i8> %bv to <4 x i32>
 %sum=add <4 x i32> %x,%y
 %small=icmp slt <4 x i32> %sum,<i32 0, i32 0, i32 0, i32 0>
 %lower=select <4 x i1> %small,<4 x i32> <i32 0, i32 0, i32 0, i32 0>,<4 x i32> %sum
 %large=icmp sgt <4 x i32> %lower,<i32 255, i32 255, i32 255, i32 255>
 %clamped=select <4 x i1> %large,<4 x i32> <i32 255, i32 255, i32 255, i32 255>,<4 x i32> %lower
 %narrow=trunc <4 x i32> %clamped to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_2(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=sext <4 x i8> %av to <4 x i32>
 %y=sext <4 x i8> %bv to <4 x i32>
 %sum=sub <4 x i32> %x,%y
 %small=icmp slt <4 x i32> %sum,<i32 -128, i32 -128, i32 -128, i32 -128>
 %lower=select <4 x i1> %small,<4 x i32> <i32 -128, i32 -128, i32 -128, i32 -128>,<4 x i32> %sum
 %large=icmp sgt <4 x i32> %lower,<i32 127, i32 127, i32 127, i32 127>
 %clamped=select <4 x i1> %large,<4 x i32> <i32 127, i32 127, i32 127, i32 127>,<4 x i32> %lower
 %narrow=trunc <4 x i32> %clamped to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_3(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=zext <4 x i8> %av to <4 x i32>
 %y=zext <4 x i8> %bv to <4 x i32>
 %sum=sub <4 x i32> %x,%y
 %small=icmp slt <4 x i32> %sum,<i32 0, i32 0, i32 0, i32 0>
 %lower=select <4 x i1> %small,<4 x i32> <i32 0, i32 0, i32 0, i32 0>,<4 x i32> %sum
 %large=icmp sgt <4 x i32> %lower,<i32 255, i32 255, i32 255, i32 255>
 %clamped=select <4 x i1> %large,<4 x i32> <i32 255, i32 255, i32 255, i32 255>,<4 x i32> %lower
 %narrow=trunc <4 x i32> %clamped to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_4(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=sext <4 x i8> %av to <4 x i32>
 %y=sext <4 x i8> %bv to <4 x i32>
 %sum=add <4 x i32> %x,%y
 %positive=icmp sge <4 x i32> %sum,zeroinitializer
 %bias=zext <4 x i1> %positive to <4 x i32>
 %biased=add <4 x i32> %sum,%bias
 %half=ashr <4 x i32> %biased,<i32 1, i32 1, i32 1, i32 1>
 %narrow=trunc <4 x i32> %half to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_5(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=zext <4 x i8> %av to <4 x i32>
 %y=zext <4 x i8> %bv to <4 x i32>
 %sum=add <4 x i32> %x,%y
 %biased=add <4 x i32> %sum,<i32 1, i32 1, i32 1, i32 1>
 %half=lshr <4 x i32> %biased,<i32 1, i32 1, i32 1, i32 1>
 %narrow=trunc <4 x i32> %half to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_6(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=zext <4 x i8> %av to <4 x i32>
 %y=zext <4 x i8> %bv to <4 x i32>
 %sum=add <4 x i32> %x,%y
 %half=lshr <4 x i32> %sum,<i32 1, i32 1, i32 1, i32 1>
 %narrow=trunc <4 x i32> %half to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_7(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=sext <4 x i8> %av to <4 x i32>
 %y=sext <4 x i8> %bv to <4 x i32>
 %choose=icmp slt <4 x i32> %x,%y
 %sum=select <4 x i1> %choose,<4 x i32> %x,<4 x i32> %y
 %narrow=trunc <4 x i32> %sum to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_8(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=zext <4 x i8> %av to <4 x i32>
 %y=zext <4 x i8> %bv to <4 x i32>
 %choose=icmp slt <4 x i32> %x,%y
 %sum=select <4 x i1> %choose,<4 x i32> %x,<4 x i32> %y
 %narrow=trunc <4 x i32> %sum to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_9(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=sext <4 x i8> %av to <4 x i32>
 %y=sext <4 x i8> %bv to <4 x i32>
 %choose=icmp sgt <4 x i32> %x,%y
 %sum=select <4 x i1> %choose,<4 x i32> %x,<4 x i32> %y
 %narrow=trunc <4 x i32> %sum to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_10(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=zext <4 x i8> %av to <4 x i32>
 %y=zext <4 x i8> %bv to <4 x i32>
 %choose=icmp sgt <4 x i32> %x,%y
 %sum=select <4 x i1> %choose,<4 x i32> %x,<4 x i32> %y
 %narrow=trunc <4 x i32> %sum to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_11(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=sext <4 x i8> %av to <4 x i32>
 %y=sext <4 x i8> %bv to <4 x i32>
 %negative=icmp slt <4 x i32> %x,zeroinitializer
 %neg=sub <4 x i32> zeroinitializer,%x
 %sum=select <4 x i1> %negative,<4 x i32> %neg,<4 x i32> %x
 %small=icmp slt <4 x i32> %sum,<i32 -128, i32 -128, i32 -128, i32 -128>
 %lower=select <4 x i1> %small,<4 x i32> <i32 -128, i32 -128, i32 -128, i32 -128>,<4 x i32> %sum
 %large=icmp sgt <4 x i32> %lower,<i32 127, i32 127, i32 127, i32 127>
 %clamped=select <4 x i1> %large,<4 x i32> <i32 127, i32 127, i32 127, i32 127>,<4 x i32> %lower
 %narrow=trunc <4 x i32> %clamped to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define internal i32 @packed_8_12(i32 %a,i32 %b) {
 %av=bitcast i32 %a to <4 x i8>
 %bv=bitcast i32 %b to <4 x i8>
 %x=sext <4 x i8> %av to <4 x i32>
 %y=sext <4 x i8> %bv to <4 x i32>
 %sum=sub <4 x i32> zeroinitializer,%x
 %small=icmp slt <4 x i32> %sum,<i32 -128, i32 -128, i32 -128, i32 -128>
 %lower=select <4 x i1> %small,<4 x i32> <i32 -128, i32 -128, i32 -128, i32 -128>,<4 x i32> %sum
 %large=icmp sgt <4 x i32> %lower,<i32 127, i32 127, i32 127, i32 127>
 %clamped=select <4 x i1> %large,<4 x i32> <i32 127, i32 127, i32 127, i32 127>,<4 x i32> %lower
 %narrow=trunc <4 x i32> %clamped to <4 x i8>
 %word=bitcast <4 x i8> %narrow to i32
 ret i32 %word
}
define void @packed_values(i32* %out,i32* %in,i32 %count) {
entry:
 %lane=call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %lanes=call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %block=call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %block_y=call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
 %grid_x=call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
 %row=mul i32 %block_y,%grid_x
 %linear_block=add i32 %block,%row
 %base=mul i32 %linear_block,%lanes
 %i=add i32 %base,%lane
 %valid=icmp ult i32 %i,%count
 br i1 %valid,label %write,label %done
write:
 %index=zext i32 %i to i64
 %offset=mul i64 %index,4
 %p=getelementptr i32,i32* %in,i64 %offset
 %p0=getelementptr i32,i32* %p,i64 0
 %a=load i32,i32* %p0,align 4
 %p1=getelementptr i32,i32* %p,i64 1
 %b=load i32,i32* %p1,align 4
 %p2=getelementptr i32,i32* %p,i64 2
 %c=load i32,i32* %p2,align 4
 %p3=getelementptr i32,i32* %p,i64 3
 %s=load i32,i32* %p3,align 4
 %v0=call i32 @llvm.nvvm.idp4a.s.s(i32 %a,i32 %b,i32 %s)
 %v1=call i32 @llvm.nvvm.idp4a.u.u(i32 %a,i32 %b,i32 %c)
 %v42=call i32 @llvm.nvvm.idp4a.s.u(i32 %a,i32 %b,i32 %s)
 %v43=call i32 @llvm.nvvm.idp4a.u.s(i32 %a,i32 %b,i32 %s)
 %v2=call i32 @llvm.nvvm.idp2a.s.s(i32 %a,i32 %b,i1 false,i32 %s)
 %v3=call i32 @llvm.nvvm.idp2a.u.u(i32 %a,i32 %b,i1 false,i32 %c)
 %v44=call i32 @llvm.nvvm.idp2a.s.u(i32 %a,i32 %b,i1 false,i32 %s)
 %v45=call i32 @llvm.nvvm.idp2a.u.s(i32 %a,i32 %b,i1 false,i32 %s)
 %v4=call i32 @llvm.nvvm.idp2a.s.s(i32 %a,i32 %b,i1 true,i32 %s)
 %v5=call i32 @llvm.nvvm.idp2a.u.u(i32 %a,i32 %b,i1 true,i32 %c)
 %v46=call i32 @llvm.nvvm.idp2a.s.u(i32 %a,i32 %b,i1 true,i32 %s)
 %v47=call i32 @llvm.nvvm.idp2a.u.s(i32 %a,i32 %b,i1 true,i32 %s)
 %sa=sext i32 %a to i64
 %sb=sext i32 %b to i64
 %ua=zext i32 %a to i64
 %ub=zext i32 %b to i64
 %ss=add i64 %sa,%sb
 %us=add i64 %ua,%ub
 %half0=ashr i64 %ss,1
 %v12=trunc i64 %half0 to i32
 %rounded1=add i64 %ss,1
 %half1=ashr i64 %rounded1,1
 %v13=trunc i64 %half1 to i32
 %half2=lshr i64 %us,1
 %v14=trunc i64 %half2 to i32
 %rounded3=add i64 %us,1
 %half3=lshr i64 %rounded3,1
 %v15=trunc i64 %half3 to i32
 %v16=call i32 @packed_16_0(i32 %a,i32 %b)
 %v17=call i32 @packed_8_0(i32 %a,i32 %b)
 %v18=call i32 @packed_16_1(i32 %a,i32 %b)
 %v19=call i32 @packed_8_1(i32 %a,i32 %b)
 %v20=call i32 @packed_16_2(i32 %a,i32 %b)
 %v21=call i32 @packed_8_2(i32 %a,i32 %b)
 %v22=call i32 @packed_16_3(i32 %a,i32 %b)
 %v23=call i32 @packed_8_3(i32 %a,i32 %b)
 %v24=call i32 @packed_16_4(i32 %a,i32 %b)
 %v25=call i32 @packed_8_4(i32 %a,i32 %b)
 %v26=call i32 @packed_16_5(i32 %a,i32 %b)
 %v27=call i32 @packed_8_5(i32 %a,i32 %b)
 %v28=call i32 @packed_16_6(i32 %a,i32 %b)
 %v29=call i32 @packed_8_6(i32 %a,i32 %b)
 %v30=call i32 @packed_16_7(i32 %a,i32 %b)
 %v31=call i32 @packed_8_7(i32 %a,i32 %b)
 %v32=call i32 @packed_16_8(i32 %a,i32 %b)
 %v33=call i32 @packed_8_8(i32 %a,i32 %b)
 %v34=call i32 @packed_16_9(i32 %a,i32 %b)
 %v35=call i32 @packed_8_9(i32 %a,i32 %b)
 %v36=call i32 @packed_16_10(i32 %a,i32 %b)
 %v37=call i32 @packed_8_10(i32 %a,i32 %b)
 %v38=call i32 @packed_16_11(i32 %a,i32 %b)
 %v39=call i32 @packed_8_11(i32 %a,i32 %b)
 %v40=call i32 @packed_16_12(i32 %a,i32 %b)
 %v41=call i32 @packed_8_12(i32 %a,i32 %b)
 %target_offset=mul i64 %index,48
 %target=getelementptr i32,i32* %out,i64 %target_offset
 %out0=getelementptr i32,i32* %target,i64 0
 store i32 %v0,i32* %out0,align 4
 %out1=getelementptr i32,i32* %target,i64 1
 store i32 %v1,i32* %out1,align 4
 %out2=getelementptr i32,i32* %target,i64 2
 store i32 %v2,i32* %out2,align 4
 %out3=getelementptr i32,i32* %target,i64 3
 store i32 %v3,i32* %out3,align 4
 %out4=getelementptr i32,i32* %target,i64 4
 store i32 %v4,i32* %out4,align 4
 %out5=getelementptr i32,i32* %target,i64 5
 store i32 %v5,i32* %out5,align 4
 %out6=getelementptr i32,i32* %target,i64 6
 store i32 %v0,i32* %out6,align 4
 %out7=getelementptr i32,i32* %target,i64 7
 store i32 %v1,i32* %out7,align 4
 %out8=getelementptr i32,i32* %target,i64 8
 store i32 %v2,i32* %out8,align 4
 %out9=getelementptr i32,i32* %target,i64 9
 store i32 %v3,i32* %out9,align 4
 %out10=getelementptr i32,i32* %target,i64 10
 store i32 %v4,i32* %out10,align 4
 %out11=getelementptr i32,i32* %target,i64 11
 store i32 %v5,i32* %out11,align 4
 %out12=getelementptr i32,i32* %target,i64 12
 store i32 %v12,i32* %out12,align 4
 %out13=getelementptr i32,i32* %target,i64 13
 store i32 %v13,i32* %out13,align 4
 %out14=getelementptr i32,i32* %target,i64 14
 store i32 %v14,i32* %out14,align 4
 %out15=getelementptr i32,i32* %target,i64 15
 store i32 %v15,i32* %out15,align 4
 %out16=getelementptr i32,i32* %target,i64 16
 store i32 %v16,i32* %out16,align 4
 %out17=getelementptr i32,i32* %target,i64 17
 store i32 %v17,i32* %out17,align 4
 %out18=getelementptr i32,i32* %target,i64 18
 store i32 %v18,i32* %out18,align 4
 %out19=getelementptr i32,i32* %target,i64 19
 store i32 %v19,i32* %out19,align 4
 %out20=getelementptr i32,i32* %target,i64 20
 store i32 %v20,i32* %out20,align 4
 %out21=getelementptr i32,i32* %target,i64 21
 store i32 %v21,i32* %out21,align 4
 %out22=getelementptr i32,i32* %target,i64 22
 store i32 %v22,i32* %out22,align 4
 %out23=getelementptr i32,i32* %target,i64 23
 store i32 %v23,i32* %out23,align 4
 %out24=getelementptr i32,i32* %target,i64 24
 store i32 %v24,i32* %out24,align 4
 %out25=getelementptr i32,i32* %target,i64 25
 store i32 %v25,i32* %out25,align 4
 %out26=getelementptr i32,i32* %target,i64 26
 store i32 %v26,i32* %out26,align 4
 %out27=getelementptr i32,i32* %target,i64 27
 store i32 %v27,i32* %out27,align 4
 %out28=getelementptr i32,i32* %target,i64 28
 store i32 %v28,i32* %out28,align 4
 %out29=getelementptr i32,i32* %target,i64 29
 store i32 %v29,i32* %out29,align 4
 %out30=getelementptr i32,i32* %target,i64 30
 store i32 %v30,i32* %out30,align 4
 %out31=getelementptr i32,i32* %target,i64 31
 store i32 %v31,i32* %out31,align 4
 %out32=getelementptr i32,i32* %target,i64 32
 store i32 %v32,i32* %out32,align 4
 %out33=getelementptr i32,i32* %target,i64 33
 store i32 %v33,i32* %out33,align 4
 %out34=getelementptr i32,i32* %target,i64 34
 store i32 %v34,i32* %out34,align 4
 %out35=getelementptr i32,i32* %target,i64 35
 store i32 %v35,i32* %out35,align 4
 %out36=getelementptr i32,i32* %target,i64 36
 store i32 %v36,i32* %out36,align 4
 %out37=getelementptr i32,i32* %target,i64 37
 store i32 %v37,i32* %out37,align 4
 %out38=getelementptr i32,i32* %target,i64 38
 store i32 %v38,i32* %out38,align 4
 %out39=getelementptr i32,i32* %target,i64 39
 store i32 %v39,i32* %out39,align 4
 %out40=getelementptr i32,i32* %target,i64 40
 store i32 %v40,i32* %out40,align 4
 %out41=getelementptr i32,i32* %target,i64 41
 store i32 %v41,i32* %out41,align 4
 %out42=getelementptr i32,i32* %target,i64 42
 store i32 %v42,i32* %out42,align 4
 %out43=getelementptr i32,i32* %target,i64 43
 store i32 %v43,i32* %out43,align 4
 %out44=getelementptr i32,i32* %target,i64 44
 store i32 %v44,i32* %out44,align 4
 %out45=getelementptr i32,i32* %target,i64 45
 store i32 %v45,i32* %out45,align 4
 %out46=getelementptr i32,i32* %target,i64 46
 store i32 %v46,i32* %out46,align 4
 %out47=getelementptr i32,i32* %target,i64 47
 store i32 %v47,i32* %out47,align 4
 br label %done
done:
 ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32*, i32)* @packed_values, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
