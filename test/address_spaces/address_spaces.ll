; Independently authored NVVM IR 2.0, LLVM 7 typed-pointer dialect.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@address_constants = addrspace(4) global [9 x i32] [i32 324508639,i32 0,i32 1,i32 2147483648,i32 4294967295,i32 305419896,i32 2309737967,i32 17,i32 3970458656], align 4
@shared_words = internal addrspace(3) global [256 x i32] undef, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i1 @llvm.nvvm.isspacep.global(i8*)
declare i1 @llvm.nvvm.isspacep.shared(i8*)
declare i1 @llvm.nvvm.isspacep.local(i8*)
declare i1 @llvm.nvvm.isspacep.const(i8*)
define internal i32 @address_mask(i32* %pointer) alwaysinline {
 %byte = bitcast i32* %pointer to i8*
 %p0 = call i1 @llvm.nvvm.isspacep.global(i8* %byte)
 %r0 = select i1 %p0, i32 1, i32 0
 %p1 = call i1 @llvm.nvvm.isspacep.shared(i8* %byte)
 %r1 = select i1 %p1, i32 2, i32 0
 %p2 = call i1 @llvm.nvvm.isspacep.local(i8* %byte)
 %r2 = select i1 %p2, i32 4, i32 0
 %p3 = call i1 @llvm.nvvm.isspacep.const(i8* %byte)
 %r3 = select i1 %p3, i32 8, i32 0
 %a = or i32 %r0,%r1
 %b = or i32 %r2,%r3
 %result = or i32 %a,%b
 ret i32 %result
}
define void @address_results(i32* %out, i32* %in, i32 %count, i32 %selector) {
entry:
 %frame = alloca [4 x i32], align 16
 %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %lanes = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
 %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %base = mul i32 %block,%lanes
 %i = add i32 %base,%lane
 %active = icmp ult i32 %i,%count
 br i1 %active, label %write, label %done
write:
 %gi = mul i32 %i,2
 %g = getelementptr i32,i32* %in,i32 %gi
 %gn = getelementptr i32,i32* %g,i32 1
 %a = load i32,i32* %g,align 4
 %b = load i32,i32* %gn,align 4
 %si0 = mul i32 %lane,4
 %si = add i32 %si0,1
 %sraw = getelementptr [256 x i32],[256 x i32] addrspace(3)* @shared_words,i32 0,i32 %si
 %s = addrspacecast i32 addrspace(3)* %sraw to i32*
 %l = getelementptr [4 x i32],[4 x i32]* %frame,i32 0,i32 1
 %ci0 = urem i32 %i,6
 %ci = add i32 %ci0,1
 %craw = getelementptr [9 x i32],[9 x i32] addrspace(4)* @address_constants,i32 0,i32 %ci
 %c = addrspacecast i32 addrspace(4)* %craw to i32*
 %s0 = xor i32 %i,324508639
 %s1 = xor i32 %a,2863311530
 %s2a = add i32 %b,%block
 %s2 = add i32 %s2a,7
 %s3 = xor i32 %a,3970458656
 %l0 = xor i32 %b,2309737967
 %l1a = mul i32 %i,17
 %l1 = add i32 %l1a,%a
 %l2a = mul i32 %i,66051
 %l2 = xor i32 %l2a,%b
 %l3 = xor i32 %i,1985229328
 %sp0 = getelementptr i32,i32* %s,i32 -1
 store i32 %s0,i32* %sp0,align 4
 %sp1 = getelementptr i32,i32* %s,i32 0
 store i32 %s1,i32* %sp1,align 4
 %sp2 = getelementptr i32,i32* %s,i32 1
 store i32 %s2,i32* %sp2,align 4
 %sp3 = getelementptr i32,i32* %s,i32 2
 store i32 %s3,i32* %sp3,align 4
 %lp0 = getelementptr i32,i32* %l,i32 -1
 store i32 %l0,i32* %lp0,align 4
 %lp1 = getelementptr i32,i32* %l,i32 0
 store i32 %l1,i32* %lp1,align 4
 %lp2 = getelementptr i32,i32* %l,i32 1
 store i32 %l2,i32* %lp2,align 4
 %lp3 = getelementptr i32,i32* %l,i32 2
 store i32 %l3,i32* %lp3,align 4
 %oi = mul i32 %i,36
 %output = getelementptr i32,i32* %out,i32 %oi
 %gmask = call i32 @address_mask(i32* %g)
 %gspecific = addrspacecast i32* %g to i32 addrspace(1)*
 %gbits = ptrtoint i32 addrspace(1)* %gspecific to i64
 %gnext = getelementptr i32,i32* %g,i32 1
 %gnextspecific = addrspacecast i32* %gnext to i32 addrspace(1)*
 %gnextbits = ptrtoint i32 addrspace(1)* %gnextspecific to i64
 %gbackraw = inttoptr i64 %gbits to i32 addrspace(1)*
 %gback = addrspacecast i32 addrspace(1)* %gbackraw to i32*
 %gplusbits = add i64 %gbits,4
 %gplusraw = inttoptr i64 %gplusbits to i32 addrspace(1)*
 %gplus = addrspacecast i32 addrspace(1)* %gplusraw to i32*
 %gequal = icmp eq i32* %g,%gback
 %gidentity = zext i1 %gequal to i32
 %gvalue = load i32,i32* %gback,align 4
 %gnextvalue = load i32,i32* %gplus,align 4
 %gdiff64 = sub i64 %gnextbits,%gbits
 %gdiff = trunc i64 %gdiff64 to i32
 %gnextequal = icmp eq i32* %gnext,%gplus
 %gnextidentity = zext i1 %gnextequal to i32
 %o0 = getelementptr i32,i32* %output,i32 0
 store i32 %gmask,i32* %o0,align 4
 %o1 = getelementptr i32,i32* %output,i32 1
 store i32 %gidentity,i32* %o1,align 4
 %o2 = getelementptr i32,i32* %output,i32 2
 store i32 %gvalue,i32* %o2,align 4
 %o3 = getelementptr i32,i32* %output,i32 3
 store i32 %gnextvalue,i32* %o3,align 4
 %o4 = getelementptr i32,i32* %output,i32 4
 store i32 %gdiff,i32* %o4,align 4
 %o5 = getelementptr i32,i32* %output,i32 5
 store i32 %gnextidentity,i32* %o5,align 4
 %smask = call i32 @address_mask(i32* %s)
 %sspecific = addrspacecast i32* %s to i32 addrspace(3)*
 %sbits = ptrtoint i32 addrspace(3)* %sspecific to i64
 %snext = getelementptr i32,i32* %s,i32 1
 %snextspecific = addrspacecast i32* %snext to i32 addrspace(3)*
 %snextbits = ptrtoint i32 addrspace(3)* %snextspecific to i64
 %sbackraw = inttoptr i64 %sbits to i32 addrspace(3)*
 %sback = addrspacecast i32 addrspace(3)* %sbackraw to i32*
 %splusbits = add i64 %sbits,4
 %splusraw = inttoptr i64 %splusbits to i32 addrspace(3)*
 %splus = addrspacecast i32 addrspace(3)* %splusraw to i32*
 %sequal = icmp eq i32* %s,%sback
 %sidentity = zext i1 %sequal to i32
 %svalue = load i32,i32* %sback,align 4
 %snextvalue = load i32,i32* %splus,align 4
 %sdiff64 = sub i64 %snextbits,%sbits
 %sdiff = trunc i64 %sdiff64 to i32
 %snextequal = icmp eq i32* %snext,%splus
 %snextidentity = zext i1 %snextequal to i32
 %o6 = getelementptr i32,i32* %output,i32 6
 store i32 %smask,i32* %o6,align 4
 %o7 = getelementptr i32,i32* %output,i32 7
 store i32 %sidentity,i32* %o7,align 4
 %o8 = getelementptr i32,i32* %output,i32 8
 store i32 %svalue,i32* %o8,align 4
 %o9 = getelementptr i32,i32* %output,i32 9
 store i32 %snextvalue,i32* %o9,align 4
 %o10 = getelementptr i32,i32* %output,i32 10
 store i32 %sdiff,i32* %o10,align 4
 %o11 = getelementptr i32,i32* %output,i32 11
 store i32 %snextidentity,i32* %o11,align 4
 %lmask = call i32 @address_mask(i32* %l)
 %lspecific = addrspacecast i32* %l to i32 addrspace(5)*
 %lbits = ptrtoint i32 addrspace(5)* %lspecific to i64
 %lnext = getelementptr i32,i32* %l,i32 1
 %lnextspecific = addrspacecast i32* %lnext to i32 addrspace(5)*
 %lnextbits = ptrtoint i32 addrspace(5)* %lnextspecific to i64
 %lbackraw = inttoptr i64 %lbits to i32 addrspace(5)*
 %lback = addrspacecast i32 addrspace(5)* %lbackraw to i32*
 %lplusbits = add i64 %lbits,4
 %lplusraw = inttoptr i64 %lplusbits to i32 addrspace(5)*
 %lplus = addrspacecast i32 addrspace(5)* %lplusraw to i32*
 %lequal = icmp eq i32* %l,%lback
 %lidentity = zext i1 %lequal to i32
 %lvalue = load i32,i32* %lback,align 4
 %lnextvalue = load i32,i32* %lplus,align 4
 %ldiff64 = sub i64 %lnextbits,%lbits
 %ldiff = trunc i64 %ldiff64 to i32
 %lnextequal = icmp eq i32* %lnext,%lplus
 %lnextidentity = zext i1 %lnextequal to i32
 %o12 = getelementptr i32,i32* %output,i32 12
 store i32 %lmask,i32* %o12,align 4
 %o13 = getelementptr i32,i32* %output,i32 13
 store i32 %lidentity,i32* %o13,align 4
 %o14 = getelementptr i32,i32* %output,i32 14
 store i32 %lvalue,i32* %o14,align 4
 %o15 = getelementptr i32,i32* %output,i32 15
 store i32 %lnextvalue,i32* %o15,align 4
 %o16 = getelementptr i32,i32* %output,i32 16
 store i32 %ldiff,i32* %o16,align 4
 %o17 = getelementptr i32,i32* %output,i32 17
 store i32 %lnextidentity,i32* %o17,align 4
 %cmask = call i32 @address_mask(i32* %c)
 %cspecific = addrspacecast i32* %c to i32 addrspace(4)*
 %cbits = ptrtoint i32 addrspace(4)* %cspecific to i64
 %cnext = getelementptr i32,i32* %c,i32 1
 %cnextspecific = addrspacecast i32* %cnext to i32 addrspace(4)*
 %cnextbits = ptrtoint i32 addrspace(4)* %cnextspecific to i64
 %cbackraw = inttoptr i64 %cbits to i32 addrspace(4)*
 %cback = addrspacecast i32 addrspace(4)* %cbackraw to i32*
 %cplusbits = add i64 %cbits,4
 %cplusraw = inttoptr i64 %cplusbits to i32 addrspace(4)*
 %cplus = addrspacecast i32 addrspace(4)* %cplusraw to i32*
 %cequal = icmp eq i32* %c,%cback
 %cidentity = zext i1 %cequal to i32
 %cvalue = load i32,i32* %cback,align 4
 %cnextvalue = load i32,i32* %cplus,align 4
 %cdiff64 = sub i64 %cnextbits,%cbits
 %cdiff = trunc i64 %cdiff64 to i32
 %cnextequal = icmp eq i32* %cnext,%cplus
 %cnextidentity = zext i1 %cnextequal to i32
 %o18 = getelementptr i32,i32* %output,i32 18
 store i32 %cmask,i32* %o18,align 4
 %o19 = getelementptr i32,i32* %output,i32 19
 store i32 %cidentity,i32* %o19,align 4
 %o20 = getelementptr i32,i32* %output,i32 20
 store i32 %cvalue,i32* %o20,align 4
 %o21 = getelementptr i32,i32* %output,i32 21
 store i32 %cnextvalue,i32* %o21,align 4
 %o22 = getelementptr i32,i32* %output,i32 22
 store i32 %cdiff,i32* %o22,align 4
 %o23 = getelementptr i32,i32* %output,i32 23
 store i32 %cnextidentity,i32* %o23,align 4
 %cgenericbits = ptrtoint i32* %c to i64
 %glow = trunc i64 %gbits to i32
 %gshift = lshr i64 %gbits,32
 %ghigh = trunc i64 %gshift to i32
 %o24 = getelementptr i32,i32* %output,i32 24
 store i32 %glow,i32* %o24,align 4
 %o25 = getelementptr i32,i32* %output,i32 25
 store i32 %ghigh,i32* %o25,align 4
 %clow = trunc i64 %cgenericbits to i32
 %cshift = lshr i64 %cgenericbits,32
 %chigh = trunc i64 %cshift to i32
 %o26 = getelementptr i32,i32* %output,i32 26
 store i32 %clow,i32* %o26,align 4
 %o27 = getelementptr i32,i32* %output,i32 27
 store i32 %chigh,i32* %o27,align 4
 %which0 = add i32 %i,%selector
 %which = and i32 %which0,3
 %isg = icmp eq i32 %which,0
 %iss = icmp eq i32 %which,1
 %isl = icmp eq i32 %which,2
 %picklc = select i1 %isl,i32* %l,i32* %c
 %pickslc = select i1 %iss,i32* %s,i32* %picklc
 %pick = select i1 %isg,i32* %g,i32* %pickslc
 %backlc = select i1 %isl,i32* %lback,i32* %cback
 %backslc = select i1 %iss,i32* %sback,i32* %backlc
 %back = select i1 %isg,i32* %gback,i32* %backslc
 %pickmask = call i32 @address_mask(i32* %pick)
 %pickvalue = load i32,i32* %pick,align 4
 %picknext = getelementptr i32,i32* %pick,i32 1
 %picknextvalue = load i32,i32* %picknext,align 4
 %same = icmp eq i32* %pick,%back
 %identity = zext i1 %same to i32
 %o28 = getelementptr i32,i32* %output,i32 28
 store i32 %pickmask,i32* %o28,align 4
 %o29 = getelementptr i32,i32* %output,i32 29
 store i32 %pickvalue,i32* %o29,align 4
 %o30 = getelementptr i32,i32* %output,i32 30
 store i32 %picknextvalue,i32* %o30,align 4
 %o31 = getelementptr i32,i32* %output,i32 31
 store i32 %identity,i32* %o31,align 4
 %guard32 = load i32,i32* %sp0,align 4
 %o32 = getelementptr i32,i32* %output,i32 32
 store i32 %guard32,i32* %o32,align 4
 %guard33 = load i32,i32* %sp3,align 4
 %o33 = getelementptr i32,i32* %output,i32 33
 store i32 %guard33,i32* %o33,align 4
 %guard34 = load i32,i32* %lp0,align 4
 %o34 = getelementptr i32,i32* %output,i32 34
 store i32 %guard34,i32* %o34,align 4
 %guard35 = load i32,i32* %lp3,align 4
 %o35 = getelementptr i32,i32* %output,i32 35
 store i32 %guard35,i32* %o35,align 4
 br label %done
done: ret void
}
!nvvmir.version = !{!0}
!nvvm.annotations = !{!1}
!0 = !{i32 2,i32 0,i32 3,i32 1}
!1 = !{void (i32*,i32*,i32,i32)* @address_results,!"kernel",i32 1}
