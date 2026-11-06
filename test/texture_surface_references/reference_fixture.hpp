#pragma once
namespace cudavm {
inline constexpr char reference_image[] = R"ptx(.version 7.0
.target sm_75
.address_size 64
.global .texref reference_texture;
.global .surfref reference_surface;
.visible .entry reference_sample(.param .u64 output) {
 .reg .pred %p; .reg .u32 %i,%x,%y; .reg .u64 %out,%offset;
 .reg .f32 %fx,%fy,%f0,%f1,%f2,%f3;
 mov.u32 %i,%tid.x; setp.ge.u32 %p,%i,32; @%p ret;
 rem.u32 %x,%i,8; div.u32 %y,%i,8;
 cvt.rn.f32.u32 %fx,%x; cvt.rn.f32.u32 %fy,%y;
 add.f32 %fx,%fx,0f3f000000; add.f32 %fy,%fy,0f3f000000;
 tex.2d.v4.f32.f32 {%f0,%f1,%f2,%f3},[reference_texture,{%fx,%fy}];
 ld.param.u64 %out,[output]; mul.wide.u32 %offset,%i,16; add.u64 %out,%out,%offset;
 st.global.v4.f32 [%out],{%f0,%f1,%f2,%f3}; ret;
}
.visible .entry reference_read(.param .u64 output) {
 .reg .pred %p; .reg .u32 %i,%x,%y,%r0,%r1,%r2,%r3; .reg .u64 %out,%offset;
 mov.u32 %i,%tid.x; setp.ge.u32 %p,%i,32; @%p ret;
 rem.u32 %x,%i,8; shl.b32 %x,%x,4; div.u32 %y,%i,8;
 suld.b.2d.v4.b32.zero {%r0,%r1,%r2,%r3},[reference_surface,{%x,%y}];
 ld.param.u64 %out,[output]; mul.wide.u32 %offset,%i,16; add.u64 %out,%out,%offset;
 st.global.v4.u32 [%out],{%r0,%r1,%r2,%r3}; ret;
}
.visible .entry reference_write(.param .u64 input) {
 .reg .pred %p; .reg .u32 %i,%x,%y,%r0,%r1,%r2,%r3; .reg .u64 %in,%offset;
 mov.u32 %i,%tid.x; setp.ge.u32 %p,%i,32; @%p ret;
 rem.u32 %x,%i,8; shl.b32 %x,%x,4; div.u32 %y,%i,8;
 ld.param.u64 %in,[input]; mul.wide.u32 %offset,%i,16; add.u64 %in,%in,%offset;
 ld.global.v4.u32 {%r0,%r1,%r2,%r3},[%in];
 sust.b.2d.v4.b32.zero [reference_surface,{%x,%y}],{%r0,%r1,%r2,%r3}; ret;
})ptx";
}
