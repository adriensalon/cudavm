// Public DPX overloads; mathematical expectations belong to the host fixture.
__device__ void write_dpx(unsigned* out,unsigned a,unsigned b,unsigned c) {
    bool hi=(a & 1) != 0,lo=(b & 1) != 0;
    out[0]=__vimax_s32_relu((int)a,(int)b);
    out[1]=__vimax_s16x2_relu(a,b);
    out[2]=__vimin_s32_relu((int)a,(int)b);
    out[3]=__vimin_s16x2_relu(a,b);
    out[4]=__vimax3_s32((int)a,(int)b,(int)c);
    out[5]=__vimax3_s16x2(a,b,c);
    out[6]=__vimax3_u32(a,b,c);
    out[7]=__vimax3_u16x2(a,b,c);
    out[8]=__vimin3_s32((int)a,(int)b,(int)c);
    out[9]=__vimin3_s16x2(a,b,c);
    out[10]=__vimin3_u32(a,b,c);
    out[11]=__vimin3_u16x2(a,b,c);
    out[12]=__vimax3_s32_relu((int)a,(int)b,(int)c);
    out[13]=__vimax3_s16x2_relu(a,b,c);
    out[14]=__vimin3_s32_relu((int)a,(int)b,(int)c);
    out[15]=__vimin3_s16x2_relu(a,b,c);
    out[16]=__viaddmax_s32((int)a,(int)b,(int)c);
    out[17]=__viaddmax_s16x2(a,b,c);
    out[18]=__viaddmax_u32(a,b,c);
    out[19]=__viaddmax_u16x2(a,b,c);
    out[20]=__viaddmin_s32((int)a,(int)b,(int)c);
    out[21]=__viaddmin_s16x2(a,b,c);
    out[22]=__viaddmin_u32(a,b,c);
    out[23]=__viaddmin_u16x2(a,b,c);
    out[24]=__viaddmax_s32_relu((int)a,(int)b,(int)c);
    out[25]=__viaddmax_s16x2_relu(a,b,c);
    out[26]=__viaddmin_s32_relu((int)a,(int)b,(int)c);
    out[27]=__viaddmin_s16x2_relu(a,b,c);
    out[28]=__vibmax_s32((int)a,(int)b,&hi);
    out[36]=hi;
    out[29]=__vibmax_s16x2(a,b,&hi,&lo);
    out[37]=hi;
    out[38]=lo;
    out[30]=__vibmax_u32(a,b,&hi);
    out[39]=hi;
    out[31]=__vibmax_u16x2(a,b,&hi,&lo);
    out[40]=hi;
    out[41]=lo;
    out[32]=__vibmin_s32((int)a,(int)b,&hi);
    out[42]=hi;
    out[33]=__vibmin_s16x2(a,b,&hi,&lo);
    out[43]=hi;
    out[44]=lo;
    out[34]=__vibmin_u32(a,b,&hi);
    out[45]=hi;
    out[35]=__vibmin_u16x2(a,b,&hi,&lo);
    out[46]=hi;
    out[47]=lo;
}
extern "C" __global__ void integer_dpx_values(unsigned* output,const unsigned* values,unsigned count) {
    unsigned id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<count)write_dpx(output+id*48,values[id*3],values[id*3+1],values[id*3+2]);
}
