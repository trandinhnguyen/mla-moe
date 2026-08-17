/* DeepSeek-V2-Lite GPU primitives for MI250 (gfx90a).
 *
 * The checkpoint stores row-major BF16 matrices while activations remain FP32
 * to match the CPU oracle. These kernels deliberately expose small C launch
 * functions so getp_run.hip can compose them without changing frozen headers.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr int BLOCK_SIZE = 256;
uint16_t *mfma_input=nullptr;
size_t mfma_input_capacity=0;

__device__ __forceinline__ float load_bf16(const uint16_t *p) {
    return __uint_as_float((uint32_t)(*p) << 16);
}

__device__ __forceinline__ float wave_sum(float value) {
    for (int offset = 32; offset > 0; offset >>= 1)
        value += __shfl_down(value, offset, 64);
    return value;
}
__device__ __forceinline__ float wave_max(float value) {
    for (int offset = 32; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down(value, offset, 64));
    return value;
}

/* Batched prefill kernels. A workgroup contains four independent wave64
 * matrix-vector reductions, amortizing scheduling overhead while sharing the
 * same weight matrix across prompt positions. */
__global__ void batched_bf16_matvec_kernel(
    float *output, const float *input, const uint16_t *weight,
    int batch, int rows, int cols) {
    int token = blockIdx.y;
    int wave = threadIdx.x >> 6;
    int lane = threadIdx.x & 63;
    int row = blockIdx.x * 4 + wave;
    if (token >= batch || row >= rows) return;
    const float *x = input + (size_t)token * cols;
    const uint16_t *w = weight + (size_t)row * cols;
    float sum = 0.0f;
    for(int col=lane*2;col<cols;col+=128){uint32_t packed=
            *reinterpret_cast<const uint32_t *>(w+col);
        sum+=x[col]*__uint_as_float((packed&0xffffu)<<16)
            +x[col+1]*__uint_as_float(packed&0xffff0000u);}
    sum = wave_sum(sum);
    if (lane == 0) output[(size_t)token * rows + row] = sum;
}

/* One wave computes a 16x16 output tile with the native gfx90a FP32 MFMA.
 * Keeping activations and accumulation in FP32 avoids the model-wide drift
 * seen with BF16 activations, while still replacing 256 scalar dot products
 * with the matrix instruction. W is [rows,K], input is [batch,K], and output
 * is [batch,rows]. The lane mapping below is the hardware mapping for
 * mfma_f32_16x16x4f32: lane%16 selects the input/output column, lane/16 the
 * K value, and the four accumulators cover rows 4*(lane/16)+{0,1,2,3}. */
using float4_acc = float __attribute__((ext_vector_type(4)));
using int4_acc = int __attribute__((ext_vector_type(4)));
using float2_reg = float __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint16_t to_bf16_rne(float value){
    uint32_t bits=__float_as_uint(value);
    bits+=0x7fffu+((bits>>16)&1u);return (uint16_t)(bits>>16);
}

__global__ void convert_bf16_kernel(uint16_t *output,const float *input,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)output[i]=to_bf16_rne(input[i]);
}

__global__ void mfma_bf16_gemm_32x64(float *output,const uint16_t *input,
    const uint16_t *weight,int batch,int rows,int cols){
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    int mt=wave&3,nt=wave>>2,tile_m=blockIdx.x*64,tile_n=blockIdx.y*64;
    int kg=(lane>>4)*4;float4_acc acc0={0,0,0,0},acc1={0,0,0,0};
    __shared__ uint16_t as[2][1024],bs[2][1024];
    for(int z=tid;z<1024;z+=512){int r=z/16,k=z%16;
        as[0][z]=tile_m+r<rows?weight[(size_t)(tile_m+r)*cols+k]:0;
        int n=z/16;bs[0][z]=tile_n+n<batch?input[(size_t)(tile_n+n)*cols+k]:0;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=16){int cur=(kb>>4)&1,nxt=cur^1;
        int ar=(mt*16+(lane&15))*16+kg;
        int br0=(nt*16+(lane&15))*16+kg,br1=br0+32*16;
        float2_reg av={__uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar)),
                       __uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar+2))};
        float2_reg bv0={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+br0)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+br0+2))};
        float2_reg bv1={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+br1)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+br1+2))};
        uint16_t na0=0,nb0=0,na1=0,nb1=0;
        if(kb+16<cols){int k=tid&15,r=tid>>4,n=r;
            if(tile_m+r<rows)na0=weight[(size_t)(tile_m+r)*cols+kb+16+k];
            if(tile_n+n<batch)nb0=input[(size_t)(tile_n+n)*cols+kb+16+k];
            r+=32;n+=32;
            if(tile_m+r<rows)na1=weight[(size_t)(tile_m+r)*cols+kb+16+k];
            if(tile_n+n<batch)nb1=input[(size_t)(tile_n+n)*cols+kb+16+k];}
        acc0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av,bv0,acc0,0,0,0);
        acc1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av,bv1,acc1,0,0,0);
        if(kb+16<cols){as[nxt][tid]=na0;bs[nxt][tid]=nb0;
            as[nxt][tid+512]=na1;bs[nxt][tid+512]=nb1;}
        __syncthreads();}
    int r=tile_m+mt*16+(lane>>4)*4,n0=tile_n+nt*16+(lane&15),n1=n0+32;
    for(int i=0;i<4;++i){if(n0<batch&&r+i<rows)output[(size_t)n0*rows+r+i]=acc0[i];
        if(n1<batch&&r+i<rows)output[(size_t)n1*rows+r+i]=acc1[i];}
}

__global__ void mfma_bf16_gemm_64x128(float *output,const uint16_t *input,
    const uint16_t *weight,int batch,int rows,int cols){
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6,mt=wave&1,nt=wave>>1;
    int tile_m=blockIdx.x*64,tile_n=blockIdx.y*128,kg=(lane>>4)*4;
    float4_acc a0={0,0,0,0},a1={0,0,0,0},b0={0,0,0,0},b1={0,0,0,0};
    __shared__ uint16_t as[2][1024],bs[2][2048];
    for(int z=tid;z<1024;z+=512){int r=z/16,k=z%16;
        as[0][z]=tile_m+r<rows?weight[(size_t)(tile_m+r)*cols+k]:0;
        int n=z/16;bs[0][z]=tile_n+n<batch?input[(size_t)(tile_n+n)*cols+k]:0;
        n+=64;bs[0][z+1024]=tile_n+n<batch?input[(size_t)(tile_n+n)*cols+k]:0;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=16){int cur=(kb>>4)&1,nxt=cur^1;
        int ai0=(mt*16+(lane&15))*16+kg,ai1=ai0+32*16;
        int bi=(nt*16+(lane&15))*16+kg,bi2=bi+64*16;
        float2_reg av0={__uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ai0)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ai0+2))};
        float2_reg av1={__uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ai1)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ai1+2))};
        float2_reg bv0={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi+2))};
        float2_reg bv1={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi2)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi2+2))};
        uint16_t na0=0,na1=0,nb0=0,nb1=0,nb2=0,nb3=0;
        if(kb+16<cols){int k=tid&15,r=tid>>4,n=r;
            if(tile_m+r<rows)na0=weight[(size_t)(tile_m+r)*cols+kb+16+k];
            if(tile_n+n<batch)nb0=input[(size_t)(tile_n+n)*cols+kb+16+k];
            r+=32;n+=32;
            if(tile_m+r<rows)na1=weight[(size_t)(tile_m+r)*cols+kb+16+k];
            if(tile_n+n<batch)nb1=input[(size_t)(tile_n+n)*cols+kb+16+k];
            n=(tid>>4)+64;if(tile_n+n<batch)nb2=input[(size_t)(tile_n+n)*cols+kb+16+k];
            n+=32;if(tile_n+n<batch)nb3=input[(size_t)(tile_n+n)*cols+kb+16+k];}
        a0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av0,bv0,a0,0,0,0);
        a1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av1,bv0,a1,0,0,0);
        b0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av0,bv1,b0,0,0,0);
        b1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av1,bv1,b1,0,0,0);
        if(kb+16<cols){as[nxt][tid]=na0;bs[nxt][tid]=nb0;
            as[nxt][tid+512]=na1;bs[nxt][tid+512]=nb1;
            bs[nxt][tid+1024]=nb2;bs[nxt][tid+1536]=nb3;}
        __syncthreads();}
    int r0=tile_m+mt*16+(lane>>4)*4,r1=r0+32;
    int n0=tile_n+nt*16+(lane&15),n1=n0+64;
    #pragma unroll
    for(int i=0;i<4;++i){
        if(n0<batch&&r0+i<rows)output[(size_t)n0*rows+r0+i]=a0[i];
        if(n0<batch&&r1+i<rows)output[(size_t)n0*rows+r1+i]=a1[i];
        if(n1<batch&&r0+i<rows)output[(size_t)n1*rows+r0+i]=b0[i];
        if(n1<batch&&r1+i<rows)output[(size_t)n1*rows+r1+i]=b1[i];}
}

__global__ void pack_head_bf16(uint16_t *output,const float *input,int batch,
    int heads,int input_stride,int cols){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)batch*heads*cols;
    if(i<total){int col=i%cols,head=(i/cols)%heads,token=i/(cols*heads);
        output[((size_t)head*batch+token)*cols+col]=
            to_bf16_rne(input[((size_t)token*heads+head)*input_stride+col]);}
}
/* Per-head GEMM for MLA absorption/value projection.  Head-major packed input
 * reuses each weight tile across 64 tokens; output is restored token-major. */
__global__ void grouped_head_mfma_32x64(float *output,const uint16_t *input,
    const uint16_t *weight,int batch,int heads,int rows,int cols,
    int weight_head_stride,int weight_row_offset,int transposed,int shared_input){
    int head=blockIdx.z,tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    int mt=wave&1,nt=wave>>1,tile_m=blockIdx.x*64,tile_n=blockIdx.y*64;
    int kg=(lane>>4)*4;float4_acc acc0={0,0,0,0},acc1={0,0,0,0};
    const uint16_t *w=weight+(size_t)head*weight_head_stride;
    const uint16_t *x=input+(shared_input?0:(size_t)head*batch*cols);
    __shared__ uint16_t as[2][1024],bs[2][1024];
    for(int z=tid;z<1024;z+=512){int r=z/16,k=z%16;
        if(tile_m+r<rows)as[0][z]=transposed?w[(size_t)k*rows+tile_m+r]:
            w[(size_t)(weight_row_offset+tile_m+r)*cols+k];else as[0][z]=0;
        int n=z/16;bs[0][z]=tile_n+n<batch?x[(size_t)(tile_n+n)*cols+k]:0;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=16){int cur=(kb>>4)&1,nxt=cur^1;
        int ar0=(mt*16+(lane&15))*16+kg;
        int ar1=((mt+2)*16+(lane&15))*16+kg,br=(nt*16+(lane&15))*16+kg;
        float2_reg av0={__uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar0)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar0+2))};
        float2_reg av1={__uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar1)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar1+2))};
        float2_reg bv={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+br)),
                       __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+br+2))};
        uint16_t na0=0,nb0=0,na1=0,nb1=0;
        if(kb+16<cols){int k=tid&15,r=tid>>4,n=r;
            if(tile_m+r<rows)na0=transposed?
                w[(size_t)(kb+16+k)*rows+tile_m+r]:
                w[(size_t)(weight_row_offset+tile_m+r)*cols+kb+16+k];
            if(tile_n+n<batch)nb0=x[(size_t)(tile_n+n)*cols+kb+16+k];
            r+=32;n+=32;
            if(tile_m+r<rows)na1=transposed?
                w[(size_t)(kb+16+k)*rows+tile_m+r]:
                w[(size_t)(weight_row_offset+tile_m+r)*cols+kb+16+k];
            if(tile_n+n<batch)nb1=x[(size_t)(tile_n+n)*cols+kb+16+k];}
        acc0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av0,bv,acc0,0,0,0);
        acc1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av1,bv,acc1,0,0,0);
        if(kb+16<cols){as[nxt][tid]=na0;bs[nxt][tid]=nb0;
            as[nxt][tid+512]=na1;bs[nxt][tid+512]=nb1;}
        __syncthreads();}
    int r0=tile_m+mt*16+(lane>>4)*4,r1=r0+32,n=tile_n+nt*16+(lane&15);
    for(int i=0;i<4;++i){if(n<batch&&r0+i<rows)
        output[((size_t)n*heads+head)*rows+r0+i]=acc0[i];
        if(n<batch&&r1+i<rows)output[((size_t)n*heads+head)*rows+r1+i]=acc1[i];}
}

/* LM-head GEMM epilogue: retain only one maximum per request/vocabulary tile.
 * This avoids writing and rereading batch*vocab FP32 logits. */
__global__ void mfma_bf16_gemm_argmax_32x64(float *partial_value,int *partial_index,
    const uint16_t *input,const uint16_t *weight,int batch,int rows,int cols,int tiles){
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6;
    int mt=wave&1,nt=wave>>1,tile_m=blockIdx.x*64,tile_n=blockIdx.y*64;
    int kg=(lane>>4)*4;float4_acc acc0={0,0,0,0},acc1={0,0,0,0};
    __shared__ uint16_t as[2][1024],bs[2][1024];
    __shared__ float wave_value[128];__shared__ int wave_index[128];
    for(int z=tid;z<1024;z+=512){int r=z/16,k=z%16;
        as[0][z]=tile_m+r<rows?weight[(size_t)(tile_m+r)*cols+k]:0;
        int n=z/16;bs[0][z]=tile_n+n<batch?input[(size_t)(tile_n+n)*cols+k]:0;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=16){int cur=(kb>>4)&1,nxt=cur^1;
        int ar0=(mt*16+(lane&15))*16+kg,ar1=((mt+2)*16+(lane&15))*16+kg;
        int br=(nt*16+(lane&15))*16+kg;
        float2_reg av0={__uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar0)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar0+2))};
        float2_reg av1={__uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar1)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(as[cur]+ar1+2))};
        float2_reg bv={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+br)),
                       __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+br+2))};
        uint16_t na0=0,nb0=0,na1=0,nb1=0;
        if(kb+16<cols){int k=tid&15,r=tid>>4,n=r;
            if(tile_m+r<rows)na0=weight[(size_t)(tile_m+r)*cols+kb+16+k];
            if(tile_n+n<batch)nb0=input[(size_t)(tile_n+n)*cols+kb+16+k];
            r+=32;n+=32;
            if(tile_m+r<rows)na1=weight[(size_t)(tile_m+r)*cols+kb+16+k];
            if(tile_n+n<batch)nb1=input[(size_t)(tile_n+n)*cols+kb+16+k];}
        acc0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av0,bv,acc0,0,0,0);
        acc1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av1,bv,acc1,0,0,0);
        if(kb+16<cols){as[nxt][tid]=na0;bs[nxt][tid]=nb0;
            as[nxt][tid+512]=na1;bs[nxt][tid+512]=nb1;}
        __syncthreads();}
    int base0=tile_m+mt*16+(lane>>4)*4,base1=base0+32;
    float best=-INFINITY;int best_i=rows;
    for(int i=0;i<4;++i){if(base0+i<rows&&(acc0[i]>best||
        (acc0[i]==best&&base0+i<best_i))){best=acc0[i];best_i=base0+i;}
        if(base1+i<rows&&(acc1[i]>best||(acc1[i]==best&&base1+i<best_i))){
            best=acc1[i];best_i=base1+i;}}
    for(int offset=16;offset<=32;offset*=2){float ov=__shfl_down(best,offset,64);
        int oi=__shfl_down(best_i,offset,64);if((lane&63)+offset<64&&
        (ov>best||(ov==best&&oi<best_i))){best=ov;best_i=oi;}}
    if(lane<16){int o=wave*16+lane;wave_value[o]=best;wave_index[o]=best_i;}
    __syncthreads();
    if(tid<64){int ntile=tid>>4,nlocal=tid&15,n=tile_n+ntile*16+nlocal;
        if(n<batch){int o0=(ntile*2)*16+nlocal,o1=o0+16;
            float v0=wave_value[o0],v1=wave_value[o1];
            int i0=wave_index[o0],i1=wave_index[o1];
            if(v1>v0||(v1==v0&&i1<i0)){v0=v1;i0=i1;}
            size_t o=(size_t)n*tiles+blockIdx.x;partial_value[o]=v0;partial_index[o]=i0;}}
}

__global__ void mfma_bf16_swiglu_32x64(float *output,const uint16_t *input,
    const uint16_t *gate,const uint16_t *up,int batch,int rows,int cols){
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6,mt=wave&1,nt=wave>>1;
    int tile_m=blockIdx.x*64,tile_n=blockIdx.y*64,kg=(lane>>4)*4;
    float4_acc ag0={0,0,0,0},au0={0,0,0,0};
    float4_acc ag1={0,0,0,0},au1={0,0,0,0};
    __shared__ uint16_t gs[2][1024],us[2][1024],bs[2][1024];
    for(int z=tid;z<1024;z+=512){int r=z/16,k=z%16;
        if(tile_m+r<rows){gs[0][z]=gate[(size_t)(tile_m+r)*cols+k];
            us[0][z]=up[(size_t)(tile_m+r)*cols+k];}else gs[0][z]=us[0][z]=0;
        int n=z/16;bs[0][z]=tile_n+n<batch?input[(size_t)(tile_n+n)*cols+k]:0;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=16){int cur=(kb>>4)&1,nxt=cur^1;
        int ai0=(mt*16+(lane&15))*16+kg;
        int ai1=((mt+2)*16+(lane&15))*16+kg,bi=(nt*16+(lane&15))*16+kg;
        float2_reg gv0={__uint_as_float(*reinterpret_cast<uint32_t *>(gs[cur]+ai0)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(gs[cur]+ai0+2))};
        float2_reg uv0={__uint_as_float(*reinterpret_cast<uint32_t *>(us[cur]+ai0)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(us[cur]+ai0+2))};
        float2_reg gv1={__uint_as_float(*reinterpret_cast<uint32_t *>(gs[cur]+ai1)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(gs[cur]+ai1+2))};
        float2_reg uv1={__uint_as_float(*reinterpret_cast<uint32_t *>(us[cur]+ai1)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(us[cur]+ai1+2))};
        float2_reg bv={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi)),
                       __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi+2))};
        uint16_t ng0=0,nu0=0,nb0=0,ng1=0,nu1=0,nb1=0;
        if(kb+16<cols){int k=tid&15,r=tid>>4,n=r;
            if(tile_m+r<rows){ng0=gate[(size_t)(tile_m+r)*cols+kb+16+k];
                nu0=up[(size_t)(tile_m+r)*cols+kb+16+k];}
            if(tile_n+n<batch)nb0=input[(size_t)(tile_n+n)*cols+kb+16+k];
            r+=32;n+=32;
            if(tile_m+r<rows){ng1=gate[(size_t)(tile_m+r)*cols+kb+16+k];
                nu1=up[(size_t)(tile_m+r)*cols+kb+16+k];}
            if(tile_n+n<batch)nb1=input[(size_t)(tile_n+n)*cols+kb+16+k];}
        ag0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv0,bv,ag0,0,0,0);
        au0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv0,bv,au0,0,0,0);
        ag1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv1,bv,ag1,0,0,0);
        au1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv1,bv,au1,0,0,0);
        if(kb+16<cols){gs[nxt][tid]=ng0;us[nxt][tid]=nu0;bs[nxt][tid]=nb0;
            gs[nxt][tid+512]=ng1;us[nxt][tid+512]=nu1;bs[nxt][tid+512]=nb1;}
        __syncthreads();}
    int r0=tile_m+mt*16+(lane>>4)*4,r1=r0+32,n=tile_n+nt*16+(lane&15);
    for(int i=0;i<4;++i){
        if(n<batch&&r0+i<rows){float g=ag0[i];
            output[(size_t)n*rows+r0+i]=(g/(1.0f+expf(-g)))*au0[i];}
        if(n<batch&&r1+i<rows){float g=ag1[i];
            output[(size_t)n*rows+r1+i]=(g/(1.0f+expf(-g)))*au1[i];}}
}

__global__ void batched_rmsnorm_kernel(
    float *output, const float *input, const uint16_t *weight,
    int batch, int count, float eps) {
    int token = blockIdx.x;
    if (token >= batch) return;
    const float *x = input + (size_t)token * count;
    float *y = output + (size_t)token * count;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        sum += x[i] * x[i];
    sum = wave_sum(sum);
    __shared__ float waves[BLOCK_SIZE / 64];
    __shared__ float inverse;
    int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    if (lane == 0) waves[wave] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE / 64; ++i) sum += waves[i];
        inverse = rsqrtf(sum / count + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        y[i] = x[i] * inverse * load_bf16(weight + i);
}

__global__ void batched_rmsnorm_strided_kernel(
    float *output, int output_stride, const float *input, int input_stride,
    const uint16_t *weight, int batch, int count, float eps) {
    int token=blockIdx.x;const float *x=input+(size_t)token*input_stride;
    float *y=output+(size_t)token*output_stride;float sum=0.0f;
    for(int i=threadIdx.x;i<count;i+=blockDim.x)sum+=x[i]*x[i];sum=wave_sum(sum);
    __shared__ float waves[4],inv;int lane=threadIdx.x&63,wave=threadIdx.x>>6;
    if(lane==0)waves[wave]=sum;__syncthreads();if(threadIdx.x==0){sum=0;for(int i=0;i<4;++i)sum+=waves[i];inv=rsqrtf(sum/count+eps);}__syncthreads();
    for(int i=threadIdx.x;i<count;i+=blockDim.x)y[i]=x[i]*inv*load_bf16(weight+i);
}

__global__ void batched_embedding_kernel(float *output, const uint16_t *table,
                                         const int *tokens, int batch,
                                         int hidden) {
    int token_index = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_index < batch && i < hidden)
        output[(size_t)token_index * hidden + i] =
            load_bf16(table + (size_t)tokens[token_index] * hidden + i);
}

__global__ void batched_residual_kernel(float *state, const float *branch,
                                        int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) state[i] += branch[i];
}

__global__ void batched_residual_rmsnorm_kernel(float *state,float *normalized,
    const float *branch,const uint16_t *weight,int count,float eps){
    int token=blockIdx.x;float *x=state+(size_t)token*count;
    const float *b=branch+(size_t)token*count;float sum=0.0f;
    for(int i=threadIdx.x;i<count;i+=blockDim.x){float value=x[i]+b[i];
        x[i]=value;sum+=value*value;}
    sum=wave_sum(sum);__shared__ float waves[4],inverse;
    int lane=threadIdx.x&63,wave=threadIdx.x>>6;
    if(lane==0)waves[wave]=sum;__syncthreads();
    if(threadIdx.x==0){sum=waves[0]+waves[1]+waves[2]+waves[3];
        inverse=rsqrtf(sum/count+eps);}__syncthreads();
    float *y=normalized+(size_t)token*count;
    for(int i=threadIdx.x;i<count;i+=blockDim.x)
        y[i]=x[i]*inverse*load_bf16(weight+i);
}

__global__ void batched_rope_kernel(float *query, int batch, int heads,
                                    int head_dim, int nope_dim, int rope_dim,
                                    const float *inverse_frequency,
                                    int sequence_length) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rope_dim / 2;
    int total = batch * heads * half;
    if (pair >= total) return;
    int j = pair % half;
    int vector = pair / half;
    int position = (vector / heads) % sequence_length;
    float *v = query + (size_t)vector * head_dim + nope_dim;
    float angle = position * inverse_frequency[j];
    float cs = cosf(angle), sn = sinf(angle);
    float x0 = v[2*j], x1 = v[2*j+1];
    v[2*j] = x0 * cs - x1 * sn;
    v[2*j+1] = x0 * sn + x1 * cs;
}

__global__ void prefill_cache_kernel(float *cache, const float *normalized,
                                     const float *compressed, int batch,
                                     int cache_stride, int latent_dim,
                                     int rope_dim,
                                     const float *inverse_frequency,
                                     int sequence_length, int request_base,
                                     int cache_sequence_length) {
    int token = blockIdx.x;
    int local_request = token / sequence_length;
    int position = token % sequence_length;
    float *dst = cache + ((size_t)(request_base + local_request) *
                          cache_sequence_length + position) * cache_stride;
    for (int i = threadIdx.x; i < latent_dim; i += blockDim.x)
        dst[i] = normalized[(size_t)token * latent_dim + i];
    for (int i = threadIdx.x; i < rope_dim; i += blockDim.x)
        dst[latent_dim+i]=compressed[(size_t)token*(latent_dim+rope_dim)+latent_dim+i];
    __syncthreads();
    for (int j = threadIdx.x; j < rope_dim / 2; j += blockDim.x) {
        float angle = position * inverse_frequency[j];
        float cs = cosf(angle), sn = sinf(angle);
        float *v=dst+latent_dim;float x0=v[2*j],x1=v[2*j+1];
        v[2*j]=x0*cs-x1*sn;v[2*j+1]=x0*sn+x1*cs;
    }
}

/* Online causal softmax: no [sequence,heads,sequence] score tensor. Each block
 * computes one query/head latent context and maintains a numerically stable
 * running maximum and normalization while streaming cache rows. */
__global__ void prefill_attention_kernel(
    float *context, const float *q_absorbed, const float *query,
    const float *cache, int batch, int cache_stride, int heads,
    int q_head_dim, int nope_dim, int latent_dim, int rope_dim, float scale,
    int sequence_length, int request_base, int cache_sequence_length) {
    int query_row = blockIdx.x, h = blockIdx.y;
    int local_request = query_row / sequence_length;
    int qpos = query_row % sequence_length;
    const float *qa = q_absorbed + ((size_t)query_row * heads + h) * latent_dim;
    const float *qp = query + ((size_t)query_row * heads + h) * q_head_dim + nope_dim;
    float acc0 = 0.0f, acc1 = 0.0f;
    float running_max = -INFINITY, running_sum = 0.0f;
    __shared__ float waves[BLOCK_SIZE / 64];
    __shared__ float score_shared, alpha_shared, beta_shared, sum_shared;
    int lane=threadIdx.x&63, wave=threadIdx.x>>6;
    for(int key=0; key<=qpos; ++key){
        const float *kv=cache+((size_t)(request_base+local_request)*
                               cache_sequence_length+key)*cache_stride;
        float score=0.0f;
        for(int i=threadIdx.x;i<latent_dim;i+=blockDim.x) score+=qa[i]*kv[i];
        for(int i=threadIdx.x;i<rope_dim;i+=blockDim.x) score+=qp[i]*kv[latent_dim+i];
        score=wave_sum(score);
        if(lane==0) waves[wave]=score;
        __syncthreads();
        if(threadIdx.x==0){
            score=0.0f; for(int i=0;i<4;++i) score+=waves[i]; score*=scale;
            float next_max=fmaxf(running_max,score);
            alpha_shared=expf(running_max-next_max);
            beta_shared=expf(score-next_max);
            sum_shared=running_sum*alpha_shared+beta_shared;
            score_shared=next_max;
        }
        __syncthreads();
        acc0=acc0*alpha_shared+beta_shared*kv[threadIdx.x];
        if(threadIdx.x+BLOCK_SIZE<latent_dim)
            acc1=acc1*alpha_shared+beta_shared*kv[threadIdx.x+BLOCK_SIZE];
        running_max=score_shared; running_sum=sum_shared;
        __syncthreads();
    }
    float *out=context+((size_t)query_row*heads+h)*latent_dim;
    out[threadIdx.x]=acc0/running_sum;
    if(threadIdx.x+BLOCK_SIZE<latent_dim) out[threadIdx.x+BLOCK_SIZE]=acc1/running_sum;
}

__global__ void decode_rope_kernel(float *query,int batch,int heads,int head_dim,
    int nope_dim,int rope_dim,const float *frequency,const int *positions){
    int pair=blockIdx.x*blockDim.x+threadIdx.x,half=rope_dim/2;
    if(pair>=batch*heads*half)return;int j=pair%half,vector=pair/half;
    int request=vector/heads;float *v=query+(size_t)vector*head_dim+nope_dim;
    float angle=positions[request]*frequency[j],cs=cosf(angle),sn=sinf(angle);
    float x0=v[2*j],x1=v[2*j+1];v[2*j]=x0*cs-x1*sn;v[2*j+1]=x0*sn+x1*cs;
}

__global__ void decode_cache_kernel(float *cache,const float *normalized,
    const float *compressed,int batch,int stride,int latent,int rope,
    const float *frequency,const int *positions,int cache_sequence){
    int request=blockIdx.x,pos=positions[request];
    float *dst=cache+((size_t)request*cache_sequence+pos)*stride;
    for(int i=threadIdx.x;i<latent;i+=blockDim.x)
        dst[i]=normalized[(size_t)request*latent+i];
    for(int i=threadIdx.x;i<rope;i+=blockDim.x)
        dst[latent+i]=compressed[(size_t)request*(latent+rope)+latent+i];
    __syncthreads();for(int j=threadIdx.x;j<rope/2;j+=blockDim.x){
        float angle=pos*frequency[j],cs=cosf(angle),sn=sinf(angle);
        float *v=dst+latent;float x0=v[2*j],x1=v[2*j+1];
        v[2*j]=x0*cs-x1*sn;v[2*j+1]=x0*sn+x1*cs;}
}

/* One wave per head.  All 16 heads in a request consume the same compressed
 * KV row, so stage it once and reuse it across the entire block. */
__global__ void decode_attention_shared_heads(float *context,const float *qa,
    const float *query,const float *cache,int batch,int stride,int heads,int qhd,
    int nope,int latent,int rope,float scale,const int *positions,int cache_sequence){
    int request=blockIdx.x,h=threadIdx.x>>6,lane=threadIdx.x&63;
    if(h>=heads)return;int pos=positions[request];
    const float *qav=qa+((size_t)request*heads+h)*latent;
    const float *qp=query+((size_t)request*heads+h)*qhd+nope;
    float acc[8]={0,0,0,0,0,0,0,0};
    float rmax=-INFINITY,rsum=0;
    __shared__ float kvs[8][576];
    const float *kv0=cache+(size_t)request*cache_sequence*stride;
    for(int i=threadIdx.x;i<stride;i+=blockDim.x)kvs[0][i]=kv0[i];
    for(int first=1;first<4&&first<=pos;++first){const float *kv=kv0+first*stride;
        for(int i=threadIdx.x;i<stride;i+=blockDim.x)kvs[first][i]=kv[i];}
    __syncthreads();
    for(int key=0;key<=pos;key+=4){
        int curbase=((key>>2)&1)*4,nxtbase=curbase^4;
        float next0=0,next1=0,next2=0,next3=0;
        if(key+4<=pos&&threadIdx.x<stride){const float *next=cache+
            ((size_t)request*cache_sequence+key+4)*stride;
            next0=next[threadIdx.x];
            if(key+5<=pos)next1=next[stride+threadIdx.x];
            if(key+6<=pos)next2=next[2*stride+threadIdx.x];
            if(key+7<=pos)next3=next[3*stride+threadIdx.x];}
        #pragma unroll
        for(int pair=0;pair<4;++pair){
            if(key+pair<=pos){float score=0;
                #pragma unroll
                for(int j=0;j<8;++j){int i=lane+j*64;
                    score+=qav[i]*kvs[curbase+pair][i];}
                if(lane<rope)score+=qp[lane]*kvs[curbase+pair][latent+lane];
                score=wave_sum(score);float a=0,b=0,nm=0,ns=0;
                if(lane==0){score*=scale;nm=fmaxf(rmax,score);
                    a=__expf(rmax-nm);b=__expf(score-nm);ns=rsum*a+b;
                    rmax=nm;rsum=ns;}
                a=__shfl(a,0,64);b=__shfl(b,0,64);
                #pragma unroll
                for(int j=0;j<8;++j)acc[j]=acc[j]*a+b*kvs[curbase+pair][lane+j*64];}}
        if(key+4<=pos){if(threadIdx.x<stride){kvs[nxtbase][threadIdx.x]=next0;
                if(key+5<=pos)kvs[nxtbase+1][threadIdx.x]=next1;
                if(key+6<=pos)kvs[nxtbase+2][threadIdx.x]=next2;
                if(key+7<=pos)kvs[nxtbase+3][threadIdx.x]=next3;}
            __syncthreads();}
    }
    float *out=context+((size_t)request*heads+h)*latent;
    float denom=__shfl(rsum,0,64);
    #pragma unroll
    for(int j=0;j<8;++j)out[lane+j*64]=acc[j]/denom;
}

__global__ void prefill_unabsorbed_attention_q4(float *output,const float *query,
    const float *key,const float *value,const float *cache,int rows,int heads,int qhd,
    int nope,int vdim,int cache_stride,int latent,int rope,int sequence_length,
    int request_base,int cache_sequence,float scale){
    int wave=threadIdx.x>>6,lane=threadIdx.x&63,tiles=(sequence_length+3)/4;
    int request=blockIdx.x/tiles,qbase=(blockIdx.x%tiles)*4,qpos=qbase+wave;
    int row=request*sequence_length+qpos,h=blockIdx.y;
    const float *q=query+((size_t)min(row,rows-1)*heads+h)*qhd;
    float acc0=0.0f,acc1=0.0f,rmax=-INFINITY,rsum=0.0f;
    __shared__ float sk[2][192],sv[2][128];
    int last=min(sequence_length-1,qbase+3);
    int kr0=request*sequence_length;
    if(threadIdx.x<nope)sk[0][threadIdx.x]=key[((size_t)kr0*heads+h)*nope+threadIdx.x];
    const float *rk0=cache+((size_t)(request_base+request)*cache_sequence)*cache_stride+latent;
    if(threadIdx.x<rope)sk[0][nope+threadIdx.x]=rk0[threadIdx.x];
    if(threadIdx.x<vdim)sv[0][threadIdx.x]=value[((size_t)kr0*heads+h)*vdim+threadIdx.x];
    __syncthreads();
    for(int k=0;k<=last;++k){int kr=request*sequence_length+k,cur=k&1,nxt=cur^1;
        float next_k=0,next_r=0,next_v=0;
        if(k<last){int knr=kr+1;
            if(threadIdx.x<nope)next_k=key[((size_t)knr*heads+h)*nope+threadIdx.x];
            const float *rkn=cache+((size_t)(request_base+request)*cache_sequence+k+1)*cache_stride+latent;
            if(threadIdx.x<rope)next_r=rkn[threadIdx.x];
            if(threadIdx.x<vdim)next_v=value[((size_t)knr*heads+h)*vdim+threadIdx.x];}
        float score=0.0f;
        if(qpos<sequence_length&&k<=qpos)
            for(int d=lane;d<nope+rope;d+=64)score+=q[d]*sk[cur][d];
        score=wave_sum(score);float a=0,b=0,nm=0,ns=0;
        if(lane==0){if(qpos<sequence_length&&k<=qpos){score*=scale;
                nm=fmaxf(rmax,score);a=__expf(rmax-nm);b=__expf(score-nm);ns=rsum*a+b;
                rmax=nm;rsum=ns;}
            else{nm=rmax;ns=rsum;a=1.0f;b=0.0f;}}
        a=__shfl(a,0,64);b=__shfl(b,0,64);
        acc0=acc0*a+b*sv[cur][lane];acc1=acc1*a+b*sv[cur][lane+64];
        if(k<last){if(threadIdx.x<nope)sk[nxt][threadIdx.x]=next_k;
            if(threadIdx.x<rope)sk[nxt][nope+threadIdx.x]=next_r;
            if(threadIdx.x<vdim)sv[nxt][threadIdx.x]=next_v;}
        __syncthreads();}
    if(qpos<sequence_length){float *out=output+((size_t)row*heads+h)*vdim;
        float denom=__shfl(rsum,0,64);
        out[lane]=acc0/denom;out[lane+64]=acc1/denom;}
}

__global__ void batched_router_topk_kernel(const float *logits, int batch,
                                           int experts, int topk,
                                           int *indices, float *weights,
                                           int *counts) {
    int token=blockIdx.x,e=threadIdx.x;
    float v=e<experts?logits[(size_t)token*experts+e]:-INFINITY,m=v;
    for(int o=32;o>0;o>>=1)m=fmaxf(m,__shfl_down(m,o,64)); m=__shfl(m,0,64);
    float x=e<experts?expf(v-m):0.0f,s=wave_sum(x); s=__shfl(s,0,64);
    float score=e<experts?x/s:-INFINITY;
    for(int rank=0;rank<topk;++rank){float best_value=score;int best=e;
        for(int offset=32;offset>0;offset>>=1){float other_value=__shfl_down(best_value,offset,64);
            int other=__shfl_down(best,offset,64);if(e+offset<64&&(other_value>best_value||
                (other_value==best_value&&other<best))){best_value=other_value;best=other;}}
        best=__shfl(best,0,64);best_value=__shfl(best_value,0,64);
        if(e==0){indices[(size_t)token*topk+rank]=best;
            weights[(size_t)token*topk+rank]=best_value;
            atomicAdd(counts+best,1);}if(e==best)score=-INFINITY;}
}

__global__ void route_prefix_descriptors_kernel(const int *counts,int *offsets,
    int *cursors,int *tile_experts,int *tile_starts,int experts,int max_tiles,
    int route_tile){
    int e=threadIdx.x;
    int count=e<experts?counts[e]:0;
    int route_sum=count;
    int tile_count=(count+route_tile-1)/route_tile,tile_sum=tile_count;
    for(int delta=1;delta<64;delta<<=1){
        int route_other=__shfl_up(route_sum,delta,64);
        int tile_other=__shfl_up(tile_sum,delta,64);
        if(e>=delta){route_sum+=route_other;tile_sum+=tile_other;}
    }
    if(e<experts){
        int route_offset=route_sum-count;
        int tile_offset=tile_sum-tile_count;
        offsets[e]=route_offset;cursors[e]=0;
        for(int tile=0;tile<tile_count;++tile){
            tile_experts[tile_offset+tile]=e;
            tile_starts[tile_offset+tile]=tile*route_tile;
        }
    }
    if(e==experts-1){
        offsets[experts]=route_sum;
        for(int d=tile_sum;d<max_tiles;++d)tile_experts[d]=-1;
    }
}

__global__ void route_scatter_kernel(const int *indices,int total,
    const int *offsets,int *cursors,int *slots){
    int route=blockIdx.x*blockDim.x+threadIdx.x;
    if(route<total){int expert=indices[route];int local=atomicAdd(cursors+expert,1);
        slots[offsets[expert]+local]=route;}
}

__global__ void pack_expert_i8_kernel(int8_t *output,float *scales,
    const float *input,const int *slots,int routes,int hidden,int topk){
    int route=blockIdx.x;if(route>=routes)return;
    int token=slots[route]/topk;const float *src=input+(size_t)token*hidden;
    float maximum=0.0f;for(int i=threadIdx.x;i<hidden;i+=blockDim.x)maximum=fmaxf(maximum,fabsf(src[i]));
    maximum=wave_max(maximum);__shared__ float waves[4],scale;
    int lane=threadIdx.x&63,wave=threadIdx.x>>6;if(lane==0)waves[wave]=maximum;__syncthreads();
    if(wave==0){maximum=lane<4?waves[lane]:0.0f;maximum=wave_max(maximum);
        if(lane==0){scale=maximum>0.0f?maximum/127.0f:1.0f;scales[route]=scale;}}
    __syncthreads();float inverse=1.0f/scale;int8_t *dst=output+(size_t)route*hidden;
    for(int i=threadIdx.x;i<hidden;i+=blockDim.x){float q=nearbyintf(src[i]*inverse);
        dst[i]=(int8_t)fmaxf(-127.0f,fminf(127.0f,q));}
}

__global__ void quantize_rows_i8_kernel(int8_t *output,float *scales,
    const float *input,int rows,int cols){
    int row=blockIdx.x;if(row>=rows)return;
    const float *src=input+(size_t)row*cols;
    float maximum=0.0f;
    for(int i=threadIdx.x;i<cols;i+=blockDim.x)
        maximum=fmaxf(maximum,fabsf(src[i]));
    maximum=wave_max(maximum);__shared__ float waves[4],scale;
    int lane=threadIdx.x&63,wave=threadIdx.x>>6;
    if(lane==0)waves[wave]=maximum;__syncthreads();
    if(wave==0){maximum=lane<4?waves[lane]:0.0f;maximum=wave_max(maximum);
        if(lane==0){scale=maximum>0.0f?maximum/127.0f:1.0f;scales[row]=scale;}}
    __syncthreads();float inverse=1.0f/scale;
    int8_t *dst=output+(size_t)row*cols;
    for(int i=threadIdx.x;i<cols;i+=blockDim.x){
        float q=nearbyintf(src[i]*inverse);
        dst[i]=(int8_t)fmaxf(-127.0f,fminf(127.0f,q));}
}

__global__ void pack_sorted_bf16_i8_kernel(int8_t *output,float *scales,
    const uint16_t *input,int routes,int hidden){
    int route=blockIdx.x;if(route>=routes)return;
    const uint16_t *src=input+(size_t)route*hidden;
    float maximum=0.0f;
    for(int i=threadIdx.x;i<hidden;i+=blockDim.x)
        maximum=fmaxf(maximum,fabsf(load_bf16(src+i)));
    maximum=wave_max(maximum);__shared__ float waves[4],scale;
    int lane=threadIdx.x&63,wave=threadIdx.x>>6;
    if(lane==0)waves[wave]=maximum;__syncthreads();
    if(wave==0){maximum=lane<4?waves[lane]:0.0f;maximum=wave_max(maximum);
        if(lane==0){scale=maximum>0.0f?maximum/127.0f:1.0f;scales[route]=scale;}}
    __syncthreads();float inverse=1.0f/scale;
    int8_t *dst=output+(size_t)route*hidden;
    for(int i=threadIdx.x;i<hidden;i+=blockDim.x){
        float q=nearbyintf(load_bf16(src+i)*inverse);
        dst[i]=(int8_t)fmaxf(-127.0f,fminf(127.0f,q));}
}

template<bool SINGLE>
__global__ void grouped_expert_down_i8_native(float *output,const int8_t *input,
    const float *input_scales,const int8_t *const *weights,
    const float *weight_scales,const int *counts,const int *slots,
    const int *offsets,const int *tile_experts,const int *tile_starts,
    int rows,int cols,int single_count){
    int descriptor=blockIdx.y,expert=SINGLE?0:tile_experts[descriptor];if(expert<0)return;
    int count=SINGLE?single_count:counts[expert],offset=SINGLE?0:offsets[expert];
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6,mt=wave&1,nt=wave>>1;
    int tile_m=blockIdx.x*64,tile_n=SINGLE?blockIdx.y*128:tile_starts[descriptor];
    int kg=(lane>>4)*4;
    int4_acc a0={0,0,0,0},a1={0,0,0,0},b0={0,0,0,0},b1={0,0,0,0};
    const int8_t *w=SINGLE?reinterpret_cast<const int8_t *>(weights):weights[expert];
    const int8_t *x=input+(size_t)offset*cols;
    __shared__ int8_t as[2][2048],bs[2][4096];
    {int z=tid*4,r=z/32,k=z%32;uint32_t av=0;
        if(tile_m+r<rows)
            av=*reinterpret_cast<const uint32_t *>(w+(size_t)(tile_m+r)*cols+k);
        *reinterpret_cast<uint32_t *>(as[0]+z)=av;
        int bz=tid*8,n=bz/32,bk=bz%32;unsigned long long bv=0;
        if(tile_n+n<count)
            bv=*reinterpret_cast<const unsigned long long *>(x+(size_t)(tile_n+n)*cols+bk);
        *reinterpret_cast<unsigned long long *>(bs[0]+bz)=bv;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=32){int cur=(kb>>5)&1,nxt=cur^1;
        uint32_t nav=0;unsigned long long nbv=0;
        if(kb+32<cols){int z=tid*4,r=z/32,k=z%32;
            if(tile_m+r<rows)
                nav=*reinterpret_cast<const uint32_t *>(w+(size_t)(tile_m+r)*cols+kb+32+k);
            int bz=tid*8,n=bz/32,bk=bz%32;
            if(tile_n+n<count)
                nbv=*reinterpret_cast<const unsigned long long *>(x+(size_t)(tile_n+n)*cols+kb+32+bk);}
        #pragma unroll
        for(int sk=0;sk<32;sk+=16){
            int ai0=(mt*16+(lane&15))*32+sk+kg,ai1=ai0+32*32;
            int bi=(nt*16+(lane&15))*32+sk+kg,bi2=bi+64*32;
            int av0=*reinterpret_cast<int *>(as[cur]+ai0);
            int av1=*reinterpret_cast<int *>(as[cur]+ai1);
            int bv0=*reinterpret_cast<int *>(bs[cur]+bi);
            int bv1=*reinterpret_cast<int *>(bs[cur]+bi2);
            a0=__builtin_amdgcn_mfma_i32_16x16x16i8(av0,bv0,a0,0,0,0);
            a1=__builtin_amdgcn_mfma_i32_16x16x16i8(av1,bv0,a1,0,0,0);
            b0=__builtin_amdgcn_mfma_i32_16x16x16i8(av0,bv1,b0,0,0,0);
            b1=__builtin_amdgcn_mfma_i32_16x16x16i8(av1,bv1,b1,0,0,0);}
        if(kb+32<cols){int z=tid*4,bz=tid*8;
            *reinterpret_cast<uint32_t *>(as[nxt]+z)=nav;
            *reinterpret_cast<unsigned long long *>(bs[nxt]+bz)=nbv;}
        __syncthreads();}
    int r0=tile_m+mt*16+(lane>>4)*4,r1=r0+32;
    int n0=tile_n+nt*16+(lane&15),n1=n0+64;
    int route0=n0<count?(SINGLE?n0:slots[offset+n0]):0;
    int route1=n1<count?(SINGLE?n1:slots[offset+n1]):0;
    float xs0=n0<count?input_scales[offset+n0]:0.0f;
    float xs1=n1<count?input_scales[offset+n1]:0.0f;
    #pragma unroll
    for(int i=0;i<4;++i){
        if(n0<count&&r0+i<rows)output[(size_t)route0*rows+r0+i]=
            (float)a0[i]*xs0*weight_scales[(size_t)expert*rows+r0+i];
        if(n0<count&&r1+i<rows)output[(size_t)route0*rows+r1+i]=
            (float)a1[i]*xs0*weight_scales[(size_t)expert*rows+r1+i];
        if(n1<count&&r0+i<rows)output[(size_t)route1*rows+r0+i]=
            (float)b0[i]*xs1*weight_scales[(size_t)expert*rows+r0+i];
        if(n1<count&&r1+i<rows)output[(size_t)route1*rows+r1+i]=
            (float)b1[i]*xs1*weight_scales[(size_t)expert*rows+r1+i];}
}

__global__ void lm_argmax_i8_kernel(float *partial_value,int *partial_index,
    const int8_t *input,const float *input_scales,const int8_t *weight,
    const float *weight_scales,int batch,int vocab,int cols,int tiles){
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6,mt=wave&1,nt=wave>>1;
    int tile_m=blockIdx.x*64,tile_n=blockIdx.y*128,kg=(lane>>4)*4;
    int4_acc a0={0,0,0,0},a1={0,0,0,0},b0={0,0,0,0},b1={0,0,0,0};
    __shared__ int8_t as[2][2048],bs[2][4096];
    __shared__ float wave_value[256];__shared__ int wave_index[256];
    {int z=tid*4,r=z/32,k=z%32;uint32_t av=0;
        if(tile_m+r<vocab)av=*reinterpret_cast<const uint32_t *>(
            weight+(size_t)(tile_m+r)*cols+k);
        *reinterpret_cast<uint32_t *>(as[0]+z)=av;
        int bz=tid*8,n=bz/32,bk=bz%32;unsigned long long bv=0;
        if(tile_n+n<batch)bv=*reinterpret_cast<const unsigned long long *>(
            input+(size_t)(tile_n+n)*cols+bk);
        *reinterpret_cast<unsigned long long *>(bs[0]+bz)=bv;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=32){int cur=(kb>>5)&1,nxt=cur^1;
        uint32_t nav=0;unsigned long long nbv=0;
        if(kb+32<cols){int z=tid*4,r=z/32,k=z%32;
            if(tile_m+r<vocab)nav=*reinterpret_cast<const uint32_t *>(
                weight+(size_t)(tile_m+r)*cols+kb+32+k);
            int bz=tid*8,n=bz/32,bk=bz%32;
            if(tile_n+n<batch)nbv=*reinterpret_cast<const unsigned long long *>(
                input+(size_t)(tile_n+n)*cols+kb+32+bk);}
        #pragma unroll
        for(int sk=0;sk<32;sk+=16){
            int ai0=(mt*16+(lane&15))*32+sk+kg,ai1=ai0+32*32;
            int bi=(nt*16+(lane&15))*32+sk+kg,bi2=bi+64*32;
            int av0=*reinterpret_cast<int *>(as[cur]+ai0);
            int av1=*reinterpret_cast<int *>(as[cur]+ai1);
            int bv0=*reinterpret_cast<int *>(bs[cur]+bi);
            int bv1=*reinterpret_cast<int *>(bs[cur]+bi2);
            a0=__builtin_amdgcn_mfma_i32_16x16x16i8(av0,bv0,a0,0,0,0);
            a1=__builtin_amdgcn_mfma_i32_16x16x16i8(av1,bv0,a1,0,0,0);
            b0=__builtin_amdgcn_mfma_i32_16x16x16i8(av0,bv1,b0,0,0,0);
            b1=__builtin_amdgcn_mfma_i32_16x16x16i8(av1,bv1,b1,0,0,0);}
        if(kb+32<cols){int z=tid*4,bz=tid*8;
            *reinterpret_cast<uint32_t *>(as[nxt]+z)=nav;
            *reinterpret_cast<unsigned long long *>(bs[nxt]+bz)=nbv;}
        __syncthreads();}
    int r0=tile_m+mt*16+(lane>>4)*4,r1=r0+32;
    int n0=tile_n+nt*16+(lane&15),n1=n0+64;
    float best0=-INFINITY,best1=-INFINITY;int best_i0=vocab,best_i1=vocab;
    float xs0=n0<batch?input_scales[n0]:0.0f;
    float xs1=n1<batch?input_scales[n1]:0.0f;
    #pragma unroll
    for(int i=0;i<4;++i){
        if(r0+i<vocab){float v=(float)a0[i]*xs0*weight_scales[r0+i];
            if(v>best0||(v==best0&&r0+i<best_i0)){best0=v;best_i0=r0+i;}
            v=(float)b0[i]*xs1*weight_scales[r0+i];
            if(v>best1||(v==best1&&r0+i<best_i1)){best1=v;best_i1=r0+i;}}
        if(r1+i<vocab){float v=(float)a1[i]*xs0*weight_scales[r1+i];
            if(v>best0||(v==best0&&r1+i<best_i0)){best0=v;best_i0=r1+i;}
            v=(float)b1[i]*xs1*weight_scales[r1+i];
            if(v>best1||(v==best1&&r1+i<best_i1)){best1=v;best_i1=r1+i;}}}
    for(int offset=16;offset<=32;offset*=2){
        float ov=__shfl_down(best0,offset,64);int oi=__shfl_down(best_i0,offset,64);
        if(lane+offset<64&&(ov>best0||(ov==best0&&oi<best_i0))){best0=ov;best_i0=oi;}
        ov=__shfl_down(best1,offset,64);oi=__shfl_down(best_i1,offset,64);
        if(lane+offset<64&&(ov>best1||(ov==best1&&oi<best_i1))){best1=ov;best_i1=oi;}}
    if(lane<16){int o=wave*16+lane;wave_value[o]=best0;wave_index[o]=best_i0;
        wave_value[128+o]=best1;wave_index[128+o]=best_i1;}
    __syncthreads();
    if(tid<128){int group=tid>>4,nlocal=tid&15,ntile=group&3;
        int base=group<4?0:128;
        int o0=base+(ntile*2)*16+nlocal,o1=o0+16;
        float v0=wave_value[o0],v1=wave_value[o1];int i0=wave_index[o0],i1=wave_index[o1];
        if(v1>v0||(v1==v0&&i1<i0)){v0=v1;i0=i1;}
        int n=tile_n+(group<4?ntile*16:64+ntile*16)+nlocal;
        if(n<batch){size_t o=(size_t)n*tiles+blockIdx.x;
            partial_value[o]=v0;partial_index[o]=i0;}}
}


__global__ void quantize_expert_down_kernel(const uint16_t *const *input,
    int8_t *output,float *scales,int rows,int cols){
    int expert=blockIdx.y,row=blockIdx.x;
    const uint16_t *src=input[expert]+(size_t)row*cols;
    float maximum=0.0f;
    for(int col=threadIdx.x;col<cols;col+=blockDim.x)
        maximum=fmaxf(maximum,fabsf(load_bf16(src+col)));
    maximum=wave_max(maximum);
    __shared__ float wave_values[4],scale;
    int lane=threadIdx.x&63,wave=threadIdx.x>>6;
    if(lane==0)wave_values[wave]=maximum;
    __syncthreads();
    if(wave==0){maximum=lane<4?wave_values[lane]:0.0f;maximum=wave_max(maximum);
        if(lane==0){scale=maximum>0.0f?maximum/127.0f:1.0f;
            scales[(size_t)expert*rows+row]=scale;}}
    __syncthreads();
    int8_t *dst=output+((size_t)expert*rows+row)*cols;
    float inverse=1.0f/scale;
    for(int col=threadIdx.x;col<cols;col+=blockDim.x){
        float value=nearbyintf(load_bf16(src+col)*inverse);
        dst[col]=(int8_t)fmaxf(-127.0f,fminf(127.0f,value));}
}

__global__ void grouped_expert_gate_up_32x128(uint16_t *gate,float *up,
    const uint16_t *input,const uint16_t *const *gate_w,
    const uint16_t *const *up_w,const int *counts,const int *offsets,
    const int *tile_experts,const int *tile_starts,int capacity,int rows,int cols,
    int single_count){
    int expert=single_count>=0?0:tile_experts[blockIdx.y];if(expert<0)return;
    int count=single_count>=0?single_count:counts[expert];
    int offset=single_count>=0?0:offsets[expert];
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6,mt=wave&1,nt=wave>>1;
    int tile_m=blockIdx.x*64,tile_n=single_count>=0?blockIdx.y*128:
        tile_starts[blockIdx.y],kg=(lane>>4)*4;
    float4_acc ag0={0,0,0,0},au0={0,0,0,0},ag1={0,0,0,0},au1={0,0,0,0};
    float4_acc bg0={0,0,0,0},bu0={0,0,0,0},bg1={0,0,0,0},bu1={0,0,0,0};
    const uint16_t *wg=single_count>=0?reinterpret_cast<const uint16_t *>(gate_w):gate_w[expert];
    const uint16_t *wu=single_count>=0?reinterpret_cast<const uint16_t *>(up_w):up_w[expert];
    const uint16_t *x=input+(size_t)offset*cols;
    __shared__ uint16_t gs[2][1024],us[2][1024],bs[2][2048];
    {int z=tid*2,r=z/16,k=z%16,n=z/16;
        uint32_t gv=0,uv=0,bv0=0,bv1=0;
        if(tile_m+r<rows){
            gv=*reinterpret_cast<const uint32_t *>(wg+(size_t)(tile_m+r)*cols+k);
            uv=*reinterpret_cast<const uint32_t *>(wu+(size_t)(tile_m+r)*cols+k);}
        if(tile_n+n<count)
            bv0=*reinterpret_cast<const uint32_t *>(x+(size_t)(tile_n+n)*cols+k);
        n+=64;if(tile_n+n<count)
            bv1=*reinterpret_cast<const uint32_t *>(x+(size_t)(tile_n+n)*cols+k);
        int p=r*16+(k^((r&3)<<2));
        *reinterpret_cast<uint32_t *>(gs[0]+p)=gv;
        *reinterpret_cast<uint32_t *>(us[0]+p)=uv;
        *reinterpret_cast<uint32_t *>(bs[0]+p)=bv0;
        *reinterpret_cast<uint32_t *>(bs[0]+p+1024)=bv1;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=16){int cur=(kb>>4)&1,nxt=cur^1;
        int ar0=mt*16+(lane&15),ar1=(mt+2)*16+(lane&15);
        int br0=nt*16+(lane&15);
        int ai0=ar0*16+(kg^((ar0&3)<<2));
        int ai1=ar1*16+(kg^((ar1&3)<<2));
        int bi=br0*16+(kg^((br0&3)<<2));
        int bi2=bi+64*16;
        float2_reg gv0={__uint_as_float(*reinterpret_cast<uint32_t *>(gs[cur]+ai0)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(gs[cur]+ai0+2))};
        float2_reg uv0={__uint_as_float(*reinterpret_cast<uint32_t *>(us[cur]+ai0)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(us[cur]+ai0+2))};
        float2_reg gv1={__uint_as_float(*reinterpret_cast<uint32_t *>(gs[cur]+ai1)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(gs[cur]+ai1+2))};
        float2_reg uv1={__uint_as_float(*reinterpret_cast<uint32_t *>(us[cur]+ai1)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(us[cur]+ai1+2))};
        float2_reg bv={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi)),
                       __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi+2))};
        float2_reg bv2={__uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi2)),
                        __uint_as_float(*reinterpret_cast<uint32_t *>(bs[cur]+bi2+2))};
        uint32_t ng=0,nu=0,nb0=0,nb1=0;
        if(kb+16<cols){int z=tid*2,k=z%16,r=z/16,n=r;
            if(tile_m+r<rows){
                ng=*reinterpret_cast<const uint32_t *>(wg+(size_t)(tile_m+r)*cols+kb+16+k);
                nu=*reinterpret_cast<const uint32_t *>(wu+(size_t)(tile_m+r)*cols+kb+16+k);}
            if(tile_n+n<count)
                nb0=*reinterpret_cast<const uint32_t *>(x+(size_t)(tile_n+n)*cols+kb+16+k);
            n+=64;if(tile_n+n<count)
                nb1=*reinterpret_cast<const uint32_t *>(x+(size_t)(tile_n+n)*cols+kb+16+k);}
        ag0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv0,bv,ag0,0,0,0);
        au0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv0,bv,au0,0,0,0);
        ag1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv1,bv,ag1,0,0,0);
        au1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv1,bv,au1,0,0,0);
        bg0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv0,bv2,bg0,0,0,0);
        bu0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv0,bv2,bu0,0,0,0);
        bg1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv1,bv2,bg1,0,0,0);
        bu1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv1,bv2,bu1,0,0,0);
        if(kb+16<cols){int z=tid*2,r=z/16,k=z%16;
            int p=r*16+(k^((r&3)<<2));
            *reinterpret_cast<uint32_t *>(gs[nxt]+p)=ng;
            *reinterpret_cast<uint32_t *>(us[nxt]+p)=nu;
            *reinterpret_cast<uint32_t *>(bs[nxt]+p)=nb0;
            *reinterpret_cast<uint32_t *>(bs[nxt]+p+1024)=nb1;}
        __syncthreads();}
    int r0=tile_m+mt*16+(lane>>4)*4,r1=r0+32,n=tile_n+nt*16+(lane&15),n2=n+64;
    #pragma unroll
    for(int i=0;i<4;++i){if(n<count&&r0+i<rows){size_t o=((size_t)offset+n)*rows+r0+i;
        float g=ag0[i];gate[o]=to_bf16_rne((g/(1.0f+expf(-g)))*au0[i]);}
        if(n<count&&r1+i<rows){size_t o=((size_t)offset+n)*rows+r1+i;
        float g=ag1[i];gate[o]=to_bf16_rne((g/(1.0f+expf(-g)))*au1[i]);}
        if(n2<count&&r0+i<rows){size_t o=((size_t)offset+n2)*rows+r0+i;
        float g=bg0[i];gate[o]=to_bf16_rne((g/(1.0f+expf(-g)))*bu0[i]);}
        if(n2<count&&r1+i<rows){size_t o=((size_t)offset+n2)*rows+r1+i;
        float g=bg1[i];gate[o]=to_bf16_rne((g/(1.0f+expf(-g)))*bu1[i]);}}
}

__global__ void grouped_expert_gate_up_i8_64x128(uint16_t *gate,
    const int8_t *input,const float *input_scales,
    const int8_t *const *gate_w,const int8_t *const *up_w,
    const float *gate_scales,const float *up_scales,const int *counts,
    const int *offsets,const int *tile_experts,const int *tile_starts,
    int rows,int cols,int single_count){
    int expert=single_count>=0?0:tile_experts[blockIdx.y];if(expert<0)return;
    int count=single_count>=0?single_count:counts[expert];
    int offset=single_count>=0?0:offsets[expert];
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6,mt=wave&1,nt=wave>>1;
    int tile_m=blockIdx.x*64,tile_n=single_count>=0?blockIdx.y*128:
        tile_starts[blockIdx.y],kg=(lane>>4)*4;
    int4_acc ag0={0,0,0,0},au0={0,0,0,0},ag1={0,0,0,0},au1={0,0,0,0};
    int4_acc bg0={0,0,0,0},bu0={0,0,0,0},bg1={0,0,0,0},bu1={0,0,0,0};
    const int8_t *wg=single_count>=0?reinterpret_cast<const int8_t *>(gate_w):
        gate_w[expert];
    const int8_t *wu=single_count>=0?reinterpret_cast<const int8_t *>(up_w):
        up_w[expert];
    const int8_t *x=input+(size_t)offset*cols;
    __shared__ int8_t gs[2][2048],us[2][2048],bs[2][4096];
    {int z=tid*4,r=z/32,k=z%32;uint32_t gv=0,uv=0;
        if(tile_m+r<rows){gv=*reinterpret_cast<const uint32_t *>(wg+(size_t)(tile_m+r)*cols+k);
            uv=*reinterpret_cast<const uint32_t *>(wu+(size_t)(tile_m+r)*cols+k);}
        *reinterpret_cast<uint32_t *>(gs[0]+z)=gv;
        *reinterpret_cast<uint32_t *>(us[0]+z)=uv;
        int bz=tid*8,n=bz/32,bk=bz%32;unsigned long long bv=0;
        if(tile_n+n<count)bv=*reinterpret_cast<const unsigned long long *>(x+(size_t)(tile_n+n)*cols+bk);
        *reinterpret_cast<unsigned long long *>(bs[0]+bz)=bv;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=32){int cur=(kb>>5)&1,nxt=cur^1;
        uint32_t ng=0,nu=0;unsigned long long nb=0;
        if(kb+32<cols){int z=tid*4,r=z/32,k=z%32;
            if(tile_m+r<rows){
                ng=*reinterpret_cast<const uint32_t *>(wg+(size_t)(tile_m+r)*cols+kb+32+k);
                nu=*reinterpret_cast<const uint32_t *>(wu+(size_t)(tile_m+r)*cols+kb+32+k);}
            int bz=tid*8,n=bz/32,bk=bz%32;
            if(tile_n+n<count)
                nb=*reinterpret_cast<const unsigned long long *>(x+(size_t)(tile_n+n)*cols+kb+32+bk);}
        #pragma unroll
        for(int sk=0;sk<32;sk+=16){
            int ai0=(mt*16+(lane&15))*32+sk+kg,ai1=ai0+32*32;
            int bi=(nt*16+(lane&15))*32+sk+kg,bi2=bi+64*32;
            int av0=*reinterpret_cast<int *>(gs[cur]+ai0);
            int av1=*reinterpret_cast<int *>(gs[cur]+ai1);
            int uv0=*reinterpret_cast<int *>(us[cur]+ai0);
            int uv1=*reinterpret_cast<int *>(us[cur]+ai1);
            int bv0=*reinterpret_cast<int *>(bs[cur]+bi);
            int bv1=*reinterpret_cast<int *>(bs[cur]+bi2);
            ag0=__builtin_amdgcn_mfma_i32_16x16x16i8(av0,bv0,ag0,0,0,0);
            au0=__builtin_amdgcn_mfma_i32_16x16x16i8(uv0,bv0,au0,0,0,0);
            ag1=__builtin_amdgcn_mfma_i32_16x16x16i8(av1,bv0,ag1,0,0,0);
            au1=__builtin_amdgcn_mfma_i32_16x16x16i8(uv1,bv0,au1,0,0,0);
            bg0=__builtin_amdgcn_mfma_i32_16x16x16i8(av0,bv1,bg0,0,0,0);
            bu0=__builtin_amdgcn_mfma_i32_16x16x16i8(uv0,bv1,bu0,0,0,0);
            bg1=__builtin_amdgcn_mfma_i32_16x16x16i8(av1,bv1,bg1,0,0,0);
            bu1=__builtin_amdgcn_mfma_i32_16x16x16i8(uv1,bv1,bu1,0,0,0);}
        if(kb+32<cols){int z=tid*4,bz=tid*8;
            *reinterpret_cast<uint32_t *>(gs[nxt]+z)=ng;
            *reinterpret_cast<uint32_t *>(us[nxt]+z)=nu;
            *reinterpret_cast<unsigned long long *>(bs[nxt]+bz)=nb;}
        __syncthreads();}
    int r0=tile_m+mt*16+(lane>>4)*4,r1=r0+32;
    int n0=tile_n+nt*16+(lane&15),n1=n0+64;
    float is0=n0<count?input_scales[offset+n0]:0.0f;
    float is1=n1<count?input_scales[offset+n1]:0.0f;
    #pragma unroll
    for(int i=0;i<4;++i){
        if(n0<count&&r0+i<rows){float g=(float)ag0[i]*is0*gate_scales[(size_t)expert*rows+r0+i];
            float u=(float)au0[i]*is0*up_scales[(size_t)expert*rows+r0+i];
            gate[((size_t)offset+n0)*rows+r0+i]=to_bf16_rne((g/(1.0f+expf(-g)))*u);}
        if(n0<count&&r1+i<rows){float g=(float)ag1[i]*is0*gate_scales[(size_t)expert*rows+r1+i];
            float u=(float)au1[i]*is0*up_scales[(size_t)expert*rows+r1+i];
            gate[((size_t)offset+n0)*rows+r1+i]=to_bf16_rne((g/(1.0f+expf(-g)))*u);}
        if(n1<count&&r0+i<rows){float g=(float)bg0[i]*is1*gate_scales[(size_t)expert*rows+r0+i];
            float u=(float)bu0[i]*is1*up_scales[(size_t)expert*rows+r0+i];
            gate[((size_t)offset+n1)*rows+r0+i]=to_bf16_rne((g/(1.0f+expf(-g)))*u);}
        if(n1<count&&r1+i<rows){float g=(float)bg1[i]*is1*gate_scales[(size_t)expert*rows+r1+i];
            float u=(float)bu1[i]*is1*up_scales[(size_t)expert*rows+r1+i];
            gate[((size_t)offset+n1)*rows+r1+i]=to_bf16_rne((g/(1.0f+expf(-g)))*u);}}
}

__global__ void batched_weighted_add_kernel(float *output,const float *experts,
                                            const float *weights,int batch,
                                            int topk,int hidden){
    int i=blockIdx.x*blockDim.x+threadIdx.x,quads=hidden/4,total=batch*quads;
    if(i>=total)return;int token=i/quads,col=(i%quads)*4;
    float4 v=*reinterpret_cast<const float4 *>(output+(size_t)token*hidden+col);
    #pragma unroll
    for(int r=0;r<topk;++r){float w=weights[(size_t)token*topk+r];
        float4 e=*reinterpret_cast<const float4 *>(experts+((size_t)token*topk+r)*hidden+col);
        v.x+=w*e.x;v.y+=w*e.y;v.z+=w*e.z;v.w+=w*e.w;}
    *reinterpret_cast<float4 *>(output+(size_t)token*hidden+col)=v;
}

extern "C" void dsv2_init_mfma_gemm(size_t capacity) {
    mfma_input_capacity=capacity;
    hipError_t status=hipMalloc(&mfma_input,capacity*sizeof(uint16_t));
    if(status!=hipSuccess){
        fprintf(stderr,"MFMA scratch allocation failed: %s\n",hipGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

extern "C" void dsv2_finish_mfma_gemm() {
    if(mfma_input){
        hipError_t status=hipFree(mfma_input);
        if(status!=hipSuccess){
            fprintf(stderr,"MFMA scratch release failed: %s\n",hipGetErrorString(status));
        }
    }
    mfma_input=nullptr;mfma_input_capacity=0;
}

extern "C" void dsv2_launch_batched_matvec(float *out,const float *in,const uint16_t *w,int batch,int rows,int cols,hipStream_t s){
    if(batch>=16){
        size_t total=(size_t)batch*cols;
        if(total<=mfma_input_capacity){
            convert_bf16_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
                mfma_input,in,total);
            if(batch>=96){
                mfma_bf16_gemm_64x128<<<dim3((rows+63)/64,(batch+127)/128),512,0,s>>>(
                    out,mfma_input,w,batch,rows,cols);
            }else{
                mfma_bf16_gemm_32x64<<<dim3((rows+63)/64,(batch+63)/64),512,0,s>>>(
                    out,mfma_input,w,batch,rows,cols);
            }
            return;
        }
    }
    batched_bf16_matvec_kernel<<<dim3((rows+3)/4,batch),BLOCK_SIZE,0,s>>>(out,in,w,batch,rows,cols);}
extern "C" void dsv2_prepare_batched_bf16(const float *in,int batch,int cols,
    hipStream_t s){
    size_t total=(size_t)batch*cols;
    if(total<=mfma_input_capacity)
        convert_bf16_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
            mfma_input,in,total);
}
extern "C" void dsv2_launch_prepared_matvec(float *out,const uint16_t *w,
    int batch,int rows,int cols,hipStream_t s){
    if(batch>=96)
        mfma_bf16_gemm_64x128<<<dim3((rows+63)/64,(batch+127)/128),512,0,s>>>(
            out,mfma_input,w,batch,rows,cols);
    else
        mfma_bf16_gemm_32x64<<<dim3((rows+63)/64,(batch+63)/64),512,0,s>>>(
            out,mfma_input,w,batch,rows,cols);
}
extern "C" void dsv2_launch_prepared_swiglu(float *out,const uint16_t *gate,
    const uint16_t *up,int batch,int rows,int cols,hipStream_t s){
    mfma_bf16_swiglu_32x64<<<dim3((rows+63)/64,(batch+63)/64),512,0,s>>>(
        out,mfma_input,gate,up,batch,rows,cols);
}
extern "C" void dsv2_launch_batched_lm_argmax_partials(float *values,int *indices,
    const float *in,const uint16_t *w,int batch,int vocab,int cols,hipStream_t s){
    size_t total=(size_t)batch*cols;int tiles=(vocab+63)/64;
    convert_bf16_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
        mfma_input,in,total);
    mfma_bf16_gemm_argmax_32x64<<<dim3(tiles,(batch+63)/64),512,0,s>>>(
        values,indices,mfma_input,w,batch,vocab,cols,tiles);
}
extern "C" void dsv2_launch_lm_argmax_i8(float *values,int *indices,
    int8_t *quantized,float *input_scales,const float *input,
    const int8_t *weight,const float *weight_scales,int batch,int vocab,int hidden,
    hipStream_t s){
    quantize_rows_i8_kernel<<<batch,256,0,s>>>(quantized,input_scales,input,batch,hidden);
    int tiles=(vocab+63)/64;
    lm_argmax_i8_kernel<<<dim3(tiles,(batch+127)/128),512,0,s>>>(
        values,indices,quantized,input_scales,weight,weight_scales,
        batch,vocab,hidden,tiles);
}
extern "C" void dsv2_launch_batched_swiglu_mfma(float *out,const float *in,
    const uint16_t *gate,const uint16_t *up,int batch,int rows,int cols,hipStream_t s){
    size_t total=(size_t)batch*cols;
    if(total<=mfma_input_capacity){
        convert_bf16_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
            mfma_input,in,total);
        mfma_bf16_swiglu_32x64<<<dim3((rows+63)/64,(batch+63)/64),512,0,s>>>(
            out,mfma_input,gate,up,batch,rows,cols);
    }}
extern "C" void dsv2_launch_batched_rmsnorm(float *out,const float *in,const uint16_t *w,int batch,int count,float eps,hipStream_t s){
    batched_rmsnorm_kernel<<<batch,BLOCK_SIZE,0,s>>>(out,in,w,batch,count,eps);}
extern "C" void dsv2_launch_batched_rmsnorm_strided(float *out,int os,const float *in,int is,const uint16_t *w,int batch,int count,float eps,hipStream_t s){
    batched_rmsnorm_strided_kernel<<<batch,BLOCK_SIZE,0,s>>>(out,os,in,is,w,batch,count,eps);}
extern "C" void dsv2_launch_batched_embedding(float *out,const uint16_t *table,const int *tokens,int batch,int hidden,hipStream_t s){
    batched_embedding_kernel<<<dim3((hidden+BLOCK_SIZE-1)/BLOCK_SIZE,batch),BLOCK_SIZE,0,s>>>(out,table,tokens,batch,hidden);}
extern "C" void dsv2_launch_batched_residual(float *state,const float *branch,int total,hipStream_t s){
    batched_residual_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(state,branch,total);}
extern "C" void dsv2_launch_batched_residual_rmsnorm(float *state,float *out,
    const float *branch,const uint16_t *weight,int batch,int count,float eps,
    hipStream_t s){
    batched_residual_rmsnorm_kernel<<<batch,BLOCK_SIZE,0,s>>>(
        state,out,branch,weight,count,eps);
}
extern "C" void dsv2_launch_batched_rope(float *q,int batch,int heads,int head_dim,int nope,int rope,const float *freq,hipStream_t s){
    int total=batch*heads*rope/2;batched_rope_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(q,batch,heads,head_dim,nope,rope,freq,batch);}
extern "C" void dsv2_launch_prefill_cache(float *cache,const float *norm,const float *comp,int batch,int stride,int latent,int rope,const float *freq,hipStream_t s){
    prefill_cache_kernel<<<batch,BLOCK_SIZE,0,s>>>(cache,norm,comp,batch,stride,latent,rope,freq,batch,0,batch);}
extern "C" void dsv2_launch_batched_absorb(float *out,const float *q,const uint16_t *w,int batch,int heads,int qhd,int nope,int value,int latent,hipStream_t s){
    size_t total=(size_t)batch*heads*nope;
    pack_head_bf16<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
        mfma_input,q,batch,heads,qhd,nope);
    grouped_head_mfma_32x64<<<dim3((latent+63)/64,(batch+63)/64,heads),512,0,s>>>(
        out,mfma_input,w,batch,heads,latent,nope,(nope+value)*latent,0,1,0);}
extern "C" void dsv2_launch_prefill_attention(float *ctx,const float *qa,const float *q,const float *cache,int batch,int stride,int heads,int qhd,int nope,int latent,int rope,float scale,hipStream_t s){
    prefill_attention_kernel<<<dim3(batch,heads),BLOCK_SIZE,0,s>>>(ctx,qa,q,cache,batch,stride,heads,qhd,nope,latent,rope,scale,batch,0,batch);}

extern "C" void dsv2_launch_group_rope(float *q,int rows,int heads,int head_dim,
    int nope,int rope,const float *freq,int sequence_length,hipStream_t s){
    int total=rows*heads*rope/2;
    batched_rope_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
        q,rows,heads,head_dim,nope,rope,freq,sequence_length);}
extern "C" void dsv2_launch_group_cache(float *cache,const float *norm,
    const float *comp,int rows,int stride,int latent,int rope,const float *freq,
    int sequence_length,int request_base,int cache_sequence_length,hipStream_t s){
    prefill_cache_kernel<<<rows,BLOCK_SIZE,0,s>>>(cache,norm,comp,rows,stride,
        latent,rope,freq,sequence_length,request_base,cache_sequence_length);}
extern "C" void dsv2_launch_group_attention(float *ctx,const float *qa,
    const float *q,const float *cache,int rows,int stride,int heads,int qhd,
    int nope,int latent,int rope,float scale,int sequence_length,int request_base,
    int cache_sequence_length,hipStream_t s){
    prefill_attention_kernel<<<dim3(rows,heads),BLOCK_SIZE,0,s>>>(ctx,qa,q,cache,
        rows,stride,heads,qhd,nope,latent,rope,scale,sequence_length,request_base,
        cache_sequence_length);}
extern "C" void dsv2_launch_decode_rope(float *q,int batch,int heads,int qhd,
    int nope,int rope,const float *freq,const int *positions,hipStream_t s){
    int total=batch*heads*rope/2;decode_rope_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,
        BLOCK_SIZE,0,s>>>(q,batch,heads,qhd,nope,rope,freq,positions);}
extern "C" void dsv2_launch_decode_cache(float *cache,const float *norm,
    const float *comp,int batch,int stride,int latent,int rope,const float *freq,
    const int *positions,int cache_sequence,hipStream_t s){
    decode_cache_kernel<<<batch,BLOCK_SIZE,0,s>>>(cache,norm,comp,batch,stride,
        latent,rope,freq,positions,cache_sequence);}
extern "C" void dsv2_launch_decode_attention(float *ctx,const float *qa,
    const float *q,const float *cache,int batch,int stride,int heads,int qhd,
    int nope,int latent,int rope,float scale,const int *positions,int cache_sequence,
    hipStream_t s){decode_attention_shared_heads<<<batch,1024,0,s>>>(
        ctx,qa,q,cache,batch,stride,heads,qhd,nope,latent,rope,scale,positions,
        cache_sequence);}
extern "C" void dsv2_launch_batched_latent_value(float *out,const float *latent,const uint16_t *w,int batch,int heads,int nope,int value,int ld,hipStream_t s){
    int chunk_max=max(1,(int)(mfma_input_capacity/((size_t)heads*ld)));
    for(int start=0;start<batch;start+=chunk_max){int n=min(chunk_max,batch-start);
        size_t total=(size_t)n*heads*ld;
        pack_head_bf16<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
            mfma_input,latent+(size_t)start*heads*ld,n,heads,ld,ld);
        grouped_head_mfma_32x64<<<dim3((value+63)/64,(n+63)/64,heads),512,0,s>>>(
            out+(size_t)start*heads*value,mfma_input,w,n,heads,value,ld,
            (nope+value)*ld,nope,0,0);}}
extern "C" void dsv2_launch_prefill_unabsorbed(float *out,float *key,float *value_out,
    const float *query,const float *latent,const float *cache,const uint16_t *w,int rows,int heads,
    int qhd,int nope,int value,int ld,int sequence_length,int request_base,
    int cache_sequence,float scale,hipStream_t s){
    int chunk_max=max(1,(int)(mfma_input_capacity/(size_t)ld));
    for(int start=0;start<rows;start+=chunk_max){int n=min(chunk_max,rows-start);
        size_t total=(size_t)n*ld;
        convert_bf16_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
            mfma_input,latent+(size_t)start*ld,total);
        grouped_head_mfma_32x64<<<dim3((nope+63)/64,(n+63)/64,heads),512,0,s>>>(
            key+(size_t)start*heads*nope,mfma_input,w,n,heads,nope,ld,
            (nope+value)*ld,0,0,1);
        grouped_head_mfma_32x64<<<dim3((value+63)/64,(n+63)/64,heads),512,0,s>>>(
            value_out+(size_t)start*heads*value,mfma_input,w,n,heads,value,ld,
            (nope+value)*ld,nope,0,1);}
    int requests=rows/sequence_length;
    prefill_unabsorbed_attention_q4<<<dim3(requests*((sequence_length+3)/4),heads),BLOCK_SIZE,0,s>>>(
        out,query,key,value_out,cache,rows,heads,qhd,nope,value,ld+qhd-nope,ld,
        qhd-nope,sequence_length,request_base,cache_sequence,scale);
}
extern "C" void dsv2_launch_batched_router_topk(const float *logits,int batch,
    int experts,int topk,int *idx,float *weights,int *counts,hipStream_t s){
    hipError_t status=hipMemsetAsync(counts,0,64*sizeof(int),s);
    if(status!=hipSuccess){fprintf(stderr,"route count reset failed: %s\n",
        hipGetErrorString(status));exit(EXIT_FAILURE);}
    batched_router_topk_kernel<<<batch,64,0,s>>>(
        logits,batch,experts,topk,idx,weights,counts);}
extern "C" void dsv2_launch_batched_weighted_add(float *out,const float *experts,const float *weights,int batch,int topk,int hidden,hipStream_t s){
    int total=batch*(hidden/4);batched_weighted_add_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(out,experts,weights,batch,topk,hidden);}
extern "C" void dsv2_launch_bucket_routes(const int *idx,int tokens,int topk,
    int *counts,int *offsets,int *cursors,int *slots,int *tile_experts,
    int *tile_starts,int max_tiles,int route_tile,hipStream_t s){
    int total=tokens*topk;
    route_prefix_descriptors_kernel<<<1,64,0,s>>>(counts,offsets,cursors,
        tile_experts,tile_starts,64,max_tiles,route_tile);
    route_scatter_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
        idx,total,offsets,cursors,slots);}
extern "C" void dsv2_launch_pack_expert_i8(int8_t *out,float *scales,
    const float *in,const int *slots,int routes,int hidden,int topk,hipStream_t s){
    pack_expert_i8_kernel<<<routes,256,0,s>>>(out,scales,in,slots,routes,hidden,topk);
}
extern "C" void dsv2_launch_pack_sorted_bf16_i8(int8_t *out,float *scales,
    const uint16_t *in,int routes,int hidden,hipStream_t s){
    pack_sorted_bf16_i8_kernel<<<routes,256,0,s>>>(out,scales,in,routes,hidden);
}
extern "C" void dsv2_launch_quantize_rows_i8(int8_t *out,float *scales,
    const float *in,int rows,int cols,hipStream_t s){
    quantize_rows_i8_kernel<<<rows,256,0,s>>>(out,scales,in,rows,cols);
}
extern "C" void dsv2_quantize_expert_down(const uint16_t *const *input,
    int8_t *output,float *scales,int experts,int rows,int cols,hipStream_t s){
    quantize_expert_down_kernel<<<dim3(rows,experts),256,0,s>>>(
        input,output,scales,rows,cols);
}
extern "C" void dsv2_launch_grouped_expert_down_i8_native(float *out,
    const int8_t *in,const float *input_scales,
    const int8_t *const *weights,const float *weight_scales,
    const int *counts,const int *slots,const int *offsets,
    const int *tile_experts,const int *tile_starts,int max_tiles,
    int rows,int cols,hipStream_t s){
    grouped_expert_down_i8_native<false><<<dim3((rows+63)/64,max_tiles),512,0,s>>>(
        out,in,input_scales,weights,weight_scales,counts,slots,offsets,
        tile_experts,tile_starts,rows,cols,0);
}
extern "C" void dsv2_launch_single_down_i8(float *out,const int8_t *in,
    const float *input_scales,const int8_t *weights,const float *weight_scales,
    int batch,int rows,int cols,hipStream_t s){
    grouped_expert_down_i8_native<true><<<dim3((rows+63)/64,(batch+127)/128),512,0,s>>>(
        out,in,input_scales,reinterpret_cast<const int8_t *const *>(weights),
        weight_scales,nullptr,nullptr,nullptr,nullptr,nullptr,rows,cols,batch);
}
extern "C" void dsv2_launch_grouped_expert_gate_up_i8(uint16_t *gate,
    const int8_t *in,const float *input_scales,const int8_t *const *gate_w,
    const int8_t *const *up_w,const float *gate_scales,const float *up_scales,
    const int *counts,const int *offsets,const int *tile_experts,
    const int *tile_starts,int max_tiles,int rows,int cols,hipStream_t s){
    grouped_expert_gate_up_i8_64x128<<<dim3((rows+63)/64,max_tiles),512,0,s>>>(
        gate,in,input_scales,gate_w,up_w,gate_scales,up_scales,counts,offsets,
        tile_experts,tile_starts,rows,cols,-1);
}
extern "C" void dsv2_launch_single_gate_up_i8(uint16_t *gate,const int8_t *in,
    const float *input_scales,const int8_t *gate_w,const int8_t *up_w,
    const float *gate_scales,const float *up_scales,int batch,int rows,int cols,
    hipStream_t s){
    grouped_expert_gate_up_i8_64x128<<<dim3((rows+63)/64,(batch+127)/128),512,0,s>>>(
        gate,in,input_scales,reinterpret_cast<const int8_t *const *>(gate_w),
        reinterpret_cast<const int8_t *const *>(up_w),gate_scales,up_scales,
        nullptr,nullptr,nullptr,nullptr,rows,cols,batch);
}
extern "C" void dsv2_launch_single_gate_up(uint16_t *gate,const uint16_t *in,
    const uint16_t *gate_w,const uint16_t *up_w,int batch,int rows,int cols,
    hipStream_t s){
    grouped_expert_gate_up_32x128<<<dim3((rows+63)/64,(batch+127)/128,1),512,0,s>>>(
        gate,nullptr,in?in:mfma_input,reinterpret_cast<const uint16_t *const *>(gate_w),
        reinterpret_cast<const uint16_t *const *>(up_w),nullptr,nullptr,nullptr,
        nullptr,batch,rows,cols,batch);}
extern "C" void dsv2_launch_bf16_input_matvec(float *out,const uint16_t *in,
    const uint16_t *w,int batch,int rows,int cols,hipStream_t s){
    if(batch>=96)
        mfma_bf16_gemm_64x128<<<dim3((rows+63)/64,(batch+127)/128),512,0,s>>>(
            out,in,w,batch,rows,cols);
    else
        mfma_bf16_gemm_32x64<<<dim3((rows+63)/64,(batch+63)/64),512,0,s>>>(
            out,in,w,batch,rows,cols);}

}  // namespace
