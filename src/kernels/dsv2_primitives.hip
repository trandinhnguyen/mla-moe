/* DeepSeek-V2-Lite GPU primitives for MI250 (gfx90a).
 *
 * The checkpoint stores row-major BF16 matrices while activations remain FP32
 * to match the CPU oracle. These kernels deliberately expose small C launch
 * functions so getp_run.hip can compose them without changing frozen headers.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int SELECTED_BLOCK_SIZE = 512;
constexpr int EXPERT_DOWN_BLOCK_SIZE = 256;
constexpr int RESNORM_BLOCK_SIZE = 1024;
constexpr int KV_CACHE_BLOCK_SIZE = 512;
constexpr int SOFTMAX_BLOCK_SIZE = 256;
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

/* One block computes one output row. Each wave first reduces its strided dot
 * product, then wave leaders finish the block reduction through LDS. This is
 * the BF16 counterpart of RadeonFlow's values-per-thread reduction pattern. */
__global__ void bf16_matvec_kernel(float *output, const float *input,
                                   const uint16_t *weight, int rows, int cols) {
    int wave = threadIdx.x >> 6;
    int lane = threadIdx.x & 63;
    int row = blockIdx.x;

    const uint16_t *weight_row = weight + (size_t)row * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x * 2; col < cols; col += BLOCK_SIZE * 2) {
        uint32_t packed = *reinterpret_cast<const uint32_t *>(weight_row + col);
        float w0 = __uint_as_float((packed & 0xffffu) << 16);
        float w1 = __uint_as_float(packed & 0xffff0000u);
        sum += input[col] * w0 + input[col + 1] * w1;
    }

    sum = wave_sum(sum);
    __shared__ float wave_sums[BLOCK_SIZE / 64];
    if (lane == 0) wave_sums[wave] = sum;
    __syncthreads();
    if (wave == 0) {
        sum = lane < BLOCK_SIZE / 64 ? wave_sums[lane] : 0.0f;
        sum = wave_sum(sum);
        if (lane == 0) output[row] = sum;
    }
}

__global__ void bf16_matvec_wave_kernel(float *output,const float *input,
                                        const uint16_t *weight,int rows,int cols){
    int wave=threadIdx.x>>6,lane=threadIdx.x&63,row=blockIdx.x*4+wave;
    if(row>=rows)return;const uint16_t *w=weight+(size_t)row*cols;
    float sum=0.0f;for(int col=lane*2;col<cols;col+=128){
        uint32_t packed=*reinterpret_cast<const uint32_t *>(w+col);
        float x0=input[col],x1=input[col+1];
        sum+=x0*__uint_as_float((packed&0xffffu)<<16)
            +x1*__uint_as_float(packed&0xffff0000u);}
    sum=wave_sum(sum);if(lane==0)output[row]=sum;
}

/* DeepSeek hidden size is 2048, so one block can calculate the RMS statistic
 * and normalize the vector without an intermediate global-memory reduction. */
__global__ void rmsnorm_kernel(float *output, const float *input,
                               const uint16_t *weight, int count, float eps) {
    float square_sum = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        float value = input[i];
        square_sum += value * value;
    }

    square_sum = wave_sum(square_sum);
    __shared__ float wave_sums[BLOCK_SIZE / 64];
    __shared__ float inverse_rms;
    int lane = threadIdx.x & 63;
    int wave = threadIdx.x >> 6;
    if (lane == 0) wave_sums[wave] = square_sum;
    __syncthreads();

    if (wave == 0) {
        square_sum = lane < BLOCK_SIZE / 64 ? wave_sums[lane] : 0.0f;
        square_sum = wave_sum(square_sum);
        if (lane == 0)
            inverse_rms = rsqrtf(square_sum / (float)count + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < count; i += blockDim.x)
        output[i] = input[i] * inverse_rms * load_bf16(weight + i);
}

__global__ void residual_add_kernel(float *state, const float *branch,
                                    int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) state[i] += branch[i];
}

__global__ void residual_rmsnorm_kernel(float *output,float *state,
                                         const float *branch,const uint16_t *weight,
                                         int count,float eps){
    float square_sum=0.0f;
    for(int i=threadIdx.x;i<count;i+=blockDim.x){float value=state[i]+branch[i];
        state[i]=value;square_sum+=value*value;}
    square_sum=wave_sum(square_sum);__shared__ float waves[RESNORM_BLOCK_SIZE/64],inverse;
    int lane=threadIdx.x&63,wave=threadIdx.x>>6;if(lane==0)waves[wave]=square_sum;
    __syncthreads();if(wave==0){square_sum=lane<RESNORM_BLOCK_SIZE/64?waves[lane]:0.0f;
        square_sum=wave_sum(square_sum);if(lane==0)inverse=rsqrtf(square_sum/count+eps);}
    __syncthreads();for(int i=threadIdx.x;i<count;i+=blockDim.x)
        output[i]=state[i]*inverse*load_bf16(weight+i);
}

__global__ void embedding_kernel(float *output, const uint16_t *table,
                                 int token, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) output[i] = load_bf16(table + (size_t)token * hidden + i);
}

__global__ void copy_kernel(float *output, const float *input, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i];
}

__global__ void silu_multiply_kernel(float *gate, const float *up, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float x = gate[i];
        gate[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}

__global__ void rope_kernel(float *vectors, int vector_stride, int rope_offset,
                            int vectors_count, int rope_dim, int position,
                            const float *inverse_frequency) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rope_dim / 2;
    int total = vectors_count * half;
    if (pair >= total) return;
    int vector = pair / half;
    int j = pair % half;
    float *v = vectors + (size_t)vector * vector_stride + rope_offset;
    float angle = (float)position * inverse_frequency[j];
    float cs = cosf(angle), sn = sinf(angle);
    float x0 = v[2 * j], x1 = v[2 * j + 1];
    v[2 * j] = x0 * cs - x1 * sn;
    v[2 * j + 1] = x0 * sn + x1 * cs;
}

__global__ void kv_cache_store_kernel(float *cache,const float *compressed,
                                      const uint16_t *norm_weight,int latent_dim,
                                      int rope_dim,int position,float eps,
                                      const float *inverse_frequency){
    float square_sum=0.0f;
    for(int i=threadIdx.x;i<latent_dim;i+=blockDim.x){float value=compressed[i];
        square_sum+=value*value;}
    square_sum=wave_sum(square_sum);
    __shared__ float waves[KV_CACHE_BLOCK_SIZE/64],inverse;
    int lane=threadIdx.x&63,wave=threadIdx.x>>6;
    if(lane==0)waves[wave]=square_sum;__syncthreads();
    if(wave==0){square_sum=lane<KV_CACHE_BLOCK_SIZE/64?waves[lane]:0.0f;
        square_sum=wave_sum(square_sum);if(lane==0)
            inverse=rsqrtf(square_sum/(float)latent_dim+eps);}
    __syncthreads();
    for(int i=threadIdx.x;i<latent_dim;i+=blockDim.x)
        cache[i]=compressed[i]*inverse*load_bf16(norm_weight+i);
    for(int j=threadIdx.x;j<rope_dim/2;j+=blockDim.x){
        float angle=(float)position*inverse_frequency[j];
        float cs=cosf(angle),sn=sinf(angle);
        float x0=compressed[latent_dim+2*j],x1=compressed[latent_dim+2*j+1];
        cache[latent_dim+2*j]=x0*cs-x1*sn;
        cache[latent_dim+2*j+1]=x0*sn+x1*cs;}
}

/* q_absorbed[h,r] = sum_d q_nope[h,d] * W_UK[h,d,r]. */
__global__ void absorb_query_kernel(float *q_absorbed, float *query,
                                    const uint16_t *kv_up, int heads,
                                    int q_head_dim, int nope_dim, int value_dim,
                                    int latent_dim,int position,
                                    const float *inverse_frequency) {
    int wave=threadIdx.x>>6,lane=threadIdx.x&63;
    int r0=blockIdx.x*8+wave*2,r1=r0+1;
    int h = blockIdx.y;
    if (h >= heads || r0 >= latent_dim) return;
    size_t head_stride = (size_t)(nope_dim + value_dim) * latent_dim;
    const uint16_t *w = kv_up + (size_t)h * head_stride;
    float *q = query + (size_t)h * q_head_dim;
    int rope_dim=q_head_dim-nope_dim;
    if(blockIdx.x==0&&wave==0&&lane<rope_dim/2){int j=lane;
        float angle=(float)position*inverse_frequency[j];
        float cs=cosf(angle),sn=sinf(angle),x0=q[nope_dim+2*j],x1=q[nope_dim+2*j+1];
        q[nope_dim+2*j]=x0*cs-x1*sn;q[nope_dim+2*j+1]=x0*sn+x1*cs;}
    float sum0=0.0f,sum1=0.0f;
    for (int d = lane; d < nope_dim; d += 64) {
        float value=q[d];
        const uint16_t *row=w+(size_t)d*latent_dim;
        uint32_t packed=*reinterpret_cast<const uint32_t *>(row+r0);
        sum0+=value*__uint_as_float((packed&0xffffu)<<16);
        if(r1<latent_dim)sum1+=value*__uint_as_float(packed&0xffff0000u);
    }
    sum0=wave_sum(sum0);sum1=wave_sum(sum1);
    if(lane==0){q_absorbed[(size_t)h*latent_dim+r0]=sum0;
        if(r1<latent_dim)q_absorbed[(size_t)h*latent_dim+r1]=sum1;}
}

__global__ void attention_score_kernel(float *scores, const float *q_absorbed,
                                       const float *query, const float *cache,
                                       int length, int cache_stride, int heads,
                                       int q_head_dim, int nope_dim,
                                       int latent_dim, int rope_dim, float scale) {
    int key = blockIdx.x;
    int h = blockIdx.y;
    if (key >= length || h >= heads) return;
    const float *qa = q_absorbed + (size_t)h * latent_dim;
    const float *qp = query + (size_t)h * q_head_dim + nope_dim;
    const float *kv = cache + (size_t)key * cache_stride;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < latent_dim; i += blockDim.x)
        sum += qa[i] * kv[i];
    for (int i = threadIdx.x; i < rope_dim; i += blockDim.x)
        sum += qp[i] * kv[latent_dim + i];
    sum = wave_sum(sum);
    __shared__ float waves[BLOCK_SIZE / 64];
    int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    if (lane == 0) waves[wave] = sum;
    __syncthreads();
    if (wave == 0) {
        sum = lane < BLOCK_SIZE / 64 ? waves[lane] : 0.0f;
        sum = wave_sum(sum);
        if (lane == 0) scores[(size_t)h * length + key] = sum * scale;
    }
}

__global__ void attention_softmax_kernel(float *scores, int length) {
    int h = blockIdx.x;
    float *row = scores + (size_t)h * length;
    float maximum = -INFINITY;
    for (int i = threadIdx.x; i < length; i += blockDim.x)
        maximum = fmaxf(maximum, row[i]);
    for (int offset = 32; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 64));
    __shared__ float waves[SOFTMAX_BLOCK_SIZE / 64];
    __shared__ float row_max, inverse_sum;
    int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    if (lane == 0) waves[wave] = maximum;
    __syncthreads();
    if (threadIdx.x == 0) {
        row_max = waves[0];
        for (int i = 1; i < SOFTMAX_BLOCK_SIZE / 64; ++i) row_max = fmaxf(row_max, waves[i]);
    }
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        float value = expf(row[i] - row_max);
        row[i] = value;
        sum += value;
    }
    sum = wave_sum(sum);
    if (lane == 0) waves[wave] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0.0f;
        for (int i = 0; i < SOFTMAX_BLOCK_SIZE / 64; ++i) sum += waves[i];
        inverse_sum = 1.0f / sum;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < length; i += blockDim.x) row[i] *= inverse_sum;
}

__global__ void latent_context_kernel(float *context, const float *scores,
                                      const float *cache, int length,
                                      int cache_stride, int latent_dim) {
    int wave=threadIdx.x>>6,lane=threadIdx.x&63;
    int r=blockIdx.x*4+wave,h=blockIdx.y;
    if(r>=latent_dim)return;
    float sum = 0.0f;
    const float *row = scores + (size_t)h * length;
    for (int k = lane; k < length; k += 64)
        sum += row[k] * cache[(size_t)k * cache_stride + r];
    sum = wave_sum(sum);
    if(lane==0)context[(size_t)h*latent_dim+r]=sum;
}

__global__ void latent_value_kernel(float *output, const float *latent,
                                    const uint16_t *kv_up, int heads,
                                    int nope_dim, int value_dim,
                                    int latent_dim) {
    int value = blockIdx.x, h = blockIdx.y;
    if (value >= value_dim || h >= heads) return;
    size_t head_stride = (size_t)(nope_dim + value_dim) * latent_dim;
    const uint16_t *row = kv_up + (size_t)h * head_stride +
                          (size_t)(nope_dim + value) * latent_dim;
    const float *x = latent + (size_t)h * latent_dim;
    float sum = 0.0f;
    for(int i=threadIdx.x*2;i<latent_dim;i+=blockDim.x*2){
        uint32_t p=*reinterpret_cast<const uint32_t *>(row+i);
        sum+=x[i]*__uint_as_float((p&0xffffu)<<16)
            +x[i+1]*__uint_as_float(p&0xffff0000u);}
    sum = wave_sum(sum);
    __shared__ float waves[BLOCK_SIZE / 64];
    int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    if (lane == 0) waves[wave] = sum;
    __syncthreads();
    if (wave == 0) {
        sum = lane < BLOCK_SIZE / 64 ? waves[lane] : 0.0f;
        sum = wave_sum(sum);
        if (lane == 0) output[(size_t)h * value_dim + value] = sum;
    }
}

__global__ void router_topk_kernel(const float *logits, int experts, int topk,
                                   int *indices, float *weights) {
    int e = threadIdx.x;
    float value = e < experts ? logits[e] : -INFINITY;
    float maximum = value;
    for (int offset = 32; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down(maximum, offset, 64));
    maximum = __shfl(maximum, 0, 64);
    float exp_value = e < experts ? expf(value - maximum) : 0.0f;
    float sum = wave_sum(exp_value);
    sum = __shfl(sum, 0, 64);
    float score=e<experts?exp_value/sum:-INFINITY;
    for(int rank=0;rank<topk;++rank){
        float best_value=score;int best=e;
        for(int offset=32;offset>0;offset>>=1){
            float other_value=__shfl_down(best_value,offset,64);
            int other=__shfl_down(best,offset,64);
            if(e+offset<64&&(other_value>best_value||
                (other_value==best_value&&other<best))){best_value=other_value;best=other;}
        }
        best=__shfl(best,0,64);best_value=__shfl(best_value,0,64);
        if(e==0){indices[rank]=best;weights[rank]=best_value;}
        if(e==best)score=-INFINITY;
    }
}

__global__ void selected_expert_wave_kernel(
    float *output,const float *input,const uint16_t *const *weights,
    const int *selected,int selections,int rows,int cols){
    int rank=blockIdx.y,wave=threadIdx.x>>6,lane=threadIdx.x&63;
    int row0=blockIdx.x*(blockDim.x/64*2)+wave*2,row1=row0+1;
    if(rank>=selections||row0>=rows)return;
    const uint16_t *base=weights[selected[rank]];
    const uint16_t *w0=base+(size_t)row0*cols,*w1=base+(size_t)row1*cols;
    const float *x=input+(size_t)rank*cols;float sum0=0.0f,sum1=0.0f;
    for(int i=lane*2;i<cols;i+=128){uint32_t p;
        float x0=x[i],x1=x[i+1];
        p=*reinterpret_cast<const uint32_t *>(w0+i);
        sum0+=x0*__uint_as_float((p&0xffffu)<<16)+x1*__uint_as_float(p&0xffff0000u);
        if(row1<rows){p=*reinterpret_cast<const uint32_t *>(w1+i);
            sum1+=x0*__uint_as_float((p&0xffffu)<<16)+x1*__uint_as_float(p&0xffff0000u);}}
    sum0=wave_sum(sum0);sum1=wave_sum(sum1);if(lane==0){
        output[(size_t)rank*rows+row0]=sum0;
        if(row1<rows)output[(size_t)rank*rows+row1]=sum1;}
}

__global__ void broadcast_kernel(float *output, const float *input,
                                 int copies, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < copies * count) output[i] = input[i % count];
}

__global__ void weighted_expert_add_kernel(float *output,
                                           const float *expert_output,
                                           const float *weights,
                                           int selections, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    float value = output[i];
    for (int rank = 0; rank < selections; ++rank)
        value += weights[rank] * expert_output[(size_t)rank * hidden + i];
    output[i] = value;
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
using float2_reg = float __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint16_t to_bf16_rne(float value){
    uint32_t bits=__float_as_uint(value);
    bits+=0x7fffu+((bits>>16)&1u);return (uint16_t)(bits>>16);
}

__global__ void convert_bf16_kernel(uint16_t *output,const float *input,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)output[i]=to_bf16_rne(input[i]);
}

__global__ void mfma_f32_bf16_weight_gemm(float *output, const float *input,
                                           const uint16_t *weight,
                                           int batch, int rows, int cols) {
    const int lane = threadIdx.x;
    const int tile_m = blockIdx.x * 16;
    const int tile_n = blockIdx.y * 16;
    const int a_row = tile_m + (lane & 15);
    const int b_row = tile_n + (lane & 15);
    const int k_lane = lane >> 4;
    float4_acc acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = k_lane; k < cols; k += 4) {
        float a = a_row < rows ? load_bf16(weight + (size_t)a_row * cols + k)
                               : 0.0f;
        float b = b_row < batch ? input[(size_t)b_row * cols + k] : 0.0f;
        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
    }

    const int out_row = tile_m + (lane >> 4) * 4;
    const int out_batch = tile_n + (lane & 15);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        if (out_batch < batch && out_row + i < rows)
            output[(size_t)out_batch * rows + out_row + i] = acc[i];
}

__global__ void mfma_bf16_gemm(float *output,const uint16_t *input,
    const uint16_t *weight,int batch,int rows,int cols){
    int lane=threadIdx.x&63,wave=threadIdx.x>>6,tile_m=blockIdx.x*16;
    int tile_n=blockIdx.y*64+wave*16;
    int row=tile_m+(lane&15),sample=tile_n+(lane&15),kg=(lane>>4)*4;
    float4_acc acc={0,0,0,0};__shared__ uint16_t shared_a[256];
    for(int kb=0;kb<cols;kb+=16){uint32_t a0=0,a1=0,b0=0,b1=0;
        int k=kb+kg;if(wave==0){uint16_t *dst=shared_a+lane*4;
            if(row<rows){const uint16_t *p=weight+(size_t)row*cols+k;
                dst[0]=p[0];dst[1]=p[1];dst[2]=p[2];dst[3]=p[3];}
            else dst[0]=dst[1]=dst[2]=dst[3]=0;}
        __syncthreads();{const uint16_t *p=shared_a+lane*4;
            a0=(uint32_t)p[0]|((uint32_t)p[1]<<16);
            a1=(uint32_t)p[2]|((uint32_t)p[3]<<16);}
        if(sample<batch){const uint16_t *p=input+(size_t)sample*cols+k;
            b0=*reinterpret_cast<const uint32_t *>(p);
            b1=*reinterpret_cast<const uint32_t *>(p+2);}
        float2_reg av={__uint_as_float(a0),__uint_as_float(a1)};
        float2_reg bv={__uint_as_float(b0),__uint_as_float(b1)};
        acc=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av,bv,acc,0,0,0);
        __syncthreads();}
    int out_row=tile_m+(lane>>4)*4,out_sample=tile_n+(lane&15);
    for(int i=0;i<4;++i)if(out_sample<batch&&out_row+i<rows)
        output[(size_t)out_sample*rows+out_row+i]=acc[i];
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

__global__ void pack_head_bf16(uint16_t *output,const float *input,int batch,
    int heads,int input_stride,int cols){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)batch*heads*cols;
    if(i<total){int col=i%cols,head=(i/cols)%heads,token=i/(cols*heads);
        output[((size_t)head*batch+token)*cols+col]=
            to_bf16_rne(input[((size_t)token*heads+head)*input_stride+col]);}
}
__global__ void pack_shared_head_bf16(uint16_t *output,const float *input,int batch,
    int heads,int cols){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t total=(size_t)batch*heads*cols;if(i<total){int col=i%cols;
        int token=(i/cols)%batch,head=i/(cols*batch);
        output[((size_t)head*batch+token)*cols+col]=to_bf16_rne(input[(size_t)token*cols+col]);}}

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

__global__ void batched_absorb_query_kernel(
    float *output, const float *query, const uint16_t *kv_up,
    int batch, int heads, int q_head_dim, int nope_dim,
    int value_dim, int latent_dim) {
    int wave=threadIdx.x>>6,lane=threadIdx.x&63;
    int r0=blockIdx.x*16+wave*2,r1=r0+1;
    int h = blockIdx.y, token = blockIdx.z;
    if(r0>=latent_dim)return;
    size_t head_stride = (size_t)(nope_dim + value_dim) * latent_dim;
    const uint16_t *w = kv_up + (size_t)h * head_stride;
    const float *q = query + ((size_t)token * heads + h) * q_head_dim;
    float sum0=0.0f,sum1=0.0f;
    for(int d=lane;d<nope_dim;d+=64){float value=q[d];
        const uint16_t *row=w+(size_t)d*latent_dim;
        uint32_t packed=*reinterpret_cast<const uint32_t *>(row+r0);
        sum0+=value*__uint_as_float((packed&0xffffu)<<16);
        if(r1<latent_dim)sum1+=value*__uint_as_float(packed&0xffff0000u);}
    sum0=wave_sum(sum0);sum1=wave_sum(sum1);
    if(lane==0){size_t base=((size_t)token*heads+h)*latent_dim;
        output[base+r0]=sum0;if(r1<latent_dim)output[base+r1]=sum1;}
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

__global__ void decode_attention_kernel(float *context,const float *qa,
    const float *query,const float *cache,int batch,int stride,int heads,int qhd,
    int nope,int latent,int rope,float scale,const int *positions,int cache_sequence){
    int request=blockIdx.x,h=blockIdx.y,pos=positions[request];
    const float *qav=qa+((size_t)request*heads+h)*latent;
    const float *qp=query+((size_t)request*heads+h)*qhd+nope;
    float acc0=0,acc1=0,rmax=-INFINITY,rsum=0;__shared__ float waves[4];
    __shared__ float next_max,alpha,beta,next_sum;int lane=threadIdx.x&63,wave=threadIdx.x>>6;
    for(int key=0;key<=pos;++key){const float *kv=cache+((size_t)request*cache_sequence+key)*stride;
        float score=0;for(int i=threadIdx.x;i<latent;i+=blockDim.x)score+=qav[i]*kv[i];
        for(int i=threadIdx.x;i<rope;i+=blockDim.x)score+=qp[i]*kv[latent+i];
        score=wave_sum(score);if(lane==0)waves[wave]=score;__syncthreads();
        if(threadIdx.x==0){score=(waves[0]+waves[1]+waves[2]+waves[3])*scale;
            next_max=fmaxf(rmax,score);alpha=expf(rmax-next_max);
            beta=expf(score-next_max);next_sum=rsum*alpha+beta;}__syncthreads();
        acc0=acc0*alpha+beta*kv[threadIdx.x];
        if(threadIdx.x+BLOCK_SIZE<latent)acc1=acc1*alpha+beta*kv[threadIdx.x+BLOCK_SIZE];
        rmax=next_max;rsum=next_sum;__syncthreads();}
    float *out=context+((size_t)request*heads+h)*latent;
    out[threadIdx.x]=acc0/rsum;if(threadIdx.x+BLOCK_SIZE<latent)out[threadIdx.x+BLOCK_SIZE]=acc1/rsum;
}

__global__ void decode_attention_head2_exact(float *context,const float *qa,
    const float *query,const float *cache,int batch,int stride,int heads,int qhd,
    int nope,int latent,int rope,float scale,const int *positions,int cache_sequence){
    int request=blockIdx.x,wave=threadIdx.x>>6,lane=threadIdx.x&63;
    int group=wave>>2,subwave=wave&3,h=blockIdx.y*2+group,pos=positions[request];
    const float *qav=qa+((size_t)request*heads+h)*latent;
    const float *qp=query+((size_t)request*heads+h)*qhd+nope;
    int t=subwave*64+lane;float acc0=0,acc1=0,rmax=-INFINITY,rsum=0;
    __shared__ float kvs[576],partials[8],alphas[2],betas[2],nmax[2],nsum[2];
    for(int key=0;key<=pos;++key){const float *kv=cache+((size_t)request*cache_sequence+key)*stride;
        for(int i=threadIdx.x;i<stride;i+=blockDim.x)kvs[i]=kv[i];__syncthreads();
        float score=qav[t]*kvs[t];if(t+256<latent)score+=qav[t+256]*kvs[t+256];
        if(t<rope)score+=qp[t]*kvs[latent+t];
        score=wave_sum(score);if(lane==0)partials[wave]=score;__syncthreads();
        if(subwave==0&&lane==0){float s=0;for(int j=0;j<4;++j)s+=partials[group*4+j];s*=scale;
            float m=fmaxf(rmax,s);nmax[group]=m;alphas[group]=expf(rmax-m);
            betas[group]=expf(s-m);nsum[group]=rsum*alphas[group]+betas[group];}__syncthreads();
        acc0=acc0*alphas[group]+betas[group]*kvs[t];
        if(t+256<latent)acc1=acc1*alphas[group]+betas[group]*kvs[t+256];
        rmax=nmax[group];rsum=nsum[group];__syncthreads();}
    float *out=context+((size_t)request*heads+h)*latent;
    out[t]=acc0/rsum;if(t+256<latent)out[t+256]=acc1/rsum;
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
    __shared__ float kvs[2][576];
    const float *kv0=cache+(size_t)request*cache_sequence*stride;
    for(int i=threadIdx.x;i<stride;i+=blockDim.x)kvs[0][i]=kv0[i];
    __syncthreads();
    for(int key=0;key<=pos;++key){int cur=key&1,nxt=cur^1;
        float next_value=0;
        if(key<pos&&threadIdx.x<stride){const float *next=cache+
            ((size_t)request*cache_sequence+key+1)*stride;
            next_value=next[threadIdx.x];}
        float score=0;
        #pragma unroll
        for(int j=0;j<8;++j){int i=lane+j*64;score+=qav[i]*kvs[cur][i];}
        if(lane<rope)score+=qp[lane]*kvs[cur][latent+lane];
        score=wave_sum(score);
        float a=0,b=0,nm=0,ns=0;
        if(lane==0){score*=scale;nm=fmaxf(rmax,score);
            a=expf(rmax-nm);b=expf(score-nm);ns=rsum*a+b;}
        a=__shfl(a,0,64);b=__shfl(b,0,64);
        nm=__shfl(nm,0,64);ns=__shfl(ns,0,64);
        #pragma unroll
        for(int j=0;j<8;++j)acc[j]=acc[j]*a+b*kvs[cur][lane+j*64];
        rmax=nm;rsum=ns;
        if(key<pos&&threadIdx.x<stride)kvs[nxt][threadIdx.x]=next_value;
        __syncthreads();
    }
    float *out=context+((size_t)request*heads+h)*latent;
    #pragma unroll
    for(int j=0;j<8;++j)out[lane+j*64]=acc[j]/rsum;
}

__global__ void batched_latent_value_kernel(
    float *output, const float *latent, const uint16_t *kv_up,
    int batch, int heads, int nope_dim, int value_dim, int latent_dim) {
    int value=blockIdx.x,h=blockIdx.y,token=blockIdx.z;
    size_t hs=(size_t)(nope_dim+value_dim)*latent_dim;
    const uint16_t *w=kv_up+(size_t)h*hs+(size_t)(nope_dim+value)*latent_dim;
    const float *x=latent+((size_t)token*heads+h)*latent_dim;
    float sum=0.0f;for(int i=threadIdx.x*2;i<latent_dim;i+=blockDim.x*2){
        uint32_t packed=*reinterpret_cast<const uint32_t *>(w+i);
        sum+=x[i]*__uint_as_float((packed&0xffffu)<<16)
            +x[i+1]*__uint_as_float(packed&0xffff0000u);}
    sum=wave_sum(sum); __shared__ float waves[4]; int lane=threadIdx.x&63,wave=threadIdx.x>>6;
    if(lane==0)waves[wave]=sum; __syncthreads(); if(wave==0){sum=lane<4?waves[lane]:0;sum=wave_sum(sum);
    if(lane==0)output[((size_t)token*heads+h)*value_dim+value]=sum;}
}

__global__ void prefill_unabsorbed_attention_kernel(float *output,const float *query,
    const float *key,const float *value,const float *cache,int rows,int heads,int qhd,
    int nope,int vdim,int cache_stride,int latent,int rope,int sequence_length,
    int request_base,int cache_sequence,float scale){
    int row=blockIdx.x,h=blockIdx.y,pos=row%sequence_length;
    const float *q=query+((size_t)row*heads+h)*qhd;
    float acc=0.0f,rmax=-INFINITY,rsum=0.0f;
    for(int k=0;k<=pos;++k){int kr=row-pos+k;
        const float *kk=key+((size_t)kr*heads+h)*nope;
        float score=0.0f;for(int d=threadIdx.x;d<nope;d+=blockDim.x)score+=q[d]*kk[d];
        const float *rk=cache+((size_t)(request_base+row/sequence_length)*cache_sequence+k)*cache_stride+latent;
        for(int d=threadIdx.x;d<rope;d+=blockDim.x)score+=q[nope+d]*rk[d];
        score=wave_sum(score);__shared__ float waves[4],alpha,beta,nextmax,nextsum;
        int lane=threadIdx.x&63,wave=threadIdx.x>>6;if(lane==0)waves[wave]=score;
        __syncthreads();if(threadIdx.x==0){score=(waves[0]+waves[1]+waves[2]+waves[3])*scale;
            nextmax=fmaxf(rmax,score);alpha=expf(rmax-nextmax);beta=expf(score-nextmax);
            nextsum=rsum*alpha+beta;}__syncthreads();
        if(threadIdx.x<vdim)acc=acc*alpha+beta*value[((size_t)kr*heads+h)*vdim+threadIdx.x];
        rmax=nextmax;rsum=nextsum;__syncthreads();}
    if(threadIdx.x<vdim)output[((size_t)row*heads+h)*vdim+threadIdx.x]=acc/rsum;
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
                nm=fmaxf(rmax,score);a=expf(rmax-nm);b=expf(score-nm);ns=rsum*a+b;}
            else{nm=rmax;ns=rsum;a=1.0f;b=0.0f;}}
        a=__shfl(a,0,64);b=__shfl(b,0,64);
        nm=__shfl(nm,0,64);ns=__shfl(ns,0,64);
        acc0=acc0*a+b*sv[cur][lane];acc1=acc1*a+b*sv[cur][lane+64];
        rmax=nm;rsum=ns;
        if(k<last){if(threadIdx.x<nope)sk[nxt][threadIdx.x]=next_k;
            if(threadIdx.x<rope)sk[nxt][nope+threadIdx.x]=next_r;
            if(threadIdx.x<vdim)sv[nxt][threadIdx.x]=next_v;}
        __syncthreads();}
    if(qpos<sequence_length){float *out=output+((size_t)row*heads+h)*vdim;
        out[lane]=acc0/rsum;out[lane+64]=acc1/rsum;}
}

__global__ void batched_router_topk_kernel(const float *logits, int batch,
                                           int experts, int topk,
                                           int *indices, float *weights) {
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
            weights[(size_t)token*topk+rank]=best_value;}if(e==best)score=-INFINITY;}
}

__global__ void bucket_routes_kernel(const int *indices,const float *weights,
    int tokens,int topk,int *counts,int *slots,float *slot_weights,int capacity){
    int route=blockIdx.x*blockDim.x+threadIdx.x,total=tokens*topk;
    if(route>=total)return;int expert=indices[route];
    int slot=atomicAdd(counts+expert,1);
    if(slot<capacity){slots[(size_t)expert*capacity+slot]=route;
        slot_weights[(size_t)expert*capacity+slot]=weights[route];}
}

__global__ void pack_expert_bf16_kernel(uint16_t *output,const float *input,
    const int *counts,const int *slots,int experts,int capacity,int hidden){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    int pairs=hidden/2;size_t total=(size_t)experts*capacity*pairs;if(i>=total)return;
    int hp=i%pairs,route=i/pairs,expert=route/capacity,slot=route%capacity;
    if(slot<min(counts[expert],capacity)){int h=hp*2,token=slots[route]/6;
        const float *src=input+(size_t)token*hidden+h;
        uint32_t packed=(uint32_t)to_bf16_rne(src[0])|
                        ((uint32_t)to_bf16_rne(src[1])<<16);
        *reinterpret_cast<uint32_t *>(output+(size_t)route*hidden+h)=packed;}
}

__global__ void convert_expert_bf16_kernel(uint16_t *output,const float *input,
    const int *counts,int experts,int capacity,int width){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t total=(size_t)experts*capacity*width;if(i>=total)return;
    int route=i/width,expert=route/capacity,slot=route%capacity;
    if(slot<min(counts[expert],capacity))output[i]=to_bf16_rne(input[i]);
}

/* Expert-sorted GEMM. grid.z selects the expert and grid.y its fixed-capacity
 * route tile; device-side counts suppress empty tiles. input_mode=0 gathers
 * token rows through slots, input_mode=1 consumes the preceding bucketed
 * activation directly. */
__global__ void grouped_expert_mfma_kernel(float *output,const uint16_t *input,
    const uint16_t *const *weights,const int *counts,const int *slots,
    int experts,int capacity,int rows,int cols,int input_mode){
    int expert=blockIdx.z,lane=threadIdx.x&63,wave=threadIdx.x>>6;
    int tile_m=blockIdx.x*16;
    int tile_slot=blockIdx.y*64+wave*16,k_lane=lane>>4;
    int out_feature=tile_m+(lane&15),slot=tile_slot+(lane&15);
    int count=min(counts[expert],capacity);
    if(blockIdx.y*64>=count)return;
    float4_acc acc={0,0,0,0};const uint16_t *w=weights[expert];
    int kg=k_lane*4;
    __shared__ uint16_t shared_a[256];
    for(int kb=0;kb<cols;kb+=16){uint32_t a0=0,a1=0,b0=0,b1=0;
        int k=kb+kg;if(wave==0){
            uint16_t *dst=shared_a+lane*4;
            if(out_feature<rows){const uint16_t *p=w+(size_t)out_feature*cols+k;
                dst[0]=p[0];dst[1]=p[1];dst[2]=p[2];dst[3]=p[3];}
            else dst[0]=dst[1]=dst[2]=dst[3]=0;}
        __syncthreads();{
            const uint16_t *p=shared_a+lane*4;
            a0=(uint32_t)p[0]|((uint32_t)p[1]<<16);
            a1=(uint32_t)p[2]|((uint32_t)p[3]<<16);}
        if(slot<count){const uint16_t *p=input+((size_t)expert*capacity+slot)*cols+k;
            b0=*reinterpret_cast<const uint32_t *>(p);
            b1=*reinterpret_cast<const uint32_t *>(p+2);}
        float2_reg av={__uint_as_float(a0),__uint_as_float(a1)};
        float2_reg bv={__uint_as_float(b0),__uint_as_float(b1)};
        acc=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av,bv,acc,0,0,0);
        __syncthreads();
    }
    int feature=tile_m+(lane>>4)*4,out_slot=tile_slot+(lane&15);
    for(int i=0;i<4;++i)if(out_slot<count&&feature+i<rows)
        output[((size_t)expert*capacity+out_slot)*rows+feature+i]=acc[i];
}

__global__ void grouped_expert_mfma_32x64(float *output,const uint16_t *input,
    const uint16_t *const *weights,const int *counts,const int *slots,
    int capacity,int rows,int cols){
    int expert=blockIdx.z,count=min(counts[expert],capacity);
    if((int)blockIdx.y*64>=count)return;
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6,mt=wave&3,nt=wave>>2;
    int tile_m=blockIdx.x*64,tile_n=blockIdx.y*64,kg=(lane>>4)*4;
    float4_acc acc0={0,0,0,0},acc1={0,0,0,0};const uint16_t *w=weights[expert];
    const uint16_t *x=input+(size_t)expert*capacity*cols;
    __shared__ uint16_t as[2][1024],bs[2][1024];
    for(int z=tid;z<1024;z+=512){int r=z/16,k=z%16;
        as[0][z]=tile_m+r<rows?w[(size_t)(tile_m+r)*cols+k]:0;
        int n=z/16;bs[0][z]=tile_n+n<count?x[(size_t)(tile_n+n)*cols+k]:0;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=16){int cur=(kb>>4)&1,nxt=cur^1;
        int ai=(mt*16+(lane&15))*16+kg;
        int bi0=(nt*16+(lane&15))*16+kg,bi1=bi0+32*16;
        uint32_t a0=*reinterpret_cast<uint32_t *>(as[cur]+ai);
        uint32_t a1=*reinterpret_cast<uint32_t *>(as[cur]+ai+2);
        uint32_t b00=*reinterpret_cast<uint32_t *>(bs[cur]+bi0);
        uint32_t b01=*reinterpret_cast<uint32_t *>(bs[cur]+bi0+2);
        uint32_t b10=*reinterpret_cast<uint32_t *>(bs[cur]+bi1);
        uint32_t b11=*reinterpret_cast<uint32_t *>(bs[cur]+bi1+2);
        uint16_t na0=0,nb0=0,na1=0,nb1=0;
        if(kb+16<cols){int k=tid&15,r=tid>>4,n=r;
            if(tile_m+r<rows)na0=w[(size_t)(tile_m+r)*cols+kb+16+k];
            if(tile_n+n<count)nb0=x[(size_t)(tile_n+n)*cols+kb+16+k];
            r+=32;n+=32;
            if(tile_m+r<rows)na1=w[(size_t)(tile_m+r)*cols+kb+16+k];
            if(tile_n+n<count)nb1=x[(size_t)(tile_n+n)*cols+kb+16+k];}
        float2_reg av={__uint_as_float(a0),__uint_as_float(a1)};
        float2_reg bv0={__uint_as_float(b00),__uint_as_float(b01)};
        float2_reg bv1={__uint_as_float(b10),__uint_as_float(b11)};
        acc0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av,bv0,acc0,0,0,0);
        acc1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av,bv1,acc1,0,0,0);
        if(kb+16<cols){as[nxt][tid]=na0;bs[nxt][tid]=nb0;
            as[nxt][tid+512]=na1;bs[nxt][tid+512]=nb1;}
        __syncthreads();}
    int r=tile_m+mt*16+(lane>>4)*4,n0=tile_n+nt*16+(lane&15),n1=n0+32;
    for(int i=0;i<4;++i){if(n0<count&&r+i<rows){int route=slots[(size_t)expert*capacity+n0];
            output[(size_t)route*rows+r+i]=acc0[i];}
        if(n1<count&&r+i<rows){int route=slots[(size_t)expert*capacity+n1];
            output[(size_t)route*rows+r+i]=acc1[i];}}
}

__global__ void grouped_expert_gate_up_32x128(uint16_t *gate,float *up,
    const uint16_t *input,const uint16_t *const *gate_w,
    const uint16_t *const *up_w,const int *counts,int capacity,int rows,int cols){
    int expert=blockIdx.z,count=min(counts[expert],capacity);
    if((int)blockIdx.y*128>=count)return;
    int tid=threadIdx.x,lane=tid&63,wave=tid>>6,mt=wave&1,nt=wave>>1;
    int tile_m=blockIdx.x*64,tile_n=blockIdx.y*128,kg=(lane>>4)*4;
    float4_acc ag0={0,0,0,0},au0={0,0,0,0},ag1={0,0,0,0},au1={0,0,0,0};
    float4_acc bg0={0,0,0,0},bu0={0,0,0,0},bg1={0,0,0,0},bu1={0,0,0,0};
    const uint16_t *wg=gate_w[expert],*wu=up_w[expert];
    const uint16_t *x=input+(size_t)expert*capacity*cols;
    __shared__ uint16_t gs[2][1024],us[2][1024],bs[2][2048];
    for(int z=tid;z<1024;z+=512){int r=z/16,k=z%16;
        if(tile_m+r<rows){gs[0][z]=wg[(size_t)(tile_m+r)*cols+k];
            us[0][z]=wu[(size_t)(tile_m+r)*cols+k];}else gs[0][z]=us[0][z]=0;
        int n=z/16;bs[0][z]=tile_n+n<count?x[(size_t)(tile_n+n)*cols+k]:0;
        n+=64;bs[0][z+1024]=tile_n+n<count?x[(size_t)(tile_n+n)*cols+k]:0;}
    __syncthreads();
    for(int kb=0;kb<cols;kb+=16){int cur=(kb>>4)&1,nxt=cur^1;
        int ai0=(mt*16+(lane&15))*16+kg;
        int ai1=((mt+2)*16+(lane&15))*16+kg,bi=(nt*16+(lane&15))*16+kg;
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
        uint16_t ng0=0,nu0=0,nb0=0,ng1=0,nu1=0,nb1=0,nb2=0,nb3=0;
        if(kb+16<cols){int k=tid&15,r=tid>>4,n=r;
            if(tile_m+r<rows){ng0=wg[(size_t)(tile_m+r)*cols+kb+16+k];
                nu0=wu[(size_t)(tile_m+r)*cols+kb+16+k];}
            if(tile_n+n<count)nb0=x[(size_t)(tile_n+n)*cols+kb+16+k];
            r+=32;n+=32;
            if(tile_m+r<rows){ng1=wg[(size_t)(tile_m+r)*cols+kb+16+k];
                nu1=wu[(size_t)(tile_m+r)*cols+kb+16+k];}
            if(tile_n+n<count)nb1=x[(size_t)(tile_n+n)*cols+kb+16+k];}
        if(kb+16<cols){int k=tid&15,n=(tid>>4)+64;
            if(tile_n+n<count)nb2=x[(size_t)(tile_n+n)*cols+kb+16+k];
            n+=32;if(tile_n+n<count)nb3=x[(size_t)(tile_n+n)*cols+kb+16+k];}
        ag0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv0,bv,ag0,0,0,0);
        au0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv0,bv,au0,0,0,0);
        ag1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv1,bv,ag1,0,0,0);
        au1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv1,bv,au1,0,0,0);
        bg0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv0,bv2,bg0,0,0,0);
        bu0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv0,bv2,bu0,0,0,0);
        bg1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(gv1,bv2,bg1,0,0,0);
        bu1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(uv1,bv2,bu1,0,0,0);
        if(kb+16<cols){gs[nxt][tid]=ng0;us[nxt][tid]=nu0;bs[nxt][tid]=nb0;
            gs[nxt][tid+512]=ng1;us[nxt][tid+512]=nu1;bs[nxt][tid+512]=nb1;
            bs[nxt][tid+1024]=nb2;bs[nxt][tid+1536]=nb3;}
        __syncthreads();}
    int r0=tile_m+mt*16+(lane>>4)*4,r1=r0+32,n=tile_n+nt*16+(lane&15),n2=n+64;
    #pragma unroll
    for(int i=0;i<4;++i){if(n<count&&r0+i<rows){size_t o=((size_t)expert*capacity+n)*rows+r0+i;
        float g=ag0[i];gate[o]=to_bf16_rne((g/(1.0f+expf(-g)))*au0[i]);}
        if(n<count&&r1+i<rows){size_t o=((size_t)expert*capacity+n)*rows+r1+i;
        float g=ag1[i];gate[o]=to_bf16_rne((g/(1.0f+expf(-g)))*au1[i]);}
        if(n2<count&&r0+i<rows){size_t o=((size_t)expert*capacity+n2)*rows+r0+i;
        float g=bg0[i];gate[o]=to_bf16_rne((g/(1.0f+expf(-g)))*bu0[i]);}
        if(n2<count&&r1+i<rows){size_t o=((size_t)expert*capacity+n2)*rows+r1+i;
        float g=bg1[i];gate[o]=to_bf16_rne((g/(1.0f+expf(-g)))*bu1[i]);}}
}

__global__ void bucket_swiglu_kernel(float *gate,const float *up,
                                      int total){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<total){float g=gate[i];
        gate[i]=(g/(1.0f+expf(-g)))*up[i];}
}

__global__ void scatter_bucket_experts_kernel(float *output,const float *expert_out,
    const int *counts,const int *slots,const float *weights,int experts,
    int capacity,int tokens,int hidden){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t routes=(size_t)experts*capacity,total=routes*hidden;
    if(i>=total)return;int h=i%hidden,route=i/hidden;
    int expert=route/capacity,slot=route%capacity;
    if(slot>=min(counts[expert],capacity))return;int original_route=slots[route];
    output[(size_t)original_route*hidden+h]=expert_out[i];
}

__global__ void expand_selected_kernel(float *output,const float *input,
                                       int batch,int topk,int hidden){
    int i=blockIdx.x*blockDim.x+threadIdx.x,total=batch*topk*hidden;
    if(i<total){int selection=i/hidden,token=selection/topk;output[i]=input[(size_t)token*hidden+i%hidden];}
}

__global__ void batched_weighted_add_kernel(float *output,const float *experts,
                                            const float *weights,int batch,
                                            int topk,int hidden){
    int i=blockIdx.x*blockDim.x+threadIdx.x,total=batch*hidden;if(i>=total)return;
    int token=i/hidden,col=i%hidden;float v=output[i];
    for(int r=0;r<topk;++r)
        v+=weights[(size_t)token*topk+r]*experts[((size_t)token*topk+r)*hidden+col];
    output[i]=v;
}

__global__ void dual_silu_matvec_kernel(float *output,const float *input,
                                        const uint16_t *gate,const uint16_t *up,
                                        int rows,int cols){
    int wave=threadIdx.x>>6,lane=threadIdx.x&63,row=blockIdx.x;float g=0.0f,u=0.0f;
    const uint16_t *wg=gate+(size_t)row*cols,*wu=up+(size_t)row*cols;
    for(int i=threadIdx.x*2;i<cols;i+=BLOCK_SIZE*2){float x0=input[i],x1=input[i+1];
        uint32_t pg=*reinterpret_cast<const uint32_t *>(wg+i),pu=*reinterpret_cast<const uint32_t *>(wu+i);
        g+=x0*__uint_as_float((pg&0xffffu)<<16)+x1*__uint_as_float(pg&0xffff0000u);
        u+=x0*__uint_as_float((pu&0xffffu)<<16)+x1*__uint_as_float(pu&0xffff0000u);}
    g=wave_sum(g);u=wave_sum(u);__shared__ float gs[BLOCK_SIZE/64],us[BLOCK_SIZE/64];
    if(lane==0){gs[wave]=g;us[wave]=u;}__syncthreads();if(wave==0){
        g=lane<BLOCK_SIZE/64?gs[lane]:0.0f;u=lane<BLOCK_SIZE/64?us[lane]:0.0f;
        g=wave_sum(g);u=wave_sum(u);if(lane==0)output[row]=(g/(1.0f+expf(-g)))*u;}
}

__global__ void dual_silu_wave_kernel(float *output,const float *input,
    const uint16_t *gate,const uint16_t *up,int rows,int cols){
    int wave=threadIdx.x>>6,lane=threadIdx.x&63,row=blockIdx.x*4+wave;
    if(row>=rows)return;const uint16_t *wg=gate+(size_t)row*cols;
    const uint16_t *wu=up+(size_t)row*cols;float g=0.0f,u=0.0f;
    for(int i=lane*2;i<cols;i+=128){float x0=input[i],x1=input[i+1];
        uint32_t pg=*reinterpret_cast<const uint32_t *>(wg+i);
        uint32_t pu=*reinterpret_cast<const uint32_t *>(wu+i);
        g+=x0*__uint_as_float((pg&0xffffu)<<16)+x1*__uint_as_float(pg&0xffff0000u);
        u+=x0*__uint_as_float((pu&0xffffu)<<16)+x1*__uint_as_float(pu&0xffff0000u);}
    g=wave_sum(g);u=wave_sum(u);if(lane==0)
        output[row]=(g/(1.0f+expf(-g)))*u;
}

__global__ void batched_dual_silu_matvec_kernel(float *output,const float *input,
                                                 const uint16_t *gate,const uint16_t *up,
                                                 int batch,int rows,int cols){
    int token=blockIdx.y,wave=threadIdx.x>>6,lane=threadIdx.x&63,row=blockIdx.x*4+wave;
    if(token>=batch||row>=rows)return;const float *x=input+(size_t)token*cols;
    const uint16_t *wg=gate+(size_t)row*cols,*wu=up+(size_t)row*cols;float g=0,u=0;
    for(int i=lane*2;i<cols;i+=128){float x0=x[i],x1=x[i+1];
        uint32_t pg=*reinterpret_cast<const uint32_t *>(wg+i),pu=*reinterpret_cast<const uint32_t *>(wu+i);
        g+=x0*__uint_as_float((pg&0xffffu)<<16)+x1*__uint_as_float(pg&0xffff0000u);
        u+=x0*__uint_as_float((pu&0xffffu)<<16)+x1*__uint_as_float(pu&0xffff0000u);}
    g=wave_sum(g);u=wave_sum(u);if(lane==0)output[(size_t)token*rows+row]=(g/(1.0f+expf(-g)))*u;
}

__global__ void selected_dual_silu_matvec_kernel(
    float *output,const float *input,const uint16_t *const *gate,
    const uint16_t *const *up,const int *selected,int selections,int rows,int cols){
    int rank=blockIdx.y,wave=threadIdx.x>>6,lane=threadIdx.x&63;
    int row0=blockIdx.x*8+wave*2,row1=row0+1;
    if(rank>=selections||row0>=rows)return;int expert=selected[rank];
    const float *x=input+(size_t)rank*cols;
    const uint16_t *wg0=gate[expert]+(size_t)row0*cols,*wu0=up[expert]+(size_t)row0*cols;
    const uint16_t *wg1=gate[expert]+(size_t)row1*cols,*wu1=up[expert]+(size_t)row1*cols;
    float g0=0,u0=0,g1=0,u1=0;
    for(int i=lane*2;i<cols;i+=128){float x0=x[i],x1=x[i+1];
        uint32_t pg=*reinterpret_cast<const uint32_t *>(wg0+i);
        uint32_t pu=*reinterpret_cast<const uint32_t *>(wu0+i);
        g0+=x0*__uint_as_float((pg&0xffffu)<<16)+x1*__uint_as_float(pg&0xffff0000u);
        u0+=x0*__uint_as_float((pu&0xffffu)<<16)+x1*__uint_as_float(pu&0xffff0000u);
        if(row1<rows){pg=*reinterpret_cast<const uint32_t *>(wg1+i);
            pu=*reinterpret_cast<const uint32_t *>(wu1+i);
            g1+=x0*__uint_as_float((pg&0xffffu)<<16)+x1*__uint_as_float(pg&0xffff0000u);
            u1+=x0*__uint_as_float((pu&0xffffu)<<16)+x1*__uint_as_float(pu&0xffff0000u);}}
    g0=wave_sum(g0);u0=wave_sum(u0);g1=wave_sum(g1);u1=wave_sum(u1);
    if(lane==0){output[(size_t)rank*rows+row0]=(g0/(1.0f+expf(-g0)))*u0;
        if(row1<rows)output[(size_t)rank*rows+row1]=(g1/(1.0f+expf(-g1)))*u1;}
}

__global__ void selected_dual_silu_shared_kernel(
    float *output,const float *input,const uint16_t *const *gate,
    const uint16_t *const *up,const int *selected,int selections,int rows,int cols){
    int rank=blockIdx.y,wave=threadIdx.x>>6,lane=threadIdx.x&63;
    int row=blockIdx.x*(blockDim.x/64)+wave;
    if(rank>=selections||row>=rows)return;int expert=selected[rank];
    const uint16_t *wg=gate[expert]+(size_t)row*cols;
    const uint16_t *wu=up[expert]+(size_t)row*cols;float g=0,u=0;
    for(int i=lane*2;i<cols;i+=128){float x0=input[i],x1=input[i+1];
        uint32_t pg=*reinterpret_cast<const uint32_t *>(wg+i),pu=*reinterpret_cast<const uint32_t *>(wu+i);
        g+=x0*__uint_as_float((pg&0xffffu)<<16)+x1*__uint_as_float(pg&0xffff0000u);
        u+=x0*__uint_as_float((pu&0xffffu)<<16)+x1*__uint_as_float(pu&0xffff0000u);}
    g=wave_sum(g);u=wave_sum(u);if(lane==0)
        output[(size_t)rank*rows+row]=(g/(1.0f+expf(-g)))*u;
}

}  // namespace

extern "C" void dsv2_init_mfma_gemm(size_t capacity) {
    mfma_input_capacity=capacity;
    if(hipMalloc(&mfma_input,capacity*sizeof(uint16_t))!=hipSuccess){
        mfma_input=nullptr;mfma_input_capacity=0;}
}

extern "C" void dsv2_finish_mfma_gemm() {
    if(mfma_input)hipFree(mfma_input);mfma_input=nullptr;mfma_input_capacity=0;
}

extern "C" void dsv2_launch_bf16_matvec(float *output, const float *input,
                                         const uint16_t *weight, int rows,
                                         int cols, hipStream_t stream) {
    if(rows>=1024)bf16_matvec_wave_kernel<<<(rows+3)/4,BLOCK_SIZE,0,stream>>>(
        output,input,weight,rows,cols);
    else bf16_matvec_kernel<<<rows,BLOCK_SIZE,0,stream>>>(
        output,input,weight,rows,cols);
}

extern "C" void dsv2_launch_rmsnorm(float *output, const float *input,
                                     const uint16_t *weight, int count,
                                     float eps, hipStream_t stream) {
    rmsnorm_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        output, input, weight, count, eps);
}

extern "C" void dsv2_launch_residual_add(float *state, const float *branch,
                                          int count, hipStream_t stream) {
    int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    residual_add_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(state, branch, count);
}
extern "C" void dsv2_launch_residual_rmsnorm(float *out,float *state,
    const float *branch,const uint16_t *weight,int count,float eps,hipStream_t stream){
    residual_rmsnorm_kernel<<<1,RESNORM_BLOCK_SIZE,0,stream>>>(out,state,branch,weight,count,eps);
}

extern "C" void dsv2_launch_embedding(float *output, const uint16_t *table,
                                       int token, int hidden, hipStream_t stream) {
    embedding_kernel<<<(hidden + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        output, table, token, hidden);
}

extern "C" void dsv2_launch_copy(float *output, const float *input, int count,
                                  hipStream_t stream) {
    copy_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(output, input, count);
}

extern "C" void dsv2_launch_silu_multiply(float *gate, const float *up,
                                           int count, hipStream_t stream) {
    silu_multiply_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(gate, up, count);
}

extern "C" void dsv2_launch_rope(float *vectors, int vector_stride,
                                  int rope_offset, int vectors_count,
                                  int rope_dim, int position,
                                  const float *inverse_frequency,
                                  hipStream_t stream) {
    int pairs = vectors_count * rope_dim / 2;
    rope_kernel<<<(pairs + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        vectors, vector_stride, rope_offset, vectors_count, rope_dim, position,
        inverse_frequency);
}

extern "C" void dsv2_launch_kv_cache_store(float *cache,const float *compressed,
    const uint16_t *norm_weight,int latent_dim,int rope_dim,int position,float eps,
    const float *inverse_frequency,hipStream_t stream){
    kv_cache_store_kernel<<<1,KV_CACHE_BLOCK_SIZE,0,stream>>>(cache,compressed,norm_weight,
        latent_dim,rope_dim,position,eps,inverse_frequency);
}

extern "C" void dsv2_launch_absorb_query(float *output, float *query,
                                          const uint16_t *kv_up, int heads,
                                          int q_head_dim, int nope_dim,
                                          int value_dim, int latent_dim,int position,
                                          const float *inverse_frequency,
                                          hipStream_t stream) {
    absorb_query_kernel<<<dim3((latent_dim+7)/8, heads), BLOCK_SIZE, 0, stream>>>(
        output,query,kv_up,heads,q_head_dim,nope_dim,value_dim,
        latent_dim,position,inverse_frequency);
}

extern "C" void dsv2_launch_attention(float *scores, float *context,
                                       const float *q_absorbed,
                                       const float *query, const float *cache,
                                       int length, int cache_stride, int heads,
                                       int q_head_dim, int nope_dim,
                                       int latent_dim, int rope_dim, float scale,
                                       hipStream_t stream) {
    attention_score_kernel<<<dim3(length, heads), BLOCK_SIZE, 0, stream>>>(
        scores, q_absorbed, query, cache, length, cache_stride, heads,
        q_head_dim, nope_dim, latent_dim, rope_dim, scale);
    attention_softmax_kernel<<<heads, SOFTMAX_BLOCK_SIZE, 0, stream>>>(scores, length);
    latent_context_kernel<<<dim3((latent_dim+3)/4, heads), BLOCK_SIZE, 0, stream>>>(
        context, scores, cache, length, cache_stride, latent_dim);
}

extern "C" void dsv2_launch_latent_value(float *output, const float *latent,
                                          const uint16_t *kv_up, int heads,
                                          int nope_dim, int value_dim,
                                          int latent_dim,
                                          hipStream_t stream) {
    latent_value_kernel<<<dim3(value_dim, heads), BLOCK_SIZE, 0, stream>>>(
        output, latent, kv_up, heads, nope_dim, value_dim, latent_dim);
}

extern "C" void dsv2_launch_router_topk(const float *logits, int experts,
                                         int topk, int *indices, float *weights,
                                         hipStream_t stream) {
    router_topk_kernel<<<1, 64, 0, stream>>>(logits, experts, topk, indices, weights);
}

extern "C" void dsv2_launch_broadcast(float *output, const float *input,
                                       int copies, int count,
                                       hipStream_t stream) {
    int total = copies * count;
    broadcast_kernel<<<(total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        output, input, copies, count);
}

extern "C" void dsv2_launch_selected_expert_matvec(
    float *output, const float *input, const uint16_t *const *weights,
    const int *selected, int selections, int rows, int cols,
    hipStream_t stream) {
    constexpr int rows_per_block=EXPERT_DOWN_BLOCK_SIZE/64*2;
    selected_expert_wave_kernel<<<dim3((rows+rows_per_block-1)/rows_per_block,selections),EXPERT_DOWN_BLOCK_SIZE,0,stream>>>(
        output, input, weights, selected, selections, rows, cols);
}

extern "C" void dsv2_launch_weighted_expert_add(
    float *output, const float *expert_output, const float *weights,
    int selections, int hidden, hipStream_t stream) {
    weighted_expert_add_kernel<<<(hidden + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                  BLOCK_SIZE, 0, stream>>>(
        output, expert_output, weights, selections, hidden);
}

extern "C" void dsv2_launch_batched_matvec(float *out,const float *in,const uint16_t *w,int batch,int rows,int cols,hipStream_t s){
    if(batch>=16){
        size_t total=(size_t)batch*cols;
        if(total<=mfma_input_capacity){
            convert_bf16_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
                mfma_input,in,total);
        mfma_bf16_gemm_32x64<<<dim3((rows+63)/64,(batch+63)/64),512,0,s>>>(
            out,mfma_input,w,batch,rows,cols);
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
extern "C" void dsv2_launch_batched_router_topk(const float *logits,int batch,int experts,int topk,int *idx,float *weights,hipStream_t s){
    batched_router_topk_kernel<<<batch,64,0,s>>>(logits,batch,experts,topk,idx,weights);}
extern "C" void dsv2_launch_expand_selected(float *out,const float *in,int batch,int topk,int hidden,hipStream_t s){
    int total=batch*topk*hidden;expand_selected_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(out,in,batch,topk,hidden);}
extern "C" void dsv2_launch_batched_weighted_add(float *out,const float *experts,const float *weights,int batch,int topk,int hidden,hipStream_t s){
    int total=batch*hidden;batched_weighted_add_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(out,experts,weights,batch,topk,hidden);}
extern "C" void dsv2_launch_bucket_routes(const int *idx,const float *weights,
    int tokens,int topk,int *counts,int *slots,float *slot_weights,int capacity,
    hipStream_t s){
    hipMemsetAsync(counts,0,64*sizeof(int),s);int total=tokens*topk;
    bucket_routes_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
        idx,weights,tokens,topk,counts,slots,slot_weights,capacity);}
extern "C" void dsv2_launch_pack_expert_bf16(uint16_t *out,const float *in,
    const int *counts,const int *slots,int experts,int capacity,int hidden,
    hipStream_t s){size_t total=(size_t)experts*capacity*(hidden/2);
    pack_expert_bf16_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
        out,in,counts,slots,experts,capacity,hidden);}
extern "C" void dsv2_launch_convert_expert_bf16(uint16_t *out,const float *in,
    const int *counts,int experts,int capacity,int width,hipStream_t s){
    size_t total=(size_t)experts*capacity*width;
    convert_expert_bf16_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
        out,in,counts,experts,capacity,width);}
extern "C" void dsv2_launch_grouped_expert_gemm(float *out,const uint16_t *in,
    const uint16_t *const *weights,const int *counts,const int *slots,int experts,
    int capacity,int rows,int cols,int input_mode,hipStream_t s){
    grouped_expert_mfma_32x64<<<dim3((rows+63)/64,(capacity+63)/64,experts),512,0,s>>>(
        out,in,weights,counts,slots,capacity,rows,cols);}
extern "C" void dsv2_launch_grouped_expert_gate_up(uint16_t *gate,float *up,
    const uint16_t *in,const uint16_t *const *gate_w,const uint16_t *const *up_w,
    const int *counts,int experts,int capacity,int rows,int cols,hipStream_t s){
    grouped_expert_gate_up_32x128<<<dim3((rows+63)/64,(capacity+127)/128,experts),
        512,0,s>>>(gate,up,in,gate_w,up_w,counts,capacity,rows,cols);}
extern "C" void dsv2_launch_bucket_swiglu(float *gate,const float *up,
    int experts,int capacity,int rows,hipStream_t s){
    int total=experts*capacity*rows;
    bucket_swiglu_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(gate,up,total);}
extern "C" void dsv2_launch_reduce_bucket_experts(float *out,const float *expert,
    const int *counts,const int *slots,const float *weights,int experts,int capacity,
    int tokens,int hidden,hipStream_t s){
    size_t total=(size_t)experts*capacity*hidden;
    scatter_bucket_experts_kernel<<<(total+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE,0,s>>>(
        out,expert,counts,slots,weights,experts,capacity,tokens,hidden);}
extern "C" void dsv2_launch_dual_silu(float *out,const float *in,const uint16_t *gate,const uint16_t *up,int rows,int cols,hipStream_t s){
    if(rows<=4096)dual_silu_wave_kernel<<<(rows+3)/4,BLOCK_SIZE,0,s>>>(
        out,in,gate,up,rows,cols);
    else dual_silu_matvec_kernel<<<rows,BLOCK_SIZE,0,s>>>(out,in,gate,up,rows,cols);}
extern "C" void dsv2_launch_batched_dual_silu(float *out,const float *in,const uint16_t *gate,const uint16_t *up,int batch,int rows,int cols,hipStream_t s){
    batched_dual_silu_matvec_kernel<<<dim3((rows+3)/4,batch),BLOCK_SIZE,0,s>>>(out,in,gate,up,batch,rows,cols);}
extern "C" void dsv2_launch_selected_dual_silu(float *out,const float *in,const uint16_t *const *gate,const uint16_t *const *up,const int *selected,int selections,int rows,int cols,hipStream_t s){
    selected_dual_silu_matvec_kernel<<<dim3((rows+7)/8,selections),BLOCK_SIZE,0,s>>>(out,in,gate,up,selected,selections,rows,cols);}
extern "C" void dsv2_launch_selected_dual_silu_shared(float *out,const float *in,const uint16_t *const *gate,const uint16_t *const *up,const int *selected,int selections,int rows,int cols,hipStream_t s){
    constexpr int rows_per_block=SELECTED_BLOCK_SIZE/64;
    selected_dual_silu_shared_kernel<<<dim3((rows+rows_per_block-1)/rows_per_block,selections),SELECTED_BLOCK_SIZE,0,s>>>(out,in,gate,up,selected,selections,rows,cols);}
