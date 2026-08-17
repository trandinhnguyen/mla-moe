/* Symmetric int8 quantisation, weights and activations.
 *
 * Weights (quantize_rows_i8) are converted once in warm_up: bf16 -> int8 with one
 * fp32 scale per output row, after which the bf16 copy is freed (~27 GB). Only
 * lm_head and moe_gate stay bf16 -- int8 there perturbs the greedy argmax and the
 * top-K routing respectively, for no measurable throughput.
 *
 * Activations are quantised per row on every use, immediately before the GEMM that
 * consumes them, so the int8 MFMA path never has to dequantise inside its K loop
 * (that VALU work used to be the MoE GEMM's bottleneck -- run19).
 * ========================================================================== */
#include "kernels.h"

/* ======================================================================== *
 *  run41: single-pass, float4 version of quant_act_rows_i8_kernel.
 *  The original reads the row TWICE (once for the abs-max, once to quantize)
 *  and one scalar float at a time. The kernel is purely memory-bound and is
 *  called on every MoE/dense activation (23760 launches, 2.86 s = 4.7% of all
 *  kernel time), so the second pass is ~half its cost. Here each thread keeps
 *  its slice in registers between the two phases and moves 16 B at a time.
 *  Requires K % 4 == 0 and K <= 4*QA_REGS*256 (2048); the caller falls back to
 *  the scalar kernel otherwise (only the dense-FFN down proj, K=10240).
 * ======================================================================== */
#define QA_REGS 2

/* Quantize a bf16 weight matrix [N][K] to int8 [N][K] + per-row fp32 scale[N]
 * (symmetric, scale = max_k|w|/127). grid = (N), block = 256. One block per row. */
__global__ void quantize_rows_i8_kernel(const bf16_t *W, int8_t *Q, float *S,
                                        int N, int K) {
    int n = blockIdx.x;
    if (n >= N) return;
    const bf16_t *wr = W + (size_t)n * K;
    int8_t *qr = Q + (size_t)n * K;
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float red[256];
    float mx = 0.0f;
    for (int k = tid; k < K; k += nt) mx = fmaxf(mx, fabsf(bf16f(wr[k])));
    red[tid] = mx;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float scale = red[0] / 127.0f;
    if (tid == 0) S[n] = scale;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int k = tid; k < K; k += nt) {
        float q = rintf(bf16f(wr[k]) * inv);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        qr[k] = (int8_t)q;
    }
}

/* ======================================================================== *
 *  run19: W8A8 int8 MoE grouped GEMM. The i8-weight kernel above dequants each
 *  weight int8 -> bf16 on the VALU before a bf16 MFMA; rocprof-compute showed the
 *  MoE GEMM is VALU-DEQUANT-bound (VALU:MFMA ~19:1, MFMA util 19.6%), so the
 *  matrix cores idle waiting on that convert. Here BOTH operands are int8 fed
 *  straight into `mfma_i32_16x16x16i8` (int32 accumulate) -- no convert at all,
 *  operands load as a single packed int per lane. Activations are quantized to
 *  int8 per-row (quant_act_rows_i8_kernel) with a per-row fp32 scale; the output
 *  int32 is scaled by x_scale[row] * w_scale[col]. Removes the entire dequant VALU
 *  path (both the weight int8->bf16 AND the activation fp32->bf16 converts).
 * ======================================================================== */

/* Quantize an fp32 activation matrix [rows][K] (row stride x_stride) to int8 +
 * per-row fp32 scale (symmetric, scale = max_k|x|/127). grid=(rows), block=256. */
__global__ void quant_act_rows_i8_kernel(const float *X, int8_t *Q, float *scale,
                                         int rows, int K, int x_stride) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float *xr = X + (size_t)r * x_stride;
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float red[256];
    float mx = 0.0f;
    for (int k = tid; k < K; k += nt) mx = fmaxf(mx, fabsf(xr[k]));
    red[tid] = mx;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    float sc = red[0] / 127.0f;
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    for (int k = tid; k < K; k += nt) {
        float q = rintf(xr[k] * inv);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        qr[k] = (int8_t)q;
    }
}

__global__ void quant_act_rows_i8_fast_kernel(const float *X, int8_t *Q, float *scale,
                                              int rows, int K, int x_stride) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float4 *xr = reinterpret_cast<const float4 *>(X + (size_t)r * x_stride);
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x;
    const int K4 = K >> 2;
    __shared__ float red[4];

    float4 v[QA_REGS];
    float mx = 0.0f;
    int n = 0;
    for (int i = tid; i < K4; i += nt, n++) {
        float4 t = xr[i];
        v[n] = t;
        mx = fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fmaxf(fabsf(t.z), fabsf(t.w)), mx));
    }
    float sc = block_reduce_max_w(mx, red) / 127.0f;   /* run47: 1 barrier, not 8 */
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    n = 0;
    for (int i = tid; i < K4; i += nt, n++) {
        char4 o;
        o.x = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v[n].x * inv)));
        o.y = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v[n].y * inv)));
        o.z = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v[n].z * inv)));
        o.w = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v[n].w * inv)));
        *reinterpret_cast<char4 *>(qr + i * 4) = o;
    }
}

/* ======================================================================== *
 *  Fused SwiGLU + activation quantisation.
 *
 *  The MoE hidden path used to run silu_mul (write the fp32 product back over
 *  hb) and then quant_act (read it again) as two kernels. Both are pure streaming
 *  work, so that round trip cost a full write plus a full read of a T x I fp32
 *  buffer -- 504 MB per layer per prefill chunk at T=40960, I=1536. Here the
 *  product is computed, reduced for the row max and quantised without ever
 *  leaving registers.
 *
 *  Same register budget rule as quant_act_rows_i8_fast_kernel: K % 4 == 0 and
 *  K <= 4*QA_REGS*256 = 2048, which covers the MoE intermediate (1536).
 * ======================================================================== */
__global__ void silu_mul_quant_kernel(const float *hb, const float *hb2,
                                      int8_t *Q, float *scale, int rows, int K) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float4 *ar = reinterpret_cast<const float4 *>(hb  + (size_t)r * K);
    const float4 *br = reinterpret_cast<const float4 *>(hb2 + (size_t)r * K);
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x;
    const int K4 = K >> 2;
    __shared__ float red[4];

    float4 v[QA_REGS];
    float mx = 0.0f;
    int n = 0;
    for (int i = tid; i < K4; i += nt, n++) {
        float4 x = ar[i], y = br[i];
        float4 t;
        t.x = silu_f(x.x) * y.x;
        t.y = silu_f(x.y) * y.y;
        t.z = silu_f(x.z) * y.z;
        t.w = silu_f(x.w) * y.w;
        v[n] = t;
        mx = fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fmaxf(fabsf(t.z), fabsf(t.w)), mx));
    }
    float sc = block_reduce_max_w(mx, red) / 127.0f;
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    n = 0;
    for (int i = tid; i < K4; i += nt, n++) {
        char4 o;
        o.x = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v[n].x * inv)));
        o.y = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v[n].y * inv)));
        o.z = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v[n].z * inv)));
        o.w = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v[n].w * inv)));
        *reinterpret_cast<char4 *>(qr + i * 4) = o;
    }
}
