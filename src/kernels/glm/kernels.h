/* Shared declarations for the GPU kernels used by src/getp_run.hip. Not part of
 * the build itself (no .hip extension) -- included from every kernel .hip file
 * and from getp_run.hip so kernel launches type-check against their definitions
 * in the other translation units. All device compute is fp32; weights are bf16
 * (converted on the fly via bf16f()), matching the CPU reference in src/run.c. */
#ifndef MLA_KERNELS_H
#define MLA_KERNELS_H

#include <hip/hip_runtime.h>
#include "tensor.h" /* bf16_t */

/* Device-side bf16->f32 (tensor.h's bf16_to_f32 is host-only). */
__device__ __forceinline__ float bf16f(bf16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

/* Device-side f32->bf16 bit pattern (truncate, matches tensor.h f32_to_bf16 /
 * PyTorch default). bf16_t is a uint16_t bit container, so a plain (bf16_t)cast
 * would integer-convert the value -- always narrow through this instead. */
__device__ __forceinline__ bf16_t f32bf(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return (bf16_t)(u >> 16);
}

/* run43: __expf (hardware exp) instead of the libm call -- silu runs on every
 * MoE/FFN hidden element (63M per prefill layer). Same fast-exp already used
 * by every softmax in this engine. */
/* run47: wave-level reductions. The LDS tree reductions in this engine cost
 * 8 __syncthreads plus 8 LDS round-trips per row, which dominates the
 * per-row kernels (quant_act was 7x above its memory floor, rmsnorm 5x).
 * A wave64 can reduce entirely in registers with 6 cross-lane shuffles, so
 * a 256-thread block needs just 2 barriers total. */
__device__ __forceinline__ float wave_reduce_sum(float v) {
#pragma unroll
    for (int m = 32; m > 0; m >>= 1) v += __shfl_xor(v, m, 64);
    return v;
}
__device__ __forceinline__ float wave_reduce_max(float v) {
#pragma unroll
    for (int m = 32; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, 64));
    return v;
}
/* Block reduction over <=4 waves; `sh` needs 4 floats. Result broadcast. */
__device__ __forceinline__ float block_reduce_sum_w(float v, float *sh) {
    int lane = threadIdx.x & 63, w = threadIdx.x >> 6, nw = blockDim.x >> 6;
    v = wave_reduce_sum(v);
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = 0.0f;
    for (int i = 0; i < nw; i++) r += sh[i];
    return r;
}
__device__ __forceinline__ float block_reduce_max_w(float v, float *sh) {
    int lane = threadIdx.x & 63, w = threadIdx.x >> 6, nw = blockDim.x >> 6;
    v = wave_reduce_max(v);
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = sh[0];
    for (int i = 1; i < nw; i++) r = fmaxf(r, sh[i]);
    return r;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }

/* ---- elementwise.hip ---- */
__global__ void embed_lookup_kernel(float *xs, const bf16_t *embed_tokens,
                                    const int *tokens, int H);
__global__ void rmsnorm_f4_kernel(float *y, const float *x, const bf16_t *w,
                                  int n, float eps, int x_row_stride,
                                  int y_row_stride);
__global__ void residual_add_f4_kernel(float *xs, const float *add, int n4);
__global__ void rmsnorm_kernel(float *y, const float *x, const bf16_t *w,
                               int n, float eps, int x_row_stride,
                               int y_row_stride);
__global__ void residual_add_kernel(float *xs, const float *add, int n);
/* Fused residual-add + RMSNorm: four buffer passes instead of five (borrowed from
 * the DeepSeek engine). Writes the updated residual AND its normalised copy. */
__global__ void residual_rmsnorm_kernel(float *x, float *y, const float *branch,
                                        const bf16_t *w, int n, float eps,
                                        int x_stride, int y_stride);
__global__ void silu_mul_kernel(float *hb, const float *hb2, int n);
__global__ void copy_strided_kernel(float *dst, const float *src, int n,
                                    int dst_row_stride, int src_row_stride);

/* ---- linear.hip ----
 * matmul_kernel: Y[row, d] = sum_i X[row*x_row_stride + i] * bf16f(W[d*n_in+i])
 *   for d in [0, d_out), row in [0, n_rows). W is row-major [d_out, n_in]
 *   (the natural nn.Linear layout: y = x @ W.T).
 * matmul_wT_kernel: y[r] = sum_d x[d] * bf16f(W[d*d_out + r]) for r in [0,d_out)
 *   -- W row-major [d_in, d_out] (used to fold W_UK into the absorbed query). */
__global__ void matmul_kernel(float *Y, const float *X, const bf16_t *W,
                              int d_out, int n_in, int x_row_stride,
                              int y_row_stride);
/* One-time W_UK transpose (grid = (KVL, NH)) into a coalesced [KVL,QKN] layout. */
__global__ void transpose_wuk_kernel(bf16_t *out, const bf16_t *in, int QKN,
                                     int KVL, int kv_stride);

/* ---- GEMM tile geometry (gemm.hip + moe.hip) -----------------------------
 * Every MFMA GEMM here tiles the output as TBM x TBN with TBK contracted per LDS
 * stage. TBM2/TBN2 is the coarse tile used for the large bandwidth-bound GEMMs:
 * with a square tile the weight and activation L2 traffic are equal and both
 * scale as 1/tile, so halving total traffic means doubling BOTH dims. */
#define MM_TILE 64   /* == TBM == TBN, the tile of the 64x64 MFMA GEMMs */
#define TBM 64
#define TBN 64
#define TBK 16       /* K contracted per LDS stage (one 16x16x16 MFMA)  */
#define TBM2 128
#define TBN2 128

/* One lane's int32 MFMA accumulator (mfma_i32_16x16x16i8 returns 4 int32). */
using i32x4b = __attribute__((__vector_size__(16))) int;
__global__ void narrow_bf16_kernel(bf16_t *dst, const float *src, int n);
/* Reduces the fused-argmax lm_head's per-tile (max,index) pairs to one token per
 * row, and takes over the decode bookkeeping argmax_batched_kernel used to do. */
__global__ void argmax_reduce_kernel(const float *pval, const int *pidx, int tiles,
                                     int *out, const int *slotmap, int *tokrec,
                                     int rec_stride, int step, const int *stepp,
                                     int *pos, int cap);

/* ---- gemm.hip ---- */
__global__ void matmul_mfma_kernel(float *Y, const float *X, const bf16_t *W,
                                   int M, int N, int K, int x_stride,
                                   int y_stride);
/* run21: 128x128-tile bf16 GEMM. Bit-identical to matmul_mfma_kernel (same fp32
 * K-accumulation) but halves the bf16 weight bandwidth for the large GEMMs
 * (lm_head, shared-expert, dense-layer). Use only when the coarser grid fills CUs. */
__global__ void matmul_mfma_128_kernel(float *Y, const float *X, const bf16_t *W,
                                       int M, int N, int K, int x_stride,
                                       int y_stride);
/* lm_head with the argmax fused into the epilogue: emits one (max,index) pair per
 * (row, 128-column tile) instead of the full logits, which removes a 714 MB buffer
 * and 1.4 GB of traffic per decode step. Borrowed from the DeepSeek engine. */
__global__ void matmul_mfma_128_xbf_argmax_kernel(float *pval, int *pidx,
                                                 const bf16_t *X, const bf16_t *W,
                                                 int M, int N, int K, int x_stride,
                                                 int tiles);
/* bf16-activation variant of matmul_mfma_128, for lm_head: its 1210 column tiles
 * re-read the activation 1210 times, and the kernel narrows X to bf16 anyway. */
/* run20 W8A8 dense GEMM: int8 activation (per-row xsc) x int8 weight (per-row wsc)
 * via mfma_i32_16x16x16i8; halves the L2 weight bytes of the bf16 matmul_mfma. */
template <typename OT>
__global__ void matmul_mfma_i8i8_kernel(OT *Y, const int8_t *Xq, const float *xsc,
                                        const int8_t *W, const float *wsc, int M, int N,
                                        int K, int x_stride, int y_stride);
/* Per-head [M=B,N=d_out]=X@W^T GEMM on the matrix cores (head = grid.z). Weight
 * tile reused across the M batch; grid = (ceil(N/64), ceil(M/64), NH),
 * block = 256. K (=n_in) a multiple of 16. */
__global__ void matmul_bh_mfma_kernel(float *Y, const float *X, const bf16_t *W,
                                      int M, int N, int K, int x_hstride,
                                      int x_bstride, int w_stride, int y_hstride,
                                      int y_bstride);
/* ---- kvcache.hip ---- */
__global__ void rope_pos_bf_kernel(bf16_t *v, int n_heads, int row_stride,
                                   int head_stride, const int *pos, int rope_dim,
                                   int interleaved, const float *inv_freq);
__global__ void quant_act_rows_i8_bf_kernel(const bf16_t *X, int8_t *Q, float *scale,
                                            int rows, int K, int x_stride);
__global__ void quant_act_rows_i8_bf_big_kernel(const bf16_t *X, int8_t *Q, float *scale,
                                                int rows, int K, int x_stride);
__global__ void silu_mul_quant_bf_big_kernel(const bf16_t *hb, const bf16_t *hb2,
                                            int8_t *Q, float *scale, int rows, int K);
__global__ void rope_pos_kernel(float *v, int n_heads, int row_stride,
                                int head_stride, const int *pos, int rope_dim,
                                int interleaved, const float *inv_freq);
__global__ void argmax_batched_kernel(const float *logits, int n, int y_stride,
                                      int *out, const int *slotmap, int *tokrec,
                                      int rec_stride, int step, const int *stepp,
                                      int *pos, int cap);
__global__ void bump_step_kernel(int *step);
__global__ void reset_step_kernel(int *step, int v);
/* KV cache is stored int8 (run16: quarters the ~24GB@B=384 fp32 cache; run15 was
 * bf16). Each cache row [KVD] is symmetric-quantized with one fp32 scale per
 * (slot,pos) in kv_scale [n_layers, B, g_cap] (per-layer base passed in;
 * g_cap = slot_stride/KVD). Writers store int8 = round(v/scale) + the row scale;
 * the decode reader multiplies int8 by that scale. (A 2-scale c_kv/k_pe variant
 * restored accuracy but its split-loop reader ran slower than bf16 -- reverted;
 * see report/run16_kvcache_int8/.) */
__global__ void kv_write_batched_kernel(int8_t *cache_base, float *kv_scale,
                                        int slot_stride, const int *pos,
                                        const float *comp, const bf16_t *kv_a_ln,
                                        int KVL, int QKR, int KVD, float eps,
                                        const float *inv_freq, int interleaved,
                                        const int *kvslot);
/* Chunked-prefill: scatter contiguous kvlin[n_tok][KVD] rows into per-slot cache
 * at (slot[t], pos[t]) as int8 + per-row scale; gather last-token rows for lm_head. */
__global__ void slot_dup_state_kernel(int *tokens, int *tokrec, int rec_stride,
                                      const int *src, const int *dst, int nfol);
__global__ void kv_scatter_kernel(int8_t *cache_base, float *kv_scale,
                                  int slot_stride, const int *slot, const int *pos,
                                  const float *kvlin, int KVD, int n_tok);
__global__ void gather_rows_kernel(float *dst, const float *src, const int *idx,
                                   int H);
/* Fused SwiGLU + per-row int8 quantisation: silu(hb)*hb2 -> int8, without the
 * fp32 round trip through HBM the two separate kernels needed. Same shape limits
 * as quant_act_rows_i8_fast_kernel (K % 4 == 0, K <= 2048). */
__global__ void silu_mul_quant_bf_kernel(const bf16_t *hb, const bf16_t *hb2,
                                      int8_t *Q, float *scale, int rows, int K);
__global__ void silu_mul_quant_kernel(const float *hb, const float *hb2,
                                      int8_t *Q, float *scale, int rows, int K);

/* ---- moe.hip ---- */
__global__ void moe_route_batched_kernel(const float *logits, int n_routed,
                                          int K, int router_sigmoid,
                                          int norm_topk, float routed_scaling,
                                          const float *bias, int *topk_idx,
                                          float *topk_wt);
__global__ void moe_hist_kernel(const int *topk_idx, int T, int *ecount);
__global__ void moe_scan_kernel(const int *ecount, int n_routed, int *eoff);
__global__ void moe_place_kernel(const int *topk_idx, const float *topk_wt,
                                 int T, int K, const int *eoff, int *cursor,
                                 int *sorted_slot, float *sorted_wt, int *inv);
/* run19 W8A8: int8 activation + int8 weight MoE GEMM via mfma_i32_16x16x16i8 (no
 * dequant VALU). quant_act_rows_i8 makes the int8 activations + per-row scale. */
__global__ void quant_act_rows_i8_kernel(const float *X, int8_t *Q, float *scale,
                                         int rows, int K, int x_stride);
/* run41: single-pass float4 activation quantizer (registers between the max
 * and the quantize phase instead of re-reading the row). K % 4 == 0 and
 * K <= 2048; callers fall back to quant_act_rows_i8_kernel otherwise. */
__global__ void quant_act_rows_i8_fast_kernel(const float *X, int8_t *Q, float *scale,
                                              int rows, int K, int x_stride);
/* 128x128-tile, bf16-activation twin of the per-head shared-X GEMM: cuts its
 * activation and weight re-reads from 4.4 to 1.6 GB per layer per prefill chunk. */
__global__ void matmul_headw_mfma_128_xbf_kernel(float *Y, const bf16_t *X, const bf16_t *W,
                                                 int M, int N, int K, int x_stride,
                                                 int w_hstride, int y_hstride);
/* Same, writing bf16: knope/value are only ever read by attn_prefill_mfma,
 * which narrows them to bf16 anyway -- bit-identical, half the bytes. */
__global__ void matmul_headw_mfma_128_xbf_bfout_kernel(bf16_t *Y, const bf16_t *X,
                                                 const bf16_t *W, int M, int N, int K,
                                                 int x_stride, int w_hstride, int y_hstride);
/* run45: prefetched (double-buffered) twin of matmul_headw_mfma_kernel.
 * Bit-identical. */
/* 128x128-tile twin of matmul_bh_mfma_kernel: halves the total L2 traffic of the
 * decode qabs/ctx calls (which were bandwidth-bound at 13.6% of MFMA peak). */
__global__ void matmul_bh_mfma_128_kernel(float *Y, const float *X, const bf16_t *W,
                                          int M, int N, int K, int x_hstride,
                                          int x_bstride, int w_stride, int y_hstride,
                                          int y_bstride);
/* run44: wide-load (int4/lane) + deep-prefetch variants of the two hot
 * 128x128 int8 GEMMs. Bit-identical; need K % 32 == 0 and 16 B-aligned row
 * strides (checked by the callers). */
template <typename OT>
__global__ void matmul_mfma_i8i8_128_w_kernel(OT *Y, const int8_t *Xq, const float *xsc,
                                              const int8_t *W, const float *wsc, int M, int N,
                                              int K, int x_stride, int y_stride);
__global__ void moe_grouped_mfma_i8i8_128_w_kernel(bf16_t *out_sorted, const int8_t *Xq,
                                                   const float *xsc, const int8_t *const *Wtbl,
                                                   const float *const *Stbl, const int *eoff,
                                                   const int *sorted_slot, int N, int K,
                                                   int x_stride, int gather);
/* ---- quant.hip ---- */
/* Quantize a bf16 weight matrix [N][K] to int8 [N][K] + per-row fp32 scale[N]
 * (symmetric, scale = max_k|w|/127). grid = (N), block = 256. */
__global__ void quantize_rows_i8_kernel(const bf16_t *W, int8_t *Q, float *S,
                                        int N, int K);
/* MoE epilogue: gather each token's K routed expert outputs (fixed k order, so
 * deterministic) plus the shared expert, in one pass. Replaces zero + atomic
 * scatter + shared add. */
__global__ void moe_gather_add_kernel(float *out, const bf16_t *down_sorted,
                                      const float *sorted_wt, const int *inv,
                                      const bf16_t *shared, int K, int H);

/* ---- attention.hip ---- */

/* MFMA flash-attention prefill (run18): 64-query tiles on the matrix cores, one
 * prompt per tile. qt_q0/qt_qs/qt_qe = per-tile first-query / prompt-start /
 * prompt-end (global). grid = (num_qtiles, NH), block = 256. Replaces the scalar
 * attn_unabsorbed_flash_varlen_kernel. */
__global__ void attn_prefill_mfma_kernel(bf16_t *ctx, const bf16_t *qall,
                                         const bf16_t *knope, const float *kv_l,
                                         const bf16_t *value, const int *qt_q0,
                                         const int *qt_qs, const int *qt_qe,
                                         int n_tok, int NH, int QHD, int QKN, int QKR,
                                         int KVD, int KVL, int VHD, float scale);

#endif /* MLA_KERNELS_H */
