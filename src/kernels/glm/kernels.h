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

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

/* ---- elementwise.hip ---- */
__global__ void embed_lookup_kernel(float *xs, const bf16_t *embed_tokens,
                                    const int *tokens, int H);
__global__ void rmsnorm_kernel(float *y, const float *x, const bf16_t *w,
                               int n, float eps, int x_row_stride,
                               int y_row_stride);
__global__ void residual_add_kernel(float *xs, const float *add, int n);
__global__ void silu_mul_kernel(float *hb, const float *hb2, int n);
__global__ void weighted_add_kernel(float *out, const float *add, float weight,
                                    int n);
__global__ void copy_strided_kernel(float *dst, const float *src, int n,
                                    int dst_row_stride, int src_row_stride);
__global__ void argmax_kernel(const float *logits, int n, int *out);

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

/* ---- batched.hip (run2 grouped/batched decode) ---- */
#define MM_TILE 64   /* TBM==TBN tile of the active 64x64 MFMA GEMMs */
__global__ void matmul_mfma_kernel(float *Y, const float *X, const bf16_t *W,
                                   int M, int N, int K, int x_stride,
                                   int y_stride);
/* run21: 128x128-tile bf16 GEMM. Bit-identical to matmul_mfma_kernel (same fp32
 * K-accumulation) but halves the bf16 weight bandwidth for the large GEMMs
 * (lm_head, shared-expert, dense-layer). Use only when the coarser grid fills CUs. */
__global__ void matmul_mfma_128_kernel(float *Y, const float *X, const bf16_t *W,
                                       int M, int N, int K, int x_stride,
                                       int y_stride);
/* run20 W8A8 dense GEMM: int8 activation (per-row xsc) x int8 weight (per-row wsc)
 * via mfma_i32_16x16x16i8; halves the L2 weight bytes of the bf16 matmul_mfma. */
__global__ void matmul_mfma_i8i8_kernel(float *Y, const int8_t *Xq, const float *xsc,
                                        const int8_t *W, const float *wsc, int M, int N,
                                        int K, int x_stride, int y_stride);
/* run21: 128x128-tile variant of the W8A8 dense GEMM. Halves TOTAL L2 traffic
 * (both weight and activation re-reads) vs the 64x64 kernel; use only when M,N
 * are large enough to keep the coarser 128x128 grid filling the CUs. */
__global__ void matmul_mfma_i8i8_128_kernel(float *Y, const int8_t *Xq, const float *xsc,
                                            const int8_t *W, const float *wsc, int M, int N,
                                            int K, int x_stride, int y_stride);
/* Per-head [M=B,N=d_out]=X@W^T GEMM on the matrix cores (head = grid.z). Weight
 * tile reused across the M batch; grid = (ceil(N/64), ceil(M/64), NH),
 * block = 256. K (=n_in) a multiple of 16. */
__global__ void matmul_bh_mfma_kernel(float *Y, const float *X, const bf16_t *W,
                                      int M, int N, int K, int x_hstride,
                                      int x_bstride, int w_stride, int y_hstride,
                                      int y_bstride);
/* MFMA per-head GEMM, SHARED X + per-head W/Y (head=grid.z): Y[h,m,n] =
 * sum_k X[m*x_stride+k]*W[h*w_hstride + n*K + k]. Folds the unabsorbed-prefill
 * knope/value per-head loops (2*NH launches -> 2). grid=(ceil(N/64),ceil(M/64),NH). */
__global__ void matmul_headw_mfma_kernel(float *Y, const float *X, const bf16_t *W,
                                         int M, int N, int K, int x_stride,
                                         int w_hstride, int y_hstride);
__global__ void rope_pos_kernel(float *v, int n_heads, int row_stride,
                                int head_stride, const int *pos, int rope_dim,
                                int interleaved, const float *inv_freq);
__global__ void argmax_batched_kernel(const float *logits, int n, int y_stride,
                                      int *out);
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
                                        const float *inv_freq, int interleaved);
/* Chunked-prefill: scatter contiguous kvlin[n_tok][KVD] rows into per-slot cache
 * at (slot[t], pos[t]) as int8 + per-row scale; gather last-token rows for lm_head. */
__global__ void kv_scatter_kernel(int8_t *cache_base, float *kv_scale,
                                  int slot_stride, const int *slot, const int *pos,
                                  const float *kvlin, int KVD, int n_tok);
/* Per-row symmetric int8 quant of a contiguous fp32 latent-KV block: n_rows rows
 * of KVD, one block per row -> dst[r*KVD] int8 + scale[r]. Used to write the
 * per-request unabsorbed-prefill scratch into the int8 cache. */
__global__ void quant_kv_rows_kernel(int8_t *dst, float *scale, const float *src,
                                     int KVD);
__global__ void gather_rows_kernel(float *dst, const float *src, const int *idx,
                                   int H);
__global__ void moe_route_batched_kernel(const float *logits, int n_routed,
                                          int K, int router_sigmoid,
                                          int norm_topk, float routed_scaling,
                                          const float *bias, int *topk_idx,
                                          float *topk_wt);
__global__ void moe_hist_kernel(const int *topk_idx, int T, int *ecount);
__global__ void moe_scan_kernel(const int *ecount, int n_routed, int *eoff);
__global__ void moe_place_kernel(const int *topk_idx, const float *topk_wt,
                                 int T, int K, const int *eoff, int *cursor,
                                 int *sorted_slot, float *sorted_wt);
/* run4: int8 weight-only quantized experts. Wtbl[e] is int8 [N][K], Stbl[e] the
 * per-output-row fp32 scale [N]. Weight int8 values are exact in bf16 (|q|<=127),
 * MFMA runs bf16, the scale is applied once in fp32 at the output write. Halves
 * the expert-weight HBM traffic (the moe_grouped GEMM's bandwidth bottleneck). */
__global__ void moe_grouped_mfma_i8_kernel(float *out_sorted, const float *X,
                                           const int8_t *const *Wtbl,
                                           const float *const *Stbl,
                                           const int *eoff, const int *sorted_slot,
                                           int N, int K, int x_stride, int gather);
/* run19 W8A8: int8 activation + int8 weight MoE GEMM via mfma_i32_16x16x16i8 (no
 * dequant VALU). quant_act_rows_i8 makes the int8 activations + per-row scale. */
__global__ void quant_act_rows_i8_kernel(const float *X, int8_t *Q, float *scale,
                                         int rows, int K, int x_stride);
__global__ void moe_grouped_mfma_i8i8_kernel(float *out_sorted, const int8_t *Xq,
                                             const float *xsc, const int8_t *const *Wtbl,
                                             const float *const *Stbl, const int *eoff,
                                             const int *sorted_slot, int N, int K,
                                             int x_stride, int gather);
/* run24: 128x128-tile W8A8 grouped MoE GEMM. Halves the L2 activation re-reads of
 * moe_grouped_mfma_i8i8_kernel (L2-BW-bound at B=1024, 133% peak) -- the run21
 * tile-128 lever extended to the MoE GEMM (DeepGEMM large-tile). Bit-identical
 * output (same fp32 K-accumulation via i32 MFMA). Toggle via MLA_MOE_T128=0. */
__global__ void moe_grouped_mfma_i8i8_128_kernel(float *out_sorted, const int8_t *Xq,
                                                 const float *xsc, const int8_t *const *Wtbl,
                                                 const float *const *Stbl, const int *eoff,
                                                 const int *sorted_slot, int N, int K,
                                                 int x_stride, int gather);
/* run33: block-wise W4A8 dense GEMM (packed int4 weight [N][K/2], scale [N][K/128]).
 * Two-level accumulation (int32 mfma per group -> float flush). MLA_DENSE_W4A8_BW=1. */
__global__ void matmul_mfma_i8i8_bw_128_kernel(float *Y, const int8_t *Xq, const float *xsc,
                                               const int8_t *W, const float *wsc, int M, int N,
                                               int K, int x_stride, int y_stride);
/* run31: TBK=64 / 128-bit-load variant of the 128x128 W8A8 MoE GEMM. Bit-identical
 * output; 4x fewer syncs/load instructions. Test of load-width on the L2-BW-bound
 * MoE GEMM. K multiple of 64. Toggle MLA_MOE_TBK64=1. */
__global__ void moe_grouped_mfma_i8i8_t64_kernel(float *out_sorted, const int8_t *Xq,
                                                 const float *xsc, const int8_t *const *Wtbl,
                                                 const float *const *Stbl, const int *eoff,
                                                 const int *sorted_slot, int N, int K,
                                                 int x_stride, int gather);
/* Quantize a bf16 weight matrix [N][K] to int8 [N][K] + per-row fp32 scale[N]
 * (symmetric, scale = max_k|w|/127). grid = (N), block = 256. */
__global__ void quantize_rows_i8_kernel(const bf16_t *W, int8_t *Q, float *S,
                                        int N, int K);
/* run29: quantize bf16 [N][K] to PACKED symmetric int4 [N][K/2] + per-row scale[N]
 * (scale = max|w|/7). Halves the weight bytes for the W4A8 MoE GEMM. K even. */
__global__ void quantize_rows_i4_kernel(const bf16_t *W, int8_t *Qp, float *S,
                                        int N, int K);
/* run29: W4A8 grouped MoE GEMM -- int4 packed weight x int8 activation (unpack
 * int4->int8 in the B stage, clean int32 MFMA). Same interface/epilogue as
 * moe_grouped_mfma_i8i8_128_kernel; Wtbl[e] is packed int4 [N][K/2], K logical. */
__global__ void moe_grouped_mfma_w4a8_128_kernel(float *out_sorted, const int8_t *Xq,
                                                 const float *xsc, const int8_t *const *Wtbl,
                                                 const float *const *Stbl, const int *eoff,
                                                 const int *sorted_slot, int N, int K,
                                                 int x_stride, int gather);
/* run32: BLOCK-WISE int4 quant -- symmetric packed int4 [N][K/2] but scale is
 * per (row, K-group of G): S is [N][K/G]. Fine-grained scale (DeepSeek/AWQ style)
 * recovers the accuracy per-row int4 lost (run29). K multiple of G. grid=(N). */
__global__ void quantize_rows_i4_bw_kernel(const bf16_t *W, int8_t *Qp, float *S,
                                           int N, int K, int G);
/* run32: block-wise W4A8 grouped MoE GEMM. Same tile/unpack as the per-row w4a8
 * kernel but the weight scale varies per K-group -> two-level accumulation: int32
 * MFMA within a group, flushed into a float accumulator scaled by S[col][group],
 * reset, next group. S is [N][K/G]. Toggle MLA_MOE_W4A8_BW=1. */
__global__ void moe_grouped_mfma_w4a8_bw_128_kernel(float *out_sorted, const int8_t *Xq,
                                                    const float *xsc, const int8_t *const *Wtbl,
                                                    const float *const *Stbl, const int *eoff,
                                                    const int *sorted_slot, int N, int K,
                                                    int x_stride, int gather);
__global__ void moe_scatter_add_kernel(float *out, const float *down_sorted,
                                       const int *sorted_slot,
                                       const float *sorted_wt, int T, int H);

/* ---- rope.hip ----
 * Applies RoPE in place to a rope_dim slice with per-row (and optional
 * per-head) stride. n_rows x n_heads grid; pos for row r is pos_base + r. */
__global__ void rope_kernel(float *v, int n_heads, int row_stride,
                            int head_stride, int pos_base, int rope_dim,
                            int interleaved, const float *inv_freq);

/* ---- attention.hip ---- */
/* Fused flash-attention prefill: score+softmax+ctx in one pass (no O(n^2) HBM
 * scratch). ctx[q,h,d] = sum_{k<=q} softmax_k(score)*value[h,k,d]. */
__global__ void attn_unabsorbed_flash_kernel(float *ctx, const float *qall,
                                             const float *knope, const float *kv_l,
                                             const float *value, int n_prompt,
                                             int NH, int QHD, int QKN, int QKR,
                                             int KVD, int VHD, float scale);
/* Varlen (chunked/batched prefill) flash: several prompts packed on one flat
 * token axis; q_seqstart[qg] bounds each query's causal key range to its prompt.
 * knope/value are [NH, n_tok, ...]; kv_l is [n_tok, KVD]. */
__global__ void attn_unabsorbed_flash_varlen_kernel(float *ctx, const float *qall,
                                                    const float *knope, const float *kv_l,
                                                    const float *value,
                                                    const int *q_seqstart, int n_tok,
                                                    int NH, int QHD, int QKN, int QKR,
                                                    int KVD, int VHD, float scale);

/* MFMA flash-attention prefill (run18): 64-query tiles on the matrix cores, one
 * prompt per tile. qt_q0/qt_qs/qt_qe = per-tile first-query / prompt-start /
 * prompt-end (global). grid = (num_qtiles, NH), block = 256. Replaces the scalar
 * attn_unabsorbed_flash_varlen_kernel. */
__global__ void attn_prefill_mfma_kernel(float *ctx, const float *qall,
                                         const float *knope, const float *kv_l,
                                         const float *value, const int *qt_q0,
                                         const int *qt_qs, const int *qt_qe,
                                         int n_tok, int NH, int QHD, int QKN, int QKR,
                                         int KVD, int KVL, int VHD, float scale);

/* Fused flash-decode (batched absorbed): score+softmax+clat for (slot,head) in one
 * online-softmax pass, no b_score materialization. grid=(NH,B), block=256, dynamic
 * shmem = KVD floats. Absorbed decode score[h,k]=qabs[h]·c_kv[k]+qpe[h]·k_pe[k]. */
__global__ void attn_flash_decode_kernel(float *clat, const float *qabs, const float *q,
                                         const int8_t *kv_base, const float *kv_scale,
                                         const int *pos, int NH,
                                         int QHD, int QKN, int QKR, int KVL, int KVD,
                                         int slot_stride, float scale);
/* run23: head-batched flash-decode (grid=(B,HG)). Stages c_kv once per slot per
 * head group into LDS -> ~HG reads/slot instead of ~2*NH, relieving the L2-read
 * bound. See attn_flash_decode_hb_kernel in attention.hip. */
__global__ void attn_flash_decode_hb_kernel(float *clat, const float *qabs, const float *q,
                                            const int8_t *kv_base, const float *kv_scale,
                                            const int *pos, int NH,
                                            int QHD, int QKN, int QKR, int KVL, int KVD,
                                            int slot_stride, float scale);
/* run27: MFMA-score variant of the head-batched flash-decode. Offloads the scalar
 * per-(head,key) KVD dot (VALU-bound at B=1024, report/run26) to the matrix cores
 * (mfma_f32_16x16x16bf16). Same signature; softmax+clat stay scalar. MLA_FDMFMA=0
 * reverts to attn_flash_decode_hb_kernel. */
__global__ void attn_flash_decode_hb_mfma_kernel(float *clat, const float *qabs, const float *q,
                                                 const int8_t *kv_base, const float *kv_scale,
                                                 const int *pos, int NH,
                                                 int QHD, int QKN, int QKR, int KVL, int KVD,
                                                 int slot_stride, float scale);

#endif /* MLA_KERNELS_H */
