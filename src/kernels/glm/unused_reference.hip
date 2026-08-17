/* ============================================================================
 * unused_reference.hip -- GLM kernels that are NO LONGER on any active code
 * path (reverted experiments / superseded implementations). Kept purely for
 * reference. This file is intentionally NOT included by glm.hip, so it is
 * never compiled or linked. To resurrect a kernel, move it back into the
 * relevant glm/<category>.hip and re-declare it in kernels.h.
 *
 * Why each was retired (see report/ + memory for the full story):
 *   attn_unabsorbed_score/ctx        -> fused into attn_unabsorbed_flash_*
 *   attn_absorbed_score/clat         -> fused into attn_flash_decode_*
 *   attn_score_batched/clat_batched  -> fused into attn_flash_decode_*
 *   attn_flash_decode_lds            -> superseded by attn_flash_decode_hb (run23)
 *   matmul_tiled                     -> superseded by matmul_mfma (matrix cores)
 *   matmul_batched/bh_batched        -> superseded by matmul_*_mfma variants
 *   matmul_ptr_batched               -> single-token MoE path (ffn_layer) retired
 *   moe_grouped_gemm/mfma            -> superseded by moe_grouped_mfma_i8/i8i8
 *   moe_route_kernel                 -> single-token router; batched path uses
 *                                       moe_route_batched_kernel
 *   weighted_reduce / softmax_row / zero -> unused elementwise helpers
 * ========================================================================== */
#include "kernels.h"


/* ---- from attention.hip ---- */

/* grid = (n_prompt [q], NH [h]); block = 256, grid-stride over k.
 * scr[(h*n_prompt+q)*n_prompt+k], causal: k>q -> -INFINITY. */
__global__ void attn_unabsorbed_score_kernel(float *scr, const float *qall,
                                             const float *knope,
                                             const float *kv_l, int n_prompt,
                                             int NH, int QHD, int QKN, int QKR,
                                             int KVD, float scale) {
    int q = blockIdx.x;
    int h = blockIdx.y;
    const float *qnope = qall + ((size_t)q * NH + h) * QHD;
    const float *qpe    = qnope + QKN;
    float *row = scr + ((size_t)h * n_prompt + q) * n_prompt;

    for (int k = threadIdx.x; k < n_prompt; k += blockDim.x) {
        if (k > q) { row[k] = -INFINITY; continue; }
        const float *knope_hk = knope + ((size_t)h * n_prompt + k) * QKN;
        const float *kpe_k    = kv_l + (size_t)k * KVD + (KVD - QKR);
        float dot = 0.0f;
        for (int d = 0; d < QKN; d++) dot += qnope[d] * knope_hk[d];
        for (int d = 0; d < QKR; d++) dot += qpe[d] * kpe_k[d];
        row[k] = dot * scale;
    }
}

/* grid = (n_prompt [q], NH [h]); block = min(VHD,1024).
 * ctx[(q*NH+h)*VHD+d] = sum_{k<=q} wgt[h,q,k] * value[h,k,d] */
__global__ void attn_unabsorbed_ctx_kernel(float *ctx, const float *wgt,
                                           const float *value, int n_prompt,
                                           int NH, int VHD) {
    int q = blockIdx.x;
    int h = blockIdx.y;
    const float *wrow = wgt + ((size_t)h * n_prompt + q) * n_prompt;
    float *out = ctx + ((size_t)q * NH + h) * VHD;

    for (int d = threadIdx.x; d < VHD; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k <= q; k++) {
            const float *val_hk = value + ((size_t)h * n_prompt + k) * VHD;
            acc += wrow[k] * val_hk[d];
        }
        out[d] = acc;
    }
}

/* grid = (NH, kv_len); block = 256 cooperating over the KVL+QKR contraction.
 * One block per (h,k) -> NH*kv_len blocks (was NH), filling the GPU for decode.
 * score[h*kv_len+k] = (qabs[h]·c_kv[k] + qpe[h]·k_pe[k]) * scale */
__global__ void attn_absorbed_score_kernel(float *score, const float *qabs,
                                           const float *q, const float *kv_l,
                                           int QHD, int QKN, int QKR, int KVL,
                                           int KVD, int kv_len, float scale) {
    int h = blockIdx.x;
    int k = blockIdx.y;
    const float *qabs_h = qabs + (size_t)h * KVL;
    const float *qpe    = q + (size_t)h * QHD + QKN;
    const float *ckv    = kv_l + (size_t)k * KVD;
    const float *kpe    = ckv + KVL;

    float p = 0.0f;
    for (int i = threadIdx.x; i < KVL + QKR; i += blockDim.x)
        p += (i < KVL) ? qabs_h[i] * ckv[i] : qpe[i - KVL] * kpe[i - KVL];
    float s = attn_block_reduce_sum(p);
    if (threadIdx.x == 0) score[(size_t)h * kv_len + k] = s * scale;
}

/* grid = (NH); block = min(KVL,1024), grid-stride over r.
 * clat[h*KVL+r] = sum_k score[h,k] * kv_l[k*KVD+r] */
__global__ void attn_absorbed_clat_kernel(float *clat, const float *score,
                                          const float *kv_l, int KVL, int KVD,
                                          int kv_len) {
    int h = blockIdx.x;
    const float *srow = score + (size_t)h * kv_len;
    float *out = clat + (size_t)h * KVL;

    for (int r = threadIdx.x; r < KVL; r += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < kv_len; k++) acc += srow[k] * kv_l[(size_t)k * KVD + r];
        out[r] = acc;
    }
}

__global__ void attn_flash_decode_lds_kernel(float *clat, const float *qabs, const float *q,
                                             const int8_t *kv_base, const float *kv_scale,
                                             const int *pos, int NH, int QHD, int QKN, int QKR,
                                             int KVL, int KVD, int slot_stride, float scale) {
    int h = blockIdx.x, s = blockIdx.y;
    int tid = threadIdx.x, nt = blockDim.x;      /* nt = 256 */
    int kv_len = pos[s] + 1;
    (void)QKR;
    const float  *qabs_h = qabs + ((size_t)s * NH + h) * KVL;
    const float  *qpe    = q + ((size_t)s * NH + h) * QHD + QKN;
    const int8_t *ckvb   = kv_base + (size_t)s * slot_stride;
    const float  *scl_s  = kv_scale + (size_t)s * (slot_stride / KVD);

    __shared__ float  qsh[640];              /* query [KVD] */
    __shared__ int8_t ckv_lds[FD_TK * 576];  /* c_kv tile [FD_TK][KVD] */
    __shared__ float  ssh[FD_TK];            /* per-tile softmax weights */
    __shared__ float  scl_lds[FD_TK];        /* per-key int8 scale */
    __shared__ float  part[256];             /* 4-way partial-dot reduction */

    for (int i = tid; i < KVD; i += nt) qsh[i] = (i < KVL) ? qabs_h[i] : qpe[i - KVL];
    __syncthreads();

    const int ndim = (KVL + nt - 1) / nt;    /* 2 */
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m = -INFINITY, l = 0.0f;

    for (int k0 = 0; k0 < kv_len; k0 += FD_TK) {
        int tk = (kv_len - k0 < FD_TK) ? (kv_len - k0) : FD_TK;
        for (int idx = tid; idx < tk * KVD; idx += nt) ckv_lds[idx] = ckvb[(size_t)k0 * KVD + idx];
        for (int j = tid; j < tk; j += nt) scl_lds[j] = scl_s[k0 + j];
        __syncthreads();
        /* Phase A: 4 threads/key partial dot over KVD (all from LDS). */
        {
            int j = tid % FD_TK, p = tid / FD_TK;         /* key, part(0..3) */
            float pd = 0.0f;
            if (j < tk) {
                int d0 = p * (KVD / 4), d1 = d0 + (KVD / 4);
                const int8_t *row = ckv_lds + (size_t)j * KVD;
                for (int d = d0; d < d1; d++) pd += qsh[d] * (float)row[d];
            }
            part[tid] = pd;
        }
        __syncthreads();
        float sc = -INFINITY;
        if (tid < FD_TK && tid < tk)
            sc = (part[tid] + part[tid + 64] + part[tid + 128] + part[tid + 192]) * scl_lds[tid] * scale;
        /* Phase B: online softmax over the tile. */
        float mnew = fmaxf(m, attn_block_reduce_max(sc));
        float corr = (m == -INFINITY) ? 0.0f : __expf(m - mnew);
        float w = (sc == -INFINITY) ? 0.0f : __expf(sc - mnew);
        if (tid < FD_TK) ssh[tid] = w;
        float tsum = attn_block_reduce_sum((tid < FD_TK) ? w : 0.0f);
        __syncthreads();
        l = l * corr + tsum;
        /* Phase C: clat contribution for owned dims, reading the LDS c_kv tile. */
        for (int d = 0; d < ndim; d++) {
            int r = tid + d * nt;
            if (r < KVL) {
                float a = 0.0f;
                for (int j = 0; j < tk; j++)
                    a += ssh[j] * scl_lds[j] * (float)ckv_lds[(size_t)j * KVD + r];
                acc[d] = acc[d] * corr + a;
            }
        }
        __syncthreads();
        m = mnew;
    }
    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int d = 0; d < ndim; d++) {
        int r = tid + d * nt;
        if (r < KVL) clat[((size_t)s * NH + h) * KVL + r] = acc[d] * inv;
    }
}


/* ---- from batched.hip ---- */

__global__ void matmul_tiled_kernel(float *Y, const float *X, const bf16_t *W,
                                    int M, int N, int K, int x_stride,
                                    int y_stride) {
    __shared__ float As[TBM][TBK];
    __shared__ float Bs[TBN][TBK];
    int tx = threadIdx.x;          /* 0..255 */
    int tr = tx >> 4, tc = tx & 15;/* 16 x 16 thread grid */
    int row0 = blockIdx.y * TBM;
    int col0 = blockIdx.x * TBN;

    /* K is always a multiple of TBK here (all contraction dims are mult of 16),
     * so K-boundary checks are unneeded; only N/M row bounds matter. Loads are
     * vectorized: fp32 activations 4-wide (float4), bf16 weights 8-wide (16B). */
    float acc[4][4] = {{0}};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        /* A tile [TBM][TBK] fp32: 64*16/4 = 256 float4 loads, one per thread. */
        {
            int r = tx / (TBK / 4), cc = (tx % (TBK / 4)) * 4;
            int gr = row0 + r, gc = k0 + cc;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gr < M) v = *reinterpret_cast<const float4 *>(&X[(size_t)gr * x_stride + gc]);
            As[r][cc + 0] = v.x; As[r][cc + 1] = v.y;
            As[r][cc + 2] = v.z; As[r][cc + 3] = v.w;
        }
        /* B tile [TBN][TBK] bf16->f32: 64*16/8 = 128 8-wide loads. */
        for (int idx = tx; idx < TBN * TBK / 8; idx += 256) {
            int r = idx / (TBK / 8), cc = (idx % (TBK / 8)) * 8;
            int gr = col0 + r, gc = k0 + cc;
            uint16_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &W[(size_t)gr * K + gc], 8 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 8; j++) Bs[r][cc + j] = (gr < N) ? bf16f(wb[j]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TBK; k++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[tr * 4 + i][k];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[tc * 4 + j][k];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int gr = row0 + tr * 4 + i;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int gc = col0 + tc * 4 + j;
            if (gc >= N) continue;
            Y[(size_t)gr * y_stride + gc] = acc[i][j];
        }
    }
}

/* ======================================================================== *
 *  Per-head batched matmul (adds a batch dim to matmul_batched_kernel):
 *   Y[s,h,d] = sum_{i<n_in} X[s*x_bstride + h*x_hstride + i] * bf16f(W[h*w_stride + d*n_in + i])
 *  grid = (d_out, NH, B). W is shared across the batch (same layer weights).
 *  Used for qabs = q_nope @ W_UK^T and ctx = clat @ W_UV^T.
 * ======================================================================== */
__global__ void matmul_bh_batched_kernel(float *Y, const float *X,
                                         const bf16_t *W, int d_out, int n_in,
                                         int x_hstride, int x_bstride,
                                         int w_stride, int y_hstride,
                                         int y_bstride) {
    (void)d_out;
    int d = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    const float *xr = X + (size_t)s * x_bstride + (size_t)h * x_hstride;
    const bf16_t *wr = W + (size_t)h * w_stride + (size_t)d * n_in;
    int tid = threadIdx.x, nt = blockDim.x;

    __shared__ float sh[256];
    float partial = 0.0f;
    for (int i = tid; i < n_in; i += nt) partial += xr[i] * bf16f(wr[i]);
    sh[tid] = partial;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) sh[tid] += sh[tid + st];
        __syncthreads();
    }
    if (tid == 0) Y[(size_t)s * y_bstride + (size_t)h * y_hstride + d] = sh[0];
}

/* ======================================================================== *
 *  Batched absorbed attention over per-request KV caches.
 *  KV cache layout: kv_cache[l][s][pos][KVD]; the caller passes the per-layer
 *  base (kv_cache + l*B*cap*KVD) plus slot_stride = cap*KVD.
 *
 *  score kernel: grid = (kv_max, NH, B). One block per (k, h, s). kv_len[s] =
 *  pos[s]+1; k >= kv_len -> -INFINITY (so the shared softmax_row over kv_max is
 *  correct). score[s][h][k] = (qabs[s,h]·c_kv[k] + qpe[s,h]·k_pe[k]) * scale.
 * ======================================================================== */
__global__ void attn_score_batched_kernel(float *score, const float *qabs,
                                           const float *q, const float *kv_base,
                                           const int *pos, int NH, int QHD,
                                           int QKN, int QKR, int KVL, int KVD,
                                           int kv_max, int slot_stride,
                                           float scale) {
    int k = blockIdx.x;
    int h = blockIdx.y;
    int s = blockIdx.z;
    float *out = score + (((size_t)s * NH + h) * kv_max) + k;
    int kv_len = pos[s] + 1;
    if (k >= kv_len) { if (threadIdx.x == 0) *out = -INFINITY; return; }

    const float *qabs_h = qabs + ((size_t)s * NH + h) * KVL;
    const float *qpe    = q + ((size_t)s * NH + h) * QHD + QKN;
    const float *ckv    = kv_base + (size_t)s * slot_stride + (size_t)k * KVD;
    const float *kpe    = ckv + KVL;

    __shared__ float sh[256];
    int tid = threadIdx.x, nt = blockDim.x;
    float p = 0.0f;
    for (int i = tid; i < KVL + QKR; i += nt)
        p += (i < KVL) ? qabs_h[i] * ckv[i] : qpe[i - KVL] * kpe[i - KVL];
    sh[tid] = p;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) sh[tid] += sh[tid + st];
        __syncthreads();
    }
    if (tid == 0) *out = sh[0] * scale;
}

/* clat[s,h,r] = sum_{k<kv_len} score[s,h,k] * c_kv[s,k,r]  (r in [0,KVL)).
 * grid = (NH, B); block strides over r. Masked score entries are 0 post-softmax. */
__global__ void attn_clat_batched_kernel(float *clat, const float *score,
                                         const float *kv_base, const int *pos,
                                         int NH, int KVL, int KVD, int kv_max,
                                         int slot_stride) {
    int h = blockIdx.x;
    int s = blockIdx.y;
    const float *srow = score + ((size_t)s * NH + h) * kv_max;
    const float *ckvb = kv_base + (size_t)s * slot_stride;
    float *out = clat + ((size_t)s * NH + h) * KVL;
    int kv_len = pos[s] + 1;
    for (int r = threadIdx.x; r < KVL; r += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < kv_len; k++) acc += srow[k] * ckvb[(size_t)k * KVD + r];
        out[r] = acc;
    }
}

/* Grouped expert GEMM: for expert e = blockIdx.z, over its tokens
 * [eoff[e], eoff[e+1]). out_sorted[p, n] = sum_k Xrow(p)[k] * bf16f(W_e[n*K+k]).
 * If gather!=0, Xrow(p) = X + sorted_slot[p]*x_stride (gate/up read xn by slot);
 * else Xrow(p) = X + p*x_stride (down reads the sorted hidden buffer directly).
 * W_e = Wtbl[e] (device expert-pointer table). grid=(ceil(N/TBN),ceil(B/TBM),n_routed). */
__global__ void moe_grouped_gemm_kernel(float *out_sorted, const float *X,
                                        const bf16_t *const *Wtbl,
                                        const int *eoff, const int *sorted_slot,
                                        int N, int K, int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt  = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM;
    if (row0 >= cnt) return;
    const bf16_t *W = Wtbl[e];

    __shared__ float As[TBM][TBK];
    __shared__ float Bs[TBN][TBK];
    int tx = threadIdx.x;
    int tr = tx >> 4, tc = tx & 15;
    int col0 = blockIdx.x * TBN;

    /* K (=H or =I) is a multiple of TBK, so no K-boundary checks. Vectorized:
     * gathered fp32 rows 4-wide, bf16 weights 8-wide. */
    float acc[4][4] = {{0}};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        {
            int r = tx / (TBK / 4), cc = (tx % (TBK / 4)) * 4;
            int p = row0 + r;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (p < cnt) {
                int srow = gather ? sorted_slot[base + p] : (base + p);
                v = *reinterpret_cast<const float4 *>(&X[(size_t)srow * x_stride + k0 + cc]);
            }
            As[r][cc + 0] = v.x; As[r][cc + 1] = v.y;
            As[r][cc + 2] = v.z; As[r][cc + 3] = v.w;
        }
        for (int idx = tx; idx < TBN * TBK / 8; idx += 256) {
            int r = idx / (TBK / 8), cc = (idx % (TBK / 8)) * 8;
            int gr = col0 + r, gc = k0 + cc;
            uint16_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &W[(size_t)gr * K + gc], 8 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 8; j++) Bs[r][cc + j] = (gr < N) ? bf16f(wb[j]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TBK; k++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[tr * 4 + i][k];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[tc * 4 + j][k];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int p = row0 + tr * 4 + i;
        if (p >= cnt) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int gc = col0 + tc * 4 + j;
            if (gc >= N) continue;
            out_sorted[(size_t)(base + p) * N + gc] = acc[i][j];
        }
    }
}

/* MFMA version of moe_grouped_gemm_kernel: same gather/expert-offset semantics,
 * but the inner product runs on the matrix cores (fp32, like matmul_mfma). Block
 * = 4 warps computing a 64x64 tile of expert e = blockIdx.z's token block.
 * grid = (ceil(N/64), ceil(rows/64), n_routed). */
__global__ void moe_grouped_mfma_kernel(float *out_sorted, const float *X,
                                        const bf16_t *const *Wtbl,
                                        const int *eoff, const int *sorted_slot,
                                        int N, int K, int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM;
    if (row0 >= cnt) return;
    const bf16_t *W = Wtbl[e];

    __shared__ __bf16 As[TBK][TBM];
    __shared__ __bf16 Bs[TBK][TBN];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int wr = warp / 2, wc = warp % 2;
    int srow = wr * 32, scol = wc * 32;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN;

    float4v acc[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        {
            int m = tx / 4, cc = (tx % 4) * 4, p = row0 + m;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (p < cnt) {
                int sr = gather ? sorted_slot[base + p] : (base + p);
                v = *reinterpret_cast<const float4 *>(&X[(size_t)sr * x_stride + k0 + cc]);
            }
            As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
            As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
        }
        for (int idx = tx; idx < TBN * TBK / 8; idx += 256) {
            int n = idx / (TBK / 8), cc = (idx % (TBK / 8)) * 8, gr = col0 + n;
            uint16_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &W[(size_t)gr * K + k0 + cc], 8 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 8; j++)
                Bs[cc + j][n] = (gr < N) ? *reinterpret_cast<const __bf16 *>(&wb[j]) : (__bf16)0.0f;
        }
        __syncthreads();
        bf16x4 a[2], b[2];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) a[i][e2] = As[trow * 4 + e2][srow + i * 16 + tcol];
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) b[j][e2] = Bs[trow * 4 + e2][scol + j * 16 + tcol];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
                acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int prow = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow + e2;
                if (p < cnt) out_sorted[(size_t)(base + p) * N + col] = acc[i][j][e2];
            }
        }
    }
}


/* ---- from linear.hip ---- */

/* Pointer-array batched matmul over the top-k selected experts: grid = (d_out, K).
 * Each expert k uses its own weight matrix Wptrs[k] (row-major [d_out, n_in]) and
 * its own input row X + k*x_stride (x_stride=0 => all experts share one row, as
 * for the gate/up projections; x_stride=n_in for the per-expert down proj).
 * Y[k, d] = sum_i X[k, i] * bf16f(Wptrs[k][d*n_in + i]). Collapses the K-expert
 * launch loop into one launch (K*d_out blocks -> full GPU). */
__global__ void matmul_ptr_batched_kernel(float *Y, const float *X,
                                          const bf16_t *const *Wptrs, int d_out,
                                          int n_in, int x_stride, int y_stride) {
    (void)d_out;
    int d = blockIdx.x;
    int k = blockIdx.y;
    const bf16_t *W = Wptrs[k] + (size_t)d * n_in;
    const float  *x = X + (size_t)k * x_stride;
    int tid = threadIdx.x, nt = blockDim.x;

    /* 8-wide coalesced bf16 loads (consecutive threads -> consecutive 16B). */
    float partial = 0.0f;
    int n8 = n_in & ~7;
    for (int i = tid * 8; i < n8; i += nt * 8) {
        uint16_t wb[8];
        __builtin_memcpy(wb, W + i, 8 * sizeof(uint16_t));
#pragma unroll
        for (int j = 0; j < 8; j++) partial += x[i + j] * bf16f(wb[j]);
    }
    for (int i = n8 + tid; i < n_in; i += nt) partial += x[i] * bf16f(W[i]);

    float acc = mm_block_reduce_sum(partial);
    if (tid == 0) Y[(size_t)k * y_stride + d] = acc;
}

/* Batched-over-heads matmul: grid = (d_out, NH). Folds the per-head absorbed
 * context loop (ctx[h] = clat[h] @ W_UV[h].T) into one launch.
 * Y[h, d] = sum_i X[h, i] * bf16f(W[h, d*n_in + i]). */
__global__ void matmul_batched_kernel(float *Y, const float *X, const bf16_t *W,
                                      int d_out, int n_in, int x_stride,
                                      int w_stride, int y_stride) {
    (void)d_out;
    int d = blockIdx.x;
    int h = blockIdx.y;
    const float  *xh = X + (size_t)h * x_stride;
    const bf16_t *Wh = W + (size_t)h * w_stride + (size_t)d * n_in;
    int tid = threadIdx.x, nt = blockDim.x;

    float partial = 0.0f;
    int n8 = n_in & ~7;
    for (int i = tid * 8; i < n8; i += nt * 8) {
        uint16_t wb[8];
        __builtin_memcpy(wb, Wh + i, 8 * sizeof(uint16_t));
#pragma unroll
        for (int j = 0; j < 8; j++) partial += xh[i + j] * bf16f(wb[j]);
    }
    for (int i = n8 + tid; i < n_in; i += nt) partial += xh[i] * bf16f(Wh[i]);

    float acc = mm_block_reduce_sum(partial);
    if (tid == 0) Y[(size_t)h * y_stride + d] = acc;
}


/* ---- from elementwise.hip ---- */

__global__ void zero_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

/* out[d] = sum_{k<K} wts[k] * src[k*H + d]. One thread per hidden dim d; the
 * k-loop runs in the reference's r=0..K-1 order so the sum matches bit-for-bit. */
__global__ void weighted_reduce_kernel(float *out, const float *src,
                                       const float *wts, int K, int H) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= H) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += wts[k] * src[(size_t)k * H + d];
    out[d] = acc;
}

/* Numerically-stable softmax over n elements per row (n_rows rows), treating
 * -INFINITY as 0 post-exp -- matches CPU softmax() exactly. */
__global__ void softmax_row_kernel(float *x, int n_rows, int n) {
    (void)n_rows;
    float *row = x + (size_t)blockIdx.x * n;

    float pm = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) pm = fmaxf(pm, row[i]);
    float mx = block_reduce_max(pm);

    float ps = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = (row[i] == -INFINITY) ? 0.0f : expf(row[i] - mx);
        row[i] = e;
        ps += e;
    }
    float sum = block_reduce_sum(ps);

    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] /= sum;
}


/* ---- from moe.hip ---- */

__global__ void moe_route_kernel(const float *logits, int n_routed, int K,
                                 int router_sigmoid, int norm_topk,
                                 float routed_scaling, const float *bias,
                                 const bf16_t *const *tbl_gate,
                                 const bf16_t *const *tbl_up,
                                 const bf16_t *const *tbl_down, float *out_wts,
                                 const bf16_t **out_gate, const bf16_t **out_up,
                                 const bf16_t **out_down) {
    __shared__ float sc[1024];
    __shared__ char  used[1024];
    __shared__ float rval[ROUTE_THREADS];
    __shared__ int   ridx[ROUTE_THREADS];
    __shared__ int   idx_sh[16];
    __shared__ float wt_sh[16];
    __shared__ float red[ROUTE_THREADS];

    int tid = threadIdx.x;
    int nt  = blockDim.x;

    /* ---- sigmoid or numerically-stable softmax over the router logits ---- */
    if (router_sigmoid) {
        for (int e = tid; e < n_routed; e += nt)
            sc[e] = 1.0f / (1.0f + expf(-logits[e]));
    } else {
        float pm = -INFINITY;
        for (int e = tid; e < n_routed; e += nt) pm = fmaxf(pm, logits[e]);
        red[tid] = pm; __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
            __syncthreads();
        }
        float mx = red[0]; __syncthreads();
        float ps = 0.0f;
        for (int e = tid; e < n_routed; e += nt) { float ex = expf(logits[e] - mx); sc[e] = ex; ps += ex; }
        red[tid] = ps; __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float sum = red[0]; __syncthreads();
        for (int e = tid; e < n_routed; e += nt) sc[e] /= sum;
    }
    for (int e = tid; e < n_routed; e += nt) used[e] = 0;
    __syncthreads();

    /* ---- greedy top-k: K rounds of block-wide argmax (bias-adjusted) ---- */
    for (int r = 0; r < K; r++) {
        float lb = -INFINITY; int li = -1;
        for (int e = tid; e < n_routed; e += nt) {
            if (used[e]) continue;
            float sel = sc[e] + (bias ? bias[e] : 0.0f);
            if (sel > lb) { lb = sel; li = e; }   /* strict > -> lowest index locally */
        }
        rval[tid] = lb; ridx[tid] = li;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float ov = rval[tid + s]; int oi = ridx[tid + s];
                if (ov > rval[tid] || (ov == rval[tid] && oi >= 0 &&
                        (ridx[tid] < 0 || oi < ridx[tid]))) {
                    rval[tid] = ov; ridx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            int best = ridx[0];
            used[best] = 1;
            idx_sh[r] = best;
            wt_sh[r]  = sc[best];
        }
        __syncthreads();
    }

    /* ---- norm_topk + routed_scaling + gather expert pointers (thread 0) ---- */
    if (tid == 0) {
        if (norm_topk) {
            float sum = 1e-20f;
            for (int r = 0; r < K; r++) sum += wt_sh[r];
            for (int r = 0; r < K; r++) wt_sh[r] /= sum;
        }
        for (int r = 0; r < K; r++) {
            int e = idx_sh[r];
            out_wts[r]  = wt_sh[r] * routed_scaling;
            out_gate[r] = tbl_gate[e];
            out_up[r]   = tbl_up[e];
            out_down[r] = tbl_down[e];
        }
    }
}



/* ======================================================================== *
 * Retired decode/prefill attention kernels (run8-run51). The engine now runs a
 * single path: attn_prefill_mfma_kernel for prefill and attn_fd_hb_q8_kernel
 * for decode. Kept for reference -- this file is NOT compiled.
 * ======================================================================== */
/* Fused flash-attention for the unabsorbed prefill path: computes causal scaled
 * scores, softmax, and the value-weighted context in one pass without ever
 * materializing the O(n_prompt^2) score matrix in HBM (replaces
 * attn_unabsorbed_score + softmax_row + attn_unabsorbed_ctx).
 *
 * grid = (n_prompt [q], NH [h]); block = ATTN_THREADS (256). Thread tid owns
 * value-dim d=tid (VHD <= ATTN_THREADS) for its running context accumulator; keys
 * are streamed in tiles of blockDim.x with online softmax (running max m, sum l).
 * ctx[(q*NH+h)*VHD + d] = sum_{k<=q} softmax_k(score)[k] * value[h,k,d]. */
__global__ void attn_unabsorbed_flash_kernel(float *ctx, const float *qall,
                                             const float *knope, const float *kv_l,
                                             const float *value, int n_prompt,
                                             int NH, int QHD, int QKN, int QKR,
                                             int KVD, int VHD, float scale) {
    int q = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    const float *qrow    = qall  + ((size_t)q * NH + h) * QHD;
    const float *knope_h = knope + (size_t)h * n_prompt * QKN;
    const float *value_h = value + (size_t)h * n_prompt * VHD;

    __shared__ float qsh[512];             /* query vector (QHD <= 512) */
    __shared__ float ssh[ATTN_THREADS];    /* per-tile softmax weights */
    for (int d = tid; d < QHD; d += blockDim.x) qsh[d] = qrow[d];
    __syncthreads();

    float acc = 0.0f;                      /* running context for dim d=tid */
    float m = -INFINITY, l = 0.0f;         /* online-softmax running max/sum */
    const int nkeys = q + 1;               /* causal: only k <= q */

    for (int k0 = 0; k0 < nkeys; k0 += blockDim.x) {
        int k = k0 + tid;
        /* Phase A: this thread's key score (or -inf if past the causal edge). */
        float s = -INFINITY;
        if (k < nkeys) {
            const float *knk = knope_h + (size_t)k * QKN;
            const float *kpe = kv_l + (size_t)k * KVD + (KVD - QKR);
            float dot = 0.0f;
            for (int d = 0; d < QKN; d++) dot += qsh[d] * knk[d];
            for (int d = 0; d < QKR; d++) dot += qsh[QKN + d] * kpe[d];
            s = dot * scale;
        }
        /* Phase B: online softmax bookkeeping for this tile. */
        float mnew = fmaxf(m, attn_block_reduce_max(s));
        float corr = expf(m - mnew);       /* rescale factor for prior tiles */
        float w = (k < nkeys) ? expf(s - mnew) : 0.0f;
        ssh[tid] = w;
        float tilesum = attn_block_reduce_sum(w);
        __syncthreads();                   /* ssh visible to all before Phase C */
        l = l * corr + tilesum;
        acc *= corr;
        /* Phase C: accumulate this tile's weighted values into dim d=tid. */
        if (tid < VHD) {
            int tk = (nkeys - k0 < (int)blockDim.x) ? (nkeys - k0) : (int)blockDim.x;
            float a = 0.0f;
            for (int j = 0; j < tk; j++)
                a += ssh[j] * value_h[(size_t)(k0 + j) * VHD + tid];
            acc += a;
        }
        __syncthreads();                   /* ssh reused next tile */
        m = mnew;
    }

    if (tid < VHD) {
        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        ctx[((size_t)q * NH + h) * VHD + tid] = acc * inv;
    }
}

/* Varlen (chunked/batched prefill) version of attn_unabsorbed_flash_kernel:
 * a single flat token axis of length n_tok packs several prompts back-to-back,
 * and each query attends only within its own prompt. q_seqstart[qg] = the global
 * index of the first token of the prompt that global query qg belongs to, so the
 * causal key range for qg is [q_seqstart[qg], qg]. knope/value are laid out
 * [NH, n_tok, ...] (head-major over the flat token axis); kv_l is [n_tok, KVD].
 * grid = (n_tok [qg], NH [h]); block = ATTN_THREADS (256), thread tid owns
 * value-dim d=tid. ctx[(qg*NH+h)*VHD + d] = sum_{k in prompt, k<=qg} softmax*value. */
__global__ void attn_unabsorbed_flash_varlen_kernel(float *ctx, const float *qall,
                                                    const float *knope, const float *kv_l,
                                                    const float *value,
                                                    const int *q_seqstart, int n_tok,
                                                    int NH, int QHD, int QKN, int QKR,
                                                    int KVD, int VHD, float scale) {
    int qg = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    const float *qrow    = qall  + ((size_t)qg * NH + h) * QHD;
    const float *knope_h = knope + (size_t)h * n_tok * QKN;
    const float *value_h = value + (size_t)h * n_tok * VHD;
    int qstart = q_seqstart[qg];
    const int nk = qg - qstart + 1;        /* keys within this prompt, causal */

    __shared__ float qsh[512];             /* query vector (QHD <= 512) */
    __shared__ float ssh[ATTN_THREADS];    /* per-tile softmax weights */
    for (int d = tid; d < QHD; d += blockDim.x) qsh[d] = qrow[d];
    __syncthreads();

    float acc = 0.0f;                      /* running context for dim d=tid */
    float m = -INFINITY, l = 0.0f;         /* online-softmax running max/sum */

    for (int t0 = 0; t0 < nk; t0 += blockDim.x) {
        int j = t0 + tid;                  /* offset within this prompt's keys */
        int kg = qstart + j;               /* global key index */
        /* Phase A: this thread's key score (or -inf past the causal edge). */
        float s = -INFINITY;
        if (j < nk) {
            const float *knk = knope_h + (size_t)kg * QKN;
            const float *kpe = kv_l + (size_t)kg * KVD + (KVD - QKR);
            float dot = 0.0f;
            for (int d = 0; d < QKN; d++) dot += qsh[d] * knk[d];
            for (int d = 0; d < QKR; d++) dot += qsh[QKN + d] * kpe[d];
            s = dot * scale;
        }
        /* Phase B: online softmax bookkeeping for this tile. */
        float mnew = fmaxf(m, attn_block_reduce_max(s));
        float corr = expf(m - mnew);
        float w = (j < nk) ? expf(s - mnew) : 0.0f;
        ssh[tid] = w;
        float tilesum = attn_block_reduce_sum(w);
        __syncthreads();
        l = l * corr + tilesum;
        acc *= corr;
        /* Phase C: accumulate this tile's weighted values into dim d=tid. */
        if (tid < VHD) {
            int tk = (nk - t0 < (int)blockDim.x) ? (nk - t0) : (int)blockDim.x;
            float a = 0.0f;
            for (int jj = 0; jj < tk; jj++)
                a += ssh[jj] * value_h[(size_t)(qstart + t0 + jj) * VHD + tid];
            acc += a;
        }
        __syncthreads();
        m = mnew;
    }

    if (tid < VHD) {
        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        ctx[((size_t)qg * NH + h) * VHD + tid] = acc * inv;
    }
}

/* Fused flash-decode for the batched absorbed path: computes score + softmax +
 * clat for one (slot s, head h) in a single online-softmax pass, replacing
 * attn_score_batched + softmax_row + attn_clat_batched (no b_score[B*NH*kv_max]
 * materialization, and NH*B fat blocks instead of kv_max*NH*B one-score blocks).
 *
 * grid = (NH, B); block = ATTN_THREADS (256); dynamic shmem = KVD floats (query).
 * For slot s the causal length is kv_len = pos[s]+1. The score is a latent-space
 * dot over KVD = KVL(c_kv) + QKR(k_pe): qsh = [qabs_h (KVL), qpe (QKR)] dotted with
 * ckv = [c_kv (KVL), k_pe (QKR)]. clat[r] = sum_{k<kv_len} softmax_k * c_kv[k,r],
 * r in [0,KVL); each thread owns dims r = tid + d*blockDim (d < ceil(KVL/blockDim)). */
__global__ void attn_flash_decode_kernel(float *clat, const float *qabs, const float *q,
                                         const int8_t *kv_base, const float *kv_scale,
                                         const int *pos, int NH,
                                         int QHD, int QKN, int QKR, int KVL, int KVD,
                                         int slot_stride, float scale) {
    int h = blockIdx.x, s = blockIdx.y;
    int tid = threadIdx.x, nt = blockDim.x;
    int kv_len = pos[s] + 1;
    (void)QKR;                             /* k_pe width is implicit in KVD - KVL */

    const float *qabs_h = qabs + ((size_t)s * NH + h) * KVL;
    const float *qpe    = q + ((size_t)s * NH + h) * QHD + QKN;
    const int8_t *ckvb  = kv_base + (size_t)s * slot_stride;              /* int8 cache */
    const float *scl_s  = kv_scale + (size_t)s * (slot_stride / KVD);     /* per-key row scale */

    extern __shared__ float qsh[];         /* [KVD] = qabs_h (KVL) ++ qpe (QKR) */
    for (int i = tid; i < KVD; i += nt) qsh[i] = (i < KVL) ? qabs_h[i] : qpe[i - KVL];
    __syncthreads();

    __shared__ float ssh[ATTN_THREADS];    /* per-tile softmax weights */
    const int ndim = (KVL + nt - 1) / nt;  /* clat dims per thread (<= 4) */
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m = -INFINITY, l = 0.0f;

    for (int k0 = 0; k0 < kv_len; k0 += nt) {
        int k = k0 + tid;
        /* Phase A: this thread's key score (latent-space dot over KVD). */
        float sc = -INFINITY;
        if (k < kv_len) {
            const int8_t *ckv = ckvb + (size_t)k * KVD;
            float dot = 0.0f;
            for (int i = 0; i < KVD; i++) dot += qsh[i] * (float)ckv[i];
            sc = dot * scl_s[k] * scale;                 /* fold per-key int8 scale */
        }
        /* Phase B: online softmax bookkeeping for this tile. */
        float mnew = fmaxf(m, attn_block_reduce_max(sc));
        float corr = expf(m - mnew);
        float w = (k < kv_len) ? expf(sc - mnew) : 0.0f;
        ssh[tid] = w;
        float tsum = attn_block_reduce_sum(w);
        __syncthreads();
        l = l * corr + tsum;
        /* Phase C: accumulate this tile's c_kv-weighted context into owned dims. */
        int tk = (kv_len - k0 < nt) ? (kv_len - k0) : nt;
        for (int d = 0; d < ndim; d++) {
            int r = tid + d * nt;
            if (r < KVL) {
                float a = 0.0f;
                for (int j = 0; j < tk; j++)
                    a += ssh[j] * scl_s[k0 + j] * (float)ckvb[(size_t)(k0 + j) * KVD + r];
                acc[d] = acc[d] * corr + a;
            }
        }
        __syncthreads();
        m = mnew;
    }

    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int d = 0; d < ndim; d++) {
        int r = tid + d * nt;
        if (r < KVL) clat[((size_t)s * NH + h) * KVL + r] = acc[d] * inv;
    }
}

/* clat regs/thread = 20         */
__global__ void attn_flash_decode_hb_kernel(float *clat, const float *qabs, const float *q,
                                            const int8_t *kv_base, const float *kv_scale,
                                            const int *pos, int NH, int QHD, int QKN, int QKR,
                                            int KVL, int KVD, int slot_stride, float scale) {
    const int s = blockIdx.x, g = blockIdx.y;
    const int h0 = g * FD_HPB;                 /* first head this block owns        */
    const int tid = threadIdx.x, nt = blockDim.x;   /* nt = 256                     */
    (void)NH; (void)QKR; (void)KVL; (void)KVD;
    const int kv_len = pos[s] + 1;
    const int8_t *ckvb  = kv_base  + (size_t)s * slot_stride;
    const float  *scl_s = kv_scale + (size_t)s * (slot_stride / FD_KVD);

    __shared__ __bf16 Qsh[FD_HPB * FD_KVD];    /* queries bf16 [HPB][KVD]           */
    __shared__ int8_t Ksh[FD_TK * FD_KVD];     /* c_kv tile int8 [TK][KVD]          */
    __shared__ float  Kscl[FD_TK];             /* per-key int8 scale                */
    __shared__ float  Ssh[FD_HPB * FD_TK];     /* scores -> softmax probs [HPB][TK] */
    __shared__ float  Msh[FD_HPB], Lsh[FD_HPB], Csh[FD_HPB];   /* running m,l,corr  */

    float oacc[FD_OACC];
#pragma unroll
    for (int e = 0; e < FD_OACC; e++) oacc[e] = 0.0f;
    if (tid < FD_HPB) { Msh[tid] = -INFINITY; Lsh[tid] = 0.0f; }

    /* stage the HPB queries into LDS as bf16 (qabs_h ++ qpe). */
    for (int i = tid; i < FD_HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i % FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        Qsh[i] = (__bf16)v;
    }
    __syncthreads();

    for (int k0 = 0; k0 < kv_len; k0 += FD_TK) {
        const int tk = (kv_len - k0 < FD_TK) ? (kv_len - k0) : FD_TK;
        /* c_kv tile into LDS ONCE (this is the L2 read shared across the head group). */
        for (int i = tid; i < tk * FD_KVD; i += nt) Ksh[i] = ckvb[(size_t)k0 * FD_KVD + i];
        for (int j = tid; j < tk; j += nt) Kscl[j] = scl_s[k0 + j];
        __syncthreads();
        /* scores: one thread computes the full KVD dot for each owned (h,k). */
        for (int hk = tid; hk < FD_HPB * tk; hk += nt) {
            int h = hk / tk, k = hk % tk;
            const __bf16 *qh = Qsh + (size_t)h * FD_KVD;
            const int8_t *kk = Ksh + (size_t)k * FD_KVD;
            float d = 0.0f;
            for (int i = 0; i < FD_KVD; i++) d += (float)qh[i] * (float)kk[i];
            Ssh[h * FD_TK + k] = d * Kscl[k] * scale;
        }
        __syncthreads();
        /* online softmax per head (one thread per head). */
        if (tid < FD_HPB) {
            int h = tid;
            float tmax = -INFINITY;
            for (int k = 0; k < tk; k++) tmax = fmaxf(tmax, Ssh[h * FD_TK + k]);
            float mo = Msh[h], mn = fmaxf(mo, tmax);
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            float sum = 0.0f;
            for (int k = 0; k < tk; k++) { float p = __expf(Ssh[h * FD_TK + k] - mn); Ssh[h * FD_TK + k] = p; sum += p; }
            Lsh[h] = Lsh[h] * corr + sum; Msh[h] = mn; Csh[h] = corr;
        }
        __syncthreads();
        /* clat: O[h][r] = O[h][r]*corr[h] + sum_k P[h,k]*scl[k]*ckv[k,r]. Flat
         * (h,r) distribution keeps the accumulator in registers. */
        int idx = tid;
#pragma unroll
        for (int e = 0; e < FD_OACC; e++, idx += nt) {
            if (idx < FD_HPB * FD_KVL) {
                int h = idx / FD_KVL, r = idx % FD_KVL;
                float a = 0.0f;
                for (int k = 0; k < tk; k++)
                    a += Ssh[h * FD_TK + k] * Kscl[k] * (float)Ksh[(size_t)k * FD_KVD + r];
                oacc[e] = oacc[e] * Csh[h] + a;
            }
        }
        __syncthreads();
    }
    /* normalize by l and write clat[s,h,r]. */
    int idx = tid;
#pragma unroll
    for (int e = 0; e < FD_OACC; e++, idx += nt) {
        if (idx < FD_HPB * FD_KVL) {
            int h = idx / FD_KVL, r = idx % FD_KVL, hg = h0 + h;
            float inv = (Lsh[h] > 0.0f) ? 1.0f / Lsh[h] : 0.0f;
            clat[((size_t)s * FD_NH + hg) * FD_KVL + r] = oacc[e] * inv;
        }
    }
}

/* Qsh row stride in __bf16 (289 dwords, gcd(289,32)=1)   */

/* run42: the same kernel, templated on (heads-per-block, key-tile) so the LDS
 * footprint -- and therefore the occupancy -- can be tuned.  After run39 killed
 * the bank conflicts this kernel issues only ~13% of peak instructions: with
 * 31.6 KB of LDS only TWO blocks fit per CU (8 waves = 2 per SIMD), far too few
 * to cover LDS latency (run38 already measured occupancy 11.5%, "Insufficient
 * CU LDS" 34%).  LDS per block = TK*580 + HPB*578*2 + ~(HPB*TK*4 + TK*4):
 *     HPB=10,TK=32 -> 31.6 KB -> 2 blocks/CU   (run39 default)
 *     HPB=10,TK=16 -> 21.7 KB -> 3 blocks/CU
 *     HPB=5, TK=32 -> 25.2 KB -> 2 blocks/CU
 *     HPB=5, TK=16 -> 15.5 KB -> 4 blocks/CU
 * Smaller HPB also doubles the block count (better load balance across the
 * ragged kv_len) at the cost of re-reading c_kv from L2 HG times per slot --
 * which run38 measured at only 1.37% of L2 peak, i.e. free.  Selected at launch
 * by MLA_FDHG / MLA_FDTK; the math is identical in every instantiation. */
template <int HPB, int TK>
__global__ void attn_fd_hb_t_kernel(float *clat, const float *qabs, const float *q,
                                    const int8_t *kv_base, const float *kv_scale,
                                    const int *pos, int NH, int QHD, int QKN, int QKR,
                                    int KVL, int KVD, int slot_stride, float scale) {
    constexpr int NGRP = HPB * FD_KVL / 4;          /* (head, 4-r) groups        */
    constexpr int CACC = (NGRP + 255) / 256;        /* clat float4s per thread   */
    const int s = blockIdx.x, g = blockIdx.y;
    const int h0 = g * HPB;
    const int tid = threadIdx.x, nt = blockDim.x;
    (void)NH; (void)QKR; (void)KVL; (void)KVD;
    const int kv_len = pos[s] + 1;
    const int8_t *ckvb  = kv_base  + (size_t)s * slot_stride;
    const float  *scl_s = kv_scale + (size_t)s * (slot_stride / FD_KVD);

    __shared__ __attribute__((aligned(16))) __bf16 Qsh[HPB * FD_QPAD];
    __shared__ __attribute__((aligned(16))) int8_t Ksh[TK * FD_KPAD];
    __shared__ float Kscl[TK];
    __shared__ float Ssh[HPB * TK];                 /* scores -> probs*keyscale  */
    __shared__ float Msh[HPB], Lsh[HPB], Csh[HPB];

    float oacc[CACC][4];
#pragma unroll
    for (int e = 0; e < CACC; e++)
#pragma unroll
        for (int j = 0; j < 4; j++) oacc[e][j] = 0.0f;
    if (tid < HPB) { Msh[tid] = -INFINITY; Lsh[tid] = 0.0f; }

    for (int i = tid; i < HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i - h * FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        Qsh[h * FD_QPAD + d] = (__bf16)v;
    }
    __syncthreads();

    const int KW = FD_KVD / 4;      /* 144 dwords per key row */
    const int KQ = FD_KVD / 16;     /* 36 int4 per key row     */
    for (int k0 = 0; k0 < kv_len; k0 += TK) {
        const int tk = (kv_len - k0 < TK) ? (kv_len - k0) : TK;
        /* run45: 16 B global loads (the GEMM lesson -- 4 B per lane leaves far too
         * few bytes in flight to cover HBM latency). The LDS side still writes
         * dwords because FD_KPAD=580 cannot be both 16 B-aligned and
         * bank-conflict-free (any multiple of 16 has gcd(stride/4,32) >= 4). */
        for (int i = tid; i < tk * KQ; i += nt) {
            int r = i / KQ, c = i - r * KQ;
            int4 v = *reinterpret_cast<const int4 *>(ckvb + (size_t)(k0 + r) * FD_KVD + c * 16);
            int8_t *d = Ksh + (size_t)r * FD_KPAD + c * 16;
            *reinterpret_cast<int *>(d)      = v.x;
            *reinterpret_cast<int *>(d + 4)  = v.y;
            *reinterpret_cast<int *>(d + 8)  = v.z;
            *reinterpret_cast<int *>(d + 12) = v.w;
        }
        for (int j = tid; j < tk; j += nt) Kscl[j] = scl_s[k0 + j];
        __syncthreads();

        for (int hk = tid; hk < HPB * tk; hk += nt) {
            int h = hk / tk, k = hk - h * tk;
            const int *kw = reinterpret_cast<const int *>(Ksh + (size_t)k * FD_KPAD);
            const int *qw = reinterpret_cast<const int *>(Qsh + (size_t)h * FD_QPAD);
            float d = 0.0f;
#pragma unroll 4
            for (int c = 0; c < KW; c++) {
                int kb = kw[c];
                int qa = qw[2 * c], qb = qw[2 * c + 1];
                d += __int_as_float(qa << 16)        * (float)(int8_t)(kb & 0xff);
                d += __int_as_float(qa & 0xffff0000) * (float)(int8_t)((kb >> 8) & 0xff);
                d += __int_as_float(qb << 16)        * (float)(int8_t)((kb >> 16) & 0xff);
                d += __int_as_float(qb & 0xffff0000) * (float)(int8_t)(kb >> 24);
            }
            Ssh[h * TK + k] = d * Kscl[k] * scale;
        }
        __syncthreads();

        if (tid < HPB) {
            int h = tid;
            float tmax = -INFINITY;
            for (int k = 0; k < tk; k++) tmax = fmaxf(tmax, Ssh[h * TK + k]);
            float mo = Msh[h], mn = fmaxf(mo, tmax);
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            float sum = 0.0f;
            for (int k = 0; k < tk; k++) {
                float p = __expf(Ssh[h * TK + k] - mn);
                sum += p;
                Ssh[h * TK + k] = p * Kscl[k];
            }
            Lsh[h] = Lsh[h] * corr + sum; Msh[h] = mn; Csh[h] = corr;
        }
        __syncthreads();

        {
            int g4 = tid;
#pragma unroll
            for (int e = 0; e < CACC; e++, g4 += nt) {
                if (CACC * 256 != NGRP && g4 >= NGRP) break;
                int h = g4 >> 7;                     /* 512/4 = 128 groups/head  */
                int r0 = (g4 & 127) << 2;
                const float *Ph = Ssh + h * TK;
                const int8_t *kb0 = Ksh + r0;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (int k = 0; k < tk; k++) {
                    int kb = *reinterpret_cast<const int *>(kb0 + (size_t)k * FD_KPAD);
                    float p = Ph[k];
                    a0 += p * (float)(int8_t)(kb & 0xff);
                    a1 += p * (float)(int8_t)((kb >> 8) & 0xff);
                    a2 += p * (float)(int8_t)((kb >> 16) & 0xff);
                    a3 += p * (float)(int8_t)(kb >> 24);
                }
                float c = Csh[h];
                oacc[e][0] = oacc[e][0] * c + a0;
                oacc[e][1] = oacc[e][1] * c + a1;
                oacc[e][2] = oacc[e][2] * c + a2;
                oacc[e][3] = oacc[e][3] * c + a3;
            }
        }
        __syncthreads();
    }
    {
        int g4 = tid;
#pragma unroll
        for (int e = 0; e < CACC; e++, g4 += nt) {
            if (CACC * 256 != NGRP && g4 >= NGRP) break;
            int h = g4 >> 7, r0 = (g4 & 127) << 2, hg = h0 + h;
            float inv = (Lsh[h] > 0.0f) ? 1.0f / Lsh[h] : 0.0f;
            float *dst = clat + ((size_t)s * FD_NH + hg) * FD_KVL + r0;
            dst[0] = oacc[e][0] * inv;
            dst[1] = oacc[e][1] * inv;
            dst[2] = oacc[e][2] * inv;
            dst[3] = oacc[e][3] * inv;
        }
    }
}

/* ======================================================================== *
 *  run50: sdot4 for the clat accumulation as well (transposed key tile).
 *
 *  After run46 the score costs one v_dot4 per 4 MACs, so clat -- which still ran
 *  scalar at ~3.5 instructions per MAC (2 LDS + 8 int8 unpacks + 4 FMA per 4
 *  MACs) -- became the whole kernel. It cannot use dot4 in the [k][d] layout
 *  because it reduces along k while d is the contiguous axis.
 *
 *  The user asked whether MFMA could take it. MFMA has the same problem AND a
 *  worse one: its B operand also wants k contiguous, so it needs the identical
 *  transpose, and on top of that D[16][512] blows the accumulator budget (32
 *  int32 + 32 float VGPRs per warp) while M=10 heads pad to 16. dot4 gets the
 *  same 4-MACs-per-instruction density with the accumulators the kernel already
 *  has, so this stages a transposed tile KT[d][key] (key contiguous, row stride
 *  20 B so gcd(5,32)=1 keeps it bank-conflict-free) and quantizes the softmax
 *  probabilities to int8 per head. clat drops from 3.5 to ~0.5 instructions per
 *  MAC. Cost: +10 KB LDS, i.e. 4 -> 2 blocks per CU.
 *
 *  MEASURED NEGATIVE, default OFF (MLA_FDCLATQ=1 to opt in): 1229.6 tok/s at
 *  TK=16 and 1248.3 at TK=8, against 1399.6 for the scalar clat -- about -12%.
 *  Two costs swamp the instruction saving: building the transpose needs 16
 *  single-byte LDS writes per 16 B loaded (~36 ds_write_b8 per thread per tile,
 *  conflict-prone), and the extra 10 KB halves occupancy. This also settles the
 *  MFMA question for clat: MFMA needs the SAME transposed operand, so it pays
 *  both of these costs, while its MAC-per-instruction density is no better than
 *  dot4's -- and it would additionally blow the accumulator budget.
 * ======================================================================== */
template <int HPB, int TK>
__global__ void attn_fd_hb_q8t_kernel(float *clat, const float *qabs, const float *q,
                                      const int8_t *kv_base, const float *kv_scale,
                                      const int *pos, int NH, int QHD, int QKN, int QKR,
                                      int KVL, int KVD, int slot_stride, float scale) {
    constexpr int KTP  = ((TK + 3) / 4) * 4;         /* KT row stride (bytes)     */
    constexpr int NGRP = HPB * FD_KVL;               /* (head, d) pairs           */
    constexpr int CACC = (NGRP + 255) / 256;
    const int s = blockIdx.x, g = blockIdx.y;
    const int h0 = g * HPB;
    const int tid = threadIdx.x, nt = blockDim.x;
    (void)NH; (void)QKR; (void)KVL; (void)KVD;
    const int kv_len = pos[s] + 1;
    const int8_t *ckvb  = kv_base  + (size_t)s * slot_stride;
    const float  *scl_s = kv_scale + (size_t)s * (slot_stride / FD_KVD);

    __shared__ __attribute__((aligned(16))) int8_t Qsh[HPB * FD_KPAD];
    __shared__ __attribute__((aligned(16))) int8_t Ksh[TK * FD_KPAD];
    __shared__ __attribute__((aligned(16))) int8_t KT[FD_KVL * KTP];   /* [d][key] */
    __shared__ __attribute__((aligned(16))) int8_t P8[HPB * KTP];      /* [h][key] */
    __shared__ float Kscl[TK];
    __shared__ float Ssh[HPB * TK];
    __shared__ float Msh[HPB], Lsh[HPB], Csh[HPB], Qsc[HPB], Psc[HPB];
    __shared__ unsigned Qmx[HPB];

    float oacc[CACC];
#pragma unroll
    for (int e = 0; e < CACC; e++) oacc[e] = 0.0f;
    if (tid < HPB) { Msh[tid] = -INFINITY; Lsh[tid] = 0.0f; Qmx[tid] = 0u; }
    __syncthreads();

    for (int i = tid; i < HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i - h * FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        atomicMax(&Qmx[h], __float_as_uint(fabsf(v)));
    }
    __syncthreads();
    if (tid < HPB) Qsc[tid] = __uint_as_float(Qmx[tid]) / 127.0f;
    __syncthreads();
    for (int i = tid; i < HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i - h * FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        float sc = Qsc[h];
        int qq = (sc > 0.0f) ? (int)rintf(v / sc) : 0;
        qq = qq > 127 ? 127 : (qq < -127 ? -127 : qq);
        Qsh[h * FD_KPAD + d] = (int8_t)qq;
    }
    __syncthreads();

    const int KW = FD_KVD / 4;
    const int KQ = FD_KVD / 16;
    for (int k0 = 0; k0 < kv_len; k0 += TK) {
        const int tk = (kv_len - k0 < TK) ? (kv_len - k0) : TK;
        /* one 16 B global load feeds both the [key][d] tile (score) and the
         * [d][key] transpose (clat). */
        /* run50b: iterate the KEY index fastest. With the d-chunk fastest (the
         * obvious order) consecutive lanes wrote KT at (d0+j)*KTP + r, i.e. 320 B
         * = 80 dwords apart, and 80 % 32 == 16 -> a 16-way bank conflict in the
         * transpose build itself, which is what sank the first measurement. With
         * the key fastest, both the KT write (stride 1) and the Ksh write (stride
         * 580 B = 145 dwords, coprime with 32) are conflict-free, and the global
         * side still reads 64 contiguous bytes per key row per wave. */
        for (int i = tid; i < tk * KQ; i += nt) {
            int c = i / tk, r = i - c * tk;
            int4 v = *reinterpret_cast<const int4 *>(ckvb + (size_t)(k0 + r) * FD_KVD + c * 16);
            int8_t *d = Ksh + (size_t)r * FD_KPAD + c * 16;
            *reinterpret_cast<int *>(d)      = v.x;
            *reinterpret_cast<int *>(d + 4)  = v.y;
            *reinterpret_cast<int *>(d + 8)  = v.z;
            *reinterpret_cast<int *>(d + 12) = v.w;
            int d0 = c * 16;
            if (d0 < FD_KVL) {
#pragma unroll
                for (int j = 0; j < 16; j++) KT[(size_t)(d0 + j) * KTP + r] = d[j];
            }
        }
        for (int j = tid; j < tk; j += nt) Kscl[j] = scl_s[k0 + j];
        __syncthreads();

        for (int hk = tid; hk < HPB * tk; hk += nt) {
            int h = hk / tk, k = hk - h * tk;
            const int *kw = reinterpret_cast<const int *>(Ksh + (size_t)k * FD_KPAD);
            const int *qw = reinterpret_cast<const int *>(Qsh + (size_t)h * FD_KPAD);
            int acc = 0;
#pragma unroll 8
            for (int c = 0; c < KW; c++) acc = __builtin_amdgcn_sdot4(qw[c], kw[c], acc, false);
            Ssh[h * TK + k] = (float)acc * Qsc[h] * Kscl[k] * scale;
        }
        __syncthreads();

        /* online softmax + int8 quantisation of this tile's probabilities. */
        if (tid < HPB) {
            int h = tid;
            float tmax = -INFINITY;
            for (int k = 0; k < tk; k++) tmax = fmaxf(tmax, Ssh[h * TK + k]);
            float mo = Msh[h], mn = fmaxf(mo, tmax);
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            float sum = 0.0f, pmx = 0.0f;
            for (int k = 0; k < tk; k++) {
                float pv = __expf(Ssh[h * TK + k] - mn);
                sum += pv;
                float pw = pv * Kscl[k];          /* fold the per-key int8 scale */
                Ssh[h * TK + k] = pw;
                pmx = fmaxf(pmx, pw);
            }
            float psc = pmx / 127.0f;
            Psc[h] = psc;
            float pinv = (psc > 0.0f) ? 1.0f / psc : 0.0f;
            for (int k = 0; k < KTP; k++) {
                int v = 0;
                if (k < tk) { v = (int)rintf(Ssh[h * TK + k] * pinv); v = v > 127 ? 127 : v; }
                P8[h * KTP + k] = (int8_t)v;      /* zero past tk -> ignored      */
            }
            Lsh[h] = Lsh[h] * corr + sum; Msh[h] = mn; Csh[h] = corr;
        }
        __syncthreads();

        /* clat: one dot4 per 4 keys, straight out of the transposed tile. */
        {
            int idx = tid;
#pragma unroll
            for (int e = 0; e < CACC; e++, idx += nt) {
                if (CACC * 256 != NGRP && idx >= NGRP) break;
                int h = idx >> 9, r = idx & (FD_KVL - 1);
                const int *pw = reinterpret_cast<const int *>(P8 + h * KTP);
                const int *kt = reinterpret_cast<const int *>(KT + (size_t)r * KTP);
                int a = 0;
#pragma unroll
                for (int c = 0; c < KTP / 4; c++) a = __builtin_amdgcn_sdot4(pw[c], kt[c], a, false);
                oacc[e] = oacc[e] * Csh[h] + (float)a * Psc[h];
            }
        }
        __syncthreads();
    }
    {
        int idx = tid;
#pragma unroll
        for (int e = 0; e < CACC; e++, idx += nt) {
            if (CACC * 256 != NGRP && idx >= NGRP) break;
            int h = idx >> 9, r = idx & (FD_KVL - 1), hg = h0 + h;
            float inv = (Lsh[h] > 0.0f) ? 1.0f / Lsh[h] : 0.0f;
            clat[((size_t)s * FD_NH + hg) * FD_KVL + r] = oacc[e] * inv;
        }
    }
}

/* ======================================================================== *
 *  run51: MFMA for the decode clat (user request).
 *
 *  Same transposed tile as run50 -- MFMA's B operand also wants the key index
 *  contiguous, so KT[d][key] is unavoidable either way -- but the accumulation
 *  itself becomes mfma_i32_16x16x16i8: M = 16 heads (HPB=10 padded), N = 16 of
 *  the 512 latent dims, K = the 16 keys of the tile, i.e. exactly ONE MFMA per
 *  (warp, n-tile). Each of the 4 warps owns 128 dims = 8 n-tiles.
 *
 *  Predicted to land level with run50's dot4: per warp per key-tile MFMA needs
 *  8 MFMA (~240 cyc) + 8 B-reads + 1 A-read + the epilogue fold, while dot4
 *  needs 20 x (4 LDS + 4 dot4) = 640 cyc -- but the fold (int32 -> float, times
 *  Psc, times the online-softmax corr, 8 x 4 lanes' worth) has to run EVERY tile
 *  because of the running rescale, and it costs about what the MFMA saves.
 *  Toggle MLA_FDMFMA2=1 (default off until measured).
 * ======================================================================== */
template <int HPB, int TK>
__global__ void attn_fd_hb_mfma2_kernel(float *clat, const float *qabs, const float *q,
                                        const int8_t *kv_base, const float *kv_scale,
                                        const int *pos, int NH, int QHD, int QKN, int QKR,
                                        int KVL, int KVD, int slot_stride, float scale) {
    constexpr int KTP = ((TK + 3) / 4) * 4;
    constexpr int NPW = FD_KVL / 4 / 16;             /* n-tiles per warp = 8      */
    const int s = blockIdx.x, g = blockIdx.y;
    const int h0 = g * HPB;
    const int tid = threadIdx.x, nt = blockDim.x;
    const int warp = tid / 64, tiw = tid % 64;
    const int trow = tiw / 16, tcol = tiw % 16;      /* MFMA lane layout          */
    (void)NH; (void)QKR; (void)KVL; (void)KVD;
    const int kv_len = pos[s] + 1;
    const int8_t *ckvb  = kv_base  + (size_t)s * slot_stride;
    const float  *scl_s = kv_scale + (size_t)s * (slot_stride / FD_KVD);

    __shared__ __attribute__((aligned(16))) int8_t Qsh[HPB * FD_KPAD];
    __shared__ __attribute__((aligned(16))) int8_t Ksh[TK * FD_KPAD];
    __shared__ __attribute__((aligned(16))) int8_t KT[FD_KVL * KTP];
    __shared__ __attribute__((aligned(16))) int8_t P8[16 * KTP];   /* M padded to 16 */
    __shared__ float Kscl[TK];
    __shared__ float Ssh[HPB * TK];
    __shared__ float Msh[HPB], Lsh[HPB], Csh[HPB], Qsc[HPB], Psc[HPB];
    __shared__ unsigned Qmx[HPB];

    float focc[NPW][4];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) focc[j][e] = 0.0f;
    if (tid < HPB) { Msh[tid] = -INFINITY; Lsh[tid] = 0.0f; Qmx[tid] = 0u; }
    for (int i = tid; i < 16 * KTP; i += nt) P8[i] = 0;      /* pad rows 10..15 */
    __syncthreads();

    for (int i = tid; i < HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i - h * FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        atomicMax(&Qmx[h], __float_as_uint(fabsf(v)));
    }
    __syncthreads();
    if (tid < HPB) Qsc[tid] = __uint_as_float(Qmx[tid]) / 127.0f;
    __syncthreads();
    for (int i = tid; i < HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i - h * FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        float sc = Qsc[h];
        int qq = (sc > 0.0f) ? (int)rintf(v / sc) : 0;
        qq = qq > 127 ? 127 : (qq < -127 ? -127 : qq);
        Qsh[h * FD_KPAD + d] = (int8_t)qq;
    }
    __syncthreads();

    const int KW = FD_KVD / 4;
    const int KQ = FD_KVD / 16;
    for (int k0 = 0; k0 < kv_len; k0 += TK) {
        const int tk = (kv_len - k0 < TK) ? (kv_len - k0) : TK;
        for (int i = tid; i < tk * KQ; i += nt) {
            int c = i / tk, r = i - c * tk;           /* key fastest: no conflicts */
            int4 v = *reinterpret_cast<const int4 *>(ckvb + (size_t)(k0 + r) * FD_KVD + c * 16);
            int8_t *d = Ksh + (size_t)r * FD_KPAD + c * 16;
            *reinterpret_cast<int *>(d)      = v.x;
            *reinterpret_cast<int *>(d + 4)  = v.y;
            *reinterpret_cast<int *>(d + 8)  = v.z;
            *reinterpret_cast<int *>(d + 12) = v.w;
            int d0 = c * 16;
            if (d0 < FD_KVL) {
#pragma unroll
                for (int j = 0; j < 16; j++) KT[(size_t)(d0 + j) * KTP + r] = d[j];
            }
        }
        for (int j = tid; j < tk; j += nt) Kscl[j] = scl_s[k0 + j];
        __syncthreads();

        for (int hk = tid; hk < HPB * tk; hk += nt) {
            int h = hk / tk, k = hk - h * tk;
            const int *kw = reinterpret_cast<const int *>(Ksh + (size_t)k * FD_KPAD);
            const int *qw = reinterpret_cast<const int *>(Qsh + (size_t)h * FD_KPAD);
            int acc = 0;
#pragma unroll 8
            for (int c = 0; c < KW; c++) acc = __builtin_amdgcn_sdot4(qw[c], kw[c], acc, false);
            Ssh[h * TK + k] = (float)acc * Qsc[h] * Kscl[k] * scale;
        }
        __syncthreads();

        if (tid < HPB) {
            int h = tid;
            float tmax = -INFINITY;
            for (int k = 0; k < tk; k++) tmax = fmaxf(tmax, Ssh[h * TK + k]);
            float mo = Msh[h], mn = fmaxf(mo, tmax);
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            float sum = 0.0f, pmx = 0.0f;
            for (int k = 0; k < tk; k++) {
                float pv = __expf(Ssh[h * TK + k] - mn);
                sum += pv;
                float pw = pv * Kscl[k];
                Ssh[h * TK + k] = pw;
                pmx = fmaxf(pmx, pw);
            }
            float psc = pmx / 127.0f;
            Psc[h] = psc;
            float pinv = (psc > 0.0f) ? 1.0f / psc : 0.0f;
            for (int k = 0; k < KTP; k++) {
                int v = 0;
                if (k < tk) { v = (int)rintf(Ssh[h * TK + k] * pinv); v = v > 127 ? 127 : v; }
                P8[h * KTP + k] = (int8_t)v;
            }
            Lsh[h] = Lsh[h] * corr + sum; Msh[h] = mn; Csh[h] = corr;
        }
        __syncthreads();

        /* clat via MFMA: one mfma_i32_16x16x16i8 per (warp, n-tile). */
        {
            const int a = *reinterpret_cast<const int *>(&P8[tcol * KTP + trow * 4]);
            float cr[4], ps[4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int h = trow * 4 + e;
                cr[e] = (h < HPB) ? Csh[h] : 0.0f;
                ps[e] = (h < HPB) ? Psc[h] : 0.0f;
            }
#pragma unroll
            for (int j = 0; j < NPW; j++) {
                int n0 = warp * (FD_KVL / 4) + j * 16;
                const int b = *reinterpret_cast<const int *>(&KT[(size_t)(n0 + tcol) * KTP + trow * 4]);
                i32x4b_attn acc = {};
                acc = __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, acc, 0, 0, 0);
#pragma unroll
                for (int e = 0; e < 4; e++)
                    focc[j][e] = focc[j][e] * cr[e] + (float)acc[e] * ps[e];
            }
        }
        __syncthreads();
    }
    {
        float iv[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int h = trow * 4 + e;
            iv[e] = (h < HPB && Lsh[h] > 0.0f) ? 1.0f / Lsh[h] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < NPW; j++) {
            int r = warp * (FD_KVL / 4) + j * 16 + tcol;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int h = trow * 4 + e;
                if (h < HPB)
                    clat[((size_t)s * FD_NH + h0 + h) * FD_KVL + r] = focc[j][e] * iv[e];
            }
        }
    }
}

/* run38: ILP variant of attn_flash_decode_hb_kernel. run26 profile showed the
 * head-batched decode is LATENCY/occupancy-bound (VALU 43.6% NOT saturated, occ
 * 22%, IPC 0.63) -- not compute-bound (so run27's MFMA-score lost). The dominant
 * VALU work is the per-(head,key) score dot: a KVD=576-long reduction into ONE
 * accumulator, i.e. a 576-deep serial FMA dependency chain that stalls on FMA
 * latency (low IPC). This breaks that chain with 4 independent accumulators (ILP)
 * so 4 FMAs are in flight -- pure latency relief, no MFMA, no extra LDS, no
 * occupancy change. Same idea on the clat tk-reduction (32-deep). NOT bit-identical
 * (fp add reassociation), but numerically equivalent (accuracy-safe, gate-checked).
 * Toggle MLA_FDILP=0 reverts to the single-accumulator attn_flash_decode_hb_kernel. */
__global__ void attn_flash_decode_hb_ilp_kernel(float *clat, const float *qabs, const float *q,
                                            const int8_t *kv_base, const float *kv_scale,
                                            const int *pos, int NH, int QHD, int QKN, int QKR,
                                            int KVL, int KVD, int slot_stride, float scale) {
    const int s = blockIdx.x, g = blockIdx.y;
    const int h0 = g * FD_HPB;
    const int tid = threadIdx.x, nt = blockDim.x;
    (void)NH; (void)QKR; (void)KVL; (void)KVD;
    const int kv_len = pos[s] + 1;
    const int8_t *ckvb  = kv_base  + (size_t)s * slot_stride;
    const float  *scl_s = kv_scale + (size_t)s * (slot_stride / FD_KVD);

    __shared__ __bf16 Qsh[FD_HPB * FD_KVD];
    __shared__ int8_t Ksh[FD_TK * FD_KVD];
    __shared__ float  Kscl[FD_TK];
    __shared__ float  Ssh[FD_HPB * FD_TK];
    __shared__ float  Msh[FD_HPB], Lsh[FD_HPB], Csh[FD_HPB];

    float oacc[FD_OACC];
#pragma unroll
    for (int e = 0; e < FD_OACC; e++) oacc[e] = 0.0f;
    if (tid < FD_HPB) { Msh[tid] = -INFINITY; Lsh[tid] = 0.0f; }

    for (int i = tid; i < FD_HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i % FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        Qsh[i] = (__bf16)v;
    }
    __syncthreads();

    for (int k0 = 0; k0 < kv_len; k0 += FD_TK) {
        const int tk = (kv_len - k0 < FD_TK) ? (kv_len - k0) : FD_TK;
        for (int i = tid; i < tk * FD_KVD; i += nt) Ksh[i] = ckvb[(size_t)k0 * FD_KVD + i];
        for (int j = tid; j < tk; j += nt) Kscl[j] = scl_s[k0 + j];
        __syncthreads();
        /* scores: 4-way ILP over the KVD=576 reduction (breaks the serial FMA chain). */
        for (int hk = tid; hk < FD_HPB * tk; hk += nt) {
            int h = hk / tk, k = hk % tk;
            const __bf16 *qh = Qsh + (size_t)h * FD_KVD;
            const int8_t *kk = Ksh + (size_t)k * FD_KVD;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll 4
            for (int i = 0; i < FD_KVD; i += 4) {   /* FD_KVD=576 divisible by 4 */
                d0 += (float)qh[i]     * (float)kk[i];
                d1 += (float)qh[i + 1] * (float)kk[i + 1];
                d2 += (float)qh[i + 2] * (float)kk[i + 2];
                d3 += (float)qh[i + 3] * (float)kk[i + 3];
            }
            Ssh[h * FD_TK + k] = ((d0 + d1) + (d2 + d3)) * Kscl[k] * scale;
        }
        __syncthreads();
        if (tid < FD_HPB) {
            int h = tid;
            float tmax = -INFINITY;
            for (int k = 0; k < tk; k++) tmax = fmaxf(tmax, Ssh[h * FD_TK + k]);
            float mo = Msh[h], mn = fmaxf(mo, tmax);
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            float sum = 0.0f;
            for (int k = 0; k < tk; k++) { float p = __expf(Ssh[h * FD_TK + k] - mn); Ssh[h * FD_TK + k] = p; sum += p; }
            Lsh[h] = Lsh[h] * corr + sum; Msh[h] = mn; Csh[h] = corr;
        }
        __syncthreads();
        int idx = tid;
#pragma unroll
        for (int e = 0; e < FD_OACC; e++, idx += nt) {
            if (idx < FD_HPB * FD_KVL) {
                int h = idx / FD_KVL, r = idx % FD_KVL;
                float a0 = 0.0f, a1 = 0.0f;
                int k = 0;
                for (; k + 1 < tk; k += 2) {
                    a0 += Ssh[h * FD_TK + k]     * Kscl[k]     * (float)Ksh[(size_t)k * FD_KVD + r];
                    a1 += Ssh[h * FD_TK + k + 1] * Kscl[k + 1] * (float)Ksh[(size_t)(k + 1) * FD_KVD + r];
                }
                for (; k < tk; k++)
                    a0 += Ssh[h * FD_TK + k] * Kscl[k] * (float)Ksh[(size_t)k * FD_KVD + r];
                oacc[e] = oacc[e] * Csh[h] + (a0 + a1);
            }
        }
        __syncthreads();
    }
    int idx = tid;
#pragma unroll
    for (int e = 0; e < FD_OACC; e++, idx += nt) {
        if (idx < FD_HPB * FD_KVL) {
            int h = idx / FD_KVL, r = idx % FD_KVL, hg = h0 + h;
            float inv = (Lsh[h] > 0.0f) ? 1.0f / Lsh[h] : 0.0f;
            clat[((size_t)s * FD_NH + hg) * FD_KVL + r] = oacc[e] * inv;
        }
    }
}

/* run27: MFMA-score variant of attn_flash_decode_hb_kernel. Profile @ B=1024
 * (report/run26) showed the head-batched decode is now VALU-BOUND (VALU 43.6%,
 * MFMA 0%, L2 2.6% -- run23 head-batching killed the L2-read bound): the scalar
 * per-(head,key) KVD=576 dot is the bottleneck. This offloads that score GEMM
 * S[HPB,tk] = Q[HPB,KVD] @ K[tk,KVD]^T onto the matrix cores via
 * mfma_f32_16x16x16bf16 (M=16 pad HPB=10, N=tk in 16-key tiles, K=KVD in 16 steps).
 * Q is already bf16 in Qsh; the int8 c_kv is cast to bf16 on the fly (exact,
 * |v|<=127) and read straight from Ksh -- NO extra LDS. Softmax + clat stay scalar
 * (separate lever). Numerically ~identical to the scalar path (Q bf16 both sides,
 * int8->bf16 exact). Toggle MLA_FDMFMA=0 reverts to attn_flash_decode_hb_kernel. */
__global__ void attn_flash_decode_hb_mfma_kernel(float *clat, const float *qabs, const float *q,
                                                 const int8_t *kv_base, const float *kv_scale,
                                                 const int *pos, int NH, int QHD, int QKN, int QKR,
                                                 int KVL, int KVD, int slot_stride, float scale) {
    const int s = blockIdx.x, g = blockIdx.y;
    const int h0 = g * FD_HPB;
    const int tid = threadIdx.x, nt = blockDim.x;
    (void)NH; (void)QKR; (void)KVL; (void)KVD;
    const int kv_len = pos[s] + 1;
    const int8_t *ckvb  = kv_base  + (size_t)s * slot_stride;
    const float  *scl_s = kv_scale + (size_t)s * (slot_stride / FD_KVD);

    __shared__ __bf16 Qsh[FD_HPB * FD_KVD];
    __shared__ int8_t Ksh[FD_TK * FD_KVD];
    __shared__ float  Kscl[FD_TK];
    __shared__ float  Ssh[FD_HPB * FD_TK];
    __shared__ float  Msh[FD_HPB], Lsh[FD_HPB], Csh[FD_HPB];

    float oacc[FD_OACC];
#pragma unroll
    for (int e = 0; e < FD_OACC; e++) oacc[e] = 0.0f;
    if (tid < FD_HPB) { Msh[tid] = -INFINITY; Lsh[tid] = 0.0f; }

    for (int i = tid; i < FD_HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i % FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        Qsh[i] = (__bf16)v;
    }
    __syncthreads();

    const int warp = tid / 64, tiw = tid % 64;
    const int trow = tiw / 16, tcol = tiw % 16;   /* MFMA lane layout */

    for (int k0 = 0; k0 < kv_len; k0 += FD_TK) {
        const int tk = (kv_len - k0 < FD_TK) ? (kv_len - k0) : FD_TK;
        for (int i = tid; i < tk * FD_KVD; i += nt) Ksh[i] = ckvb[(size_t)k0 * FD_KVD + i];
        for (int j = tid; j < tk; j += nt) Kscl[j] = scl_s[k0 + j];
        __syncthreads();
        /* --- score via MFMA: warp w owns key-tile w (16 keys). S[head, key] --- */
        const int ntiles = (tk + 15) / 16;                 /* <= FD_TK/16 (=2) */
        if (warp < ntiles) {
            const int key_base = warp * 16;
            float4va sacc = {0.f, 0.f, 0.f, 0.f};
            for (int kk0 = 0; kk0 < FD_KVD; kk0 += 16) {
                bf16x4a a, b;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int d = kk0 + trow * 4 + e;             /* contraction index */
                    int m = tcol;                          /* head (M) */
                    a[e] = (m < FD_HPB) ? Qsh[(size_t)m * FD_KVD + d] : (__bf16)0.0f;
                    int key = key_base + tcol;             /* key (N) */
                    b[e] = (key < tk) ? (__bf16)(float)Ksh[(size_t)key * FD_KVD + d] : (__bf16)0.0f;
                }
                sacc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, sacc, 0, 0, 0);
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int head = trow * 4 + e, key = key_base + tcol;
                if (head < FD_HPB && key < tk)
                    Ssh[head * FD_TK + key] = sacc[e] * Kscl[key] * scale;
            }
        }
        __syncthreads();
        /* online softmax per head (one thread per head). */
        if (tid < FD_HPB) {
            int h = tid;
            float tmax = -INFINITY;
            for (int k = 0; k < tk; k++) tmax = fmaxf(tmax, Ssh[h * FD_TK + k]);
            float mo = Msh[h], mn = fmaxf(mo, tmax);
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            float sum = 0.0f;
            for (int k = 0; k < tk; k++) { float p = __expf(Ssh[h * FD_TK + k] - mn); Ssh[h * FD_TK + k] = p; sum += p; }
            Lsh[h] = Lsh[h] * corr + sum; Msh[h] = mn; Csh[h] = corr;
        }
        __syncthreads();
        /* clat: O[h][r] = O[h][r]*corr[h] + sum_k P[h,k]*scl[k]*ckv[k,r]. */
        int idx = tid;
#pragma unroll
        for (int e = 0; e < FD_OACC; e++, idx += nt) {
            if (idx < FD_HPB * FD_KVL) {
                int h = idx / FD_KVL, r = idx % FD_KVL;
                float a = 0.0f;
                for (int k = 0; k < tk; k++)
                    a += Ssh[h * FD_TK + k] * Kscl[k] * (float)Ksh[(size_t)k * FD_KVD + r];
                oacc[e] = oacc[e] * Csh[h] + a;
            }
        }
        __syncthreads();
    }
    int idx = tid;
#pragma unroll
    for (int e = 0; e < FD_OACC; e++, idx += nt) {
        if (idx < FD_HPB * FD_KVL) {
            int h = idx / FD_KVL, r = idx % FD_KVL, hg = h0 + h;
            float inv = (Lsh[h] > 0.0f) ? 1.0f / Lsh[h] : 0.0f;
            clat[((size_t)s * FD_NH + hg) * FD_KVL + r] = oacc[e] * inv;
        }
    }
}


/* ======================================================================== *
 * Retired GEMM / quantisation kernels (run4-run52). The engine now runs one
 * 128x128 int8 GEMM family (matmul_mfma_i8i8_128_w / moe_grouped_..._128_w,
 * wide int4 loads + 32-K double-buffered LDS tile) with the 64x64 kernels as
 * the small-shape fallback. int4/W4A8, TBK=64 and gate+up fusion all measured
 * negative -- see report/run39_bankconflict_widegemm/. NOT compiled.
 * ======================================================================== */
/* ======================================================================== *
 *  run21: 128x128-tile variant of matmul_mfma_i8i8_kernel. Same math/output
 *  (Y = (Xq @ Wq^T) * xsc[row] * wsc[col]) but each block computes a 128x128
 *  output tile instead of 64x64. For a square GEMM the L2 traffic through the
 *  weight and through the activation are EQUAL (weight = N*K*M/TBM, act =
 *  M*K*N/TBN), so with square tiles both scale as 1/tile: doubling BOTH tile
 *  dims halves TOTAL L2 traffic -- the exact relief for the L2-BW-bound dense
 *  GEMM at B=1024 (only worthwhile when M,N are large enough to still fill the
 *  104 CUs at the coarser 128x128 grid; small-N projections stay on the 64x64
 *  kernel). block = 256 (4 warps, 2x2 warp grid, each warp a 64x64 = 4x4 grid
 *  of 16x16 MFMA tiles). grid = (ceil(N/128), ceil(M/128)). K a multiple of 16.
 * ======================================================================== */
__global__ void matmul_mfma_i8i8_128_kernel(float *Y, const int8_t *Xq, const float *xsc,
                                            const int8_t *W, const float *wsc, int M, int N,
                                            int K, int x_stride, int y_stride) {
    __shared__ int8_t As[TBM2][TBK];   /* [m][k], k contiguous */
    __shared__ int8_t Bs[TBN2][TBK];   /* [n][k], k contiguous */
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;   /* 2x2 warp grid, 64x64 each */
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM2, col0 = blockIdx.x * TBN2;

    i32x4b acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        /* A tile: 128 rows, TBK=16 -> 512 int-loads / 256 threads = 2 passes. */
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            int v = 0;
            if (gr < M) v = *reinterpret_cast<const int *>(&Xq[(size_t)gr * x_stride + k0 + cc]);
            *reinterpret_cast<int *>(&As[m][cc]) = v;
        }
#pragma unroll
        for (int r = 0; r < TBN2; r += 64) {
            int n = r + tx / 4, cc = (tx % 4) * 4, gr = col0 + n;
            int v = 0;
            if (gr < N) v = *reinterpret_cast<const int *>(&W[(size_t)gr * K + k0 + cc]);
            *reinterpret_cast<int *>(&Bs[n][cc]) = v;
        }
        __syncthreads();
        int a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_row = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float wc = wsc[col];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = base_row + e;
                if (row < M) Y[(size_t)row * y_stride + col] = (float)acc[i][j][e] * wc * xsc[row];
            }
        }
    }
}

/* ======================================================================== *
 *  run40: DOUBLE-BUFFERED 128x128 W8A8 dense GEMM (bit-identical to
 *  matmul_mfma_i8i8_128_kernel -- same tiles, same K order, same MFMA calls).
 *
 *  The run21/24 kernels run the K-loop as
 *      global load -> LDS store -> sync -> 8x ds_read + 16x MFMA -> sync
 *  so (a) there are TWO barriers per K-step and (b) the global load for step
 *  k+1 cannot issue until every wave has cleared the second barrier of step k,
 *  leaving the ~800-cycle L2/HBM latency fully exposed in front of only ~485
 *  cycles of MFMA. Measured effect: both hot GEMMs sit at ~31% of MFMA peak
 *  (prefill MoE 17.8 s + dense 13.1 s) even though their L2 traffic accounts
 *  for only ~14% of that time -- i.e. LATENCY-bound, not bandwidth-bound
 *  (which is why run24's "pipelining is useless, it's L2-BW-bound" call was
 *  wrong, and why run31's wider loads did not help either).
 *
 *  Fix: two LDS tile buffers. Each iteration issues the NEXT tile's global
 *  loads into registers first, runs the 16 MFMAs on the current buffer while
 *  those loads are in flight, then parks the registers in the other buffer.
 *  One barrier per K-step instead of two, and the memory latency is covered by
 *  the MFMA block. LDS grows 4KB -> 8KB per block (occupancy unchanged: these
 *  kernels are VGPR-limited by acc[4][4]). Toggle MLA_GEMMDB=0 -> run21/24.
 * ======================================================================== */
__global__ void matmul_mfma_i8i8_128_db_kernel(float *Y, const int8_t *Xq, const float *xsc,
                                               const int8_t *W, const float *wsc, int M, int N,
                                               int K, int x_stride, int y_stride) {
    __shared__ int8_t As[2][TBM2][TBK];
    __shared__ int8_t Bs[2][TBN2][TBK];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM2, col0 = blockIdx.x * TBN2;
    const int m0 = tx / 4, cc = (tx % 4) * 4;      /* this thread's tile slot */

    /* prologue: stage k0 = 0 into buffer 0. */
#pragma unroll
    for (int r = 0; r < TBM2; r += 64) {
        int gr = row0 + r + m0; int v = 0;
        if (gr < M) v = *reinterpret_cast<const int *>(&Xq[(size_t)gr * x_stride + cc]);
        *reinterpret_cast<int *>(&As[0][r + m0][cc]) = v;
    }
#pragma unroll
    for (int r = 0; r < TBN2; r += 64) {
        int gr = col0 + r + m0; int v = 0;
        if (gr < N) v = *reinterpret_cast<const int *>(&W[(size_t)gr * K + cc]);
        *reinterpret_cast<int *>(&Bs[0][r + m0][cc]) = v;
    }
    __syncthreads();

    i32x4b acc[4][4] = {};
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += TBK, buf ^= 1) {
        const int kn = k0 + TBK;
        /* 1. issue the next tile's global loads (in flight during the MFMAs). */
        int pa[TBM2 / 64], pb[TBN2 / 64];
        if (kn < K) {
#pragma unroll
            for (int r = 0, i = 0; r < TBM2; r += 64, i++) {
                int gr = row0 + r + m0; pa[i] = 0;
                if (gr < M) pa[i] = *reinterpret_cast<const int *>(&Xq[(size_t)gr * x_stride + kn + cc]);
            }
#pragma unroll
            for (int r = 0, i = 0; r < TBN2; r += 64, i++) {
                int gr = col0 + r + m0; pb[i] = 0;
                if (gr < N) pb[i] = *reinterpret_cast<const int *>(&W[(size_t)gr * K + kn + cc]);
            }
        }
        /* 2. MFMA on the buffer staged last round. */
        int a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[buf][srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[buf][scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        /* 3. park the prefetched tile in the other buffer; one barrier closes it. */
        if (kn < K) {
#pragma unroll
            for (int r = 0, i = 0; r < TBM2; r += 64, i++)
                *reinterpret_cast<int *>(&As[buf ^ 1][r + m0][cc]) = pa[i];
#pragma unroll
            for (int r = 0, i = 0; r < TBN2; r += 64, i++)
                *reinterpret_cast<int *>(&Bs[buf ^ 1][r + m0][cc]) = pb[i];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_row = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float wc = wsc[col];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = base_row + e;
                if (row < M) Y[(size_t)row * y_stride + col] = (float)acc[i][j][e] * wc * xsc[row];
            }
        }
    }
}

/* run33: BLOCK-WISE W4A8 dense GEMM. Packed int4 weight [N][K/2] with a
 * per-(row,K-group) scale wsc=[N][K/GW4] (GW4=128). Same structure as
 * matmul_mfma_i8i8_128_kernel but two-level accumulation: int32 mfma within a
 * group (weight pure int4), flush (float)acc * wsc[col][g] into a float
 * accumulator at each group boundary, reset int32 acc, continue. The int8
 * activation scale xsc is per-row -> factors out to the epilogue. K % GW4 == 0.
 * Doubles the float-accumulator registers (occupancy risk, see run32). MLA_DENSE_W4A8_BW=1. */
__global__ void matmul_mfma_i8i8_bw_128_kernel(float *Y, const int8_t *Xq, const float *xsc,
                                               const int8_t *W, const float *wsc, int M, int N,
                                               int K, int x_stride, int y_stride) {
    __shared__ int8_t As[TBM2][TBK];
    __shared__ int8_t Bs[TBN2][TBK];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM2, col0 = blockIdx.x * TBN2;
    const int Kp = K / 2, ng = K / GW4;

    i32x4b acc[4][4] = {};
    float  facc[4][4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            int v = 0;
            if (gr < M) v = *reinterpret_cast<const int *>(&Xq[(size_t)gr * x_stride + k0 + cc]);
            *reinterpret_cast<int *>(&As[m][cc]) = v;
        }
#pragma unroll
        for (int r = 0; r < TBN2; r += 64) {
            int n = r + tx / 4, cc = (tx % 4) * 4, gr = col0 + n;
            unsigned pk = 0;
            if (gr < N) pk = *reinterpret_cast<const unsigned short *>(&W[(size_t)gr * Kp + ((k0 + cc) >> 1)]);
            Bs[n][cc + 0] = (int8_t)i4sext(pk & 0xF);
            Bs[n][cc + 1] = (int8_t)i4sext((pk >> 4) & 0xF);
            Bs[n][cc + 2] = (int8_t)i4sext((pk >> 8) & 0xF);
            Bs[n][cc + 3] = (int8_t)i4sext((pk >> 12) & 0xF);
        }
        __syncthreads();
        int a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
        if ((k0 + TBK) % GW4 == 0) {
            int g = (k0 + TBK) / GW4 - 1;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int col = col0 + scol + j * 16 + tcol;
                float sc = (col < N) ? wsc[(size_t)col * ng + g] : 0.0f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int e = 0; e < 4; e++)
                        facc[i][j][e] += (float)acc[i][j][e] * sc;
                    acc[i][j] = i32x4b{};
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int base_row = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = base_row + e;
                if (row < M) Y[(size_t)row * y_stride + col] = facc[i][j][e] * xsc[row];
            }
        }
    }
}

/* ======================================================================== *
 *  MFMA per-head GEMM with SHARED X, per-head W/Y (head = blockIdx.z). Same math
 *  as matmul_mfma_kernel but W and the output are offset by head while X (the
 *  common input) is not: Y[h, m, n] = sum_k X[m*x_stride + k] * W[h*w_hstride + n*K + k].
 *  Used to fold the unabsorbed-prefill per-head loops knope[h] = kvlin @ W_UK[h]^T
 *  and value[h] = kvlin @ W_UV[h]^T (kvlin shared across heads) from 2*NH launches
 *  into 2 launches. Y is contiguous per head (row stride N, head stride y_hstride).
 *  grid = (ceil(N/64), ceil(M/64), NH), block = 256. K a multiple of 16.
 * ======================================================================== */
__global__ void matmul_headw_mfma_kernel(float *Y, const float *X, const bf16_t *W,
                                         int M, int N, int K, int x_stride,
                                         int w_hstride, int y_hstride) {
    int h = blockIdx.z;
    const bf16_t *Wh = W + (size_t)h * w_hstride;
    float        *Yh = Y + (size_t)h * y_hstride;

    __shared__ __bf16 As[TBK][TBM + 4];   /* run41: +4 pad removes the LDS bank conflict on the MFMA operand read (stride 64 bf16 = 32 dwords -> bank = col/2 only -> 4-way). */
    __shared__ __bf16 Bs[TBK][TBN + 4];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int wr = warp / 2, wc = warp % 2;
    int srow = wr * 32, scol = wc * 32;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM, col0 = blockIdx.x * TBN;

    float4v acc[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        /* A tile: X[row0+m][k0+k] -> As[k][m] (shared across heads). */
        {
            int m = tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gr < M) v = *reinterpret_cast<const float4 *>(&X[(size_t)gr * x_stride + k0 + cc]);
            As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
            As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
        }
        /* B tile: Wh[col0+n][k0+k] -> Bs[k][n], bf16 straight in. */
        for (int idx = tx; idx < TBN * TBK / 8; idx += 256) {
            int n = idx / (TBK / 8), cc = (idx % (TBK / 8)) * 8, gr = col0 + n;
            uint16_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &Wh[(size_t)gr * K + k0 + cc], 8 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 8; j++)
                Bs[cc + j][n] = (gr < N) ? *reinterpret_cast<const __bf16 *>(&wb[j]) : (__bf16)0.0f;
        }
        __syncthreads();
        bf16x4 a[2], b[2];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) a[i][e] = As[trow * 4 + e][srow + i * 16 + tcol];
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) b[j][e] = Bs[trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
                acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int base_row = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = base_row + e;
                if (row < M) Yh[(size_t)row * N + col] = acc[i][j][e];
            }
        }
    }
}

/* run4: int8 weight-only version of moe_grouped_mfma_kernel. Same tiling / gather
 * / expert-offset semantics, but W_e is int8 [N][K] with a per-output-row fp32
 * scale S_e[N]. Weight int8 values load straight into the bf16 LDS (exact, since
 * |q|<=127 < 256), the MFMA runs bf16 x bf16 -> fp32, and the scale is folded in
 * once at the output write. HBM weight bytes are halved vs the bf16 kernel. */
__global__ void moe_grouped_mfma_i8_kernel(float *out_sorted, const float *X,
                                           const int8_t *const *Wtbl,
                                           const float *const *Stbl,
                                           const int *eoff, const int *sorted_slot,
                                           int N, int K, int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM;
    if (row0 >= cnt) return;
    const int8_t *W = Wtbl[e];
    const float  *S = Stbl[e];

    __shared__ __bf16 As[TBK][TBM + 4];
    __shared__ __bf16 Bs[TBK][TBN + 4];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int wr = warp / 2, wc = warp % 2;
    int srow = wr * 32, scol = wc * 32;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN;

    float4v acc[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        {
            int m = tx / 4, cc = (tx % 4) * 4, p = row0 + m;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (p < cnt) {
                int sr = gather ? sorted_slot[base + p] : (base + p);
                v = *reinterpret_cast<const float4 *>(&X[(size_t)sr * x_stride + k0 + cc]);
            }
            As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
            As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
        }
        /* B tile: int8 weight -> bf16 (exact int value), 8-wide (8 bytes). */
        for (int idx = tx; idx < TBN * TBK / 8; idx += 256) {
            int n = idx / (TBK / 8), cc = (idx % (TBK / 8)) * 8, gr = col0 + n;
            int8_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &W[(size_t)gr * K + k0 + cc], 8);
#pragma unroll
            for (int j = 0; j < 8; j++)
                Bs[cc + j][n] = (gr < N) ? (__bf16)(float)wb[j] : (__bf16)0.0f;
        }
        __syncthreads();
        bf16x4 a[2], b[2];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) a[i][e2] = As[trow * 4 + e2][srow + i * 16 + tcol];
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) b[j][e2] = Bs[trow * 4 + e2][scol + j * 16 + tcol];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
                acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int prow = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float sc = S[col];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow + e2;
                if (p < cnt) out_sorted[(size_t)(base + p) * N + col] = acc[i][j][e2] * sc;
            }
        }
    }
}

/* W8A8 grouped expert GEMM. Xq = int8 activations [rows][K] (row stride x_stride)
 * with per-row scale xsc[]; W = int8 weight [N][K] with per-output-row scale S[N].
 * out[base+p][col] = (Σ_k Xq[sr][k]*W[col][k]) * xsc[sr] * S[col].  As2/Bs2 store
 * int8 with K contiguous so each MFMA operand is one packed-int load; no converts.
 * grid = (ceil(N/TBN), ceil(rows/TBM), n_routed), block = 256. */
__global__ void moe_grouped_mfma_i8i8_kernel(float *out_sorted, const int8_t *Xq,
                                             const float *xsc, const int8_t *const *Wtbl,
                                             const float *const *Stbl, const int *eoff,
                                             const int *sorted_slot, int N, int K,
                                             int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM;
    if (row0 >= cnt) return;
    const int8_t *W = Wtbl[e];
    const float  *S = Stbl[e];

    __shared__ int8_t As2[TBM][TBK];   /* [m][k], k contiguous */
    __shared__ int8_t Bs2[TBN][TBK];   /* [n][k], k contiguous */
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int wr = warp / 2, wc = warp % 2;
    int srow = wr * 32, scol = wc * 32;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN;

    i32x4 acc[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        {   /* A tile: Xq[sr][k0+cc..+3] -> As2[m][cc..+3] (one int copy) */
            int m = tx / 4, cc = (tx % 4) * 4, p = row0 + m;
            int v = 0;
            if (p < cnt) {
                int sr = gather ? sorted_slot[base + p] : (base + p);
                v = *reinterpret_cast<const int *>(&Xq[(size_t)sr * x_stride + k0 + cc]);
            }
            *reinterpret_cast<int *>(&As2[m][cc]) = v;
        }
        {   /* B tile: W[col0+n][k0+cc..+3] -> Bs2[n][cc..+3] (one int copy) */
            int n = tx / 4, cc = (tx % 4) * 4, gr = col0 + n;
            int v = 0;
            if (gr < N) v = *reinterpret_cast<const int *>(&W[(size_t)gr * K + k0 + cc]);
            *reinterpret_cast<int *>(&Bs2[n][cc]) = v;
        }
        __syncthreads();
        int a[2], b[2];
#pragma unroll
        for (int i = 0; i < 2; i++) a[i] = *reinterpret_cast<const int *>(&As2[srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 2; j++) b[j] = *reinterpret_cast<const int *>(&Bs2[scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int prow = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float wsc = S[col];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow + e2;
                if (p < cnt) {
                    int sr = gather ? sorted_slot[base + p] : (base + p);
                    out_sorted[(size_t)(base + p) * N + col] = (float)acc[i][j][e2] * wsc * xsc[sr];
                }
            }
        }
    }
}

/* run24: 128x128-tile variant of the W8A8 grouped MoE GEMM. Profile @ B=1024
 * showed moe_grouped_mfma_i8i8_kernel L2-BANDWIDTH-BOUND (L2 BW 133.7% peak, MFMA
 * only 24%); the 64x64 tile re-reads the activation ceil(N/64) times. The coarse
 * 128x128 tile halves those re-reads (act read ceil(N/128) times), the same L2
 * relief run21 gave the dense GEMM -- extended here to the MoE grouped GEMM (the
 * DeepGEMM large-tile idea). Identical epilogue to moe_grouped_mfma_i8i8_kernel
 * (expert gather via sorted_slot, scale S[col]*xsc[sr]); acc[4][4], 2x2 warp grid.
 * A row-block beyond an expert's cnt returns early -- M-waste when cnt<128 costs
 * only idle MFMA rows (MFMA is 24% util, not the bound), no extra L2 bytes. */
__global__ void moe_grouped_mfma_i8i8_128_kernel(float *out_sorted, const int8_t *Xq,
                                                 const float *xsc, const int8_t *const *Wtbl,
                                                 const float *const *Stbl, const int *eoff,
                                                 const int *sorted_slot, int N, int K,
                                                 int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM2;
    if (row0 >= cnt) return;
    const int8_t *W = Wtbl[e];
    const float  *S = Stbl[e];

    __shared__ int8_t As[TBM2][TBK];   /* [m][k], k contiguous */
    __shared__ int8_t Bs[TBN2][TBK];   /* [n][k], k contiguous */
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;   /* 2x2 warp grid, 64x64 each */
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN2;

    i32x4b acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        /* A tile: 128 gathered rows, TBK=16 -> 512 int-loads / 256 threads = 2 passes. */
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 4, p = row0 + m;
            int v = 0;
            if (p < cnt) {
                int sr = gather ? sorted_slot[base + p] : (base + p);
                v = *reinterpret_cast<const int *>(&Xq[(size_t)sr * x_stride + k0 + cc]);
            }
            *reinterpret_cast<int *>(&As[m][cc]) = v;
        }
#pragma unroll
        for (int r = 0; r < TBN2; r += 64) {
            int n = r + tx / 4, cc = (tx % 4) * 4, gr = col0 + n;
            int v = 0;
            if (gr < N) v = *reinterpret_cast<const int *>(&W[(size_t)gr * K + k0 + cc]);
            *reinterpret_cast<int *>(&Bs[n][cc]) = v;
        }
        __syncthreads();
        int a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int prow_base = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float wc = S[col];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow_base + e2;
                if (p < cnt) {
                    int sr = gather ? sorted_slot[base + p] : (base + p);
                    out_sorted[(size_t)(base + p) * N + col] = (float)acc[i][j][e2] * wc * xsc[sr];
                }
            }
        }
    }
}

/* ======================================================================== *
 *  run40: DOUBLE-BUFFERED 128x128 grouped MoE GEMM. Bit-identical to
 *  moe_grouped_mfma_i8i8_128_kernel (same tiles, same K order, same MFMAs);
 *  see the dense twin above for why the two-barrier / no-prefetch K-loop left
 *  these kernels latency-bound at ~31% of MFMA peak.
 *
 *  Two extra wins specific to the grouped kernel:
 *   - the gathered row index sorted_slot[base+p] does NOT depend on k0, but the
 *     original loop re-loaded it from global memory on every one of the K/16 =
 *     128 iterations; it is hoisted into a register here.
 *   - the same hoisted index serves the epilogue's xsc[sr] lookup.
 *  Toggle MLA_GEMMDB=0 reverts to moe_grouped_mfma_i8i8_128_kernel.
 * ======================================================================== */
__global__ void moe_grouped_mfma_i8i8_128_db_kernel(float *out_sorted, const int8_t *Xq,
                                                    const float *xsc, const int8_t *const *Wtbl,
                                                    const float *const *Stbl, const int *eoff,
                                                    const int *sorted_slot, int N, int K,
                                                    int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM2;
    if (row0 >= cnt) return;
    const int8_t *W = Wtbl[e];
    const float  *S = Stbl[e];

    __shared__ int8_t As[2][TBM2][TBK];
    __shared__ int8_t Bs[2][TBN2][TBK];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN2;
    const int m0 = tx / 4, cc = (tx % 4) * 4;

    /* hoist the gather indices: they are k-invariant (2 rows per thread). */
    size_t arow[TBM2 / 64];
#pragma unroll
    for (int r = 0, i = 0; r < TBM2; r += 64, i++) {
        int p = row0 + r + m0;
        arow[i] = (size_t)-1;
        if (p < cnt) arow[i] = (size_t)(gather ? sorted_slot[base + p] : (base + p));
    }

    /* prologue: stage k0 = 0 into buffer 0. */
#pragma unroll
    for (int r = 0, i = 0; r < TBM2; r += 64, i++) {
        int v = 0;
        if (arow[i] != (size_t)-1) v = *reinterpret_cast<const int *>(&Xq[arow[i] * x_stride + cc]);
        *reinterpret_cast<int *>(&As[0][r + m0][cc]) = v;
    }
#pragma unroll
    for (int r = 0; r < TBN2; r += 64) {
        int gr = col0 + r + m0; int v = 0;
        if (gr < N) v = *reinterpret_cast<const int *>(&W[(size_t)gr * K + cc]);
        *reinterpret_cast<int *>(&Bs[0][r + m0][cc]) = v;
    }
    __syncthreads();

    i32x4b acc[4][4] = {};
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += TBK, buf ^= 1) {
        const int kn = k0 + TBK;
        int pa[TBM2 / 64], pb[TBN2 / 64];
        if (kn < K) {
#pragma unroll
            for (int r = 0, i = 0; r < TBM2; r += 64, i++) {
                pa[i] = 0;
                if (arow[i] != (size_t)-1)
                    pa[i] = *reinterpret_cast<const int *>(&Xq[arow[i] * x_stride + kn + cc]);
            }
#pragma unroll
            for (int r = 0, i = 0; r < TBN2; r += 64, i++) {
                int gr = col0 + r + m0; pb[i] = 0;
                if (gr < N) pb[i] = *reinterpret_cast<const int *>(&W[(size_t)gr * K + kn + cc]);
            }
        }
        int a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[buf][srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[buf][scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        if (kn < K) {
#pragma unroll
            for (int r = 0, i = 0; r < TBM2; r += 64, i++)
                *reinterpret_cast<int *>(&As[buf ^ 1][r + m0][cc]) = pa[i];
#pragma unroll
            for (int r = 0, i = 0; r < TBN2; r += 64, i++)
                *reinterpret_cast<int *>(&Bs[buf ^ 1][r + m0][cc]) = pb[i];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int prow_base = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float wc = S[col];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow_base + e2;
                if (p < cnt) {
                    int sr = gather ? sorted_slot[base + p] : (base + p);
                    out_sorted[(size_t)(base + p) * N + col] = (float)acc[i][j][e2] * wc * xsc[sr];
                }
            }
        }
    }
}

/* run31: TBK=64 variant of the 128x128 W8A8 MoE GEMM. Loads the A/B tiles 128-bit
 * wide (int4 = 16 int8 per thread) and stages 64 K per iteration, so the K-loop
 * has 4x fewer __syncthreads and load instructions than the TBK=16 kernel. The
 * MFMA still consumes 16-K slices (4 per staged tile), so the int32 accumulation
 * order -- and thus the output -- is BIT-IDENTICAL to moe_grouped_mfma_i8i8_128.
 * A test of whether the narrow 32-bit int8 loads (vs the bf16 path's 128-bit)
 * cost anything; expected ~neutral since the kernel is L2-BW-bound (bytes, not
 * transactions). K must be a multiple of 64 (H=2048, I=1536 both are). Toggle
 * (retired: measured -1.5%, see unused_reference.hip.) */
#define TBK4 64
__global__ void moe_grouped_mfma_i8i8_t64_kernel(float *out_sorted, const int8_t *Xq,
                                                 const float *xsc, const int8_t *const *Wtbl,
                                                 const float *const *Stbl, const int *eoff,
                                                 const int *sorted_slot, int N, int K,
                                                 int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM2;
    if (row0 >= cnt) return;
    const int8_t *W = Wtbl[e];
    const float  *S = Stbl[e];

    __shared__ int8_t As[TBM2][TBK4];   /* [m][k], k contiguous (64 wide) */
    __shared__ int8_t Bs[TBN2][TBK4];   /* [n][k], k contiguous (64 wide) */
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN2;

    i32x4b acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK4) {
        /* A tile: 128 rows x 64 K -> 128-bit (int4 = 16 int8) loads, 4 per row. */
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 16, p = row0 + m;
            int4 v = {0, 0, 0, 0};
            if (p < cnt) {
                int sr = gather ? sorted_slot[base + p] : (base + p);
                v = *reinterpret_cast<const int4 *>(&Xq[(size_t)sr * x_stride + k0 + cc]);
            }
            *reinterpret_cast<int4 *>(&As[m][cc]) = v;
        }
#pragma unroll
        for (int r = 0; r < TBN2; r += 64) {
            int n = r + tx / 4, cc = (tx % 4) * 16, gr = col0 + n;
            int4 v = {0, 0, 0, 0};
            if (gr < N) v = *reinterpret_cast<const int4 *>(&W[(size_t)gr * K + k0 + cc]);
            *reinterpret_cast<int4 *>(&Bs[n][cc]) = v;
        }
        __syncthreads();
        /* 4 MFMA k-slices of 16 over the staged 64 (same order as TBK=16 kernel). */
#pragma unroll
        for (int kk = 0; kk < TBK4; kk += 16) {
            int a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[srow + i * 16 + tcol][kk + trow * 4]);
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[scol + j * 16 + tcol][kk + trow * 4]);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int prow_base = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float wc = S[col];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow_base + e2;
                if (p < cnt) {
                    int sr = gather ? sorted_slot[base + p] : (base + p);
                    out_sorted[(size_t)(base + p) * N + col] = (float)acc[i][j][e2] * wc * xsc[sr];
                }
            }
        }
    }
}

/* ======================================================================== *
 *  run52: FUSED gate+up grouped MoE GEMM (re-test of run11 on the run44 kernel).
 *
 *  gate and up share the same activation rows, the same expert row range and the
 *  same shape, so one block can stage the A tile once and run it against BOTH
 *  weight tiles: A traffic halves (12 column-blocks instead of 2 x 12) and the
 *  launch count halves. run11 measured this -2.3% back when the kernel was a
 *  64x64 bf16 tile with narrow loads; the kernel has changed completely since
 *  (128x128, int8 MFMA, int4 loads, 32-K double-buffered LDS), so it is worth
 *  re-measuring rather than inheriting the old verdict.
 *
 *  The cost is unchanged in kind though: two i32x4b acc[4][4] sets = 128 VGPRs
 *  of accumulator (vs 64), and three LDS tiles instead of two = 24 KB (vs 16),
 *  which takes the block residency from 4 per CU to 2. Toggle MLA_MOEGU=1.
 * ======================================================================== */
__global__ __launch_bounds__(256, 1) void moe_grouped_gateup_w_kernel(float *out_g, float *out_u,
                                            const int8_t *Xq, const float *xsc,
                                            const int8_t *const *Wgt, const float *const *Sgt,
                                            const int8_t *const *Wut, const float *const *Sut,
                                            const int *eoff, const int *sorted_slot,
                                            int N, int K, int x_stride) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM2;
    if (row0 >= cnt) return;
    const int8_t *Wg = Wgt[e], *Wu = Wut[e];
    const float  *Sg = Sgt[e], *Su = Sut[e];

    __shared__ __attribute__((aligned(16))) int8_t As[2][TBM2][32];
    __shared__ __attribute__((aligned(16))) int8_t Bg[2][TBN2][32];
    __shared__ __attribute__((aligned(16))) int8_t Bu[2][TBN2][32];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN2;
    const int m0 = tx >> 1, half = tx & 1;
    const int p0 = row0 + m0, bcol = col0 + m0;
    const int bok = (bcol < N);
    size_t arow = (size_t)-1;
    if (p0 < cnt) arow = (size_t)sorted_slot[base + p0];
    const int aok = (arow != (size_t)-1);
    const int NG = K >> 5;

    int4 ra = {0, 0, 0, 0}, rg = {0, 0, 0, 0}, ru = {0, 0, 0, 0}, na, ng, nu;
    if (aok) ra = *reinterpret_cast<const int4 *>(&Xq[arow * x_stride + half * 16]);
    if (bok) {
        rg = *reinterpret_cast<const int4 *>(&Wg[(size_t)bcol * K + half * 16]);
        ru = *reinterpret_cast<const int4 *>(&Wu[(size_t)bcol * K + half * 16]);
    }
    *reinterpret_cast<int4 *>(&As[0][m0][half * 16]) = ra;
    *reinterpret_cast<int4 *>(&Bg[0][m0][half * 16]) = rg;
    *reinterpret_cast<int4 *>(&Bu[0][m0][half * 16]) = ru;
    __syncthreads();

    i32x4b accg[4][4] = {}, accu[4][4] = {};
    int buf = 0;
    for (int g = 0; g < NG; g++) {
        na = make_int4(0, 0, 0, 0); ng = make_int4(0, 0, 0, 0); nu = make_int4(0, 0, 0, 0);
        if (g + 1 < NG) {
            const size_t off = (size_t)(g + 1) * 32 + half * 16;
            if (aok) na = *reinterpret_cast<const int4 *>(&Xq[arow * x_stride + off]);
            if (bok) {
                ng = *reinterpret_cast<const int4 *>(&Wg[(size_t)bcol * K + off]);
                nu = *reinterpret_cast<const int4 *>(&Wu[(size_t)bcol * K + off]);
            }
        }
#pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            int a[4], bg[4], bu[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i]  = *reinterpret_cast<const int *>(&As[buf][srow + i * 16 + tcol][sub * 16 + trow * 4]);
#pragma unroll
            for (int j = 0; j < 4; j++) bg[j] = *reinterpret_cast<const int *>(&Bg[buf][scol + j * 16 + tcol][sub * 16 + trow * 4]);
#pragma unroll
            for (int j = 0; j < 4; j++) bu[j] = *reinterpret_cast<const int *>(&Bu[buf][scol + j * 16 + tcol][sub * 16 + trow * 4]);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    accg[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], bg[j], accg[i][j], 0, 0, 0);
                    accu[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], bu[j], accu[i][j], 0, 0, 0);
                }
        }
        if (g + 1 < NG) {
            ra = na; rg = ng; ru = nu;
            *reinterpret_cast<int4 *>(&As[buf ^ 1][m0][half * 16]) = ra;
            *reinterpret_cast<int4 *>(&Bg[buf ^ 1][m0][half * 16]) = rg;
            *reinterpret_cast<int4 *>(&Bu[buf ^ 1][m0][half * 16]) = ru;
        }
        __syncthreads();
        buf ^= 1;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int prow_base = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float wg = Sg[col], wu = Su[col];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow_base + e2;
                if (p < cnt) {
                    int sr = sorted_slot[base + p];
                    float xs = xsc[sr];
                    out_g[(size_t)(base + p) * N + col] = (float)accg[i][j][e2] * wg * xs;
                    out_u[(size_t)(base + p) * N + col] = (float)accu[i][j][e2] * wu * xs;
                }
            }
        }
    }
}

/* run29: quantize bf16 weight [N][K] to symmetric int4 PACKED [N][K/2] + per-row
 * fp32 scale[N] (scale = max_k|w|/7; int4 range [-7,7]). Two int4 per byte: even k
 * -> low nibble, odd k -> high nibble. K assumed even. grid=(N), block=256. This
 * halves the weight L2 bytes vs int8 -- the bandwidth bound of the MoE GEMM. */
__global__ void quantize_rows_i4_kernel(const bf16_t *W, int8_t *Qp, float *S,
                                        int N, int K) {
    int n = blockIdx.x;
    if (n >= N) return;
    const bf16_t *wr = W + (size_t)n * K;
    int8_t *qr = Qp + (size_t)n * (K / 2);
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
    float scale = red[0] / 7.0f;
    if (tid == 0) S[n] = scale;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int j = tid; j < K / 2; j += nt) {
        float q0 = rintf(bf16f(wr[2 * j])     * inv);
        float q1 = rintf(bf16f(wr[2 * j + 1]) * inv);
        int i0 = (int)fminf(7.0f, fmaxf(-7.0f, q0));
        int i1 = (int)fminf(7.0f, fmaxf(-7.0f, q1));
        qr[j] = (int8_t)((i0 & 0xF) | ((i1 & 0xF) << 4));
    }
}

/* run29: W4A8 grouped MoE GEMM -- int4 PACKED weight (per-row scale) x int8
 * activation (per-row scale), unpacking int4->int8 in the B-tile stage then a clean
 * int32 mfma_i32_16x16x16i8. 128x128 tile (mirrors moe_grouped_mfma_i8i8_128).
 * Wtbl[e] points to packed int4 [N][K/2]; the logical K is passed (K even). Halves
 * the weight L2 traffic (MoE GEMM is L2-BW-bound at B=1024). VALU headroom absorbs
 * the nibble unpack. Toggle MLA_MOE_W4A8=1. */
__global__ void moe_grouped_mfma_w4a8_128_kernel(float *out_sorted, const int8_t *Xq,
                                                 const float *xsc, const int8_t *const *Wtbl,
                                                 const float *const *Stbl, const int *eoff,
                                                 const int *sorted_slot, int N, int K,
                                                 int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM2;
    if (row0 >= cnt) return;
    const int8_t *W = Wtbl[e];        /* packed int4 [N][K/2] */
    const float  *S = Stbl[e];
    const int Kp = K / 2;

    __shared__ int8_t As[TBM2][TBK];
    __shared__ int8_t Bs[TBN2][TBK];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN2;

    i32x4b acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 4, p = row0 + m;
            int v = 0;
            if (p < cnt) {
                int sr = gather ? sorted_slot[base + p] : (base + p);
                v = *reinterpret_cast<const int *>(&Xq[(size_t)sr * x_stride + k0 + cc]);
            }
            *reinterpret_cast<int *>(&As[m][cc]) = v;
        }
#pragma unroll
        for (int r = 0; r < TBN2; r += 64) {
            int n = r + tx / 4, cc = (tx % 4) * 4, gr = col0 + n;
            /* 4 int8 weights (k0+cc..+3) live in 2 packed bytes at (k0+cc)/2. */
            unsigned pk = 0;
            if (gr < N) pk = *reinterpret_cast<const unsigned short *>(&W[(size_t)gr * Kp + ((k0 + cc) >> 1)]);
            Bs[n][cc + 0] = (int8_t)i4sext(pk & 0xF);
            Bs[n][cc + 1] = (int8_t)i4sext((pk >> 4) & 0xF);
            Bs[n][cc + 2] = (int8_t)i4sext((pk >> 8) & 0xF);
            Bs[n][cc + 3] = (int8_t)i4sext((pk >> 12) & 0xF);
        }
        __syncthreads();
        int a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int prow_base = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
            float wc = S[col];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow_base + e2;
                if (p < cnt) {
                    int sr = gather ? sorted_slot[base + p] : (base + p);
                    out_sorted[(size_t)(base + p) * N + col] = (float)acc[i][j][e2] * wc * xsc[sr];
                }
            }
        }
    }
}

/* run32: BLOCK-WISE int4 quant. Same packed layout as quantize_rows_i4_kernel
 * (even k -> low nibble, odd k -> high nibble) but a separate fp32 scale per
 * K-group of G (scale = max|w| in group / 7). S is [N][K/G]. Fine-grained scale
 * (DeepSeek/AWQ) so 16 int4 levels track local weight magnitude -> recovers the
 * accuracy per-row int4 threw away (run29). K multiple of G. grid=(N), block=256. */
/* run29: sign-extend a 4-bit value (0..15 nibble, two's complement) to int8. */
static __device__ __forceinline__ int i4sext(int nib) { return (int)((int8_t)(nib << 4) >> 4); }

#define GW4 128    /* run32/33: block-wise int4 quant group size (K per weight scale) */

__global__ void quantize_rows_i4_bw_kernel(const bf16_t *W, int8_t *Qp, float *S,
                                           int N, int K, int G) {
    int n = blockIdx.x;
    if (n >= N) return;
    const bf16_t *wr = W + (size_t)n * K;
    int8_t *qr = Qp + (size_t)n * (K / 2);
    int ng = K / G;
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float red[256];
    for (int g = 0; g < ng; g++) {
        int k0 = g * G;
        float mx = 0.0f;
        for (int k = tid; k < G; k += nt) mx = fmaxf(mx, fabsf(bf16f(wr[k0 + k])));
        red[tid] = mx;
        __syncthreads();
        for (int st = nt / 2; st > 0; st >>= 1) {
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
            __syncthreads();
        }
        float scale = red[0] / 7.0f;
        if (tid == 0) S[(size_t)n * ng + g] = scale;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int j = tid; j < G / 2; j += nt) {
            float q0 = rintf(bf16f(wr[k0 + 2 * j])     * inv);
            float q1 = rintf(bf16f(wr[k0 + 2 * j + 1]) * inv);
            int i0 = (int)fminf(7.0f, fmaxf(-7.0f, q0));
            int i1 = (int)fminf(7.0f, fmaxf(-7.0f, q1));
            qr[k0 / 2 + j] = (int8_t)((i0 & 0xF) | ((i1 & 0xF) << 4));
        }
        __syncthreads();   /* before reusing red[] next group */
    }
}

/* run32: block-wise W4A8 grouped MoE GEMM. Packed int4 weight [N][K/2] with a
 * per-(row,K-group) scale S=[N][K/G]. Because the weight scale changes every G
 * along K, a single int32 accumulation over all K is no longer valid: we do
 * two-level accumulation -- int32 mfma within a group (weight is pure int4, no
 * scale), and at each group boundary flush (float)acc * S[col][g] into a float
 * accumulator, reset the int32 acc, continue. The int8 activation scale xsc is
 * per-row so it still factors out to the epilogue. GW4 = G = 128 = 8*TBK. Doubles
 * accumulator registers (i32 acc + float facc) -> occupancy risk. MLA_MOE_W4A8_BW=1. */
__global__ void moe_grouped_mfma_w4a8_bw_128_kernel(float *out_sorted, const int8_t *Xq,
                                                    const float *xsc, const int8_t *const *Wtbl,
                                                    const float *const *Stbl, const int *eoff,
                                                    const int *sorted_slot, int N, int K,
                                                    int x_stride, int gather) {
    int e = blockIdx.z;
    int base = eoff[e];
    int cnt = eoff[e + 1] - base;
    if (cnt <= 0) return;
    int row0 = blockIdx.y * TBM2;
    if (row0 >= cnt) return;
    const int8_t *W = Wtbl[e];        /* packed int4 [N][K/2] */
    const float  *S = Stbl[e];        /* [N][K/GW4] */
    const int Kp = K / 2, ng = K / GW4;

    __shared__ int8_t As[TBM2][TBK];
    __shared__ int8_t Bs[TBN2][TBK];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN2;

    i32x4b acc[4][4] = {};
    float  facc[4][4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 4, p = row0 + m;
            int v = 0;
            if (p < cnt) {
                int sr = gather ? sorted_slot[base + p] : (base + p);
                v = *reinterpret_cast<const int *>(&Xq[(size_t)sr * x_stride + k0 + cc]);
            }
            *reinterpret_cast<int *>(&As[m][cc]) = v;
        }
#pragma unroll
        for (int r = 0; r < TBN2; r += 64) {
            int n = r + tx / 4, cc = (tx % 4) * 4, gr = col0 + n;
            unsigned pk = 0;
            if (gr < N) pk = *reinterpret_cast<const unsigned short *>(&W[(size_t)gr * Kp + ((k0 + cc) >> 1)]);
            Bs[n][cc + 0] = (int8_t)i4sext(pk & 0xF);
            Bs[n][cc + 1] = (int8_t)i4sext((pk >> 4) & 0xF);
            Bs[n][cc + 2] = (int8_t)i4sext((pk >> 8) & 0xF);
            Bs[n][cc + 3] = (int8_t)i4sext((pk >> 12) & 0xF);
        }
        __syncthreads();
        int a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
        if ((k0 + TBK) % GW4 == 0) {          /* group boundary: flush int32 -> float */
            int g = (k0 + TBK) / GW4 - 1;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int col = col0 + scol + j * 16 + tcol;
                float sc = (col < N) ? S[(size_t)col * ng + g] : 0.0f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int e2 = 0; e2 < 4; e2++)
                        facc[i][j][e2] += (float)acc[i][j][e2] * sc;
                    acc[i][j] = i32x4b{};
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int prow_base = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow_base + e2;
                if (p < cnt) {
                    int sr = gather ? sorted_slot[base + p] : (base + p);
                    out_sorted[(size_t)(base + p) * N + col] = facc[i][j][e2] * xsc[sr];
                }
            }
        }
    }
}

/* Per-row int8 quant of a contiguous [n_rows][KVD] block (grid = n_rows). */
__global__ void quant_kv_rows_kernel(int8_t *dst, float *scale, const float *src,
                                     int KVD) {
    int r = blockIdx.x;
    __shared__ float red[256];
    quant_row_i8(dst + (size_t)r * KVD, scale + r, src + (size_t)r * KVD, KVD,
                 threadIdx.x, blockDim.x, red);
}


/* ======================================================================== *
 * Retired elementwise kernels. The float4 / register-cached variants (run43)
 * measured -0.5%: these kernels are already bandwidth-limited, so the wider
 * loads bought less than the extra registers cost in occupancy. NOT compiled.
 * ======================================================================== */
__global__ void rmsnorm_fast_kernel(float *y, const float *x, const bf16_t *w,
                                    int n, float eps, int x_row_stride,
                                    int y_row_stride) {
    int row = blockIdx.x;
    const float4 *xr = reinterpret_cast<const float4 *>(x + (size_t)row * x_row_stride);
    float4 *yr = reinterpret_cast<float4 *>(y + (size_t)row * y_row_stride);
    int tid = threadIdx.x, nt = blockDim.x;
    const int n4 = n >> 2;

    float4 v[RN_REGS];
    float partial = 0.0f;
    int c = 0;
    for (int i = tid; i < n4; i += nt, c++) {
        float4 t = xr[i];
        v[c] = t;
        partial += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }
    float ss = block_reduce_sum(partial);
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    c = 0;
    for (int i = tid; i < n4; i += nt, c++) {
        int b = i << 2;
        float4 o;
        o.x = v[c].x * inv * bf16f(w[b + 0]);
        o.y = v[c].y * inv * bf16f(w[b + 1]);
        o.z = v[c].z * inv * bf16f(w[b + 2]);
        o.w = v[c].w * inv * bf16f(w[b + 3]);
        yr[i] = o;
    }
}

/* run43: float4 forms of the streaming elementwise kernels. The scalar versions
 * move 4 B per thread and measured ~260 GB/s (a fifth of HBM peak) because one
 * dword per lane leaves far too few bytes in flight to cover memory latency.
 * n must be a multiple of 4; the launchers fall back otherwise. */
__global__ void residual_add4_kernel(float4 *xs, const float4 *add, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 a = xs[i], b = add[i];
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        xs[i] = a;
    }
}

__global__ void weighted_add4_kernel(float4 *out, const float4 *add, float weight, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 a = out[i], b = add[i];
        a.x += weight * b.x; a.y += weight * b.y;
        a.z += weight * b.z; a.w += weight * b.w;
        out[i] = a;
    }
}

__global__ void silu_mul4_kernel(float4 *hb, const float4 *hb2, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 a = hb[i], b = hb2[i];
        a.x = silu_f(a.x) * b.x; a.y = silu_f(a.y) * b.y;
        a.z = silu_f(a.z) * b.z; a.w = silu_f(a.w) * b.w;
        hb[i] = a;
    }
}

/* Greedy argmax over `n` logits (lowest index on ties, matching sample() in
 * src/run.c). Single block, grid-stride + LDS reduction; writes one int so the
 * per-token DtoH shrinks from vocab_size floats to 4 bytes. */
__global__ void argmax_kernel(const float *logits, int n, int *out) {
    __shared__ float sv[REDUCE_THREADS];
    __shared__ int   si[REDUCE_THREADS];
    int tid = threadIdx.x;
    float bv = -INFINITY; int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = logits[i];
        if (v > bv) { bv = v; bi = i; }
    }
    sv[tid] = bv; si[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out = si[0];
}


/* ======================================================================== *
 * Retired: fixed-base RoPE. The batched path uses rope_pos_kernel (per-row
 * absolute position) in batched.hip. NOT compiled.
 * ======================================================================== */
__global__ void rope_kernel(float *v, int n_heads, int row_stride,
                            int head_stride, int pos_base, int rope_dim,
                            int interleaved, const float *inv_freq) {
    (void)n_heads;
    int row = blockIdx.x;
    int head = blockIdx.y;
    int pos = pos_base + row;
    int half = rope_dim / 2;
    int j = threadIdx.x;

    float *base = v + (size_t)row * row_stride + (size_t)head * head_stride;
    if (j >= half) return;

    float ang = (float)pos * inv_freq[j];
    float c = cosf(ang), s = sinf(ang);

    if (!interleaved) {
        float x0 = base[2 * j], x1 = base[2 * j + 1];
        base[2 * j]     = x0 * c - x1 * s;
        base[2 * j + 1] = x0 * s + x1 * c;
        return;
    }

    extern __shared__ float smem[]; /* rope_dim floats */
    float even = base[2 * j], odd = base[2 * j + 1];
    smem[j]        = even * c - odd * s;
    smem[half + j]  = odd  * c + even * s;
    __syncthreads();
    base[j]        = smem[j];
    base[half + j] = smem[half + j];
}


/* ======================================================================== *
 * Retired with the MoE epilogue rewrite: the shared expert is now folded into
 * moe_gather_add_kernel, so nothing needs a standalone weighted add.
 * ======================================================================== */
__global__ void weighted_add_kernel(float *out, const float *add, float weight,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += weight * add[i];
}


/* ======================================================================== *
 * Retired: the per-head shared-X GEMM now always runs the 128-tile bf16-in/bf16-out
 * variant (run64). This fp32 version is kept for reference only.
 * ======================================================================== */
/* run45: prefetched twin of matmul_headw_mfma_kernel. Same tiles / K order /
 * MFMA calls (BIT-IDENTICAL); the K-loop keeps two LDS tile buffers and issues
 * the next tile's global loads before the MFMAs so the ~800-cycle miss is
 * covered -- the same lever that took the two hot int8 GEMMs from 982 to 1210
 * tok/s. This kernel is 4.0 s of prefill (per-head W_UK/W_UV materialisation).
 * of prefill (per-head W_UK/W_UV materialisation). */
__global__ void matmul_headw_mfma_pf_kernel(float *Y, const float *X, const bf16_t *W,
                                            int M, int N, int K, int x_stride,
                                            int w_hstride, int y_hstride) {
    int h = blockIdx.z;
    const bf16_t *Wh = W + (size_t)h * w_hstride;
    float        *Yh = Y + (size_t)h * y_hstride;

    __shared__ __bf16 As[2][TBK][TBM + 4];
    __shared__ __bf16 Bs[2][TBK][TBN + 4];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 32, scol = (warp % 2) * 32;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM, col0 = blockIdx.x * TBN;
    const int am = tx / 4, acc4 = (tx % 4) * 4;          /* A: 64 rows x 4 k-quads */
    const int bn = tx / (TBK / 8), bcc = (tx % (TBK / 8)) * 8;  /* B: 8 bf16 each   */
    const int bact = (tx < TBN * TBK / 8);

    float4 va; uint16_t wb[8];
    auto loadA = [&](int k0) {
        va = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row0 + am < M) va = *reinterpret_cast<const float4 *>(&X[(size_t)(row0 + am) * x_stride + k0 + acc4]);
    };
    auto loadB = [&](int k0) {
#pragma unroll
        for (int j = 0; j < 8; j++) wb[j] = 0;
        if (bact && col0 + bn < N)
            __builtin_memcpy(wb, &Wh[(size_t)(col0 + bn) * K + k0 + bcc], 8 * sizeof(uint16_t));
    };
    auto store = [&](int buf) {
        As[buf][acc4 + 0][am] = (__bf16)va.x; As[buf][acc4 + 1][am] = (__bf16)va.y;
        As[buf][acc4 + 2][am] = (__bf16)va.z; As[buf][acc4 + 3][am] = (__bf16)va.w;
        if (bact)
#pragma unroll
            for (int j = 0; j < 8; j++) Bs[buf][bcc + j][bn] = *reinterpret_cast<const __bf16 *>(&wb[j]);
    };

    loadA(0); loadB(0); store(0);
    __syncthreads();

    float4v acc[2][2] = {};
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += TBK, buf ^= 1) {
        const int kn = k0 + TBK;
        if (kn < K) { loadA(kn); loadB(kn); }
        bf16x4 a[2], b[2];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) a[i][e] = As[buf][trow * 4 + e][srow + i * 16 + tcol];
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) b[j][e] = Bs[buf][trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
                acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], acc[i][j], 0, 0, 0);
        if (kn < K) store(buf ^ 1);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int base_row = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = base_row + e;
                if (row < M) Yh[(size_t)row * N + col] = acc[i][j][e];
            }
        }
    }
}

/* ======================================================================== *
 * Retired (run65): plain bf16 lm_head. Superseded by
 * matmul_mfma_128_xbf_argmax_kernel + launch_lm_head_argmax, which folds the
 * argmax into the GEMM epilogue so the [rows][154880] logits buffer (714 MB, more
 * than the whole remaining VRAM headroom) is never materialised. Its last caller
 * was the prefill first-token head: 26 launches, 363 ms, 1.0% of kernel time.
 * ======================================================================== */
static void launch_lm_head(float *Y, const float *X, const bf16_t *W,
                           int n_rows, int d_out, int n_in)
{
    int b128 = ceil_div(d_out, 128) * ceil_div(n_rows, 128);
    if (n_rows >= 128 && d_out >= 128 && b128 >= 96) {
        bf16_t *xbf = (bf16_t *)sbuf(B_XBF16, (size_t)n_rows * n_in * sizeof(bf16_t));
        int n = n_rows * n_in;
        narrow_bf16_kernel<<<ceil_div(n, 256), 256, 0, g_str>>>(xbf, X, n);
        dim3 grid(ceil_div(d_out, 128), ceil_div(n_rows, 128));
        matmul_mfma_128_xbf_kernel<<<grid, 256, 0, g_str>>>(Y, xbf, W, n_rows, d_out,
                                                           n_in, n_in, d_out);
    } else {
        launch_matmul(Y, X, W, n_rows, d_out, n_in);
    }
}

__global__ void matmul_mfma_128_xbf_kernel(float *Y, const bf16_t *X, const bf16_t *W,
                                           int M, int N, int K, int x_stride,
                                           int y_stride) {
    __shared__ __bf16 As[TBK][TBM2 + 4];
    __shared__ __bf16 Bs[TBK][TBN2 + 4];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM2, col0 = blockIdx.x * TBN2;

    float4v acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            uint16_t xb[4] = {0, 0, 0, 0};
            if (gr < M) __builtin_memcpy(xb, &X[(size_t)gr * x_stride + k0 + cc], 4 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 4; j++)
                As[cc + j][m] = (gr < M) ? *reinterpret_cast<const __bf16 *>(&xb[j]) : (__bf16)0.0f;
        }
        {
            int n = tx / (TBK / 8), cc = (tx % (TBK / 8)) * 8, gr = col0 + n;
            uint16_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &W[(size_t)gr * K + k0 + cc], 8 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 8; j++)
                Bs[cc + j][n] = (gr < N) ? *reinterpret_cast<const __bf16 *>(&wb[j]) : (__bf16)0.0f;
        }
        __syncthreads();
        bf16x4 a[4], b[4];
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++)
#pragma unroll
            for (int e = 0; e < 4; e++) a[i2][e] = As[trow * 4 + e][srow + i2 * 16 + tcol];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) b[j][e] = Bs[trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i2][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i2], b[j], acc[i2][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i2 = 0; i2 < 4; i2++) {
        int base_row = row0 + srow + i2 * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col0 + scol + j * 16 + tcol;
            if (col >= N) continue;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = base_row + e;
                if (row < M) Y[(size_t)row * y_stride + col] = acc[i2][j][e];
            }
        }
    }
}


/* ======================================================================== *
 * Retired (run65): the 32-query-tile prefill attention. MEASURED, do not retry.
 *
 * The counter data said the 64-query kernel is latency-bound (MFMA 4.9%, VALU
 * 18.3%, LDS BW 7.1%, occupancy 18.8%, IPC 0.30) and that its register floor --
 * O[64][256] over 256 threads = 64 accumulators per lane, 149 VGPRs -- was the one
 * thing pinning it at 3 waves/SIMD. This halves that floor: 80 VGPRs and LDS 13.8
 * -> 8.1 KB, i.e. 4 waves/SIMD with no spill (5 waves needs 102 regs and spills 32
 * B/lane; the launch_bounds sweep is in the report).
 *
 * It got the occupancy and still LOST: 1608.4 vs 1620.5 tok/s (-0.75%) at
 * B=1152/dup2q/PFCHUNK=16384, same binary, same GPU, toggled by env. Total MFMA and
 * softmax work is identical -- what doubles is the number of (query-tile, key-tile)
 * pairs, hence the barrier count, and two extra barriers per key tile cost more
 * than the extra resident wave buys. The conclusion for this kernel is therefore
 * that BARRIERS, not occupancy, are what is left to attack.
 *
 * Output was bit-identical to the 64-query version (verified by cmp on the full
 * 1152-request run), which also confirms the tiling rewrite is correct: the extra
 * fully-masked key tiles the 64-query version processes contribute exactly zero.
 * ======================================================================== */
/* ======================================================================== *
 *  32-QUERY-TILE prefill attention.
 *
 *  rocprof-compute on the 64-query version: nothing saturated (MFMA 4.9%, VALU
 *  18.3%, LDS BW 7.1%), occupancy 18.8%, IPC 0.30 -- latency-bound, wanting
 *  resident waves. Its LDS was reduced to 13.8 KB (4 blocks/CU) but occupancy did
 *  not move, because the register floor took over: O[64][256] spread over 256
 *  threads is 64 registers per lane, and 512/149 caps the kernel at 3 waves/SIMD.
 *
 *  Halving the query tile halves that floor. min-waves 4 is the most the register
 *  budget allows without spilling (5 waves = 102 regs = 32 B/lane of scratch). Per block: S[32][64], O[32][256] =
 *  32 registers of accumulator, Psh 4 KB, LDS ~8 KB. Warp grid is 2x2 over
 *  (16 rows) x (32 cols), so each warp holds 1 m-tile x 2 n-tiles instead of 2x2.
 *  Twice as many blocks, each doing half the work: total MFMA and softmax work is
 *  unchanged, only the barrier count doubles -- which is the trade being measured.
 * ======================================================================== */
__global__ __launch_bounds__(256, 4) void attn_prefill_mfma_q32_kernel(
        float *ctx, const float *qall, const bf16_t *knope, const float *kv_l,
        const bf16_t *value, const int *qt_q0, const int *qt_qs, const int *qt_qe,
        int n_tok, int NH, int QHD, int QKN, int QKR, int KVD, int KVL, int VHD,
        float scale) {
    const int qt = blockIdx.x, h = blockIdx.y, tx = threadIdx.x;
    const int warp = tx / 64, tiw = tx % 64;
    const int srow = (warp / 2) * 16, scol = (warp % 2) * 32;
    const int trow = tiw / 16, tcol = tiw % 16;
    (void)QKR;

    const int q0 = qt_q0[qt], qs = qt_qs[qt], qe = qt_qe[qt];
    const bf16_t *knope_h = knope + (size_t)h * n_tok * QKN;
    const bf16_t *value_h = value + (size_t)h * n_tok * VHD;

    __shared__ __bf16 Psh[32][64];
    __shared__ __bf16 As[16][32];
    __shared__ __bf16 Bs[16][64];
    __shared__ float  m_[32], l_[32], corr_[32];
    __shared__ float  rmax[2][32], rsum[2][32];

    if (tx < 32) { m_[tx] = -INFINITY; l_[tx] = 0.0f; }
    float4va oacc[4][2] = {};
    __syncthreads();

    const int kmax = (q0 + 31 < qe - 1) ? (q0 + 31) : (qe - 1);
    const int chalf = scol / 32;

    for (int kt0 = qs; kt0 <= kmax; kt0 += 64) {
        float4va sacc[2] = {};
        for (int k0 = 0; k0 < QHD; k0 += 16) {
            if (tx < 128) {          /* A tile: 32 queries x 16 k, 4 floats/lane */
                int m = tx / 4, cc = (tx % 4) * 4, g = q0 + m;
                float4 v = {0.f, 0.f, 0.f, 0.f};
                if (g < qe) v = *reinterpret_cast<const float4 *>(
                                    &qall[((size_t)g * NH + h) * QHD + k0 + cc]);
                As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
                As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
            }
            {                        /* B tile: 64 keys x 16 k */
                int n = tx / 4, cc = (tx % 4) * 4, kg = kt0 + n;
                if (kg < qe && k0 < QKN) {
                    uint16_t kb[4];
                    __builtin_memcpy(kb, &knope_h[(size_t)kg * QKN + k0 + cc], 4 * sizeof(uint16_t));
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        Bs[cc + j][n] = *reinterpret_cast<const __bf16 *>(&kb[j]);
                } else {
                    float4 v = {0.f, 0.f, 0.f, 0.f};
                    if (kg < qe)
                        v = *reinterpret_cast<const float4 *>(&kv_l[(size_t)kg * KVD + KVL + (k0 - QKN) + cc]);
                    Bs[cc + 0][n] = (__bf16)v.x; Bs[cc + 1][n] = (__bf16)v.y;
                    Bs[cc + 2][n] = (__bf16)v.z; Bs[cc + 3][n] = (__bf16)v.w;
                }
            }
            __syncthreads();
            bf16x4a a, b[2];
#pragma unroll
            for (int e = 0; e < 4; e++) a[e] = As[trow * 4 + e][srow + tcol];
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) b[j][e] = Bs[trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
            for (int j = 0; j < 2; j++)
                sacc[j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b[j], sacc[j], 0, 0, 0);
            __syncthreads();
        }
        /* row max from registers */
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int row = srow + trow * 4 + e, qg = q0 + row;
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                int kg = kt0 + scol + j * 16 + tcol;
                if (!(kg > qg || kg >= qe)) mx = fmaxf(mx, sacc[j][e] * scale);
            }
#pragma unroll
            for (int m = 1; m < 16; m <<= 1) mx = fmaxf(mx, __shfl_xor(mx, m, 64));
            if (tcol == 0) rmax[chalf][row] = mx;
        }
        __syncthreads();
        /* probs -> LDS (bf16) + row sums */
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int row = srow + trow * 4 + e, qg = q0 + row;
            float mn = fmaxf(m_[row], fmaxf(rmax[0][row], rmax[1][row]));
            float sm = 0.0f;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                int col = scol + j * 16 + tcol, kg = kt0 + col;
                float p = 0.0f;
                if (!(kg > qg || kg >= qe)) p = __expf(sacc[j][e] * scale - mn);
                Psh[row][col] = (__bf16)p;
                sm += p;
            }
#pragma unroll
            for (int m = 1; m < 16; m <<= 1) sm += __shfl_xor(sm, m, 64);
            if (tcol == 0) rsum[chalf][row] = sm;
        }
        __syncthreads();
        if (tx < 32) {
            int row = tx;
            float mo = m_[row], mn = fmaxf(mo, fmaxf(rmax[0][row], rmax[1][row]));
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            l_[row] = l_[row] * corr + rsum[0][row] + rsum[1][row];
            m_[row] = mn;
            corr_[row] = corr;
        }
        __syncthreads();
        /* rescale running O, then O += P @ V over 4 column blocks of 64 */
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    oacc[nb][j][e] *= corr_[srow + trow * 4 + e];
        /* nb and k0 MUST both unroll: oacc stays in registers only while its
         * first index is a compile-time constant (a runtime index spilled
         * 80 B/lane to scratch). */
#pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            int col0 = nb * 64;
#pragma unroll
            for (int k0 = 0; k0 < 64; k0 += 16) {
                {
                    int n = tx / 4, cc = (tx % 4) * 4;
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int kg = kt0 + k0 + cc + e;
                        Bs[cc + e][n] = (kg < qe)
                            ? *reinterpret_cast<const __bf16 *>(&value_h[(size_t)kg * VHD + col0 + n])
                            : (__bf16)0.0f;
                    }
                }
                __syncthreads();
                bf16x4a a, b[2];
#pragma unroll
                for (int e = 0; e < 4; e++) a[e] = Psh[srow + tcol][k0 + trow * 4 + e];
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) b[j][e] = Bs[trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
                for (int j = 0; j < 2; j++)
                    oacc[nb][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b[j], oacc[nb][j], 0, 0, 0);
                __syncthreads();
            }
        }
    }
    /* normalise and write ctx[query, head, d] */
#pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        int col0 = nb * 64;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int col = col0 + scol + j * 16 + tcol;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = srow + trow * 4 + e, qg = q0 + row;
                if (qg >= qe) continue;
                float inv = (l_[row] > 0.0f) ? 1.0f / l_[row] : 0.0f;
                ctx[((size_t)qg * NH + h) * VHD + col] = oacc[nb][j][e] * inv;
            }
        }
    }
}


/* ======================================================================== *
 * Retired (run65): the two device-side weight quantisers. Both took a bf16 copy
 * that was already resident on the device; quant_from_host streams from the host
 * mmap through one 6 MB staging buffer instead, which is what removed the 58 GB
 * warm_up peak. quant_keep additionally LEFT the bf16 alive (2.6 GB) for a bf16
 * fallback path that no longer exists.
 * ======================================================================== */
/* run20: quantize a bf16 weight [rows][cols] to int8 + per-row scale but KEEP the
 * bf16 (lm_head and moe_gate deliberately stay bf16). */
static void quant_keep(const bf16_t *src, int rows, int cols, int8_t **out_q, float **out_s)
{
    HIP_CHECK(hipMalloc((void **)out_q, (size_t)rows * cols * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)out_s, (size_t)rows * sizeof(float)));
    quantize_rows_i8_kernel<<<rows, 256>>>(src, *out_q, *out_s, rows, cols);
}



/* Quantize one bf16 weight [rows][cols] (device) into a fresh int8 buffer + fp32
 * per-row scale buffer (device), then free the bf16 source. */
static void quant_one_expert(bf16_t *src, int rows, int cols, int8_t **out_q, float **out_s)
{
    HIP_CHECK(hipMalloc((void **)out_q, (size_t)rows * cols * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)out_s, (size_t)rows * sizeof(float)));
    quantize_rows_i8_kernel<<<rows, 256>>>(src, *out_q, *out_s, rows, cols);
    HIP_CHECK(hipFree(src));
}






/* Retired (run65): the standalone SwiGLU. Its last caller was the fp32 fallback for
 * FFN rows too wide for the register-resident quantiser; silu_mul_quant_bf_big_kernel
 * now covers that case in bf16 and fuses the quantise, so the fp32 round trip
 * through hb is gone entirely.
 * ======================================================================== */
static void launch_silu_mul(float *hb, const float *hb2, int n)
{
    silu_mul_kernel<<<ceil_div(n, 256), 256, 0, g_str>>>(hb, hb2, n);
}

/* ======================================================================== *
 * Retired (run66): the v_dot4 score version of the absorbed flash decode. Same
 * kernel otherwise; attn_fd_mfma_kernel replaces its score loop with
 * mfma_i32_16x16x16i8 and is 1846.1 vs 1835.5 tok/s (+0.6%) at B=1792 with a
 * BIT-IDENTICAL output (the int32 score sum is exact either way, only its order
 * changes). dot4 needed 144 instructions plus 288 LDS reads per thread per key
 * tile; the MFMA form needs 9 MFMAs plus 18 LDS reads per wave, and gives most of
 * that back in the LDS atomics that sum the four waves' partial tiles and in the
 * block per CU that Sq costs.
 *
 * ALSO retired here (measured, do not retry): MFMA for BOTH matmuls, i.e. the clat
 * accumulate on MFMA as well. That needs a transposed [d][key] tile; paying for it
 * by moving the score's K operand to global reads cost -6.2% (1720.7 vs 1835.1),
 * because a 4-byte-per-lane global read is exactly the starved access pattern the
 * staged version exists to avoid. run39 had already measured the transpose at
 * -1.5% with the score left on dot4.
 * ======================================================================== */
__global__ void attn_fd_hb_q8_kernel(float *clat, const float *qabs, const float *q,
                                     const int8_t *kv_base, const float *kv_scale,
                                     const int *pos, int NH, int QHD, int QKN, int QKR,
                                     int KVL, int KVD, int slot_stride, float scale) {
    constexpr int NGRP = FD_HPB * FD_KVL / 4;   /* (head, 4-r) groups        */
    constexpr int CACC = NGRP / 256;            /* = 5 clat float4s/thread   */
    const int s = blockIdx.x, g = blockIdx.y;
    const int h0 = g * FD_HPB;
    const int tid = threadIdx.x, nt = blockDim.x;
    (void)NH; (void)QKR; (void)KVL; (void)KVD;
    const int kv_len = pos[s] + 1;
    const int8_t *ckvb  = kv_base  + (size_t)s * slot_stride;
    const float  *scl_s = kv_scale + (size_t)s * (slot_stride / FD_KVD);

    __shared__ __attribute__((aligned(16))) int8_t Qsh[FD_HPB * FD_KPAD];
    __shared__ __attribute__((aligned(16))) int8_t Ksh[FD_TK * FD_KPAD];
    __shared__ float Kscl[FD_TK];
    __shared__ float Ssh[FD_HPB * FD_TK];
    __shared__ float Msh[FD_HPB], Lsh[FD_HPB], Csh[FD_HPB], Qsc[FD_HPB];
    __shared__ unsigned Qmx[FD_HPB];

    float oacc[CACC][4];
#pragma unroll
    for (int e = 0; e < CACC; e++)
#pragma unroll
        for (int j = 0; j < 4; j++) oacc[e][j] = 0.0f;
    if (tid < FD_HPB) { Msh[tid] = -INFINITY; Lsh[tid] = 0.0f; Qmx[tid] = 0u; }
    __syncthreads();

    /* ---- quantize this block's FD_HPB query rows to int8 (per-head scale) ---- */
    for (int i = tid; i < FD_HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i - h * FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        atomicMax(&Qmx[h], __float_as_uint(fabsf(v)));
    }
    __syncthreads();
    if (tid < FD_HPB) Qsc[tid] = __uint_as_float(Qmx[tid]) / 127.0f;
    __syncthreads();
    for (int i = tid; i < FD_HPB * FD_KVD; i += nt) {
        int h = i / FD_KVD, d = i - h * FD_KVD, hg = h0 + h;
        float v = (d < FD_KVL) ? qabs[((size_t)s * FD_NH + hg) * FD_KVL + d]
                               : q[((size_t)s * FD_NH + hg) * QHD + QKN + (d - FD_KVL)];
        float sc = Qsc[h];
        int qq = (sc > 0.0f) ? (int)rintf(v / sc) : 0;
        qq = qq > 127 ? 127 : (qq < -127 ? -127 : qq);
        Qsh[h * FD_KPAD + d] = (int8_t)qq;
    }
    __syncthreads();

    const int KW = FD_KVD / 4;
    const int KQ = FD_KVD / 16;
    for (int k0 = 0; k0 < kv_len; k0 += FD_TK) {
        const int tk = (kv_len - k0 < FD_TK) ? (kv_len - k0) : FD_TK;
        for (int i = tid; i < tk * KQ; i += nt) {
            int r = i / KQ, c = i - r * KQ;
            int4 v = *reinterpret_cast<const int4 *>(ckvb + (size_t)(k0 + r) * FD_KVD + c * 16);
            int8_t *d = Ksh + (size_t)r * FD_KPAD + c * 16;
            *reinterpret_cast<int *>(d)      = v.x;
            *reinterpret_cast<int *>(d + 4)  = v.y;
            *reinterpret_cast<int *>(d + 8)  = v.z;
            *reinterpret_cast<int *>(d + 12) = v.w;
        }
        for (int j = tid; j < tk; j += nt) Kscl[j] = scl_s[k0 + j];
        __syncthreads();

        /* scores: 4 int8 MACs per instruction, no unpack, no convert. */
        for (int hk = tid; hk < FD_HPB * tk; hk += nt) {
            int h = hk / tk, k = hk - h * tk;
            const int *kw = reinterpret_cast<const int *>(Ksh + (size_t)k * FD_KPAD);
            const int *qw = reinterpret_cast<const int *>(Qsh + (size_t)h * FD_KPAD);
            int acc = 0;
#pragma unroll 8
            for (int c = 0; c < KW; c++) acc = __builtin_amdgcn_sdot4(qw[c], kw[c], acc, false);
            Ssh[h * FD_TK + k] = (float)acc * Qsc[h] * Kscl[k] * scale;
        }
        __syncthreads();

        if (tid < FD_HPB) {
            int h = tid;
            float tmax = -INFINITY;
            for (int k = 0; k < tk; k++) tmax = fmaxf(tmax, Ssh[h * FD_TK + k]);
            float mo = Msh[h], mn = fmaxf(mo, tmax);
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            float sum = 0.0f;
            for (int k = 0; k < tk; k++) {
                float p = __expf(Ssh[h * FD_TK + k] - mn);
                sum += p;
                Ssh[h * FD_TK + k] = p * Kscl[k];
            }
            Lsh[h] = Lsh[h] * corr + sum; Msh[h] = mn; Csh[h] = corr;
        }
        __syncthreads();

        {
            int g4 = tid;
#pragma unroll
            for (int e = 0; e < CACC; e++, g4 += nt) {
                int h = g4 >> 7;
                int r0 = (g4 & 127) << 2;
                const float *Ph = Ssh + h * FD_TK;
                const int8_t *kb0 = Ksh + r0;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (int k = 0; k < tk; k++) {
                    int kb = *reinterpret_cast<const int *>(kb0 + (size_t)k * FD_KPAD);
                    float p = Ph[k];
                    a0 += p * (float)(int8_t)(kb & 0xff);
                    a1 += p * (float)(int8_t)((kb >> 8) & 0xff);
                    a2 += p * (float)(int8_t)((kb >> 16) & 0xff);
                    a3 += p * (float)(int8_t)(kb >> 24);
                }
                float c = Csh[h];
                oacc[e][0] = oacc[e][0] * c + a0;
                oacc[e][1] = oacc[e][1] * c + a1;
                oacc[e][2] = oacc[e][2] * c + a2;
                oacc[e][3] = oacc[e][3] * c + a3;
            }
        }
        __syncthreads();
    }
    {
        int g4 = tid;
#pragma unroll
        for (int e = 0; e < CACC; e++, g4 += nt) {
            if (CACC * 256 != NGRP && g4 >= NGRP) break;
            int h = g4 >> 7, r0 = (g4 & 127) << 2, hg = h0 + h;
            float inv = (Lsh[h] > 0.0f) ? 1.0f / Lsh[h] : 0.0f;
            float *dst = clat + ((size_t)s * FD_NH + hg) * FD_KVL + r0;
            dst[0] = oacc[e][0] * inv;
            dst[1] = oacc[e][1] * inv;
            dst[2] = oacc[e][2] * inv;
            dst[3] = oacc[e][3] * inv;
        }
    }
}

/* ======================================================================== *
 * MEASURED THREE TIMES, DO NOT RETRY: MFMA for the flash-decode clat accumulate.
 *
 * O[h][r] = sum_k P[h][k] * V[k][r] contracts along k, and an int8 MFMA needs 4
 * consecutive CONTRACTION values packed per lane, so it needs a [r][key] copy of the
 * key tile that the [key][d] staging (which the score wants) cannot provide.
 *
 *   run39: transpose + MFMA accumulate, score left on dot4      -1.5%
 *   run66: MFMA for both, score's K operand moved to global to
 *          pay for the transpose's LDS (4 B per lane per MFMA)  -6.2%  (1720.7)
 *   run66: MFMA for both, score's K operand still staged, VT
 *          built from Ksh in LDS, LDS 25 KB -> 2 blocks/CU      -3.7%  (1797.8)
 *
 * The last of these is the cleanest version of the idea and it still loses: the
 * accumulate really does drop from ~720 VALU instructions per thread per key tile to
 * 8 MFMAs plus a 64-instruction int32->fp32 fold, but the transpose costs 32 LDS
 * dword writes per thread per tile, P has to be quantised to int8 (which also
 * perturbs the output), and the 8 KB it needs takes the kernel from 3 blocks/CU to
 * 2. What DOES pay is MFMA for the score alone (bit-identical, +0.6%), because that
 * one needs no layout it does not already have.
 * ======================================================================== */

/* ======================================================================== *
 * Superseded (run67, same session): the KV byte-copy path for duplicate prompts.
 * It gave every follower slot its own copy of the leader's prefilled rows (8.4 GB
 * of D2D at B=1792, ~14 ms) and got 3258.7 tok/s. Sharing the leader's KV SLOT
 * instead does the same thing for free AND makes the cache cost one slot per
 * distinct prompt instead of one per request (26 GB -> 7.5 GB at B=1792), which is
 * what lets the batch grow. Kept here because the copy form is the one to fall back
 * to if a future change makes duplicate slots diverge (the sharing relies on their
 * decode being bit-identical, so their writes to the shared rows are the same
 * bytes).
 * ======================================================================== */
__global__ void kv_dup_copy_kernel(int8_t *kv, const int *src, const int *dst,
                                   const int *np, int slot_stride, size_t layer_stride,
                                   int KVD) {
    const int f = blockIdx.y, l = blockIdx.z;
    const int n = np[f] * KVD;                       /* bytes for this follower */
    const size_t base = (size_t)l * layer_stride;
    const int8_t *s = kv + base + (size_t)src[f] * slot_stride;
    int8_t       *d = kv + base + (size_t)dst[f] * slot_stride;
    int i0 = (int)((blockIdx.x * blockDim.x + threadIdx.x) * 16);
    if (i0 + 16 <= n)
        *reinterpret_cast<int4 *>(d + i0) = *reinterpret_cast<const int4 *>(s + i0);
    else
        for (int i = i0; i < n; i++) d[i] = s[i];
}

__global__ void kv_dup_scale_kernel(float *sc, const int *src, const int *dst,
                                    const int *np, int cap, size_t layer_stride) {
    const int f = blockIdx.y, l = blockIdx.z;
    const int n = np[f];
    const size_t base = (size_t)l * layer_stride;
    const float *s = sc + base + (size_t)src[f] * cap;
    float       *d = sc + base + (size_t)dst[f] * cap;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}

