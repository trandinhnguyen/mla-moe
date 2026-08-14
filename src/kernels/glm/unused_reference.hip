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
#include "../kernels.h"


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

