/* MLA attention, both paths from src/run.c ported to GPU:
 *  - unabsorbed (prefill): explicit per-head K_nope/V, causal scores, softmax,
 *    weighted context (attn_unabsorbed_{score,ctx}_kernel).
 *  - absorbed (decode): W_UK folded into the query, attention in latent space
 *    (attn_absorbed_{score,clat}_kernel); softmax_row_kernel (elementwise.hip)
 *    handles the softmax step for both paths. */
#include "kernels.h"

#define ATTN_THREADS 256
__device__ __forceinline__ float attn_block_reduce_sum(float val) {
    __shared__ float sh[ATTN_THREADS];
    int tid = threadIdx.x;
    sh[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    return sh[0];
}

__device__ __forceinline__ float attn_block_reduce_max(float val) {
    __shared__ float sh[ATTN_THREADS];
    int tid = threadIdx.x;
    sh[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    return sh[0];
}


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

/* ======================================================================== *
 *  MFMA flash-attention for the unabsorbed prefill path (run18). Replaces the
 *  scalar-FP32 attn_unabsorbed_flash_varlen_kernel (1 query/block, 256-dim dots
 *  on the vector ALU, no K/V reuse across queries -> the single biggest kernel at
 *  B=1024, ~27% of wall time and compute-bound). Here a block owns a 64-query tile
 *  of ONE prompt and runs FlashAttention-2 on the CDNA2 matrix cores:
 *    S[64,64] = Q[64,D] @ K[64,D]^T   (mfma_f32_16x16x16bf16, D=QHD=256)
 *    online softmax of S in LDS (fp32), P written back as bf16
 *    O[64,VHD] += P[64,64] @ V[64,VHD]  (mfma, VHD=256 as 4 N-blocks of 64)
 *  Each key tile's K/V is loaded once and reused across all 64 queries (the reuse
 *  the scalar kernel lacked). bf16 inputs / fp32 accumulate; numerics stay within
 *  the accuracy gate. Q = [qnope(QKN) || qpe(QKR)] is contiguous in qall; K per
 *  key = [knope(QKN) || k_pe(QKR)] with k_pe living in kv_l[key*KVD + KVL..]; the
 *  QKN=192 split falls on a TBK=16 boundary so the contraction branches on the
 *  d-tile once. Causal + prompt-boundary mask applied when S is stored to LDS.
 *
 *  Work list (host-built): qt_q0[qt] = global index of the tile's first query,
 *  qt_qs[qt] = its prompt's global start, qt_qe[qt] = its prompt's global end
 *  (exclusive). Tiles never cross a prompt boundary. grid = (num_qtiles, NH);
 *  block = 256 (4 warps, 2x2), matching matmul_mfma's lane layout.
 * ======================================================================== */
using bf16x4a = __attribute__((__vector_size__(8))) __bf16;
using float4va = __attribute__((__vector_size__(16))) float;

__global__ void attn_prefill_mfma_kernel(float *ctx, const float *qall,
                                         const float *knope, const float *kv_l,
                                         const float *value, const int *qt_q0,
                                         const int *qt_qs, const int *qt_qe,
                                         int n_tok, int NH, int QHD, int QKN, int QKR,
                                         int KVD, int KVL, int VHD, float scale) {
    const int qt = blockIdx.x, h = blockIdx.y, tx = threadIdx.x;
    const int warp = tx / 64, tiw = tx % 64;
    const int srow = (warp / 2) * 32, scol = (warp % 2) * 32;
    const int trow = tiw / 16, tcol = tiw % 16;

    const int q0 = qt_q0[qt];      /* global first query row of this tile */
    const int qs = qt_qs[qt];      /* prompt start (global) */
    const int qe = qt_qe[qt];      /* prompt end (global, exclusive) */

    const float *knope_h = knope + (size_t)h * n_tok * QKN;
    const float *value_h = value + (size_t)h * n_tok * VHD;

    /* run28: Ssh doubles as scores (pre-softmax) then softmax probs (post-softmax).
     * Dropping the separate bf16 Psh[64][64] frees 8KB LDS (28.75->20.75KB/block ->
     * 2->3 blocks/CU) to relieve the occupancy-bound prefill (report/run26: occ
     * 16.7%). Bit-identical: P@V converts the fp32 prob to bf16 at load, same value
     * as storing bf16 then reading it. */
    __shared__ float  Ssh[64][64];     /* scores (fp32) -> softmax probs (fp32)   */
    __shared__ __bf16 As[16][64];      /* staged A tile [k][m]                     */
    __shared__ __bf16 Bs[16][64];      /* staged B tile [k][n]                     */
    __shared__ float  m_[64], l_[64], corr_[64];

    if (tx < 64) { m_[tx] = -INFINITY; l_[tx] = 0.0f; }
    float4va oacc[4][2][2] = {};       /* O[64,256]: nblk x (2x2 warp tiles)       */
    __syncthreads();

    const int kmax = (q0 + 63 < qe - 1) ? (q0 + 63) : (qe - 1);  /* causal key ceiling */

    for (int kt0 = qs; kt0 <= kmax; kt0 += 64) {
        /* ---- S[64,64] = Q[q0..][d] . K[kt0..][d] over d=0..QHD-1 (bf16 MFMA) ---- */
        float4va sacc[2][2] = {};
        for (int k0 = 0; k0 < QHD; k0 += 16) {
            {   /* A[k][m] = Q[q0+m][k0+k] (contiguous in qall row) */
                int m = tx / 4, cc = (tx % 4) * 4, g = q0 + m;
                float4 v = {0.f, 0.f, 0.f, 0.f};
                if (g < qe) v = *reinterpret_cast<const float4 *>(
                                    &qall[((size_t)g * NH + h) * QHD + k0 + cc]);
                As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
                As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
            }
            {   /* B[k][n] = K[kt0+n][k0+k]; k_pe half lives in kv_l past KVL */
                int n = tx / 4, cc = (tx % 4) * 4, kg = kt0 + n;
                float4 v = {0.f, 0.f, 0.f, 0.f};
                if (kg < qe) {
                    if (k0 < QKN)
                        v = *reinterpret_cast<const float4 *>(&knope_h[(size_t)kg * QKN + k0 + cc]);
                    else
                        v = *reinterpret_cast<const float4 *>(&kv_l[(size_t)kg * KVD + KVL + (k0 - QKN) + cc]);
                }
                Bs[cc + 0][n] = (__bf16)v.x; Bs[cc + 1][n] = (__bf16)v.y;
                Bs[cc + 2][n] = (__bf16)v.z; Bs[cc + 3][n] = (__bf16)v.w;
            }
            __syncthreads();
            bf16x4a a[2], b[2];
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
                    sacc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], sacc[i][j], 0, 0, 0);
            __syncthreads();
        }
        /* ---- store S to LDS with scale + causal/prompt mask ---- */
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int row = srow + i * 16 + trow * 4 + e, col = scol + j * 16 + tcol;
                    int kg = kt0 + col, qg = q0 + row;
                    float s = sacc[i][j][e] * scale;
                    if (kg > qg || kg >= qe) s = -INFINITY;   /* causal + within prompt */
                    Ssh[row][col] = s;
                }
        __syncthreads();
        /* ---- online softmax over this key tile (one thread per query row) ---- */
        if (tx < 64) {
            int row = tx;
            float tmax = -INFINITY;
#pragma unroll
            for (int c = 0; c < 64; c++) tmax = fmaxf(tmax, Ssh[row][c]);
            float mo = m_[row], mn = fmaxf(mo, tmax);
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < 64; c++) {
                float p = (Ssh[row][c] == -INFINITY) ? 0.0f : __expf(Ssh[row][c] - mn);
                Ssh[row][c] = p;   /* overwrite dead score with prob (fp32, in place) */
                sum += p;
            }
            l_[row] = l_[row] * corr + sum;
            m_[row] = mn;
            corr_[row] = corr;
        }
        __syncthreads();
        /* ---- rescale running O by corr, then O += P @ V (4 N-blocks of 64) ---- */
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++)
                        oacc[nb][i][j][e] *= corr_[srow + i * 16 + trow * 4 + e];
        for (int nb = 0; nb < 4; nb++) {
            int col0 = nb * 64;
            for (int k0 = 0; k0 < 64; k0 += 16) {
                {   /* B[k][n] = V[kt0+k0+k][col0+n] (strided over keys) */
                    int n = tx / 4, cc = (tx % 4) * 4;
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        int kg = kt0 + k0 + cc + e;
                        Bs[cc + e][n] = (kg < qe) ? (__bf16)value_h[(size_t)kg * VHD + col0 + n] : (__bf16)0.0f;
                    }
                }
                __syncthreads();
                bf16x4a a[2], b[2];
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int e = 0; e < 4; e++)
                        a[i][e] = (__bf16)Ssh[srow + i * 16 + tcol][k0 + trow * 4 + e];  /* P[query][key] */
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) b[j][e] = Bs[trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++)
                        oacc[nb][i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], oacc[nb][i][j], 0, 0, 0);
                __syncthreads();
            }
        }
    }
    /* ---- normalize and write ctx[query, head, d] ---- */
#pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        int col0 = nb * 64;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int base_row = srow + i * 16 + trow * 4;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                int col = col0 + scol + j * 16 + tcol;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int row = base_row + e, qg = q0 + row;
                    if (qg < qe) {
                        float inv = (l_[row] > 0.0f) ? 1.0f / l_[row] : 0.0f;
                        ctx[((size_t)qg * NH + h) * VHD + col] = oacc[nb][i][j][e] * inv;
                    }
                }
            }
        }
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

/* ======================================================================== *
 * run23: HEAD-BATCHED flash-decode. rocprof-compute (B=1024, run22) showed
 * attn_flash_decode_kernel is L2-READ-BANDWIDTH-BOUND (L2 BW 137.5% peak,
 * Data-Return 99%, VALU 37%, MFMA 0%, HBM 3%, occupancy 92%): grid=(NH,B) reads
 * each slot's int8 c_kv cache ~2x per head (score row + clat cols) => ~2*NH reads
 * per slot saturate L2. Here grid=(B,HG): one block owns a group of FD_HPB heads
 * of ONE slot and stages each c_kv key-tile into LDS ONCE, shared across the whole
 * head group -> the redundant reads move from L2 (3.5 TB/s, saturated) to LDS
 * (22.6 TB/s, was 11%). c_kv is read from L2 HG times per slot instead of 2*NH
 * (HG=2 => 20x fewer). Scalar dots (compute was NOT the bound); model dims fixed
 * so the clat accumulator stays in registers. A/B via MLA_FDHB.
 *
 * Latent MLA: query[h] = [qabs_h (KVL) ++ qpe (QKR)] = KVD; c_kv[k] = [c_kv (KVL)
 * ++ k_pe (QKR)] = KVD. score[h,k] = (q[h].ckv[k]) * scl[k] * scale; softmax over
 * k (online, per head); clat[h,r] = sum_k P[h,k] * scl[k] * ckv[k,r], r in [0,KVL).
 * grid = (B, HG); block = 256. Model dims: NH=20, KVL=512, KVD=576. ==========*/
#define FD_NH   20
#define FD_KVL  512
#define FD_KVD  576
#define FD_HG   2                       /* head groups per slot (NH % HG == 0)     */
#define FD_HPB  (FD_NH / FD_HG)          /* heads per block = 10                    */
#define FD_TK   32                       /* key tile                                */
#define FD_OACC ((FD_HPB * FD_KVL + 255) / 256)   /* clat regs/thread = 20         */
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
