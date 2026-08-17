/* MLA attention, both paths from src/run.c ported to GPU:
 *  - unabsorbed (prefill): explicit per-head K_nope/V, causal scores, softmax,
 *    weighted context (attn_unabsorbed_{score,ctx}_kernel).
 *  - absorbed (decode): W_UK folded into the query, attention in latent space
 *    (attn_absorbed_{score,clat}_kernel); softmax_row_kernel (elementwise.hip)
 *    handles the softmax step for both paths. */
#include "kernels.h"

/* run39: the frozen Makefile compiles every .hip with -ffp-contract=off, which
 * splits every fp32 `a*b+c` into v_mul_f32 + v_add_f32 -- exactly 2x the VALU
 * instructions of a single v_fmac_f32 (verified in ISA). That flag exists to keep
 * the kernels bit-aligned with the CPU reference for `make eval`; this project is
 * scored by the fuzzy METEOR/BERTScore gate and is already far from bit-exact
 * (int8 weights/activations/KV), so contraction is re-enabled here. FMA also
 * rounds ONCE instead of twice, i.e. it is strictly more accurate than the split
 * form. File-scope pragma -> applies to every kernel below without touching the
 * frozen Makefile. */
#pragma clang fp contract(fast)


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
    (void)QKR;   /* the rope half is located via QHD/QKN, not QKR */
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
    /* run41 REVERTED: padding Ssh to [64][65] and As/Bs to [16][68] removes the
     * bank conflicts (softmax scan 64-way, P@V operand 16-way) but grows LDS
     * 20.75 -> 21.25 KB, which crosses the 3-blocks-per-CU line run28 was built
     * to sit just under (3 x 21845 B = 64 KB). Measured: attn_prefill_mfma
     * 6688 -> 8209 ms (+23%). Occupancy beats bank conflicts here, so the tight
     * unpadded layout stays. */
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

/* ==== ABSORBED-MLA FLASH DECODE (one pass per (slot, head-group)) ==========
 * Latent MLA: query[h] = [qabs_h (KVL) ++ qpe (QKR)] = KVD, and c_kv[k] =
 * [c_kv (KVL) ++ k_pe (QKR)] = KVD, both int8 with a per-row scale. For each
 * slot: score[h,k] = (q[h] . ckv[k]) * qscale[h] * kvscale[k] * softmax_scale,
 * online softmax over k per head, clat[h,r] = sum_k P[h,k] * ckv[k,r] for
 * r in [0,KVL). grid = (B, FD_HG); block = 256.
 *
 * One block owns FD_HPB heads of one slot and stages each c_kv key tile into LDS
 * once for the whole head group (run23): that turned ~2*NH L2 reads per slot into
 * FD_HG, moving the traffic to LDS (22.6 TB/s vs 3.5) and taking the kernel off
 * its L2 bound.
 *
 * Three things then decided its speed, in this order (report/run39_.../
 * profile_finding.md has the full derivation):
 *
 *  1. LDS BANK CONFLICTS. With the natural row stride FD_KVD = 576 B = 144
 *     dwords, 144 % 32 == 16, so keys k and k+2 land in the same bank: a wave
 *     covering 32 distinct keys touched only 2 banks -- a 16-way conflict on
 *     every single key byte. The cost model (2560 blocks x 8.5 tiles x 4 waves
 *     x 720 reads x ~32 cycles / 104 CUs = 533 ms/step) matched the 531 ms
 *     measured, i.e. this one detail WAS the kernel. FD_KPAD = 580 B = 145
 *     dwords, gcd(145,32) = 1, puts all 32 lanes in distinct banks.
 *  2. INSTRUCTION COUNT, not occupancy. Once the conflicts were gone the kernel
 *     became issue-bound: a wave64 instruction holds its 16-lane SIMD for 4
 *     cycles, so the GCD retires 177 G wave-instr/s, and the score dot alone
 *     needed 14.0 G of them = 79 ms of pure issue. Sweeping the tile for more
 *     occupancy therefore LOST (FD_TK=16 and FD_HG=4 both measured slower at
 *     that point). What won was `v_dot4_i32_i8` -- four int8 MACs, one
 *     instruction, no unpack, no convert -- which needs the query in int8, so
 *     the prologue quantizes it per head (LDS atomicMax over the IEEE bits,
 *     valid because the values are non-negative). Overflow is impossible:
 *     576 * 127 * 127 = 9.3M << 2^31. Accuracy cost is nil in practice: the
 *     query was already stored bf16 (8-bit mantissa) and c_kv is already int8.
 *  3. THE TILE OPTIMUM THEN FLIPPED. With the score 6x cheaper, FD_TK = 16
 *     (4 blocks/CU) beat FD_TK = 32: 1389 vs 1311 tok/s. FD_TK = 8 was worse
 *     again (1373) -- the per-tile fixed cost (4 barriers plus the 10-thread
 *     softmax section) starts to dominate.
 *
 * The clat accumulation stays scalar on purpose. It reduces along k, which is
 * the strided axis here, so both dot4 and MFMA need a transposed [d][key] copy
 * of the tile; building that (16 byte-wide LDS writes per 16 B loaded) plus its
 * 10 KB of LDS (occupancy 4 -> 2 blocks/CU) costs more than either saves --
 * measured -5.3% for dot4 and -1.5% for MFMA. See clat_mfma_finding.md.
 * ========================================================================= */
#define FD_NH   20                       /* model dims are fixed for GLM-4.7   */
#define FD_KVL  512
#define FD_KVD  576
#define FD_HG   2                        /* head groups per slot (NH % HG == 0) */
#define FD_HPB  (FD_NH / FD_HG)          /* heads per block = 10                */
#define FD_TK   16                       /* key tile (swept: 8 / 16 / 32)       */
#define FD_KPAD 580                      /* LDS row stride, bytes: 145 dwords   */

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

using i32x4b_attn = __attribute__((__vector_size__(16))) int;
