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

/* __launch_bounds__(256, 3): rocprof-compute showed this kernel spilling -- VGPRs
 * capped at 128, AGPRs 0, 96 bytes/lane of scratch, 40 registers spilled -- while
 * its occupancy is limited by LDS ("Insufficient CU LDS" 27.7%, peak 61.6%), not by
 * registers ("Insufficient SIMD VGPRs" 0%). LDS (21.2 KB) allows 3 blocks/CU = 3
 * waves/SIMD, so asking for a minimum of 3 waves/SIMD lets the compiler spend up to
 * 512/3 = 170 registers per lane -- enough for the 168 it wants -- and the spill
 * disappears at no cost in occupancy. Leaving it to the compiler cost both. */
__global__ __launch_bounds__(256, 3) void attn_prefill_mfma_kernel(bf16_t *ctx, const bf16_t *qall,
                                         const bf16_t *knope, const float *kv_l,
                                         const bf16_t *value, const int *qt_q0,
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

    /* knope/value arrive in bf16: their producer (matmul_headw) narrows them with
     * the same cast this kernel used to apply when staging into LDS, so this is
     * bit-identical and halves the bytes on both sides -- they are re-read once per
     * (query tile, head, key tile). */
    const bf16_t *knope_h = knope + (size_t)h * n_tok * QKN;
    const bf16_t *value_h = value + (size_t)h * n_tok * VHD;

    /* The scores never reach LDS. rocprof-compute showed this kernel's occupancy
     * capped by LDS ("Insufficient CU LDS" 27.7%, peak 61.6%) with nothing else
     * saturated (MFMA 4.9%, VALU 18.3%, LDS BW 7.1%) -- i.e. it is latency-bound and
     * wants more resident waves. The old layout kept a full fp32 Ssh[64][64] = 16 KB
     * purely so one thread per row could scan it for the softmax max and sum.
     *
     * Instead the S tile stays in the MFMA accumulators, the row max and sum are
     * cross-lane reductions (each row's 64 columns live across 16 tcol lanes and the
     * two column-halves, so 4 shuffles plus a 2-entry LDS merge), and LDS holds only
     * what the P@V MFMA actually consumes: P in bf16. 21.2 KB -> 13.8 KB, so 4 blocks
     * per CU instead of 3, and the softmax stops being one wave's serial 128-element
     * LDS scan while the other three idle. */
    /* run65: rows are padded to 66 bf16 = 132 B = 33 dwords, and gcd(33,32) = 1, so
     * every row of a tile starts on a different bank. Unpadded (64 bf16 = 32 dwords)
     * every row starts on the SAME bank, and the P@V A-operand read
     * Psh[srow+i*16+tcol][k0+trow*4+e] walks 16 rows across the 16 tcol lanes of a
     * wave -- a 16-WAY conflict on the kernel's densest LDS access (128 reads per
     * key tile against 12 for the score tiles). The three pads together cost 592 B:
     * LDS goes 13.8 -> 14.1 KB, and 149 VGPRs already cap this kernel at 3
     * waves/SIMD = 3 blocks/CU, which leaves room up to 21.3 KB. That headroom is
     * why run41's padding attempt lost and this one does not: back then Ssh was
     * still a 16 KB fp32 tile, so the same pad crossed 21.3 KB and cost a block. */
    __shared__ __bf16 Psh[64][66];     /* softmax probs, the P@V A-operand          */
    /* run65: 64 rows, not 16. The score loop still stages 16 k-slices at a time (its
     * depth is capped by As), but the P@V loop stages a whole 64-key slab of V per
     * column block, so its four MFMA steps run with no barrier between them: 32
     * barriers per key tile become 8. Barriers, not occupancy, are what this kernel
     * is short of -- the 32-query experiment proved that by buying an extra resident
     * wave at the cost of twice the barriers and losing 0.75%. LDS 14.1 -> 20.3 KB,
     * and 3 blocks/CU (the register cap) needs <= 21.0 KB, so the 6 KB is free. */
    __shared__ __bf16 Bs[64][66];      /* staged B tile [k][n]: K (16 rows) or V (64) */
    __shared__ float  rmax[2][64], rsum[2][64];   /* per column-half row reductions */
    __shared__ float  m_[64], l_[64], corr_[64];

    if (tx < 64) { m_[tx] = -INFINITY; l_[tx] = 0.0f; }
    float4va oacc[4][2][2] = {};       /* O[64,256]: nblk x (2x2 warp tiles)       */
    __syncthreads();

    const int kmax = (q0 + 63 < qe - 1) ? (q0 + 63) : (qe - 1);  /* causal key ceiling */

    for (int kt0 = qs; kt0 <= kmax; kt0 += 64) {
        /* ---- S[64,64] = Q[q0..][d] . K[kt0..][d] over d=0..QHD-1 (bf16 MFMA) ---- */
        float4va sacc[2][2] = {};
        for (int kk = 0; kk < QHD; kk += 64) {
            {   /* K slab: 64 k-slices x 64 keys, 16 contiguous values per thread */
                int n = tx / 4, kc = (tx % 4) * 16, kg = kt0 + n;
                if (kg >= qe) {
#pragma unroll
                    for (int j = 0; j < 16; j++) Bs[kc + j][n] = (__bf16)0.0f;
                } else if (kk < QKN) {
                    uint16_t kb[16];
                    __builtin_memcpy(kb, &knope_h[(size_t)kg * QKN + kk + kc],
                                     16 * sizeof(uint16_t));
#pragma unroll
                    for (int j = 0; j < 16; j++)
                        Bs[kc + j][n] = *reinterpret_cast<const __bf16 *>(&kb[j]);
                } else {
                    const float *src = &kv_l[(size_t)kg * KVD + KVL + (kk - QKN) + kc];
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        float4 v = *reinterpret_cast<const float4 *>(src + c * 4);
                        Bs[kc + c * 4 + 0][n] = (__bf16)v.x;
                        Bs[kc + c * 4 + 1][n] = (__bf16)v.y;
                        Bs[kc + c * 4 + 2][n] = (__bf16)v.z;
                        Bs[kc + c * 4 + 3][n] = (__bf16)v.w;
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int k0 = 0; k0 < 64; k0 += 16) {
                bf16x4a a[2], b[2];
#pragma unroll
                for (int i = 0; i < 2; i++) {   /* A straight from global: 4 k per lane */
                    int g = q0 + srow + i * 16 + tcol;
                    uint16_t qb[4] = {0, 0, 0, 0};
                    if (g < qe)
                        __builtin_memcpy(qb, &qall[((size_t)g * NH + h) * QHD + kk + k0 + trow * 4],
                                         4 * sizeof(uint16_t));
#pragma unroll
                    for (int e = 0; e < 4; e++) a[i][e] = *reinterpret_cast<const __bf16 *>(&qb[e]);
                }
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) b[j][e] = Bs[k0 + trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++)
                        sacc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], sacc[i][j], 0, 0, 0);
            }
            __syncthreads();
        }
        /* ---- row max over this key tile, from registers ---- */
        const int chalf = scol / 32;        /* which 32-column half this warp owns */
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = srow + i * 16 + trow * 4 + e, qg = q0 + row;
                float mx = -INFINITY;
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    int kg = kt0 + scol + j * 16 + tcol;
                    float sv = sacc[i][j][e] * scale;
                    if (!(kg > qg || kg >= qe)) mx = fmaxf(mx, sv);
                }
#pragma unroll
                for (int m = 1; m < 16; m <<= 1) mx = fmaxf(mx, __shfl_xor(mx, m, 64));
                if (tcol == 0) rmax[chalf][row] = mx;
            }
        __syncthreads();
        /* ---- probs into LDS (bf16) + row sums, still from registers ---- */
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int row = srow + i * 16 + trow * 4 + e, qg = q0 + row;
                float mo = m_[row], mn = fmaxf(mo, fmaxf(rmax[0][row], rmax[1][row]));
                float sm = 0.0f;
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    int col = scol + j * 16 + tcol, kg = kt0 + col;
                    float p = 0.0f;
                    if (!(kg > qg || kg >= qe)) p = __expf(sacc[i][j][e] * scale - mn);
                    Psh[row][col] = (__bf16)p;
                    sm += p;
                }
#pragma unroll
                for (int m = 1; m < 16; m <<= 1) sm += __shfl_xor(sm, m, 64);
                if (tcol == 0) rsum[chalf][row] = sm;
            }
        __syncthreads();
        /* ---- fold this tile into the running (m, l) and publish corr ---- */
        if (tx < 64) {
            int row = tx;
            float mo = m_[row], mn = fmaxf(mo, fmaxf(rmax[0][row], rmax[1][row]));
            float corr = (mo == -INFINITY) ? 0.0f : __expf(mo - mn);
            l_[row] = l_[row] * corr + rsum[0][row] + rsum[1][row];
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
            {   /* V slab: 64 keys x 64 cols staged at once, 16 contiguous bf16/thread */
                int k = tx / 4, n0 = (tx % 4) * 16, kg = kt0 + k;
                if (kg < qe) {
                    uint16_t vb[16];
                    __builtin_memcpy(vb, &value_h[(size_t)kg * VHD + col0 + n0],
                                     16 * sizeof(uint16_t));
#pragma unroll
                    for (int j = 0; j < 16; j++)
                        Bs[k][n0 + j] = *reinterpret_cast<const __bf16 *>(&vb[j]);
                } else {
#pragma unroll
                    for (int j = 0; j < 16; j++) Bs[k][n0 + j] = (__bf16)0.0f;
                }
            }
            __syncthreads();
#pragma unroll
            for (int k0 = 0; k0 < 64; k0 += 16) {
                bf16x4a a[2], b[2];
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int e = 0; e < 4; e++)
                        a[i][e] = Psh[srow + i * 16 + tcol][k0 + trow * 4 + e];  /* P[query][key] */
#pragma unroll
                for (int j = 0; j < 2; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) b[j][e] = Bs[k0 + trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++)
                        oacc[nb][i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], oacc[nb][i][j], 0, 0, 0);
            }
            __syncthreads();
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
                        __bf16 o = (__bf16)(oacc[nb][i][j][e] * inv);
                ctx[((size_t)qg * NH + h) * VHD + col] = *reinterpret_cast<const bf16_t *>(&o);
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

using i32x4b_attn = __attribute__((__vector_size__(16))) int;

__global__ __launch_bounds__(256, 3) void attn_fd_mfma_kernel(float *clat, const float *qabs, const float *q,
                                     const int8_t *kv_base, const float *kv_scale,
                                     const int *pos, int NH, int QHD, int QKN, int QKR,
                                     int KVL, int KVD, int slot_stride, float scale,
                                     const int *kvslot) {
    constexpr int NGRP = FD_HPB * FD_KVL / 4;   /* (head, 4-r) groups        */
    constexpr int CACC = NGRP / 256;            /* = 5 clat float4s/thread   */
    const int s = blockIdx.x, g = blockIdx.y;
    const int h0 = g * FD_HPB;
    const int tid = threadIdx.x, nt = blockDim.x;
    (void)NH; (void)QKR; (void)KVL; (void)KVD;
    const int kv_len = pos[s] + 1;
    const int ks = kvslot ? kvslot[s] : s;      /* duplicate prompts share a KV slot */
    const int8_t *ckvb  = kv_base  + (size_t)ks * slot_stride;
    const float  *scl_s = kv_scale + (size_t)ks * (slot_stride / FD_KVD);

    __shared__ __attribute__((aligned(16))) int8_t Qsh[FD_HPB * FD_KPAD];
    __shared__ __attribute__((aligned(16))) int8_t Ksh[FD_TK * FD_KPAD];
    __shared__ float Kscl[FD_TK];
    __shared__ int   Sq[FD_TK][16];        /* MFMA score tile, [key][head]      */
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

    const int wavew = tid >> 6, tiww = tid & 63, troww = tiww >> 4, tcolw = tiww & 15;
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
        if (tid < FD_TK * 16) Sq[tid >> 4][tid & 15] = 0;
        __syncthreads();

        /* run66: scores on the matrix cores. v_dot4 does 4 MACs per instruction and
         * needs TWO LDS reads to feed them, so this loop was 144 dot4 + 288 LDS
         * reads per thread per key tile -- 3.5 s of the kernel's 9.0 s at B=1792,
         * pure issue cost. One mfma_i32_16x16x16i8 does 4096 MACs off two dwords.
         *
         * S[k][h] = sum_d K[k][d] * Q[h][d]: contraction d, so A = Ksh[k][d] (m = k,
         * exactly 16 keys) and B = Qsh[h][d] (n = h, 10 of 16 columns used -- the
         * index is clamped rather than padding Qsh to 16 rows, which would cost 3.5
         * KB of LDS and a block per CU). The 36 contraction steps split 9 per wave
         * and the four partial tiles are summed into Sq by LDS atomics.
         *
         * The clat accumulation deliberately stays scalar: it reduces along k, so
         * MFMA there needs a transposed [d][key] tile, and that costs more than it
         * saves -- measured -1.5% in run39 and again -6.2% in run66 when the score
         * operand was moved to global to pay for its LDS. */
        {
            const int hcl = (tcolw < FD_HPB) ? tcolw : (FD_HPB - 1);
            i32x4b_attn sacc = {0, 0, 0, 0};
            const int d0w = wavew * (FD_KVD / 4);
#pragma unroll
            for (int st = 0; st < (FD_KVD / 4) / 16; st++) {
                int d0 = d0w + st * 16;
                int a = *reinterpret_cast<const int *>(Ksh + (size_t)tcolw * FD_KPAD + d0 + troww * 4);
                int b = *reinterpret_cast<const int *>(Qsh + (size_t)hcl * FD_KPAD + d0 + troww * 4);
                sacc = __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, sacc, 0, 0, 0);
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int k = troww * 4 + e;
                if (k < tk && tcolw < FD_HPB) atomicAdd(&Sq[k][tcolw], sacc[e]);
            }
        }
        __syncthreads();
        for (int hk = tid; hk < FD_HPB * tk; hk += nt) {
            int h = hk / tk, k = hk - h * tk;
            Ssh[h * FD_TK + k] = (float)Sq[k][h] * Qsc[h] * Kscl[k] * scale;
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


