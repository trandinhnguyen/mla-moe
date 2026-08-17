/* Fused MoE (vLLM-style), the whole expert path for one layer:
 *
 *   1. moe_route_batched  per-token router + top-K -> topk_idx[T], topk_wt[T]
 *   2. moe_hist           expert histogram (atomic) over topk_idx
 *   3. moe_scan           exclusive prefix sum -> eoff[0..n_routed]
 *   4. moe_place          counting-sort the (token, weight) pairs into expert order
 *   5. moe_grouped_gemm   ONE tiled GEMM per expert over its slice of the sorted
 *                         rows, so each expert's weight is streamed exactly once
 *   6. moe_scatter_add    weighted accumulate of the expert outputs back per token
 *
 * The sort is what makes step 5 possible: without it each (token, expert) pair
 * would be an independent GEMV and every expert weight would be re-read per token.
 * ========================================================================== */
#include "kernels.h"

/* ======================================================================== *
 *  Fused MoE (vLLM-style) for a batch of B tokens.
 *   1. moe_route_batched : per-token router + top-K -> topk_idx[B*K], topk_wt[B*K]
 *   2. moe_hist          : expert histogram (atomic) from topk_idx
 *   3. moe_scan          : exclusive prefix sum -> eoff[0..n_routed]
 *   4. moe_place         : scatter (token,weight) into expert-grouped order
 *   5. moe_grouped_gemm  : one tiled GEMM per expert (gather rows by slot)
 *   6. silu_mul (shared) : elementwise on the sorted hidden buffers
 *   7. moe_scatter_add   : atomically add weighted down-proj back to out[slot]
 * ======================================================================== */
__global__ void moe_route_batched_kernel(const float *logits, int n_routed,
                                          int K, int router_sigmoid,
                                          int norm_topk, float routed_scaling,
                                          const float *bias, int *topk_idx,
                                          float *topk_wt) {
    int s = blockIdx.x;
    const float *lg = logits + (size_t)s * n_routed;
    __shared__ float sc[1024];
    __shared__ char  used[1024];
    __shared__ float rval[256];
    __shared__ int   ridx[256];
    __shared__ int   idx_sh[16];
    __shared__ float wt_sh[16];
    __shared__ float red[256];
    int tid = threadIdx.x, nt = blockDim.x;

    if (router_sigmoid) {
        for (int e = tid; e < n_routed; e += nt)
            sc[e] = 1.0f / (1.0f + expf(-lg[e]));
    } else {
        float pm = -INFINITY;
        for (int e = tid; e < n_routed; e += nt) pm = fmaxf(pm, lg[e]);
        red[tid] = pm; __syncthreads();
        for (int st = nt / 2; st > 0; st >>= 1) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid+st]); __syncthreads(); }
        float mx = red[0]; __syncthreads();
        float ps = 0.0f;
        for (int e = tid; e < n_routed; e += nt) { float ex = expf(lg[e]-mx); sc[e]=ex; ps+=ex; }
        red[tid] = ps; __syncthreads();
        for (int st = nt / 2; st > 0; st >>= 1) { if (tid < st) red[tid]+=red[tid+st]; __syncthreads(); }
        float sum = red[0]; __syncthreads();
        for (int e = tid; e < n_routed; e += nt) sc[e] /= sum;
    }
    for (int e = tid; e < n_routed; e += nt) used[e] = 0;
    __syncthreads();

    for (int r = 0; r < K; r++) {
        float lb = -INFINITY; int li = -1;
        for (int e = tid; e < n_routed; e += nt) {
            if (used[e]) continue;
            float sel = sc[e] + (bias ? bias[e] : 0.0f);
            if (sel > lb) { lb = sel; li = e; }
        }
        rval[tid] = lb; ridx[tid] = li;
        __syncthreads();
        for (int st = nt / 2; st > 0; st >>= 1) {
            if (tid < st) {
                float ov = rval[tid+st]; int oi = ridx[tid+st];
                if (ov > rval[tid] || (ov == rval[tid] && oi >= 0 &&
                        (ridx[tid] < 0 || oi < ridx[tid]))) { rval[tid]=ov; ridx[tid]=oi; }
            }
            __syncthreads();
        }
        if (tid == 0) { int best = ridx[0]; used[best]=1; idx_sh[r]=best; wt_sh[r]=sc[best]; }
        __syncthreads();
    }
    if (tid == 0) {
        if (norm_topk) {
            float sum = 1e-20f;
            for (int r = 0; r < K; r++) sum += wt_sh[r];
            for (int r = 0; r < K; r++) wt_sh[r] /= sum;
        }
        for (int r = 0; r < K; r++) {
            topk_idx[(size_t)s * K + r] = idx_sh[r];
            topk_wt[(size_t)s * K + r]  = wt_sh[r] * routed_scaling;
        }
    }
}

__global__ void moe_hist_kernel(const int *topk_idx, int T, int *ecount) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) atomicAdd(&ecount[topk_idx[t]], 1);
}

/* exclusive scan of ecount[n_routed] -> eoff[n_routed+1]; single block. */
__global__ void moe_scan_kernel(const int *ecount, int n_routed, int *eoff) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (int e = 0; e < n_routed; e++) { eoff[e] = acc; acc += ecount[e]; }
    eoff[n_routed] = acc;
}

/* Scatter each (token t = s*K+r) into its expert's grouped slot. sorted_slot[p]
 * = originating batch row s; sorted_wt[p] = routed weight. cursor is a per-expert
 * running index seeded from eoff. */
__global__ void moe_place_kernel(const int *topk_idx, const float *topk_wt,
                                 int T, int K, const int *eoff, int *cursor,
                                 int *sorted_slot, float *sorted_wt, int *inv) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int e = topk_idx[t];
    int p = eoff[e] + atomicAdd(&cursor[e], 1);
    sorted_slot[p] = t / K;          /* originating batch row */
    sorted_wt[p]   = topk_wt[t];
    /* Inverse of the sort, keyed by the ORIGINAL assignment index t = row*K + k.
     * The epilogue uses it to gather a token's K expert outputs in a fixed order
     * instead of having each expert atomically add into the token's row. */
    inv[t] = p;
}

__global__ void moe_grouped_mfma_i8i8_128_w_kernel(bf16_t *out_sorted, const int8_t *Xq,
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

    __shared__ __attribute__((aligned(16))) int8_t As[2][TBM2][32];
    __shared__ __attribute__((aligned(16))) int8_t Bs[2][TBN2][32];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int col0 = blockIdx.x * TBN2;
    const int m0 = tx >> 1, half = tx & 1;
    const int p0 = row0 + m0, bcol = col0 + m0;
    const int bok = (bcol < N);
    /* the gathered row index is k-invariant -- hoisted out of the K loop. */
    size_t arow = (size_t)-1;
    if (p0 < cnt) arow = (size_t)(gather ? sorted_slot[base + p0] : (base + p0));
    const int aok = (arow != (size_t)-1);
    const int NG = K >> 5;

    /* run66: THREE-stage pipeline. The old form issued the load for group g+1 just
     * before the MFMAs of group g, so it had one group of matrix work (~960 cycles)
     * to cover an HBM miss that costs 1700-3400. Loading g+2 instead doubles that
     * cover for 8 more VGPRs; LDS is what caps this kernel at 4 blocks/CU, not
     * registers. */
    int4 ra = {0, 0, 0, 0}, rb = {0, 0, 0, 0}, na = ra, nb = rb, pa = ra, pb = rb;
    if (aok) ra = *reinterpret_cast<const int4 *>(&Xq[arow * x_stride + half * 16]);
    if (bok) rb = *reinterpret_cast<const int4 *>(&W[(size_t)bcol * K + half * 16]);
    if (NG > 1) {
        const size_t off1 = (size_t)32 + half * 16;
        if (aok) na = *reinterpret_cast<const int4 *>(&Xq[arow * x_stride + off1]);
        if (bok) nb = *reinterpret_cast<const int4 *>(&W[(size_t)bcol * K + off1]);
    }
    /* run49: stage the WHOLE 32-K group (both halves) at once, so the K loop
     * needs ONE barrier per 32 MFMAs instead of one per 16. LDS 8 -> 16 KB. */
    *reinterpret_cast<int4 *>(&As[0][m0][half * 16]) = ra;
    *reinterpret_cast<int4 *>(&Bs[0][m0][half * 16]) = rb;
    __syncthreads();

    i32x4b acc[4][4] = {};
    int buf = 0;
    for (int g = 0; g < NG; g++) {
        pa = make_int4(0, 0, 0, 0); pb = make_int4(0, 0, 0, 0);
        if (g + 2 < NG) {
            const size_t off = (size_t)(g + 2) * 32 + half * 16;
            if (aok) pa = *reinterpret_cast<const int4 *>(&Xq[arow * x_stride + off]);
            if (bok) pb = *reinterpret_cast<const int4 *>(&W[(size_t)bcol * K + off]);
        }
#pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            int a[4], b[4];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[buf][scol + j * 16 + tcol][sub * 16 + trow * 4]);
            /* Skip the 16-row sub-tiles that lie entirely past this expert's row
             * count. At decode an expert gets about T*K/NR = 72 rows, so 44% of a
             * 128-row tile is padding: three of the eight 16-row sub-tiles are
             * completely dead. The loops stay unrolled (the predicate is per
             * unrolled body, so acc[][] keeps its registers). */
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[buf][srow + i * 16 + tcol][sub * 16 + trow * 4]);
            /* Keep the four A loads issued together (batching them is what hides the
             * LDS latency -- moving each next to its own MFMAs cost 3.8%) and only
             * predicate the MFMA group: at decode an expert gets about T*K/NR = 72
             * rows, so three of the eight 16-row sub-tiles of a 128-row tile are
             * entirely padding. Loops stay unrolled so acc[][] keeps its registers. */
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (row0 + srow + i * 16 >= cnt) continue;
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
            }
        }
        if (g + 1 < NG) {
            ra = na; rb = nb; na = pa; nb = pb;
            *reinterpret_cast<int4 *>(&As[buf ^ 1][m0][half * 16]) = ra;
            *reinterpret_cast<int4 *>(&Bs[buf ^ 1][m0][half * 16]) = rb;
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
            float wc = S[col];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                int p = prow_base + e2;
                if (p < cnt) {
                    int sr = gather ? sorted_slot[base + p] : (base + p);
                    /* run65: bf16 out. Every consumer of these three buffers either
                     * quantises to int8 next (gate/up -> silu_mul_quant) or scales
                     * and sums into an fp32 residual (down -> moe_gather_add), so
                     * fp32 here was 8 extra bits nobody read: 402 MB written + 804
                     * MB read back per layer per chunk at T=16384, halved. */
                    __bf16 v = (__bf16)((float)acc[i][j][e2] * wc * xsc[sr]);
                    out_sorted[(size_t)(base + p) * N + col] = *reinterpret_cast<const bf16_t *>(&v);
                }
            }
        }
    }
}

/* out[slot] += sorted_wt[p] * down_sorted[p]  (atomic; slots repeat across K).
 * grid = (ceil(H/256), T). */
/* MoE epilogue, one pass. For each token row: sum its K routed expert outputs
 * (located through the inverse sort map, so in a fixed order) plus the shared
 * expert, and store.
 *
 * This replaces three passes -- zero the output, have every expert atomically add
 * into its tokens' rows, then add the shared expert -- which together moved about
 * 168 MB per layer at B=1280 against 63 MB here, and were the last source of
 * run-to-run output variation in the engine: atomicAdd fixes no summation order,
 * so the same input produced slightly different logits between runs, which greedy
 * decoding then amplified into different tokens. Gathering in k order makes the
 * result deterministic. */
__global__ void moe_gather_add_kernel(float *out, const bf16_t *down_sorted,
                                      const float *sorted_wt, const int *inv,
                                      const bf16_t *shared, int K, int H) {
    /* run67: row on grid.x, hidden on grid.y -- grid.y is capped at 65535 and row is
     * the batch here, so the old order put a ceiling on the batch size. */
    int row = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= H) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        int p = inv[row * K + k];
        acc += sorted_wt[p] * bf16f(down_sorted[(size_t)p * H + d]);
    }
    out[(size_t)row * H + d] = acc + bf16f(shared[(size_t)row * H + d]);
}
