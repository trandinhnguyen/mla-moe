/* MFMA GEMMs -- every matrix multiply in the engine except attention.
 *
 * All of them compute Y = X @ W^T (nn.Linear's layout) on the CDNA2 matrix cores,
 * accumulating in fp32 (bf16 weights) or int32 (int8 weights). They differ only in
 * tile size and in which dimension is batched:
 *
 *   matmul_mfma            64x64   bf16    the small/narrow projections
 *   matmul_mfma_128       128x128  bf16    lm_head and the large bf16 GEMMs
 *   matmul_mfma_i8i8       64x64   int8    narrow W8A8 projections (q_a, kv_a)
 *   matmul_mfma_i8i8_128_w 128x128 int8    the hot W8A8 path: 16 B loads, 32-K
 *                                          double-buffered LDS tile
 *   matmul_bh_mfma         64x64   bf16    per-head weight AND per-head X (qabs, ctx)
 *   matmul_headw_mfma_pf   64x64   bf16    per-head weight, SHARED X (prefill W_UK/W_UV)
 *
 * Tile choice is a bandwidth argument: with a square tile the L2 traffic through
 * the weight (N*K*M/TBM) equals that through the activation (M*K*N/TBN) and both
 * scale as 1/tile, so only doubling BOTH dims halves total traffic -- 128x64 buys
 * a useless 25%. The coarser tile is therefore used only when its grid still fills
 * the 104 CUs; the callers in glm_engine.hip apply that test.
 * ========================================================================== */
#include "kernels.h"

using float4v = __attribute__((__vector_size__(16))) float;
/* 4 packed bf16 = one lane's operand for mfma_f32_16x16x16bf16_1k. bf16_t is a
 * uint16_t holding a bfloat16 bit pattern, identical to the native __bf16, so a
 * weight can be bit-reinterpreted into this vector without a widen/narrow. */
using bf16x4 = __attribute__((__vector_size__(8))) __bf16;

/* bf16-native MFMA GEMM (run3). Same math / tiling / output layout as the fp32
 * matmul_mfma_kernel, but the matrix cores run mfma_f32_16x16x16bf16_1k: bf16
 * inputs (weights stay bf16 -- no widen to fp32 in LDS; activations narrowed to
 * bf16), fp32 accumulate. K=16 consumed per instruction (vs 4 for f32x4), so the
 * TBK tile is a single MFMA. Lane->K layout verified: lane l holds K=(l/16)*4+e. */
__global__ void matmul_mfma_kernel(float *Y, const float *X, const bf16_t *W,
                                   int M, int N, int K, int x_stride, int y_stride) {
    __shared__ __bf16 As[TBK][TBM + 4];   /* A_s_transposed[k][m] */
    __shared__ __bf16 Bs[TBK][TBN + 4];   /* B_s[k][n] = W[n][k]  */
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int wr = warp / 2, wc = warp % 2;           /* 2x2 warp grid */
    int srow = wr * 32, scol = wc * 32;         /* warp tile origin in the block */
    int trow = tiw / 16, tcol = tiw % 16;       /* lane position in a 16x16 tile */
    int row0 = blockIdx.y * TBM, col0 = blockIdx.x * TBN;

    float4v acc[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        /* A tile: A[row0+m][k0+k] -> As[k][m], fp32 read 4-wide, narrowed to bf16. */
        {
            int m = tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gr < M) v = *reinterpret_cast<const float4 *>(&X[(size_t)gr * x_stride + k0 + cc]);
            As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
            As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
        }
        /* B tile: W[col0+n][k0+k] -> Bs[k][n], bf16 copied straight in (no widen). */
        for (int idx = tx; idx < TBN * TBK / 8; idx += 256) {
            int n = idx / (TBK / 8), cc = (idx % (TBK / 8)) * 8, gr = col0 + n;
            uint16_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &W[(size_t)gr * K + k0 + cc], 8 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 8; j++)
                Bs[cc + j][n] = (gr < N) ? *reinterpret_cast<const __bf16 *>(&wb[j]) : (__bf16)0.0f;
        }
        __syncthreads();
        /* One MFMA per warp-tile pair consumes the whole TBK=16 K-slice. */
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
                if (row < M) Y[(size_t)row * y_stride + col] = acc[i][j][e];
            }
        }
    }
}

/* ======================================================================== *
 *  run21: 128x128-tile variant of the bf16 matmul_mfma_kernel. Same math/output
 *  (Y = X @ W.T, bf16 weight, fp32 accumulate) but each block computes a 128x128
 *  tile so each weight tile is re-read half as many times -> halves the bf16
 *  weight bandwidth (L2 for the smaller-weight GEMMs, HBM for lm_head whose
 *  634MB weight is far bigger than L2). Bit-identical to matmul_mfma_kernel (same
 *  fp32 K-accumulation order) -> argmax/logits unchanged. block = 256 (4 warps,
 *  2x2 grid, each warp a 64x64 = 4x4 grid of 16x16 MFMA tiles). Only worthwhile
 *  when M,N large enough to keep the coarser 128x128 grid filling the CUs.
 *  grid = (ceil(N/128), ceil(M/128)), K a multiple of 16.
 * ======================================================================== */
__global__ void matmul_mfma_128_kernel(float *Y, const float *X, const bf16_t *W,
                                       int M, int N, int K, int x_stride, int y_stride) {
    __shared__ __bf16 As[TBK][TBM2 + 4];   /* A_s_transposed[k][m] */
    __shared__ __bf16 Bs[TBK][TBN2 + 4];   /* B_s[k][n] = W[n][k]  */
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;   /* 2x2 warp grid, 64x64 each */
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM2, col0 = blockIdx.x * TBN2;

    float4v acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        /* A tile: 128 rows -> 2 passes (m and m+64), fp32 read 4-wide -> bf16. */
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gr < M) v = *reinterpret_cast<const float4 *>(&X[(size_t)gr * x_stride + k0 + cc]);
            As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
            As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
        }
        /* B tile: 128 cols x TBK/8 = 256 groups = one pass, bf16 copied straight in. */
        for (int idx = tx; idx < TBN2 * TBK / 8; idx += 256) {
            int n = idx / (TBK / 8), cc = (idx % (TBK / 8)) * 8, gr = col0 + n;
            uint16_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &W[(size_t)gr * K + k0 + cc], 8 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 8; j++)
                Bs[cc + j][n] = (gr < N) ? *reinterpret_cast<const __bf16 *>(&wb[j]) : (__bf16)0.0f;
        }
        __syncthreads();
        bf16x4 a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) a[i][e] = As[trow * 4 + e][srow + i * 16 + tcol];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) b[j][e] = Bs[trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
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
                if (row < M) Y[(size_t)row * y_stride + col] = acc[i][j][e];
            }
        }
    }
}

/* ======================================================================== *
 *  bf16-ACTIVATION variant of matmul_mfma_128_kernel, for lm_head.
 *
 *  lm_head is M=1280, N=154880, K=2048 -- 1210 column tiles, so the activation is
 *  re-read 1210 times: 12.7 GB per decode step against 6.3 GB for the weight, and
 *  together that is essentially all of its 19 ms. The kernel narrows X to bf16 in
 *  LDS anyway, so it never needed the fp32 bytes; taking X in already-bf16 halves
 *  the dominant term. One small pass (narrow_bf16_kernel) prepares it.
 *
 *  Numerically identical to the fp32-input kernel: same narrowing (truncation),
 *  same K order, same fp32 accumulate -- which matters because lm_head feeds the
 *  greedy argmax.
 * ======================================================================== */
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

__global__ void matmul_mfma_i8i8_kernel(float *Y, const int8_t *Xq, const float *xsc,
                                        const int8_t *W, const float *wsc, int M, int N,
                                        int K, int x_stride, int y_stride) {
    __shared__ int8_t As[TBM][TBK];   /* [m][k], k contiguous */
    __shared__ int8_t Bs[TBN][TBK];   /* [n][k], k contiguous */
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 32, scol = (warp % 2) * 32;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM, col0 = blockIdx.x * TBN;

    i32x4b acc[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        {   int m = tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            int v = 0;
            if (gr < M) v = *reinterpret_cast<const int *>(&Xq[(size_t)gr * x_stride + k0 + cc]);
            *reinterpret_cast<int *>(&As[m][cc]) = v;
        }
        {   int n = tx / 4, cc = (tx % 4) * 4, gr = col0 + n;
            int v = 0;
            if (gr < N) v = *reinterpret_cast<const int *>(&W[(size_t)gr * K + k0 + cc]);
            *reinterpret_cast<int *>(&Bs[n][cc]) = v;
        }
        __syncthreads();
        int a[2], b[2];
#pragma unroll
        for (int i = 0; i < 2; i++) a[i] = *reinterpret_cast<const int *>(&As[srow + i * 16 + tcol][trow * 4]);
#pragma unroll
        for (int j = 0; j < 2; j++) b[j] = *reinterpret_cast<const int *>(&Bs[scol + j * 16 + tcol][trow * 4]);
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++)
                acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int base_row = row0 + srow + i * 16 + trow * 4;
#pragma unroll
        for (int j = 0; j < 2; j++) {
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
 *  run44: WIDE-LOAD / DEEP-PREFETCH 128x128 int8 MFMA GEMMs.
 *
 *  The run40 double-buffered kernels still load only ONE dword (4 B) per lane
 *  per K-step and issue it just one MFMA block (~480 cycles) ahead of its use,
 *  while an L2/HBM miss costs ~800-1000 cycles. By Little's law the memory pipe
 *  stays starved: the decode MoE GEMM streams its 604 MB of expert weight per
 *  layer at an effective ~192 GB/s (about 15% of HBM peak) and the prefill GEMMs
 *  sit at ~31% of MFMA peak.
 *
 *  Here each lane loads an int4 (16 B) covering half of a 32-K group, so the
 *  same tile costs 4x fewer, 4x wider requests, and a group is fetched TWO
 *  sub-steps (~960 cycles) before it is consumed -- enough to cover the miss.
 *  LDS stays double-buffered at 16 K per tile (8 KB/block, occupancy unchanged);
 *  only ~12 extra VGPRs hold the in-flight group.
 *
 *  Tiles, K order and MFMA calls are untouched, so the output is BIT-IDENTICAL.
 *  Requires K % 32 == 0 and 16 B-aligned row strides; the callers check both.
 * ======================================================================== */
__global__ void matmul_mfma_i8i8_128_w_kernel(float *Y, const int8_t *Xq, const float *xsc,
                                              const int8_t *W, const float *wsc, int M, int N,
                                              int K, int x_stride, int y_stride) {
    __shared__ __attribute__((aligned(16))) int8_t As[2][TBM2][32];
    __shared__ __attribute__((aligned(16))) int8_t Bs[2][TBN2][32];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM2, col0 = blockIdx.x * TBN2;
    const int m0 = tx >> 1, half = tx & 1;          /* 128 rows x 2 half-groups   */
    const int arow = row0 + m0, bcol = col0 + m0;
    const int aok = (arow < M), bok = (bcol < N);
    const int NG = K >> 5;                           /* 32-K groups                */

    int4 ra = {0, 0, 0, 0}, rb = {0, 0, 0, 0}, na, nb;
    if (aok) ra = *reinterpret_cast<const int4 *>(&Xq[(size_t)arow * x_stride + half * 16]);
    if (bok) rb = *reinterpret_cast<const int4 *>(&W[(size_t)bcol * K + half * 16]);
    /* run49: stage the WHOLE 32-K group (both halves) at once, so the K loop
     * needs ONE barrier per 32 MFMAs instead of one per 16. LDS 8 -> 16 KB. */
    *reinterpret_cast<int4 *>(&As[0][m0][half * 16]) = ra;
    *reinterpret_cast<int4 *>(&Bs[0][m0][half * 16]) = rb;
    __syncthreads();

    i32x4b acc[4][4] = {};
    int buf = 0;
    for (int g = 0; g < NG; g++) {
        na = make_int4(0, 0, 0, 0); nb = make_int4(0, 0, 0, 0);
        if (g + 1 < NG) {
            const size_t off = (size_t)(g + 1) * 32 + half * 16;
            if (aok) na = *reinterpret_cast<const int4 *>(&Xq[(size_t)arow * x_stride + off]);
            if (bok) nb = *reinterpret_cast<const int4 *>(&W[(size_t)bcol * K + off]);
        }
#pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            int a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const int *>(&As[buf][srow + i * 16 + tcol][sub * 16 + trow * 4]);
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = *reinterpret_cast<const int *>(&Bs[buf][scol + j * 16 + tcol][sub * 16 + trow * 4]);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x16i8(a[i], b[j], acc[i][j], 0, 0, 0);
        }
        if (g + 1 < NG) {
            ra = na; rb = nb;
            *reinterpret_cast<int4 *>(&As[buf ^ 1][m0][half * 16]) = ra;
            *reinterpret_cast<int4 *>(&Bs[buf ^ 1][m0][half * 16]) = rb;
        }
        __syncthreads();
        buf ^= 1;
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
 *  MFMA per-head batched GEMM: same math/output as matmul_bh_batched_kernel
 *   Y[s,h,d] = sum_{i<K} X[s*x_bstride + h*x_hstride + i] * bf16f(W[h*w_stride + d*K + i])
 *  but run as a real tiled GEMM on the matrix cores instead of one block per
 *  output scalar. Each head h = blockIdx.z is an independent [M=B, N=d_out]
 *  = X[M,K] @ W[N,K]^T GEMM (X/W/Y offset by the per-head strides); the weight
 *  tile is reused across the whole M batch, so W[h] is read once per 64-row tile
 *  instead of once per (output, batch-row) as the GEMV kernel did. Body is
 *  identical to matmul_mfma_kernel with x_stride->x_bstride and the head bases.
 *  grid = (ceil(N/64), ceil(M/64), NH), block = 256. K a multiple of 16.
 * ======================================================================== */
__global__ void matmul_bh_mfma_kernel(float *Y, const float *X, const bf16_t *W,
                                      int M, int N, int K, int x_hstride,
                                      int x_bstride, int w_stride, int y_hstride,
                                      int y_bstride) {
    int h = blockIdx.z;
    const float  *Xh = X + (size_t)h * x_hstride;
    const bf16_t *Wh = W + (size_t)h * w_stride;
    float        *Yh = Y + (size_t)h * y_hstride;

    __shared__ __bf16 As[TBK][TBM + 4];   /* A_s_transposed[k][m] */
    __shared__ __bf16 Bs[TBK][TBN + 4];   /* B_s[k][n] = W[n][k]  */
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int wr = warp / 2, wc = warp % 2;
    int srow = wr * 32, scol = wc * 32;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM, col0 = blockIdx.x * TBN;

    float4v acc[2][2] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        /* A tile: X[row0+m][k0+k] -> As[k][m], batch-row stride = x_bstride. */
        {
            int m = tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gr < M) v = *reinterpret_cast<const float4 *>(&Xh[(size_t)gr * x_bstride + k0 + cc]);
            As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
            As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
        }
        /* B tile: W[col0+n][k0+k] -> Bs[k][n], bf16 copied straight in. */
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
                if (row < M) Yh[(size_t)row * y_bstride + col] = acc[i][j][e];
            }
        }
    }
}

/* ======================================================================== *
 *  128x128-tile version of matmul_bh_mfma_kernel (per-head weight AND per-head X).
 *
 *  The 64x64 kernel measured 13.6% of MFMA peak on the two decode calls (qabs and
 *  ctx, 21.4 ms per step): it is bandwidth-bound, not compute-bound. With N=512
 *  and a 64-wide tile the activation is re-read once per column tile -- 8 times --
 *  which at B=1280 is ~157 MB of the ~290 MB the qabs call moves per layer. The
 *  square-tile rule from run21 applies: weight traffic (N*K*M/TBM) and activation
 *  traffic (M*K*N/TBN) are equal and both scale as 1/tile, so doubling BOTH dims
 *  halves the total. 4 warps in a 2x2 grid, each warp a 64x64 region = 4x4 MFMA
 *  tiles, acc[4][4]. grid = (ceil(N/128), ceil(M/128), NH).
 * ======================================================================== */
__global__ void matmul_bh_mfma_128_kernel(float *Y, const float *X, const bf16_t *W,
                                          int M, int N, int K, int x_hstride,
                                          int x_bstride, int w_stride, int y_hstride,
                                          int y_bstride) {
    int h = blockIdx.z;
    const float  *Xh = X + (size_t)h * x_hstride;
    const bf16_t *Wh = W + (size_t)h * w_stride;
    float        *Yh = Y + (size_t)h * y_hstride;

    __shared__ __bf16 As[TBK][TBM2 + 4];
    __shared__ __bf16 Bs[TBK][TBN2 + 4];
    int tx = threadIdx.x;
    int warp = tx / 64, tiw = tx % 64;
    int srow = (warp / 2) * 64, scol = (warp % 2) * 64;
    int trow = tiw / 16, tcol = tiw % 16;
    int row0 = blockIdx.y * TBM2, col0 = blockIdx.x * TBN2;

    float4v acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += TBK) {
        /* A: 128 rows x 16 K, fp32 read 4-wide and narrowed to bf16 (2 passes). */
#pragma unroll
        for (int r = 0; r < TBM2; r += 64) {
            int m = r + tx / 4, cc = (tx % 4) * 4, gr = row0 + m;
            float4 v = {0.f, 0.f, 0.f, 0.f};
            if (gr < M) v = *reinterpret_cast<const float4 *>(&Xh[(size_t)gr * x_bstride + k0 + cc]);
            As[cc + 0][m] = (__bf16)v.x; As[cc + 1][m] = (__bf16)v.y;
            As[cc + 2][m] = (__bf16)v.z; As[cc + 3][m] = (__bf16)v.w;
        }
        /* B: 128 cols x 16 K, bf16 straight in, 8 per thread = one pass. */
        {
            int n = tx / (TBK / 8), cc = (tx % (TBK / 8)) * 8, gr = col0 + n;
            uint16_t wb[8] = {0};
            if (gr < N) __builtin_memcpy(wb, &Wh[(size_t)gr * K + k0 + cc], 8 * sizeof(uint16_t));
#pragma unroll
            for (int j = 0; j < 8; j++)
                Bs[cc + j][n] = (gr < N) ? *reinterpret_cast<const __bf16 *>(&wb[j]) : (__bf16)0.0f;
        }
        __syncthreads();
        bf16x4 a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) a[i][e] = As[trow * 4 + e][srow + i * 16 + tcol];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) b[j][e] = Bs[trow * 4 + e][scol + j * 16 + tcol];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[i], b[j], acc[i][j], 0, 0, 0);
        __syncthreads();
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
                if (row < M) Yh[(size_t)row * y_bstride + col] = acc[i][j][e];
            }
        }
    }
}

/* ======================================================================== *
 *  128x128-tile, bf16-ACTIVATION twin of matmul_headw_mfma_pf_kernel.
 *
 *  This is the prefill W_UK/W_UV materialisation, and it is bandwidth-bound on
 *  re-reading its SHARED activation. With a 64-wide tile the two calls together
 *  span (3 + 4) column tiles, and X is re-read once per (column tile, head), so
 *  140 times: 2.94 GB per layer per chunk at T=10240, against 1.46 GB for the
 *  weights. Two changes, both pure byte reductions:
 *
 *    - 128-wide tile: (2 + 2) x 20 = 80 re-reads instead of 140, and half the
 *      weight re-reads too (80 row tiles instead of 160). N=192 wastes half of
 *      its second column tile on MFMA work, which is free here -- the kernel is
 *      nowhere near compute-bound.
 *    - bf16 activation: the kernel narrows X to bf16 in LDS anyway, so the fp32
 *      bytes were never needed. One narrow_bf16_kernel pass prepares it.
 *
 *  4.4 GB -> 1.6 GB per layer per chunk. X is shared across heads (that is the
 *  point of this kernel: it folds 2*NH launches into 2), so only W and Y are
 *  offset by head. grid = (ceil(N/128), ceil(M/128), NH).
 * ======================================================================== */
__global__ void matmul_headw_mfma_128_xbf_kernel(float *Y, const bf16_t *X, const bf16_t *W,
                                                 int M, int N, int K, int x_stride,
                                                 int w_hstride, int y_hstride) {
    int h = blockIdx.z;
    const bf16_t *Wh = W + (size_t)h * w_hstride;
    float        *Yh = Y + (size_t)h * y_hstride;

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
            if (gr < N) __builtin_memcpy(wb, &Wh[(size_t)gr * K + k0 + cc], 8 * sizeof(uint16_t));
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
                if (row < M) Yh[(size_t)row * N + col] = acc[i2][j][e];
            }
        }
    }
}

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
