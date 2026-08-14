/* Batched (grouped) decode kernels for run2 continuous/grouped batching.
 *
 * The run1 engine processed one request at a time (batch-1 GEMV, memory-bound,
 * re-streaming every active weight per token). run2 groups B requests and decodes
 * them together so weights are read once per B tokens. The enablers here:
 *   - matmul_tiled_kernel : real tiled GEMM (Y = X @ W.T) that reuses each bf16
 *     weight tile across the whole M=B batch (fp32 accumulate -> low drift).
 *   - matmul_bh_batched   : per-head matmul with an added batch dim (qabs / ctx).
 *   - rope_pos_kernel     : RoPE with a per-row absolute position (each request in
 *     the batch sits at a different decode position).
 *   - attn_score/clat_batched : absorbed attention over per-request KV caches with
 *     per-request kv_len (masked to kv_max so the shared softmax_row works).
 *   - moe_route_batched + the moe_* sort/gemm/scatter kernels : a fused MoE that
 *     counting-sorts the B*K (token,expert) assignments by expert and runs one
 *     grouped GEMM per expert (weights read once, vLLM-style), instead of B*K
 *     independent GEMVs.
 */
#include "../kernels.h"

/* ======================================================================== *
 *  Tiled GEMM:  Y[m,n] = sum_k X[m*x_stride + k] * bf16f(W[n*K + k])
 *  X is [M,K] fp32 (row stride x_stride), W is [N,K] bf16 (nn.Linear y=x@W.T),
 *  Y is [M,N] fp32 (row stride y_stride). Each block computes a TBM x TBN output
 *  tile; the W tile in LDS is reused across all TBM rows -> the batched-decode
 *  win. grid = (ceil(N/TBN), ceil(M/TBM)), block = 256 (16x16 threads, 4x4 each).
 * ======================================================================== */
#define TBM 64
#define TBN 64
#define TBK 16
#define TBM2 128   /* run21: coarse 128x128 tile for the large bandwidth-bound GEMMs */
#define TBN2 128


/* ======================================================================== *
 *  MFMA tiled GEMM: same math as matmul_tiled_kernel (Y = X @ W.T) but the inner
 *  product runs on the CDNA2 matrix cores (mfma_f32_16x16x4f32, fp32 in/out so
 *  numerics stay close to the scalar path). Block = 4 warps (2x2) computing a
 *  64x64 output tile; each warp does a 32x32 region as 2x2 16x16 MFMA tiles.
 *  Weights are bf16 (converted to f32 in LDS); K is a multiple of 16 here.
 *  grid = (ceil(N/64), ceil(M/64)), block = 256.
 * ======================================================================== */
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
    __shared__ __bf16 As[TBK][TBM];   /* A_s_transposed[k][m] */
    __shared__ __bf16 Bs[TBK][TBN];   /* B_s[k][n] = W[n][k]  */
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
    __shared__ __bf16 As[TBK][TBM2];   /* A_s_transposed[k][m] */
    __shared__ __bf16 Bs[TBK][TBN2];   /* B_s[k][n] = W[n][k]  */
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
 *  run20: W8A8 dense GEMM (mfma_i32_16x16x16i8). Same math/output as
 *  matmul_mfma_kernel (Y = X @ W.T) but BOTH operands int8 -> no dequant/convert
 *  and int8 weight halves the L2-cache bytes. rocprof-compute (B=1024) showed the
 *  bf16 matmul_mfma is L2-BANDWIDTH-BOUND (L2 BW 133% of peak, MFMA only 24%,
 *  fully occupied) from re-reading bf16 weights out of L2, so 1-byte weights
 *  relieve the exact bottleneck. Activation is int8 (per-row scale xsc); weight
 *  int8 (per-output-row scale wsc). out[m][n] = (Σ_k Xq[m][k]*W[n][k]) * xsc[m] *
 *  wsc[n]. grid = (ceil(N/TBN), ceil(M/TBM)), block = 256.
 * ======================================================================== */
using i32x4b = __attribute__((__vector_size__(16))) int;
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

    __shared__ __bf16 As[TBK][TBM];   /* A_s_transposed[k][m] */
    __shared__ __bf16 Bs[TBK][TBN];   /* B_s[k][n] = W[n][k]  */
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

    __shared__ __bf16 As[TBK][TBM];
    __shared__ __bf16 Bs[TBK][TBN];
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


/* ======================================================================== *
 *  RoPE with a per-row absolute position. grid = (B, n_heads); block = half
 *  threads; dynamic shared = rope_dim floats (interleaved only). Row s applies
 *  RoPE at absolute position pos[s]. Mirrors rope_kernel bit-for-bit (GLM
 *  interleaved: read adjacent pairs (2j,2j+1), write split-half (j,j+half)).
 * ======================================================================== */
__global__ void rope_pos_kernel(float *v, int n_heads, int row_stride,
                                int head_stride, const int *pos, int rope_dim,
                                int interleaved, const float *inv_freq) {
    (void)n_heads;
    int s = blockIdx.x;
    int h = blockIdx.y;
    int half = rope_dim / 2;
    int j = threadIdx.x;
    if (j >= half) return;
    float *base = v + (size_t)s * row_stride + (size_t)h * head_stride;
    float ang = (float)pos[s] * inv_freq[j];
    float c = cosf(ang), sn = sinf(ang);
    if (!interleaved) {
        float x0 = base[2 * j], x1 = base[2 * j + 1];
        base[2 * j]     = x0 * c - x1 * sn;
        base[2 * j + 1] = x0 * sn + x1 * c;
        return;
    }
    extern __shared__ float smem[];
    float even = base[2 * j], odd = base[2 * j + 1];
    smem[j]        = even * c - odd * sn;
    smem[half + j] = odd  * c + even * sn;
    __syncthreads();
    base[j]        = smem[j];
    base[half + j] = smem[half + j];
}

/* int8 row quant helper (defined below with the scatter helpers). */
__device__ __forceinline__ void quant_row_i8(int8_t *dst, float *scale_out,
                                             const float *src, int KVD,
                                             int tid, int nt, float *red);

/* ======================================================================== *
 *  Batched KV-cache write: for each slot s, RMSNorm the compressed latent and
 *  copy + RoPE the k_pe slice straight into that slot's cache row at position
 *  pos[s] (cache[s][pos[s]] within the per-layer base). grid = (B); block=256.
 *  Fuses launch_rmsnorm + copy_strided + launch_rope of the decode KV path,
 *  which the single-request engine did against a contiguous cache row but which
 *  the batched cache (address depends on pos[s]) needs done per slot.
 * ======================================================================== */
__global__ void kv_write_batched_kernel(int8_t *cache_base, float *kv_scale,
                                        int slot_stride, const int *pos,
                                        const float *comp, const bf16_t *kv_a_ln,
                                        int KVL, int QKR, int KVD, float eps,
                                        const float *inv_freq, int interleaved) {
    int s = blockIdx.x;
    int p = pos[s];
    int8_t *dst = cache_base + (size_t)s * slot_stride + (size_t)p * KVD;   /* int8 cache */
    const float *cs = comp + (size_t)s * KVD;
    int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ float srow[];   /* [KVD] fp32 row, built then quantized */
    __shared__ float red[256];
    __shared__ float rtmp[256];       /* QKR <= 256 */
    /* rmsnorm the compressed c_kv part into srow[0..KVL). */
    float ss = 0.0f;
    for (int i = tid; i < KVL; i += nt) { float v = cs[i]; ss += v * v; }
    red[tid] = ss; __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid+st]; __syncthreads(); }
    float inv = 1.0f / sqrtf(red[0] / (float)KVL + eps);
    for (int i = tid; i < KVL; i += nt) srow[i] = cs[i] * inv * bf16f(kv_a_ln[i]);
    for (int i = tid; i < QKR; i += nt) srow[KVL + i] = cs[KVL + i];
    __syncthreads();
    /* RoPE the k_pe slice in fp32 (single head). */
    int half = QKR / 2;
    float *pe = srow + KVL;
    if (!interleaved) {
        for (int j = tid; j < half; j += nt) {
            float ang = (float)p * inv_freq[j];
            float c = cosf(ang), sn = sinf(ang);
            float x0 = pe[2 * j], x1 = pe[2 * j + 1];
            pe[2 * j]     = x0 * c - x1 * sn;
            pe[2 * j + 1] = x0 * sn + x1 * c;
        }
    } else {
        for (int j = tid; j < half; j += nt) {
            float ang = (float)p * inv_freq[j];
            float c = cosf(ang), sn = sinf(ang);
            rtmp[j]        = pe[2 * j] * c - pe[2 * j + 1] * sn;
            rtmp[half + j] = pe[2 * j + 1] * c + pe[2 * j] * sn;
        }
        __syncthreads();
        for (int j = tid; j < half; j += nt) { pe[j] = rtmp[j]; pe[half + j] = rtmp[half + j]; }
    }
    __syncthreads();
    /* int8 quant (single per-row scale) of the built row. */
    float *scl = kv_scale + (size_t)s * (slot_stride / KVD) + p;
    quant_row_i8(dst, scl, srow, KVD, tid, nt, red);
}

/* ======================================================================== *
 *  Chunked/batched-prefill helpers.
 *
 *  kv_scatter: after batched prefill builds one contiguous latent-KV buffer
 *  kvlin[n_tok][KVD] (rmsnorm'd + rope'd), copy each token's row into its slot's
 *  cache region at its local position, so the absorbed decode path can read it.
 *  grid = (n_tok); block = 256. slot[t]/pos[t] give the destination slot & pos.
 *
 *  gather_rows: pull the last-token hidden row of each prompt in the chunk into a
 *  dense [nseq][H] buffer (idx[i] = global row index) for the final lm_head.
 *  grid = (nseq); block = 256.
 * ======================================================================== */
/* Block-wide symmetric int8 quant of one KVD row (256 threads). Writes dst[KVD]
 * int8 and *scale_out (single per-row scale). Uses a shared max-reduction. A
 * 2-scale variant (c_kv vs k_pe) restored accuracy but its split-loop reader
 * regressed throughput below bf16 -- see report/run16_kvcache_int8/. */
__device__ __forceinline__ void quant_row_i8(int8_t *dst, float *scale_out,
                                              const float *src, int KVD,
                                              int tid, int nt, float *red) {
    float mx = 0.0f;
    for (int i = tid; i < KVD; i += nt) mx = fmaxf(mx, fabsf(src[i]));
    red[tid] = mx; __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid+st]); __syncthreads(); }
    float scale = red[0] / 127.0f;
    if (tid == 0) *scale_out = scale;
    float qinv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int i = tid; i < KVD; i += nt) {
        float q = rintf(src[i] * qinv);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        dst[i] = (int8_t)q;
    }
}

__global__ void kv_scatter_kernel(int8_t *cache_base, float *kv_scale,
                                  int slot_stride, const int *slot, const int *pos,
                                  const float *kvlin, int KVD, int n_tok) {
    int t = blockIdx.x;
    if (t >= n_tok) return;
    int8_t *dst = cache_base + (size_t)slot[t] * slot_stride + (size_t)pos[t] * KVD;
    float *scl  = kv_scale + (size_t)slot[t] * (slot_stride / KVD) + pos[t];
    const float *src = kvlin + (size_t)t * KVD;
    __shared__ float red[256];
    quant_row_i8(dst, scl, src, KVD, threadIdx.x, blockDim.x, red);
}

/* Per-row int8 quant of a contiguous [n_rows][KVD] block (grid = n_rows). */
__global__ void quant_kv_rows_kernel(int8_t *dst, float *scale, const float *src,
                                     int KVD) {
    int r = blockIdx.x;
    __shared__ float red[256];
    quant_row_i8(dst + (size_t)r * KVD, scale + r, src + (size_t)r * KVD, KVD,
                 threadIdx.x, blockDim.x, red);
}

__global__ void gather_rows_kernel(float *dst, const float *src, const int *idx,
                                   int H) {
    int i = blockIdx.x;
    const float *s = src + (size_t)idx[i] * H;
    float *d = dst + (size_t)i * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) d[j] = s[j];
}


/* Batched greedy argmax: grid = (B), one block per row -> next token per row. */
__global__ void argmax_batched_kernel(const float *logits, int n, int y_stride,
                                      int *out) {
    int s = blockIdx.x;
    const float *row = logits + (size_t)s * y_stride;
    __shared__ float vsh[256];
    __shared__ int   ish[256];
    int tid = threadIdx.x, nt = blockDim.x;
    float bv = -INFINITY; int bi = 0;
    for (int i = tid; i < n; i += nt)
        if (row[i] > bv) { bv = row[i]; bi = i; }
    vsh[tid] = bv; ish[tid] = bi;
    __syncthreads();
    for (int st = nt / 2; st > 0; st >>= 1) {
        if (tid < st) {
            if (vsh[tid + st] > vsh[tid] ||
                (vsh[tid + st] == vsh[tid] && ish[tid + st] < ish[tid])) {
                vsh[tid] = vsh[tid + st]; ish[tid] = ish[tid + st];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[s] = ish[0];
}

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
                                 int *sorted_slot, float *sorted_wt) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int e = topk_idx[t];
    int p = eoff[e] + atomicAdd(&cursor[e], 1);
    sorted_slot[p] = t / K;          /* originating batch row */
    sorted_wt[p]   = topk_wt[t];
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

using i32x4 = __attribute__((__vector_size__(16))) int;

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

/* out[slot] += sorted_wt[p] * down_sorted[p]  (atomic; slots repeat across K).
 * grid = (ceil(H/256), T). */
__global__ void moe_scatter_add_kernel(float *out, const float *down_sorted,
                                       const int *sorted_slot,
                                       const float *sorted_wt, int T, int H) {
    int p = blockIdx.y;
    if (p >= T) return;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= H) return;
    int slot = sorted_slot[p];
    atomicAdd(&out[(size_t)slot * H + d], sorted_wt[p] * down_sorted[(size_t)p * H + d]);
}
