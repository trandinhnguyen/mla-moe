/* GEMV-per-output-row kernels: one block computes one output scalar via a
 * blockDim-wide reduction over the contraction dimension. Mirrors matmul() in
 * src/run.c (y[d] = sum_i x[i] * W[d,i]) but parallel over (row, d_out). */
#include "../kernels.h"

#define MM_THREADS 256

__device__ __forceinline__ float mm_block_reduce_sum(float val) {
    __shared__ float sh[MM_THREADS];
    int tid = threadIdx.x;
    sh[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    return sh[0];
}

/* grid = (d_out, n_rows); Y[row, d] = sum_i X[row*x_row_stride+i] * bf16f(W[d*n_in+i]) */
__global__ void matmul_kernel(float *Y, const float *X, const bf16_t *W,
                              int d_out, int n_in, int x_row_stride,
                              int y_row_stride) {
    (void)d_out;
    int d   = blockIdx.x;
    int row = blockIdx.y;
    const float  *xr = X + (size_t)row * x_row_stride;
    const bf16_t *wr = W + (size_t)d * n_in;
    int tid = threadIdx.x, nt = blockDim.x;

    /* 8-wide coalesced bf16 loads (consecutive threads -> consecutive 16B). */
    float partial = 0.0f;
    int n8 = n_in & ~7;
    for (int i = tid * 8; i < n8; i += nt * 8) {
        uint16_t wb[8];
        __builtin_memcpy(wb, wr + i, 8 * sizeof(uint16_t));
#pragma unroll
        for (int j = 0; j < 8; j++) partial += xr[i + j] * bf16f(wb[j]);
    }
    for (int i = n8 + tid; i < n_in; i += nt) partial += xr[i] * bf16f(wr[i]);

    float acc = mm_block_reduce_sum(partial);
    if (tid == 0) Y[(size_t)row * y_row_stride + d] = acc;
}

/* One-time transpose of the per-head W_UK block from [QKN, KVL] (d contiguous
 * along KVL) to [KVL, QKN] (d contiguous), so the absorbed q-fold becomes a
 * standard coalesced/vectorizable matmul_batched. grid = (KVL, NH).
 * out[h, r, d] = in[h*kv_stride + d*KVL + r]. */
__global__ void transpose_wuk_kernel(bf16_t *out, const bf16_t *in, int QKN,
                                     int KVL, int kv_stride) {
    int r = blockIdx.x;
    int h = blockIdx.y;
    bf16_t       *o = out + ((size_t)h * KVL + r) * QKN;
    const bf16_t *ih = in + (size_t)h * kv_stride + r;
    for (int d = threadIdx.x; d < QKN; d += blockDim.x)
        o[d] = ih[(size_t)d * KVL];
}


