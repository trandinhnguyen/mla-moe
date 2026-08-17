/* Embedding lookup, RMSNorm, residual/elementwise ops, and a generic causal-
 * aware row softmax. All mirror the scalar CPU reference in src/run.c op for
 * op (same reduction order per row) so results stay within eval tolerance. */
#include "kernels.h"

#define REDUCE_THREADS 256

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float sh[REDUCE_THREADS];
    int tid = threadIdx.x;
    sh[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    return sh[0];
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float sh[REDUCE_THREADS];
    int tid = threadIdx.x;
    sh[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    return sh[0];
}

__global__ void embed_lookup_kernel(float *xs, const bf16_t *embed_tokens,
                                    const int *tokens, int H) {
    int row = blockIdx.x;
    const bf16_t *src = embed_tokens + (size_t)tokens[row] * H;
    float *dst = xs + (size_t)row * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = bf16f(src[i]);
}

__global__ void rmsnorm_kernel(float *y, const float *x, const bf16_t *w,
                               int n, float eps, int x_row_stride,
                               int y_row_stride) {
    int row = blockIdx.x;
    const float *xr = x + (size_t)row * x_row_stride;
    float *yr = y + (size_t)row * y_row_stride;

    float partial = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) partial += xr[i] * xr[i];
    float ss = block_reduce_sum(partial);
    float inv = 1.0f / sqrtf(ss / (float)n + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = xr[i] * inv * bf16f(w[i]);
}

__global__ void residual_add_kernel(float *xs, const float *add, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xs[i] += add[i];
}

__global__ void silu_mul_kernel(float *hb, const float *hb2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hb[i] = silu_f(hb[i]) * hb2[i];
}

__global__ void weighted_add_kernel(float *out, const float *add, float weight,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += weight * add[i];
}

__global__ void copy_strided_kernel(float *dst, const float *src, int n,
                                    int dst_row_stride, int src_row_stride) {
    int row = blockIdx.x;
    const float *s = src + (size_t)row * src_row_stride;
    float *d = dst + (size_t)row * dst_row_stride;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
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


