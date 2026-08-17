/* Embedding lookup, RMSNorm, residual/elementwise ops, and a generic causal-
 * aware row softmax. All mirror the scalar CPU reference in src/run.c op for
 * op (same reduction order per row) so results stay within eval tolerance. */
#include "kernels.h"

/* run43: re-enable FMA contraction here too -- the frozen Makefile passes
 * -ffp-contract=off, which doubles the VALU cost of every fp32 multiply-add.
 * See the note at the top of attention.hip. */
#pragma clang fp contract(fast)


__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float sh[4];
    return block_reduce_sum_w(val, sh);
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float sh[4];
    return block_reduce_max_w(val, sh);
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

/* run43: single-pass float4 RMSNorm. The kernel above streams the row TWICE
 * (sum-of-squares, then normalize) one scalar at a time; it is purely
 * memory-bound, so the second read is ~half its cost. Here each thread keeps its
 * slice in registers across the block reduction and moves 16 B per instruction.
 * Needs n % 4 == 0, strides % 4 == 0 and n <= 4*RN_REGS*256 (= 2048, i.e. the
 * hidden size); callers fall back to the scalar kernel otherwise. */
#define RN_REGS 2


__global__ void residual_add_kernel(float *xs, const float *add, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xs[i] += add[i];
}


__global__ void silu_mul_kernel(float *hb, const float *hb2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hb[i] = silu_f(hb[i]) * hb2[i];
}


__global__ void copy_strided_kernel(float *dst, const float *src, int n,
                                    int dst_row_stride, int src_row_stride) {
    int row = blockIdx.x;
    const float *s = src + (size_t)row * src_row_stride;
    float *d = dst + (size_t)row * dst_row_stride;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

__global__ void gather_rows_kernel(float *dst, const float *src, const int *idx,
                                   int H) {
    int i = blockIdx.x;
    const float *s = src + (size_t)idx[i] * H;
    float *d = dst + (size_t)i * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) d[j] = s[j];
}

/* Batched greedy argmax: grid = (B), one block per row -> next token per row. */
/* Greedy argmax per row, plus the decode loop's two pieces of bookkeeping so the
 * host never has to read a token back at all: `tokrec` records this step's
 * token at [slot][step] and `pos` advances to the next cache position (clamped to
 * cap-1, which only bites if a slot ran out of budget -- its output is trimmed on
 * the host anyway). Pass tokrec = NULL for the prefill's first-token argmax. */
__global__ void argmax_batched_kernel(const float *logits, int n, int y_stride,
                                      int *out, const int *slotmap, int *tokrec,
                                      int rec_stride, int step, const int *stepp,
                                      int *pos, int cap) {
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
    if (tid == 0) {
        /* slotmap != NULL during prefill: row s is the s-th prompt of the chunk,
         * but its token belongs to cache slot slotmap[s]. */
        int d = slotmap ? slotmap[s] : s;
        out[d] = ish[0];
        /* stepp != NULL: the column index lives on the device, which is what
         * lets one captured hipGraph replay for every decode step. */
        if (tokrec) tokrec[(size_t)d * rec_stride + (stepp ? *stepp : step)] = ish[0];
        if (pos) {
            int np = pos[d] + 1;
            pos[d] = np < cap ? np : cap - 1;
        }
    }
}

/* Advances the device-side decode step counter. One thread, launched after the
 * argmax so the two never race; exists only so the captured graph is
 * step-invariant. */
__global__ void bump_step_kernel(int *step) { if (threadIdx.x == 0) *step += 1; }

/* Seeds it. The decode steps write columns 1.. of the record: column 0 already
 * holds the prefill's first token. */
__global__ void reset_step_kernel(int *step, int v) { if (threadIdx.x == 0) *step = v; }

/* fp32 -> bf16 for the GEMMs that take their activation pre-narrowed (lm_head,
 * the per-head W_UK/W_UV pair). Uses the SAME conversion those kernels apply when
 * they narrow in LDS -- the compiler's `(__bf16)` cast, which rounds to nearest,
 * not the truncating f32bf() used for weights -- so pre-narrowing is numerically
 * identical to the path it replaces rather than half an LSB off. */
__global__ void narrow_bf16_kernel(bf16_t *dst, const float *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __bf16 v = (__bf16)src[i];
        dst[i] = *reinterpret_cast<const bf16_t *>(&v);
    }
}
