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

/* run65: single-pass float4 RMSNorm. The kernel above streams the row TWICE
 * (sum-of-squares, then normalize) one scalar at a time; it is purely
 * memory-bound, so the second read is ~half its cost, and 4-byte loads leave the
 * memory pipe starved on top of that. Here each thread keeps its slice in
 * registers across the block reduction and moves 16 B per instruction.
 *
 * The remaining scalar-kernel callers after the residual+norm fusion are
 * q_a_layernorm (n=1536) and kv_a_layernorm (n=512, stride 576) once per layer per
 * forward -- ~4100 launches. Needs n % 4 == 0, strides % 4 == 0 and
 * n <= 4*RN_REGS*256 = 2048; launch_rmsnorm falls back otherwise.
 *
 * NOT bit-identical to the scalar kernel: the sum of squares is over the same
 * values but grouped 4-at-a-time per thread instead of stride-256, so it rounds
 * differently in the last bits. Accuracy gate re-run. Slots are NAMED for the
 * same reason as in quant.hip -- an indexed array lands in scratch. */
#define RN_REGS 2

__global__ void rmsnorm_f4_kernel(float *y, const float *x, const bf16_t *w,
                                  int n, float eps, int x_row_stride,
                                  int y_row_stride) {
    int row = blockIdx.x;
    const float4 *xr = reinterpret_cast<const float4 *>(x + (size_t)row * x_row_stride);
    float4 *yr = reinterpret_cast<float4 *>(y + (size_t)row * y_row_stride);
    const int tid = threadIdx.x, nt = blockDim.x, n4 = n >> 2;
    const int i0 = tid, i1 = tid + nt;
    float4 v0 = {0.0f, 0.0f, 0.0f, 0.0f}, v1 = v0;
    float ss = 0.0f;
    if (i0 < n4) { v0 = xr[i0]; ss += v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w; }
    if (i1 < n4) { v1 = xr[i1]; ss += v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w; }
    float inv = 1.0f / sqrtf(block_reduce_sum(ss) / (float)n + eps);
    if (i0 < n4) {
        float4 g = {bf16f(w[i0 * 4 + 0]), bf16f(w[i0 * 4 + 1]),
                    bf16f(w[i0 * 4 + 2]), bf16f(w[i0 * 4 + 3])};
        float4 o = {v0.x * inv * g.x, v0.y * inv * g.y, v0.z * inv * g.z, v0.w * inv * g.w};
        yr[i0] = o;
    }
    if (i1 < n4) {
        float4 g = {bf16f(w[i1 * 4 + 0]), bf16f(w[i1 * 4 + 1]),
                    bf16f(w[i1 * 4 + 2]), bf16f(w[i1 * 4 + 3])};
        float4 o = {v1.x * inv * g.x, v1.y * inv * g.y, v1.z * inv * g.z, v1.w * inv * g.w};
        yr[i1] = o;
    }
}


/* Fused residual-add + RMSNorm (technique borrowed from the DeepSeek engine's
 * batched_residual_rmsnorm_kernel). The two used to be separate kernels, so the
 * updated residual was written to HBM and immediately read back: five buffer
 * passes where four suffice. Writes both the updated residual (needed by the next
 * layer) and the normalised copy. */
__global__ void residual_rmsnorm_kernel(float *x, float *y, const float *branch,
                                        const bf16_t *w, int n, float eps,
                                        int x_stride, int y_stride) {
    int row = blockIdx.x;
    float *xr = x + (size_t)row * x_stride;
    const float *br = branch + (size_t)row * x_stride;
    float *yr = y + (size_t)row * y_stride;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i] + br[i];
        xr[i] = v;
        partial += v * v;
    }
    float inv = 1.0f / sqrtf(block_reduce_sum(partial) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = xr[i] * inv * bf16f(w[i]);
}

__global__ void residual_add_kernel(float *xs, const float *add, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xs[i] += add[i];
}

/* float4 form of the above (n % 4 == 0, both pointers 16 B aligned). Same adds in
 * the same order, so bit-identical; only the load/store width changes. */
__global__ void residual_add_f4_kernel(float *xs, const float *add, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 a = reinterpret_cast<float4 *>(xs)[i];
    float4 b = reinterpret_cast<const float4 *>(add)[i];
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    reinterpret_cast<float4 *>(xs)[i] = a;
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

/* Second half of the fused-argmax lm_head: reduce the per-tile (max,index) pairs
 * to one token per row, then do exactly what argmax_batched_kernel did with it --
 * record it in the decode history and advance the cache position. One block per
 * row; `tiles` is ceil(vocab/128), about 1210. */
__global__ void argmax_reduce_kernel(const float *pval, const int *pidx, int tiles,
                                     int *out, const int *slotmap, int *tokrec,
                                     int rec_stride, int step, const int *stepp,
                                     int *pos, int cap) {
    int s = blockIdx.x;
    const float *vr = pval + (size_t)s * tiles;
    const int   *ir = pidx + (size_t)s * tiles;
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ float shv[4];
    __shared__ int   shi[4];

    float bv = -INFINITY; int bi = 0;
    for (int t = tid; t < tiles; t += nt) {
        float v = vr[t]; int i = ir[t];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    int lane = tid & 63, wave = tid >> 6, nw = nt >> 6;
#pragma unroll
    for (int m = 1; m < 64; m <<= 1) {
        float ov = __shfl_xor(bv, m, 64);
        int   oi = __shfl_xor(bi, m, 64);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { shv[wave] = bv; shi[wave] = bi; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < nw; w++)
            if (shv[w] > bv || (shv[w] == bv && shi[w] < bi)) { bv = shv[w]; bi = shi[w]; }
        int d = slotmap ? slotmap[s] : s;
        out[d] = bi;
        if (tokrec) tokrec[(size_t)d * rec_stride + (stepp ? *stepp : step)] = bi;
        if (pos) { int np = pos[d] + 1; pos[d] = np < cap ? np : cap - 1; }
    }
}
