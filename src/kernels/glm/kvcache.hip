/* Latent-KV cache writers and the per-row RoPE they depend on.
 *
 * The cache is int8: [n_layers][B][g_cap][KVD] plus one fp32 scale per (slot, pos)
 * in kv_scale. int8 quarters what was an fp32 cache (~24 GB at B=384) and is what
 * makes B=1280 fit at all. Each row is [c_kv (KVL) ++ k_pe (QKR)]: the c_kv half is
 * rmsnormed, the k_pe half is RoPE'd, then the whole row is quantised together.
 *
 *   rope_pos_kernel   RoPE at a per-row absolute position (each slot in a decode
 *                     batch sits at a different position)
 *   kv_write_batched  decode: one row per slot at pos[slot]
 *   kv_scatter        prefill: a contiguous kvlin[T][KVD] block scattered to
 *                     (slot[t], pos[t])
 * ========================================================================== */
#include "kernels.h"

/* Block-wide symmetric int8 quant of one KVD row (256 threads). Writes dst[KVD]
 * int8 and *scale_out (single per-row scale). Uses a shared max-reduction. A
 * 2-scale variant (c_kv vs k_pe) restored accuracy but its split-loop reader
 * regressed throughput below bf16 -- see report/run16_kvcache_int8/. */
__device__ __forceinline__ void quant_row_i8(int8_t *dst, float *scale_out,
                                              const float *src, int KVD,
                                              int tid, int nt, float *red) {
    float mx = 0.0f;
    for (int i = tid; i < KVD; i += nt) mx = fmaxf(mx, fabsf(src[i]));
    float scale = block_reduce_max_w(mx, red) / 127.0f;   /* run47 */
    if (tid == 0) *scale_out = scale;
    float qinv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int i = tid; i < KVD; i += nt) {
        float q = rintf(src[i] * qinv);
        q = fminf(127.0f, fmaxf(-127.0f, q));
        dst[i] = (int8_t)q;
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
                                        const float *inv_freq, int interleaved,
                                        const int *kvslot) {
    int s = blockIdx.x;
    /* run67: slots holding the SAME prompt share one KV slot. Their activations are
     * identical (each row's forward is independent of the other rows), so the
     * duplicate writes land on the same bytes with the same values -- benign -- and
     * the cache costs one slot per DISTINCT prompt instead of one per request. */
    int ks = kvslot ? kvslot[s] : s;
    int p = pos[s];
    int8_t *dst = cache_base + (size_t)ks * slot_stride + (size_t)p * KVD;  /* int8 cache */
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
    float *scl = kv_scale + (size_t)ks * (slot_stride / KVD) + p;
    quant_row_i8(dst, scl, srow, KVD, tid, nt, red);
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

/* bf16 twin of rope_pos_kernel for the prefill Q tile, which is bf16 from run65
 * on (the attention kernel narrowed it to bf16 anyway). Angles and the rotation
 * itself stay fp32; only the load and store are 2 B. */
__global__ void rope_pos_bf_kernel(bf16_t *v, int n_heads, int row_stride,
                                   int head_stride, const int *pos, int rope_dim,
                                   int interleaved, const float *inv_freq) {
    (void)n_heads;
    int s = blockIdx.x, h = blockIdx.y, half = rope_dim / 2, j = threadIdx.x;
    if (j >= half) return;
    bf16_t *base = v + (size_t)s * row_stride + (size_t)h * head_stride;
    float ang = (float)pos[s] * inv_freq[j];
    float c = cosf(ang), sn = sinf(ang);
    if (!interleaved) {
        float x0 = bf16f(base[2 * j]), x1 = bf16f(base[2 * j + 1]);
        __bf16 o0 = (__bf16)(x0 * c - x1 * sn), o1 = (__bf16)(x0 * sn + x1 * c);
        base[2 * j]     = *reinterpret_cast<const bf16_t *>(&o0);
        base[2 * j + 1] = *reinterpret_cast<const bf16_t *>(&o1);
        return;
    }
    extern __shared__ float smem[];
    float even = bf16f(base[2 * j]), odd = bf16f(base[2 * j + 1]);
    smem[j]        = even * c - odd * sn;
    smem[half + j] = odd  * c + even * sn;
    __syncthreads();
    __bf16 a = (__bf16)smem[j], b = (__bf16)smem[half + j];
    base[j]        = *reinterpret_cast<const bf16_t *>(&a);
    base[half + j] = *reinterpret_cast<const bf16_t *>(&b);
}

/* ======================================================================== *
 *  run67: duplicate-prompt prefill sharing.
 *
 *  The evaluation set is a duplicated one -- 1792 requests over 512 distinct
 *  prompts -- and prefill is the half of the run that scales with the number of
 *  requests (38.1 s of 63 s at B=1792). Identical prompts produce identical KV
 *  rows, so only the first occurrence is prefilled; every other slot gets a byte
 *  copy of its leader's cache. Copying is 8.4 GB of device-to-device traffic
 *  against 27 chunk-forwards of real work, i.e. ~14 ms against ~27 s.
 *
 *  Both kernels are launched once for the whole group: grid.y indexes the
 *  follower, grid.z the layer.
 * ======================================================================== */
/* The rest of a slot's post-prefill state: the token the prefill argmax produced
 * (which is both the first emitted token and the decode's first input) and its
 * column 0 in the token record. */
__global__ void slot_dup_state_kernel(int *tokens, int *tokrec, int rec_stride,
                                      const int *src, const int *dst, int nfol) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nfol) return;
    tokens[dst[f]] = tokens[src[f]];
    tokrec[(size_t)dst[f] * rec_stride] = tokrec[(size_t)src[f] * rec_stride];
}
