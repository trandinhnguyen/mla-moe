/* Symmetric int8 quantisation, weights and activations.
 *
 * Weights (quantize_rows_i8) are converted once in warm_up: bf16 -> int8 with one
 * fp32 scale per output row, after which the bf16 copy is freed (~27 GB). Only
 * lm_head and moe_gate stay bf16 -- int8 there perturbs the greedy argmax and the
 * top-K routing respectively, for no measurable throughput.
 *
 * Activations are quantised per row on every use, immediately before the GEMM that
 * consumes them, so the int8 MFMA path never has to dequantise inside its K loop
 * (that VALU work used to be the MoE GEMM's bottleneck -- run19).
 * ========================================================================== */
#include "kernels.h"

/* Symmetric int8 quantisation of four values with a precomputed 1/scale. */
__device__ __forceinline__ char4 quant4(float4 v, float inv) {
    char4 o;
    o.x = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v.x * inv)));
    o.y = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v.y * inv)));
    o.z = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v.z * inv)));
    o.w = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(v.w * inv)));
    return o;
}

/* ======================================================================== *
 *  run41: single-pass, float4 version of quant_act_rows_i8_kernel.
 *  The original reads the row TWICE (once for the abs-max, once to quantize)
 *  and one scalar float at a time. The kernel is purely memory-bound and is
 *  called on every MoE/dense activation (23760 launches, 2.86 s = 4.7% of all
 *  kernel time), so the second pass is ~half its cost. Here each thread keeps
 *  its slice in registers between the two phases and moves 16 B at a time.
 *  Requires K % 4 == 0 and K <= 4*QA_REGS*256; the caller falls back to the
 *  scalar kernel otherwise.
 *
 *  run65: QA_REGS 2 -> 4, i.e. the limit goes 2048 -> 4096. At 2048 the o_proj
 *  activation (K = NH*VHD = 2560) missed the fast path and took the scalar
 *  two-pass kernel -- 1968 launches, 664 ms, 1.8% of all kernel time, at 4 B per
 *  load. Two more named slots cost 8 VGPRs on a kernel that has registers to
 *  spare. Only the dense-FFN intermediate (10240) still falls back, and that is
 *  one layer of 47.
 * ======================================================================== */
#define QA_REGS 4

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

__global__ void quant_act_rows_i8_fast_kernel(const float *X, int8_t *Q, float *scale,
                                             int rows, int K, int x_stride) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float4 *xr = reinterpret_cast<const float4 *>(X + (size_t)r * x_stride);
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x;
    const int K4 = K >> 2;
    __shared__ float red[4];

    /* The slots are NAMED, not an indexed array: with `float4 v[QA_REGS]` and a
     * runtime index the compiler cannot promote them to registers and puts the array
     * in scratch (measured: 48 bytes/lane, VGPRs 10) -- which defeats the entire
     * point of this kernel, whose only job is to keep the row in registers between
     * the abs-max pass and the quantise pass. */
    const int i0 = tid, i1 = tid + nt, i2 = tid + 2 * nt, i3 = tid + 3 * nt;
    float4 v0 = {0.0f, 0.0f, 0.0f, 0.0f}, v1 = v0, v2 = v0, v3 = v0;
    float mx = 0.0f;
    if (i0 < K4) {
        v0 = xr[i0];
        mx = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    }
    if (i1 < K4) {
        v1 = xr[i1];
        mx = fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fmaxf(fabsf(v1.z), fabsf(v1.w)), mx));
    }
    if (i2 < K4) {
        v2 = xr[i2];
        mx = fmaxf(fmaxf(fabsf(v2.x), fabsf(v2.y)), fmaxf(fmaxf(fabsf(v2.z), fabsf(v2.w)), mx));
    }
    if (i3 < K4) {
        v3 = xr[i3];
        mx = fmaxf(fmaxf(fabsf(v3.x), fabsf(v3.y)), fmaxf(fmaxf(fabsf(v3.z), fabsf(v3.w)), mx));
    }
    float sc = block_reduce_max_w(mx, red) / 127.0f;
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    if (i0 < K4) *reinterpret_cast<char4 *>(qr + i0 * 4) = quant4(v0, inv);
    if (i1 < K4) *reinterpret_cast<char4 *>(qr + i1 * 4) = quant4(v1, inv);
    if (i2 < K4) *reinterpret_cast<char4 *>(qr + i2 * 4) = quant4(v2, inv);
    if (i3 < K4) *reinterpret_cast<char4 *>(qr + i3 * 4) = quant4(v3, inv);
}

/* ======================================================================== *
 *  Fused SwiGLU + activation quantisation.
 *
 *  The MoE hidden path used to run silu_mul (write the fp32 product back over
 *  hb) and then quant_act (read it again) as two kernels. Both are pure streaming
 *  work, so that round trip cost a full write plus a full read of a T x I fp32
 *  buffer -- 504 MB per layer per prefill chunk at T=40960, I=1536. Here the
 *  product is computed, reduced for the row max and quantised without ever
 *  leaving registers.
 *
 *  Same register budget rule as quant_act_rows_i8_fast_kernel: K % 4 == 0 and
 *  K <= 4*QA_REGS*256 = 2048, which covers the MoE intermediate (1536).
 * ======================================================================== */
/* bf16-input twin of the kernel above, for the MoE path whose two grouped GEMMs
 * now write bf16 (see moe.hip). Reads 8 B per lane per slot instead of 16 and is
 * otherwise identical -- the product is quantised to int8 immediately, so the bf16
 * mantissa is not the limiting error. */
__global__ void silu_mul_quant_bf_kernel(const bf16_t *hb, const bf16_t *hb2,
                                        int8_t *Q, float *scale, int rows, int K) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const bf16_t *ar = hb  + (size_t)r * K;
    const bf16_t *br = hb2 + (size_t)r * K;
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x;
    const int K4 = K >> 2;
    __shared__ float red[4];

    const int i0 = tid, i1 = tid + nt, i2 = tid + 2 * nt, i3 = tid + 3 * nt;
    float4 v0 = {0.0f, 0.0f, 0.0f, 0.0f}, v1 = v0, v2 = v0, v3 = v0;
    float mx = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 4; slot++) {
        int idx = (slot == 0) ? i0 : (slot == 1) ? i1 : (slot == 2) ? i2 : i3;
        if (idx >= K4) continue;
        uint16_t xb[4], yb[4];
        __builtin_memcpy(xb, ar + idx * 4, 4 * sizeof(uint16_t));
        __builtin_memcpy(yb, br + idx * 4, 4 * sizeof(uint16_t));
        float4 o;
        o.x = silu_f(bf16f(xb[0])) * bf16f(yb[0]);
        o.y = silu_f(bf16f(xb[1])) * bf16f(yb[1]);
        o.z = silu_f(bf16f(xb[2])) * bf16f(yb[2]);
        o.w = silu_f(bf16f(xb[3])) * bf16f(yb[3]);
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w))));
        if (slot == 0) v0 = o; else if (slot == 1) v1 = o; else if (slot == 2) v2 = o; else v3 = o;
    }
    float sc = block_reduce_max_w(mx, red) / 127.0f;
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    if (i0 < K4) *reinterpret_cast<char4 *>(qr + i0 * 4) = quant4(v0, inv);
    if (i1 < K4) *reinterpret_cast<char4 *>(qr + i1 * 4) = quant4(v1, inv);
    if (i2 < K4) *reinterpret_cast<char4 *>(qr + i2 * 4) = quant4(v2, inv);
    if (i3 < K4) *reinterpret_cast<char4 *>(qr + i3 * 4) = quant4(v3, inv);
}

__global__ void silu_mul_quant_kernel(const float *hb, const float *hb2,
                                      int8_t *Q, float *scale, int rows, int K) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float4 *ar = reinterpret_cast<const float4 *>(hb  + (size_t)r * K);
    const float4 *br = reinterpret_cast<const float4 *>(hb2 + (size_t)r * K);
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x;
    const int K4 = K >> 2;
    __shared__ float red[4];

    /* Named slots, not an indexed array -- see quant_act_rows_i8_fast_kernel. */
    const int i0 = tid, i1 = tid + nt;
    float4 v0 = {0.0f, 0.0f, 0.0f, 0.0f}, v1 = v0;
    float mx = 0.0f;
    if (i0 < K4) {
        float4 x = ar[i0], y = br[i0];
        v0.x = silu_f(x.x) * y.x; v0.y = silu_f(x.y) * y.y;
        v0.z = silu_f(x.z) * y.z; v0.w = silu_f(x.w) * y.w;
        mx = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
    }
    if (i1 < K4) {
        float4 x = ar[i1], y = br[i1];
        v1.x = silu_f(x.x) * y.x; v1.y = silu_f(x.y) * y.y;
        v1.z = silu_f(x.z) * y.z; v1.w = silu_f(x.w) * y.w;
        mx = fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fmaxf(fabsf(v1.z), fabsf(v1.w)), mx));
    }
    float sc = block_reduce_max_w(mx, red) / 127.0f;
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    if (i0 < K4) *reinterpret_cast<char4 *>(qr + i0 * 4) = quant4(v0, inv);
    if (i1 < K4) *reinterpret_cast<char4 *>(qr + i1 * 4) = quant4(v1, inv);
}

/* bf16-activation twins (run65). Same math, half the load bytes: used where the
 * producer now writes bf16 (the prefill attention context, the FFN intermediates).
 * The int8 result is bit-identical to quantising the bf16 values in fp32 form. */
__global__ void quant_act_rows_i8_bf_kernel(const bf16_t *X, int8_t *Q, float *scale,
                                            int rows, int K, int x_stride) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const bf16_t *xr = X + (size_t)r * x_stride;
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x;
    const int K4 = K >> 2;
    __shared__ float red[4];
    const int i0 = tid, i1 = tid + nt, i2 = tid + 2 * nt, i3 = tid + 3 * nt;
    float4 v0 = {0.0f, 0.0f, 0.0f, 0.0f}, v1 = v0, v2 = v0, v3 = v0;
    float mx = 0.0f;
#pragma unroll
    for (int slot = 0; slot < 4; slot++) {
        int idx = (slot == 0) ? i0 : (slot == 1) ? i1 : (slot == 2) ? i2 : i3;
        if (idx >= K4) continue;
        uint16_t xb[4];
        __builtin_memcpy(xb, xr + idx * 4, 4 * sizeof(uint16_t));
        float4 o = {bf16f(xb[0]), bf16f(xb[1]), bf16f(xb[2]), bf16f(xb[3])};
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w))));
        if (slot == 0) v0 = o; else if (slot == 1) v1 = o; else if (slot == 2) v2 = o; else v3 = o;
    }
    float sc = block_reduce_max_w(mx, red) / 127.0f;
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    if (i0 < K4) *reinterpret_cast<char4 *>(qr + i0 * 4) = quant4(v0, inv);
    if (i1 < K4) *reinterpret_cast<char4 *>(qr + i1 * 4) = quant4(v1, inv);
    if (i2 < K4) *reinterpret_cast<char4 *>(qr + i2 * 4) = quant4(v2, inv);
    if (i3 < K4) *reinterpret_cast<char4 *>(qr + i3 * 4) = quant4(v3, inv);
}

/* Two-pass form for rows too wide to keep in registers: the dense-layer FFN
 * intermediate (10240), one layer of 47. Reads the row twice but 8 B at a time. */
__global__ void silu_mul_quant_bf_big_kernel(const bf16_t *hb, const bf16_t *hb2,
                                            int8_t *Q, float *scale, int rows, int K) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const bf16_t *ar = hb + (size_t)r * K, *br = hb2 + (size_t)r * K;
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x, K4 = K >> 2;
    __shared__ float red[4];
    float mx = 0.0f;
    for (int i = tid; i < K4; i += nt) {
        uint16_t xb[4], yb[4];
        __builtin_memcpy(xb, ar + i * 4, 4 * sizeof(uint16_t));
        __builtin_memcpy(yb, br + i * 4, 4 * sizeof(uint16_t));
#pragma unroll
        for (int j = 0; j < 4; j++)
            mx = fmaxf(mx, fabsf(silu_f(bf16f(xb[j])) * bf16f(yb[j])));
    }
    float sc = block_reduce_max_w(mx, red) / 127.0f;
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    for (int i = tid; i < K4; i += nt) {
        uint16_t xb[4], yb[4];
        __builtin_memcpy(xb, ar + i * 4, 4 * sizeof(uint16_t));
        __builtin_memcpy(yb, br + i * 4, 4 * sizeof(uint16_t));
        float4 o;
        o.x = silu_f(bf16f(xb[0])) * bf16f(yb[0]);
        o.y = silu_f(bf16f(xb[1])) * bf16f(yb[1]);
        o.z = silu_f(bf16f(xb[2])) * bf16f(yb[2]);
        o.w = silu_f(bf16f(xb[3])) * bf16f(yb[3]);
        *reinterpret_cast<char4 *>(qr + i * 4) = quant4(o, inv);
    }
}

/* Two-pass bf16 quantiser for rows wider than the register form can hold
 * (4*QA_REGS*256 = 4096). The attention context is NH*VHD = 5120 wide, so this is
 * the o_proj activation path -- reading the row twice is still half the bytes the
 * fp32 scalar kernel moved, and 8 B per load instead of 4. */
__global__ void quant_act_rows_i8_bf_big_kernel(const bf16_t *X, int8_t *Q, float *scale,
                                                int rows, int K, int x_stride) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const bf16_t *xr = X + (size_t)r * x_stride;
    int8_t *qr = Q + (size_t)r * K;
    int tid = threadIdx.x, nt = blockDim.x, K4 = K >> 2;
    __shared__ float red[4];
    float mx = 0.0f;
    for (int i = tid; i < K4; i += nt) {
        uint16_t xb[4];
        __builtin_memcpy(xb, xr + i * 4, 4 * sizeof(uint16_t));
#pragma unroll
        for (int j = 0; j < 4; j++) mx = fmaxf(mx, fabsf(bf16f(xb[j])));
    }
    float sc = block_reduce_max_w(mx, red) / 127.0f;
    if (tid == 0) scale[r] = sc;
    float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    for (int i = tid; i < K4; i += nt) {
        uint16_t xb[4];
        __builtin_memcpy(xb, xr + i * 4, 4 * sizeof(uint16_t));
        float4 o = {bf16f(xb[0]), bf16f(xb[1]), bf16f(xb[2]), bf16f(xb[3])};
        *reinterpret_cast<char4 *>(qr + i * 4) = quant4(o, inv);
    }
}
