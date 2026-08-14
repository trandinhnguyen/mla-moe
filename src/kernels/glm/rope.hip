/* In-place RoPE, ported from rope_apply() in src/run.c. grid = (n_rows,
 * n_heads); block = rope_dim/2 threads. `v` points at the rope_dim slice for
 * row 0 / head 0; row_stride/head_stride locate row blockIdx.x / head
 * blockIdx.y. pos for row r is pos_base + r (prefill: pos_base=0, r=p;
 * decode: pos_base=pos, n_rows=1). */
#include "../kernels.h"

__global__ void rope_kernel(float *v, int n_heads, int row_stride,
                            int head_stride, int pos_base, int rope_dim,
                            int interleaved, const float *inv_freq) {
    (void)n_heads;
    int row = blockIdx.x;
    int head = blockIdx.y;
    int pos = pos_base + row;
    int half = rope_dim / 2;
    int j = threadIdx.x;

    float *base = v + (size_t)row * row_stride + (size_t)head * head_stride;
    if (j >= half) return;

    float ang = (float)pos * inv_freq[j];
    float c = cosf(ang), s = sinf(ang);

    if (!interleaved) {
        float x0 = base[2 * j], x1 = base[2 * j + 1];
        base[2 * j]     = x0 * c - x1 * s;
        base[2 * j + 1] = x0 * s + x1 * c;
        return;
    }

    extern __shared__ float smem[]; /* rope_dim floats */
    float even = base[2 * j], odd = base[2 * j + 1];
    smem[j]        = even * c - odd * s;
    smem[half + j]  = odd  * c + even * s;
    __syncthreads();
    base[j]        = smem[j];
    base[half + j] = smem[half + j];
}
