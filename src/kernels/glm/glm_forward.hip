/* ==========================================================================
 * glm_forward.hip -- the two forward passes and the launch wrappers they use.
 *
 *   device_forward_prefill_batched   several whole prompts packed on one flat
 *                                    token axis -> one forward; attention is
 *                                    causal within each prompt; writes each
 *                                    prompt's latent KV into its cache slot and
 *                                    its first token into b_tokens[slot].
 *   device_forward_absorbed_batched  one decode step for B slots in lockstep;
 *                                    attention runs absorbed (W_UK folded into
 *                                    the query) over each slot's own KV cache.
 *
 * The launch wrappers above them exist to keep those two readable: each one hides
 * a shape decision (which tile, int8 or bf16, fast or scalar quantiser) that is
 * argued for at its definition.
 * ==========================================================================*/
/* ---------------------------------------------------------------------------
 * Host-side launch helpers -- one kernel-launch-sequence per CPU reference op.
 * ------------------------------------------------------------------------- */

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static void launch_rmsnorm(float *y, const float *x, const bf16_t *w, int n_rows,
                           int n, float eps, int x_stride = -1, int y_stride = -1)
{
    if (x_stride < 0) x_stride = n;
    if (y_stride < 0) y_stride = n;
    if ((n & 3) == 0 && (x_stride & 3) == 0 && (y_stride & 3) == 0 && n <= 4 * RN_REGS * 256)
        rmsnorm_f4_kernel<<<n_rows, 256, 0, g_str>>>(y, x, w, n, eps, x_stride, y_stride);
    else
        rmsnorm_kernel<<<n_rows, 256, 0, g_str>>>(y, x, w, n, eps, x_stride, y_stride);
}

/* residual-add immediately followed by the RMSNorm of the same buffer: one kernel
 * (see elementwise.hip). Both call sites per layer have this shape -- after
 * attention, and after the FFN where the norm belongs to the next layer. */
static void launch_residual_rmsnorm(float *x, float *y, const float *branch,
                                    const bf16_t *w, int n_rows, int n, float eps)
{
    residual_rmsnorm_kernel<<<n_rows, 256, 0, g_str>>>(x, y, branch, w, n, eps, n, n);
}

static void launch_residual_add(float *xs, const float *add, int n)
{
    if ((n & 3) == 0)
        residual_add_f4_kernel<<<ceil_div(n / 4, 256), 256, 0, g_str>>>(xs, add, n / 4);
    else
        residual_add_kernel<<<ceil_div(n, 256), 256, 0, g_str>>>(xs, add, n);
}


/* Per-head GEMM (per-head weight and per-head X): qabs = q_nope @ W_UK^T and
 * ctx = clat @ W_UV^T. Picks the 128x128 tile when its coarser grid still fills
 * the CUs -- these two are bandwidth-bound, so halving the re-reads is the whole
 * game (see matmul_bh_mfma_128_kernel). */
static void launch_matmul_bh(float *Y, const float *X, const bf16_t *W, int M, int N,
                             int K, int x_hstride, int x_bstride, int w_stride,
                             int y_hstride, int y_bstride, int NH)
{
    if (M >= 128 && N >= 128 && ceil_div(N, 128) * ceil_div(M, 128) * NH >= 96) {
        dim3 grid(ceil_div(N, 128), ceil_div(M, 128), NH);
        matmul_bh_mfma_128_kernel<<<grid, 256, 0, g_str>>>(Y, X, W, M, N, K, x_hstride,
                                                           x_bstride, w_stride, y_hstride,
                                                           y_bstride);
    } else {
        dim3 grid(ceil_div(N, MM_TILE), ceil_div(M, MM_TILE), NH);
        matmul_bh_mfma_kernel<<<grid, 256, 0, g_str>>>(Y, X, W, M, N, K, x_hstride,
                                                       x_bstride, w_stride, y_hstride,
                                                       y_bstride);
    }
}

static void launch_matmul(float *Y, const float *X, const bf16_t *W, int n_rows,
                          int d_out, int n_in, int x_stride = -1, int y_stride = -1)
{
    if (x_stride < 0) x_stride = n_in;
    if (y_stride < 0) y_stride = d_out;
    int b128 = ceil_div(d_out, 128) * ceil_div(n_rows, 128);
    if (n_rows >= 128 && d_out >= 128 && b128 >= 96)
    {
        /* run21: coarse 128x128 tile halves the bf16 weight bandwidth for the large
         * bandwidth-bound GEMMs (lm_head/shared-expert/dense-layer); bit-identical. */
        dim3 grid(ceil_div(d_out, 128), ceil_div(n_rows, 128));
        matmul_mfma_128_kernel<<<grid, 256, 0, g_str>>>(Y, X, W, n_rows, d_out, n_in, x_stride, y_stride);
    }
    else if (n_rows >= 4)
    {
        dim3 grid(ceil_div(d_out, MM_TILE), ceil_div(n_rows, MM_TILE));
        matmul_mfma_kernel<<<grid, 256, 0, g_str>>>(Y, X, W, n_rows, d_out, n_in, x_stride, y_stride);
    }
    else
    {
        dim3 grid(d_out, n_rows);
        matmul_kernel<<<grid, 256, 0, g_str>>>(Y, X, W, d_out, n_in, x_stride, y_stride);
    }
}

/* run20: W8A8 dense matmul. Routes to the int8 GEMM when g_dense_w8a8 && the int8
 * weight exists && n_rows >= 4 (MFMA path); else falls back to bf16 launch_matmul.
 * Quantizes the fp32 activation X (K = n_in, stride x_stride) into b_act_i8 first. */
/* Per-row symmetric int8 quantisation of an activation block. The fast kernel
 * keeps the row in registers between the abs-max and the quantise pass (one read
 * instead of two) and moves 16 B per instruction, but that caps the row at
 * 4*QA_REGS*256 = 4096 elements. That covers every hot call (hidden 2048, MoE
 * intermediate 1536, o_proj 2560); only the dense-layer FFN down projection
 * (K = 10240) takes the scalar kernel, and that is one layer of 47. */
static void launch_quant_act(const float *X, int8_t *Q, float *scale,
                             int rows, int K, int x_stride)
{
    if ((K & 3) == 0 && (x_stride & 3) == 0 && K <= 4 * QA_REGS * 256)
        quant_act_rows_i8_fast_kernel<<<rows, 256, 0, g_str>>>(X, Q, scale, rows, K, x_stride);
    else
        quant_act_rows_i8_kernel<<<rows, 256, 0, g_str>>>(X, Q, scale, rows, K, x_stride);
}

/* run65: bf16 activation in. Same register-budget split as the fp32 pair above --
 * the attention context is NH*VHD = 5120 wide, past what one register slot set can
 * hold, and quantising only the first 4096 columns of it is exactly the bug this
 * split exists to prevent. */
static void launch_quant_act(const bf16_t *X, int8_t *Q, float *scale,
                             int rows, int K, int x_stride)
{
    if ((K & 3) == 0 && (x_stride & 3) == 0 && K <= 4 * QA_REGS * 256)
        quant_act_rows_i8_bf_kernel<<<rows, 256, 0, g_str>>>(X, Q, scale, rows, K, x_stride);
    else
        quant_act_rows_i8_bf_big_kernel<<<rows, 256, 0, g_str>>>(X, Q, scale, rows, K, x_stride);
}

/* W8A8 dense matmul. Routes on int8-table presence: a NULL Wq means the caller
 * deliberately kept that GEMM in bf16 (lm_head, moe_gate), so it falls through to
 * launch_matmul. Otherwise the activation is quantised per row and the int8 MFMA
 * GEMM runs -- the 128x128 tile when its coarser grid still fills the 104 CUs
 * (halves total L2 traffic: with a square tile the weight and activation re-reads
 * are equal and both scale as 1/tile, so both dims must double), else the 64x64
 * tile, which keeps occupancy on the narrow projections (q_a N=768, kv_a N=576). */
/* lm_head + argmax, fused. Deliberately bf16 weights (int8 here perturbs the
 * greedy argmax for +0.4% throughput -- measured, rejected), activation narrowed
 * to bf16 because with N = vocab there are 1210 column tiles and the kernel
 * narrows in LDS anyway, and the argmax folded into the GEMM epilogue so the
 * logits are never materialised: that buffer was 714 MB -- more than the whole
 * remaining VRAM headroom -- and it was written and read back every step.
 * Technique borrowed from the DeepSeek engine (mfma_bf16_gemm_argmax_32x64). */
static void launch_lm_head_argmax(const float *X, const bf16_t *W, int rows, int d_out,
                                  int n_in, int *out, const int *slotmap, int *tokrec,
                                  int rec_stride, int step, const int *stepp,
                                  int *pos, int cap)
{
    const int tiles = ceil_div(d_out, 128);
    int n = rows * n_in;
    bf16_t *xbf = (bf16_t *)sbuf(B_XBF16, (size_t)n * sizeof(bf16_t));
    narrow_bf16_kernel<<<ceil_div(n, 256), 256, 0, g_str>>>(xbf, X, n);
    float *pv = (float *)sbuf(B_LMPV, (size_t)rows * tiles * sizeof(float));
    int   *pi = (int *)  sbuf(B_LMPI, (size_t)rows * tiles * sizeof(int));
    dim3 grid(tiles, ceil_div(rows, 128));
    matmul_mfma_128_xbf_argmax_kernel<<<grid, 256, 0, g_str>>>(pv, pi, xbf, W, rows,
                                                              d_out, n_in, n_in, tiles);
    argmax_reduce_kernel<<<rows, 256, 0, g_str>>>(pv, pi, tiles, out, slotmap, tokrec,
                                                 rec_stride, step, stepp, pos, cap);
}

template <typename OT>
static void launch_matmul_i8_pre(OT *Y, const int8_t *Xq, const float *xsc,
                                 const int8_t *Wq, const float *Ws,
                                 int n_rows, int d_out, int n_in, int y_stride);

/* W8A8 with an explicit output/activation type. Used where the consumer narrows
 * anyway (bf16 Q tile, bf16 FFN intermediate) or the producer already did (bf16
 * attention context). The int8 weight table is mandatory here -- there is no bf16
 * weight resident to fall back to. */
template <typename OT, typename XT>
static void launch_matmul_w8a8_t(OT *Y, const XT *X, const int8_t *Wq, const float *Ws,
                                 int n_rows, int d_out, int n_in,
                                 int x_stride = -1, int y_stride = -1)
{
    if (x_stride < 0) x_stride = n_in;
    if (y_stride < 0) y_stride = d_out;
    launch_quant_act(X, b_act_i8, b_act_sc, n_rows, n_in, x_stride);
    launch_matmul_i8_pre(Y, b_act_i8, b_act_sc, Wq, Ws, n_rows, d_out, n_in, y_stride);
}

static void launch_matmul_w8a8(float *Y, const float *X, const bf16_t *Wbf,
                               const int8_t *Wq, const float *Ws, int n_rows, int d_out,
                               int n_in, int x_stride = -1, int y_stride = -1)
{
    if (x_stride < 0) x_stride = n_in;
    if (y_stride < 0) y_stride = d_out;
    /* run65: no n_rows floor any more. The bf16 weight is not even uploaded now
     * (see glm_weights.hip), so there is nothing to fall back TO -- the small-M
     * escape hatch existed for the 1-row warm-up decode and would dereference a
     * NULL weight. int8 is the only path for every weight that has a table. */
    if (Wq)
    {
        launch_quant_act(X, b_act_i8, b_act_sc, n_rows, n_in, x_stride);
        launch_matmul_i8_pre(Y, b_act_i8, b_act_sc, Wq, Ws, n_rows, d_out, n_in, y_stride);
    }
    else
    {
        launch_matmul(Y, X, Wbf, n_rows, d_out, n_in, x_stride, y_stride);
    }
}

/* The int8 GEMM alone, for callers that already produced quantised activations
 * (the SwiGLU path fuses silu+quantise into one kernel). */
template <typename OT>
static void launch_matmul_i8_pre(OT *Y, const int8_t *Xq, const float *xsc,
                                 const int8_t *Wq, const float *Ws,
                                 int n_rows, int d_out, int n_in, int y_stride)
{
    {
        int b128 = ceil_div(d_out, 128) * ceil_div(n_rows, 128);
        if (n_rows >= 128 && d_out >= 128 && b128 >= 96 && (n_in & 31) == 0)
        {
            dim3 grid(ceil_div(d_out, 128), ceil_div(n_rows, 128));
            matmul_mfma_i8i8_128_w_kernel<OT><<<grid, 256, 0, g_str>>>(Y, Xq, xsc, Wq, Ws,
                                                         n_rows, d_out, n_in, n_in, y_stride);
        }
        else
        {
            dim3 grid(ceil_div(d_out, MM_TILE), ceil_div(n_rows, MM_TILE));
            matmul_mfma_i8i8_kernel<OT><<<grid, 256, 0, g_str>>>(Y, Xq, xsc, Wq, Ws,
                                                   n_rows, d_out, n_in, n_in, y_stride);
        }
    }
}

/* run65: the two hidden buffers are bf16. They are written by the gate/up GEMMs and
 * read by silu+quantise, which turns them into int8 -- fp32 was 2x the bytes for
 * mantissa nobody read. (void)gate/up/down: the bf16 weights are no longer resident,
 * the int8 tables are mandatory. */
template <typename OT>
static void launch_swiglu_w8a8(OT *dst, const float *x,
                               const bf16_t *gate, const int8_t *gq, const float *gs,
                               const bf16_t *up,   const int8_t *uq, const float *us,
                               const bf16_t *down, const int8_t *dq, const float *ds,
                               int n_rows, int inter, int hidden, bf16_t *hb, bf16_t *hb2)
{
    (void)gate; (void)up; (void)down;
    /* Quantise x ONCE: gate and up take the same activation, and going through
     * launch_matmul_w8a8_t twice quantised it twice into the same buffer -- a full
     * extra read of x plus an extra int8 write per FFN per layer. */
    launch_quant_act(x, b_act_i8, b_act_sc, n_rows, hidden, hidden);
    launch_matmul_i8_pre(hb,  b_act_i8, b_act_sc, gq, gs, n_rows, inter, hidden, inter);
    launch_matmul_i8_pre(hb2, b_act_i8, b_act_sc, uq, us, n_rows, inter, hidden, inter);
    if (inter <= 4 * QA_REGS * 256)
        silu_mul_quant_bf_kernel<<<n_rows, 256, 0, g_str>>>(hb, hb2, b_act_i8, b_act_sc,
                                                            n_rows, inter);
    else   /* dense layer only: inter = 10240, too wide for the register form */
        silu_mul_quant_bf_big_kernel<<<n_rows, 256, 0, g_str>>>(hb, hb2, b_act_i8, b_act_sc,
                                                                n_rows, inter);
    launch_matmul_i8_pre(dst, b_act_i8, b_act_sc, dq, ds, n_rows, hidden, inter, hidden);
}

/* bf16 Q tile (prefill): the attention kernel converts Q to bf16 for its MFMA
 * anyway, so writing it in bf16 halves both the store and every re-read. */
static void launch_project_q_bf(const Config *c, int l, const float *xn, bf16_t *q,
                                int n_rows, float *qa_scratch)
{
    launch_matmul_w8a8_t(qa_scratch, xn, dq_qa[l], ds_qa[l], n_rows, c->q_lora_rank,
                         c->hidden_size);
    launch_rmsnorm(qa_scratch, qa_scratch, d_w.q_a_layernorm[l], n_rows,
                   c->q_lora_rank, c->mla_norm_eps);
    launch_matmul_w8a8_t(q, qa_scratch, dq_qb[l], ds_qb[l], n_rows,
                         c->n_heads * (c->qk_nope_head_dim + c->qk_rope_head_dim),
                         c->q_lora_rank);
}

static void launch_project_q(const Config *c, int l, const float *xn, float *q,
                             int n_rows, float *qa_scratch)
{
    int q_dim = c->n_heads * (c->qk_nope_head_dim + c->qk_rope_head_dim);
    if (c->q_lora_rank > 0)
    {
        launch_matmul_w8a8(qa_scratch, xn, d_w.q_a_proj[l], dq_qa ? dq_qa[l] : NULL,
                           dq_qa ? ds_qa[l] : NULL, n_rows, c->q_lora_rank, c->hidden_size);
        launch_rmsnorm(qa_scratch, qa_scratch, d_w.q_a_layernorm[l], n_rows,
                       c->q_lora_rank, c->mla_norm_eps);
        launch_matmul_w8a8(q, qa_scratch, d_w.q_b_proj[l], dq_qb ? dq_qb[l] : NULL,
                           dq_qb ? ds_qb[l] : NULL, n_rows, q_dim, c->q_lora_rank);
    }
    else
    {
        launch_matmul(q, xn, d_w.q_proj[l], n_rows, q_dim, c->hidden_size);
    }
}

/* Widest FFN hidden buffer needed across dense / per-expert / shared-expert SwiGLU. */
static int inter_max_of(const Config *c)
{
    int inter = c->dense_inter_size;
    int shared = c->moe_inter_size * c->n_shared_experts;
    if (shared > inter) inter = shared;
    if (c->moe_inter_size > inter) inter = c->moe_inter_size;
    return inter;
}

/* Batched FFN shared by prefill and decode. Dense layers -> a single batched
 * SwiGLU (weights reused across all n_rows). MoE layers -> fused expert path:
 * batched router, counting-sort the n_rows*K (token,expert) pairs by expert, one
 * grouped GEMM per expert (each expert's weights read once), weighted scatter,
 * plus the shared expert. `out` is overwritten. hb/hb2 hold >= n_rows*inter_max
 * floats (caller's scratch). Uses the persistent b_* MoE buffers (sized g_cap). */
static void ffn_batched(const Config *c, int l, const float *xn, float *out,
                        int n_rows, bf16_t *hb, bf16_t *hb2)
{
    const int H = c->hidden_size;
    if (l < c->first_k_dense)
    {
        /* The dense FFN is the only place inter_max (10240) is used -- one layer of
         * 47 -- and it is row-parallel, so run it in row blocks of DENSE_ROWS
         * instead of all T rows at once. That is what sizes hb/hb2: at T=16384 the
         * full-width version needed T*10240*4 = 671 MB each, 1.34 GB of scratch for
         * a single layer, on an engine that runs with 0.6 GB of VRAM to spare. */
        const int step = DENSE_ROWS;
        for (int r0 = 0; r0 < n_rows; r0 += step)
        {
            int nr = n_rows - r0 < step ? n_rows - r0 : step;
            launch_swiglu_w8a8(out + (size_t)r0 * H, xn + (size_t)r0 * H,
                d_w.dense_gate[l], dq_dg?dq_dg[l]:NULL, dq_dg?ds_dg[l]:NULL,
                d_w.dense_up[l],   dq_du?dq_du[l]:NULL, dq_du?ds_du[l]:NULL,
                d_w.dense_down[l], dq_dd?dq_dd[l]:NULL, dq_dd?ds_dd[l]:NULL,
                nr, c->dense_inter_size, H, hb, hb2);
        }
        return;
    }
    const int NR = c->n_routed_experts, K = c->n_experts_per_tok, I = c->moe_inter_size;
    const int T = n_rows * K;

    launch_matmul(b_moelog, xn, d_w.moe_gate[l], n_rows, NR, H);
    moe_route_batched_kernel<<<n_rows, 256, 0, g_str>>>(b_moelog, NR, K, c->router_sigmoid,
                                             c->norm_topk, c->routed_scaling, d_moe_bias[l],
                                             b_topk_idx, b_topk_wt);
    HIP_CHECK(hipMemsetAsync(b_ecount, 0, (size_t)NR * sizeof(int), g_str));
    moe_hist_kernel<<<ceil_div(T, 256), 256, 0, g_str>>>(b_topk_idx, T, b_ecount);
    moe_scan_kernel<<<1, 1, 0, g_str>>>(b_ecount, NR, b_eoff);
    HIP_CHECK(hipMemsetAsync(b_cursor, 0, (size_t)NR * sizeof(int), g_str));
    int *b_inv = (int *)sbuf(B_MOEINV, (size_t)T * sizeof(int));
    moe_place_kernel<<<ceil_div(T, 256), 256, 0, g_str>>>(b_topk_idx, b_topk_wt, T, K, b_eoff,
                                               b_cursor, b_sorted_slot, b_sorted_wt, b_inv);
    /* Grouped expert GEMM, W8A8: int8 activation (per-row scale) x int8 expert
     * weight (per-output-row scale) straight into mfma_i32_16x16x16i8 -- no
     * dequant in the loop, which is what the weight-only-int8 kernel used to
     * spend most of its VALU on (run19). One 128x128 tile per (expert, col-tile):
     * with a square tile the weight and activation L2 re-reads are equal, so
     * doubling BOTH dims halves total traffic (run24). Rows per expert are only
     * ~T/NR, so most row-tiles exit immediately -- that "M waste" is free here
     * (the kernel is nowhere near MFMA-bound) and the wide grid is what keeps the
     * CUs fed (run14).
     *
     * Three variants of this were measured and lost: TBK=64 staging (run31),
     * a double-buffered gather (run40b, +0.8% slower -- the indirect A load adds
     * address latency the MFMA block cannot hide) and fusing gate+up into one
     * launch (run11, re-measured run52: -8.2%, two accumulator sets spill or halve
     * occupancy). They live in unused_reference.hip. */
    launch_quant_act(xn, b_xn_i8, b_xn_scale, n_rows, H, H);
    dim3 gGU(ceil_div(I, 128), ceil_div(n_rows, 128), NR);
    dim3 gD(ceil_div(H, 128), ceil_div(n_rows, 128), NR);
    moe_grouped_mfma_i8i8_128_w_kernel<<<gGU, 256, 0, g_str>>>(b_hbk, b_xn_i8, b_xn_scale,
        d_epq_gate[l], d_eps_gate[l], b_eoff, b_sorted_slot, I, H, H, 1);
    moe_grouped_mfma_i8i8_128_w_kernel<<<gGU, 256, 0, g_str>>>(b_hb2k, b_xn_i8, b_xn_scale,
        d_epq_up[l], d_eps_up[l], b_eoff, b_sorted_slot, I, H, H, 1);
    /* silu(gate) * up, reduced and quantised in one pass (see quant.hip). The
     * fp32 fallback is gone: warm_up hard-checks moe_inter <= 4*QA_REGS*256. */
    silu_mul_quant_bf_kernel<<<T, 256, 0, g_str>>>(b_hbk, b_hb2k, b_hbk_i8, b_hbk_scale, T, I);
    moe_grouped_mfma_i8i8_128_w_kernel<<<gD, 256, 0, g_str>>>(b_downk, b_hbk_i8, b_hbk_scale,
        d_epq_down[l], d_eps_down[l], b_eoff, b_sorted_slot, H, I, I, 0);
    /* Shared expert first: the gather below folds it in, so the routed sum, the
     * shared expert and the store all happen in one pass. */
    launch_swiglu_w8a8(b_shared, xn,
        d_w.shared_gate[l], dq_sg?dq_sg[l]:NULL, dq_sg?ds_sg[l]:NULL,
        d_w.shared_up[l],   dq_su?dq_su[l]:NULL, dq_su?ds_su[l]:NULL,
        d_w.shared_down[l], dq_sd?dq_sd[l]:NULL, dq_sd?ds_sd[l]:NULL,
        n_rows, c->n_shared_experts * c->moe_inter_size, H, hb, hb2);
    dim3 gSc(n_rows, ceil_div(H, 256));
    moe_gather_add_kernel<<<gSc, 256, 0, g_str>>>(out, b_downk, b_sorted_wt, b_inv,
                                                 b_shared, K, H);
}

/* Chunked/batched prefill: prefill nseq whole prompts packed on one flat token
 * axis of length T in a SINGLE forward pass, so every layer's weights are read
 * once for the whole chunk instead of once per request. Everything but attention
 * is row-parallel over the T tokens (dense GEMMs, MoE, FFN); attention uses the
 * varlen flash kernel (each query attends only within its own prompt via the
 * per-token q_seqstart). The latent KV is built contiguously (kvlin) and scattered
 * into each prompt's slot cache so the absorbed decode path can read it. The first
 * greedy token of each prompt is written straight into b_tokens[slot] (the decode
 * loop's input) and into column 0 of the token record, so nothing comes back to
 * the host here -- see the decode loop in glm_inference.
 *
 * Host inputs (all length T except last[nseq]): tokcat = concatenated prompt token
 * ids; seqstart[t] = global index of the first token of t's prompt; pos[t] = t's
 * position within its prompt (0-based, for RoPE + cache write); slot[t] = t's
 * prompt's decode slot; last[i] = global index of prompt i's final token. */
static void device_forward_prefill_batched(Transformer *t, int T, int nseq,
                                           const int *tokcat, const int *seqstart,
                                           const int *pos, const int *slot,
                                           const int *last, const int *cslot,
                                           int rec_stride)
{
    const Config *c = &t->config;
    const int H = c->hidden_size, NH = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim, QHD = QKN + QKR;
    const int VHD = c->v_head_dim, KVL = c->kv_lora_rank, KVD = KVL + QKR, VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const int kv_stride = (QKN + VHD) * KVL;
    const size_t layer_stride = (size_t)(g_kvslots + 1) * g_cap * KVD;   /* run67 */
    const int slot_stride = g_cap * KVD;
    const int inter_max = inter_max_of(c);

    /* Upload the per-token index arrays. */
    int *d_tok = (int *)sbuf(B_TOKENS, (size_t)T * sizeof(int));
    int *d_seq = (int *)sbuf(B_PSEQ,   (size_t)T * sizeof(int));
    int *d_pos = (int *)sbuf(B_PPOS,   (size_t)T * sizeof(int));
    int *d_slt = (int *)sbuf(B_PSLOT,  (size_t)T * sizeof(int));
    int *d_lst = (int *)sbuf(B_PLAST,  (size_t)nseq * sizeof(int));
    HIP_CHECK(hipMemcpyAsync(d_tok, tokcat,   (size_t)T * sizeof(int), hipMemcpyHostToDevice, g_str));
    HIP_CHECK(hipMemcpyAsync(d_seq, seqstart, (size_t)T * sizeof(int), hipMemcpyHostToDevice, g_str));
    HIP_CHECK(hipMemcpyAsync(d_pos, pos,      (size_t)T * sizeof(int), hipMemcpyHostToDevice, g_str));
    /* run18: build the MFMA prefill-attention work list — one 64-query tile per
     * (prompt, tile), never crossing a prompt boundary. qt_q0/qs/qe are global
     * first-query / prompt-start / prompt-end (exclusive). */
    static int *h_qtq0 = NULL, *h_qtqs = NULL, *h_qtqe = NULL;
    static int h_qtcap = 0;
    int qt_bound = nseq + (T + 63) / 64 + 1;
    if (qt_bound > h_qtcap) {
        h_qtq0 = (int *)realloc(h_qtq0, (size_t)qt_bound * sizeof(int));
        h_qtqs = (int *)realloc(h_qtqs, (size_t)qt_bound * sizeof(int));
        h_qtqe = (int *)realloc(h_qtqe, (size_t)qt_bound * sizeof(int));
        h_qtcap = qt_bound;
    }
    int n_qtile = 0;
    for (int i = 0; i < nseq; i++) {
        int e = last[i] + 1, s = seqstart[last[i]];
        for (int q0 = s; q0 < e; q0 += 64) {
            h_qtq0[n_qtile] = q0; h_qtqs[n_qtile] = s; h_qtqe[n_qtile] = e; n_qtile++;
        }
    }
    int *d_qtq0 = (int *)sbuf(B_QTQ0, (size_t)n_qtile * sizeof(int));
    int *d_qtqs = (int *)sbuf(B_QTQS, (size_t)n_qtile * sizeof(int));
    int *d_qtqe = (int *)sbuf(B_QTQE, (size_t)n_qtile * sizeof(int));
    HIP_CHECK(hipMemcpyAsync(d_qtq0, h_qtq0, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice, g_str));
    HIP_CHECK(hipMemcpyAsync(d_qtqs, h_qtqs, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice, g_str));
    HIP_CHECK(hipMemcpyAsync(d_qtqe, h_qtqe, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice, g_str));
    HIP_CHECK(hipMemcpyAsync(d_slt, slot,     (size_t)T * sizeof(int), hipMemcpyHostToDevice, g_str));
    HIP_CHECK(hipMemcpyAsync(d_lst, last,     (size_t)nseq * sizeof(int), hipMemcpyHostToDevice, g_str));

    float *qa    = c->q_lora_rank > 0 ? (float *)sbuf(B_QA, (size_t)T * c->q_lora_rank * sizeof(float)) : NULL;
    float *xs    = (float *)sbuf(B_XS,    (size_t)T * H * sizeof(float));
    float *xb    = (float *)sbuf(B_XB,    (size_t)T * H * sizeof(float));
    float *comp  = (float *)sbuf(B_COMP,  (size_t)T * KVD * sizeof(float));
    bf16_t *qall = (bf16_t *)sbuf(B_QALL, (size_t)T * NH * QHD * sizeof(bf16_t));
    float *kvlin = (float *)sbuf(B_KVLIN, (size_t)T * KVD * sizeof(float));
    /* bf16: only attn_prefill_mfma reads these and it narrows them anyway. */
    bf16_t *knope = (bf16_t *)sbuf(B_KNOPE, (size_t)NH * T * QKN * sizeof(bf16_t));
    bf16_t *value = (bf16_t *)sbuf(B_VALUE, (size_t)NH * T * VHD * sizeof(bf16_t));
    bf16_t *ctx  = (bf16_t *)sbuf(B_CTX,  (size_t)T * NH * VHD * sizeof(bf16_t));
    float *ao    = (float *)sbuf(B_AO,    (size_t)T * H * sizeof(float));
    float *mo    = (float *)sbuf(B_MO,    (size_t)T * H * sizeof(float));
    /* hb/hb2 hold one SwiGLU hidden block. The MoE layers' shared expert needs
     * T x moe_inter; the single dense layer needs DENSE_ROWS x dense_inter because
     * ffn_batched chunks its rows (see there). Size for whichever is larger. */
    const size_t hb_elems = (size_t)T * c->moe_inter_size > (size_t)DENSE_ROWS * inter_max
                          ? (size_t)T * c->moe_inter_size : (size_t)DENSE_ROWS * inter_max;
    bf16_t *hb   = (bf16_t *)sbuf(B_HB,   hb_elems * sizeof(bf16_t));
    bf16_t *hb2  = (bf16_t *)sbuf(B_HB2,  hb_elems * sizeof(bf16_t));
    float *lastx = (float *)sbuf(B_PLASTX,(size_t)nseq * H * sizeof(float));

    const size_t rope_sh = c->rope_interleaved ? (size_t)QKR * sizeof(float) : 0;

    embed_lookup_kernel<<<T, 256, 0, g_str>>>(xs, d_w.embed_tokens, d_tok, H);

    for (int l = 0; l < c->n_layers; l++)
    {
        int8_t *cache_base = d_kv_cache + (size_t)l * layer_stride;
        float  *scale_base = d_kv_scale + (size_t)l * (g_kvslots + 1) * g_cap;

        /* Layer 0 only: every later input norm is fused with the FFN residual
         * that precedes it, at the bottom of this loop. */
        if (l == 0) launch_rmsnorm(xb, xs, d_w.input_layernorm[0], T, H, eps);
        launch_project_q_bf(c, l, xb, qall, T, qa);
        rope_pos_bf_kernel<<<dim3(T, NH), QKR / 2, rope_sh, g_str>>>(qall + QKN, NH, NH * QHD, QHD,
                                                           d_pos, QKR, c->rope_interleaved, d_inv_freq);

        launch_matmul_w8a8(comp, xb, d_w.kv_a_proj[l], dq_kva?dq_kva[l]:NULL, dq_kva?ds_kva[l]:NULL, T, KVD, H);
        /* Build the contiguous latent KV (rmsnorm the compressed part, copy + RoPE
         * the k_pe slice at each token's own position), then scatter into slots. */
        launch_rmsnorm(kvlin, comp, d_w.kv_a_layernorm[l], T, KVL, c->mla_norm_eps, KVD, KVD);
        copy_strided_kernel<<<T, 256, 0, g_str>>>(kvlin + KVL, comp + KVL, QKR, KVD, KVD);
        rope_pos_kernel<<<dim3(T, 1), QKR / 2, rope_sh, g_str>>>(kvlin + KVL, 1, KVD, 0,
                                                          d_pos, QKR, c->rope_interleaved, d_inv_freq);
        kv_scatter_kernel<<<T, 256, 0, g_str>>>(cache_base, scale_base, slot_stride, d_slt, d_pos, kvlin, KVD, T);

        /* Per-head knope[h] = kvlin @ W_UK[h]^T and value[h] = kvlin @ W_UV[h]^T
         * (kvlin shared across heads) folded into 2 head-batched MFMA GEMMs. */
        /* Both are bandwidth-bound on re-reading that shared kvlin, so: the 128-wide
         * tile (halves the re-reads), kvlin narrowed to bf16 once up front (the
         * kernel narrows it in LDS anyway), and bf16 OUTPUT (attention narrows it
         * too). 4.4 GB -> 0.8 GB per layer per chunk. See gemm.hip. */
        {
            {
                bf16_t *kvbf = (bf16_t *)sbuf(B_KVBF, (size_t)T * KVD * sizeof(bf16_t));
                narrow_bf16_kernel<<<ceil_div(T * KVD, 256), 256, 0, g_str>>>(kvbf, kvlin, T * KVD);
                dim3 gK2(ceil_div(QKN, 128), ceil_div(T, 128), NH);
                matmul_headw_mfma_128_xbf_bfout_kernel<<<gK2, 256, 0, g_str>>>(knope, kvbf, d_w.W_UK[l],
                                                       T, QKN, KVL, KVD, kv_stride, T * QKN);
                dim3 gV2(ceil_div(VHD, 128), ceil_div(T, 128), NH);
                matmul_headw_mfma_128_xbf_bfout_kernel<<<gV2, 256, 0, g_str>>>(value, kvbf, d_w.W_UV[l],
                                                       T, VHD, KVL, KVD, kv_stride, T * VHD);
            }
        }

        /* run18: FlashAttention-2 on the matrix cores -- one block per 64-query tile
         * of one prompt, causal within the prompt, softmax online in LDS so the
         * O(n^2) score matrix never reaches HBM. Replaced a scalar fp32 kernel that
         * was 34.6% of wall time (45.5 s -> 8.1 s). */
        attn_prefill_mfma_kernel<<<dim3(n_qtile, NH), 256, 0, g_str>>>(ctx, qall, knope, kvlin, value,
                                                             d_qtq0, d_qtqs, d_qtqe, T, NH,
                                                             QHD, QKN, QKR, KVD, KVL, VHD, scale);
        launch_matmul_w8a8_t(ao, ctx, dq_op[l], ds_op[l], T, H, NH * VHD);
        launch_residual_rmsnorm(xs, xb, ao, d_w.post_attn_norm[l], T, H, eps);
        ffn_batched(c, l, xb, mo, T, hb, hb2);
        /* Fuse this residual with the next layer's input norm. The final norm is
         * applied to the gathered last-token rows instead, so the last layer's
         * residual stays a plain add. */
        if (l + 1 < c->n_layers)
            launch_residual_rmsnorm(xs, xb, mo, d_w.input_layernorm[l + 1], T, H, eps);
        else
            launch_residual_add(xs, mo, T * H);
    }

    /* First token of each prompt = argmax of its last position's logits. Gather
     * those rows, norm, lm_head (batched over nseq), argmax. */
    gather_rows_kernel<<<nseq, 256, 0, g_str>>>(lastx, xs, d_lst, H);
    launch_rmsnorm(xb, lastx, d_w.norm, nseq, H, eps);
    /* Scatter each prompt's first token to its cache slot: b_tokens[slot] seeds the
     * decode loop and column 0 of the record is the first emitted token. */
    int *d_cslot = (int *)sbuf(B_PCSLOT, (size_t)nseq * sizeof(int));
    HIP_CHECK(hipMemcpyAsync(d_cslot, cslot, (size_t)nseq * sizeof(int), hipMemcpyHostToDevice, g_str));
    int *tokrec = (int *)sbuf(B_TOKREC, (size_t)g_batch * rec_stride * sizeof(int));
    launch_lm_head_argmax(xb, d_w.lm_head, nseq, VOC, H, b_tokens, d_cslot, tokrec,
                          rec_stride, 0, NULL, NULL, 0);
    /* The caller repacks the pf_* staging arrays for the next chunk, so the
     * async uploads issued above must be consumed before returning. */
    HIP_CHECK(hipStreamSynchronize(g_str));
}

/* Batched absorbed decode: advance B slots by one token each. b_tokens holds the
 * per-slot input token; b_pos the position at which to write its KV latent. On
 * return b_tokens holds each slot's next greedy token, b_pos has advanced, and
 * b_tokrec[slot][step] has this step's token. `step`/`nrec` locate that record
 * the B slots (host-tracked) bounds the attention/softmax width. Every dense
 * matmul now reuses each weight tile across the whole B batch (matmul_tiled), and
 * the MoE experts are gathered/grouped so each expert's weights are read once. */
static void device_forward_absorbed_batched(Transformer *t, int B, int nrec)
{
    const Config *c = &t->config;
    const int H = c->hidden_size, NH = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim, QHD = QKN + QKR;
    const int VHD = c->v_head_dim, KVL = c->kv_lora_rank, KVD = KVL + QKR, VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const int kv_stride = (QKN + VHD) * KVL;
    const size_t layer_stride = (size_t)(g_kvslots + 1) * g_cap * KVD;   /* run67 */
    const int slot_stride = g_cap * KVD;

    embed_lookup_kernel<<<B, 256, 0, g_str>>>(b_x, d_w.embed_tokens, b_tokens, H);

    for (int l = 0; l < c->n_layers; l++)
    {
        int8_t *kv_base = d_kv_cache + (size_t)l * layer_stride;
        float  *kv_scale_base = d_kv_scale + (size_t)l * (g_kvslots + 1) * g_cap;

        if (l == 0) launch_rmsnorm(b_xb, b_x, d_w.input_layernorm[0], B, H, eps);
        launch_project_q(c, l, b_xb, b_q, B, b_qa);
        {
            dim3 grid(B, NH);
            size_t shmem = c->rope_interleaved ? (size_t)QKR * sizeof(float) : 0;
            rope_pos_kernel<<<grid, QKR / 2, shmem, g_str>>>(b_q + QKN, NH, NH * QHD, QHD,
                                                      b_pos, QKR, c->rope_interleaved, d_inv_freq);
        }
        launch_matmul_w8a8(b_comp, b_xb, d_w.kv_a_proj[l], dq_kva?dq_kva[l]:NULL, dq_kva?ds_kva[l]:NULL, B, KVD, H);
        kv_write_batched_kernel<<<B, 256, (size_t)KVD * sizeof(float), g_str>>>(
                                            kv_base, kv_scale_base, slot_stride, b_pos, b_comp,
                                            d_w.kv_a_layernorm[l], KVL, QKR, KVD,
                                            c->mla_norm_eps, d_inv_freq, c->rope_interleaved,
                                            b_kvslot);

        /* qabs[s,h] = q_nope[s,h] @ W_UK[h]^T  (W_UK pre-transposed to [KVL,QKN]). */
        launch_matmul_bh(b_qabs, b_q, d_wuk_t[l], B, KVL, QKN,
                         QHD, NH * QHD, KVL * QKN, KVL, NH * KVL, NH);
        /* Absorbed attention over each slot's own KV cache (masked to kv_len[s]):
         * one fused flash-decode pass per (slot,head) -> b_clat (no b_score HBM).
         * NB: a head-fused variant (1 block/slot, ckv read once for all heads) was
         * tried but is SLOWER -- grid=(NH,B) already schedules a slot's heads as
         * consecutive blocks so the "redundant" ckv reads hit L2, and fusing trades
         * that cheap reuse for worse thread utilization. See report/run8. */
        {
            /* Absorbed attention over each slot's own KV cache, one fused pass per
             * (slot, head-group): grid = (B, FD_HG). See attention.hip for why this
             * shape and this instruction mix -- it went from 531 to 104 ms per decode
             * step over runs 39-49, and the model dims it assumes are asserted in
             * warm_up. */
            attn_fd_mfma_kernel<<<dim3(B, FD_HG), 256, 0, g_str>>>(b_clat, b_qabs, b_q, kv_base,
                                                          kv_scale_base, b_pos, NH, QHD, QKN, QKR,
                                                          KVL, KVD, slot_stride, scale, b_kvslot);
        }
        /* ctx[s,h] = clat[s,h] @ W_UV[h]^T. */
        launch_matmul_bh(b_ctx, b_clat, d_w.W_UV[l], B, VHD, KVL,
                         KVL, NH * KVL, kv_stride, VHD, NH * VHD, NH);
        launch_matmul_w8a8(b_ao, b_ctx, d_w.o_proj[l], dq_op?dq_op[l]:NULL, dq_op?ds_op[l]:NULL, B, H, NH * VHD);
        launch_residual_rmsnorm(b_x, b_xb, b_ao, d_w.post_attn_norm[l], B, H, eps);
        ffn_batched(c, l, b_xb, b_mo, B, b_hb, b_hb2);
        /* Fuse with the norm that follows: the next layer's input norm, or the
         * final norm after the last layer. */
        launch_residual_rmsnorm(b_x, b_xb, b_mo,
                                l + 1 < c->n_layers ? d_w.input_layernorm[l + 1] : d_w.norm,
                                B, H, eps);
    }


    int *tokrec = (int *)sbuf(B_TOKREC, (size_t)g_batch * nrec * sizeof(int));
    launch_lm_head_argmax(b_xb, d_w.lm_head, B, VOC, H, b_tokens, NULL, tokrec,
                          nrec, 0, b_step, b_pos, g_cap);
    bump_step_kernel<<<1, 1, 0, g_str>>>(b_step);
}
