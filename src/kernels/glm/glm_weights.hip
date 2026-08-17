/* ==========================================================================
 * glm_weights.hip -- getting the model onto the device, once, in warm_up.
 *
 * Upload is bf16 straight from the host mmap, then the pieces that run in int8
 * are quantised ON the device and their bf16 copy is freed immediately -- holding
 * both at once would not fit (the experts alone are ~55 of the 58 GB). What ends
 * up resident: int8 experts + int8 attention projections + int8 shared/dense FFN,
 * each with an fp32 scale per output row, plus bf16 lm_head, moe_gate and the
 * norms. Also builds the per-layer device pointer tables the MoE GEMM indexes by
 * expert, and the one-off W_UK transpose the absorbed decode path needs.
 * ==========================================================================*/

static void device_malloc_and_copy(void **dst, void *src, size_t size)
{
    HIP_CHECK(hipMalloc(dst, size));
    HIP_CHECK(hipMemcpy(*dst, src, size, hipMemcpyHostToDevice));
}

/* ==== DEVICE WEIGHT UPLOAD & POINTER TABLES (shared by both models) ======== */

static void device_malloc_model_weights(Transformer *t)
{
    Config *c = &t->config;
    ModelWeights *w = &t->weights;

    device_malloc_and_copy((void **)&d_w.embed_tokens, w->embed_tokens,
                           (size_t)c->vocab_size * c->hidden_size * sizeof(bf16_t));
    device_malloc_and_copy((void **)&d_w.norm, w->norm,
                           (size_t)c->hidden_size * sizeof(bf16_t));
    device_malloc_and_copy((void **)&d_w.lm_head, w->lm_head,
                           (size_t)c->hidden_size * c->vocab_size * sizeof(bf16_t));

    d_w.q_proj          = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.q_a_proj        = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.q_b_proj        = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.kv_a_proj       = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.W_UK            = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.W_UV            = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.o_proj          = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.input_layernorm = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.post_attn_norm  = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.q_a_layernorm   = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.kv_a_layernorm  = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.dense_gate      = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.dense_up        = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.dense_down      = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.moe_gate        = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.moe_gate_bias   = (float **)malloc(c->n_layers * sizeof(float *));
    d_w.shared_gate     = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.shared_up       = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.shared_down     = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    d_w.expert_gate     = (bf16_t ***)malloc(c->n_layers * sizeof(bf16_t **));
    d_w.expert_up       = (bf16_t ***)malloc(c->n_layers * sizeof(bf16_t **));
    d_w.expert_down     = (bf16_t ***)malloc(c->n_layers * sizeof(bf16_t **));

    for (int l = 0; l < c->n_layers; l++)
    {
        if (c->q_lora_rank > 0)
        {
            device_malloc_and_copy((void **)&d_w.q_a_proj[l], w->q_a_proj[l],
                                   (size_t)c->hidden_size * c->q_lora_rank * sizeof(bf16_t));
            device_malloc_and_copy(
                (void **)&d_w.q_b_proj[l], w->q_b_proj[l],
                (size_t)c->q_lora_rank * c->n_heads *
                    (c->qk_nope_head_dim + c->qk_rope_head_dim) * sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.q_a_layernorm[l], w->q_a_layernorm[l],
                                   (size_t)c->q_lora_rank * sizeof(bf16_t));
            d_w.q_proj[l] = NULL;
        }
        else
        {
            d_w.q_a_proj[l] = d_w.q_b_proj[l] = d_w.q_a_layernorm[l] = NULL;
            device_malloc_and_copy(
                (void **)&d_w.q_proj[l], w->q_proj[l],
                (size_t)c->hidden_size * c->n_heads *
                    (c->qk_nope_head_dim + c->qk_rope_head_dim) * sizeof(bf16_t));
        }

        device_malloc_and_copy((void **)&d_w.kv_a_proj[l], w->kv_a_proj[l],
                               (size_t)c->hidden_size *
                                   (c->kv_lora_rank + c->qk_rope_head_dim) * sizeof(bf16_t));

        device_malloc_and_copy((void **)&d_w.W_UK[l], w->W_UK[l],
                               (size_t)c->n_heads * c->kv_lora_rank *
                                   (c->qk_nope_head_dim + c->v_head_dim) * sizeof(bf16_t));
        d_w.W_UV[l] = d_w.W_UK[l] + (size_t)c->qk_nope_head_dim * c->kv_lora_rank;

        device_malloc_and_copy((void **)&d_w.o_proj[l], w->o_proj[l],
                               (size_t)c->n_heads * c->v_head_dim * c->hidden_size * sizeof(bf16_t));

        device_malloc_and_copy((void **)&d_w.input_layernorm[l], w->input_layernorm[l],
                               (size_t)c->hidden_size * sizeof(bf16_t));
        device_malloc_and_copy((void **)&d_w.post_attn_norm[l], w->post_attn_norm[l],
                               (size_t)c->hidden_size * sizeof(bf16_t));
        device_malloc_and_copy((void **)&d_w.kv_a_layernorm[l], w->kv_a_layernorm[l],
                               (size_t)c->kv_lora_rank * sizeof(bf16_t));

        /* moe_gate_bias (GLM router bias) is only ever read host-side during
         * top-k selection -- t->weights.moe_gate_bias[l] (the original host
         * pointer, already resolved by model_load.c) is used directly for
         * that, so no device copy is needed here. */
        d_w.moe_gate_bias[l] = NULL;

        if (l < c->first_k_dense)
        {
            device_malloc_and_copy((void **)&d_w.dense_gate[l], w->dense_gate[l],
                                   (size_t)c->hidden_size * c->dense_inter_size * sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.dense_up[l], w->dense_up[l],
                                   (size_t)c->hidden_size * c->dense_inter_size * sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.dense_down[l], w->dense_down[l],
                                   (size_t)c->dense_inter_size * c->hidden_size * sizeof(bf16_t));

            d_w.moe_gate[l] = NULL;
            d_w.shared_gate[l] = d_w.shared_up[l] = d_w.shared_down[l] = NULL;
            d_w.expert_gate[l] = d_w.expert_up[l] = d_w.expert_down[l] = NULL;
            continue;
        }
        d_w.dense_gate[l] = d_w.dense_up[l] = d_w.dense_down[l] = NULL;

        device_malloc_and_copy((void **)&d_w.moe_gate[l], w->moe_gate[l],
                               (size_t)c->hidden_size * c->n_routed_experts * sizeof(bf16_t));

        device_malloc_and_copy((void **)&d_w.shared_gate[l], w->shared_gate[l],
                               (size_t)c->hidden_size * c->n_shared_experts *
                                   c->moe_inter_size * sizeof(bf16_t));
        device_malloc_and_copy((void **)&d_w.shared_up[l], w->shared_up[l],
                               (size_t)c->hidden_size * c->n_shared_experts *
                                   c->moe_inter_size * sizeof(bf16_t));
        device_malloc_and_copy((void **)&d_w.shared_down[l], w->shared_down[l],
                               (size_t)c->n_shared_experts * c->moe_inter_size *
                                   c->hidden_size * sizeof(bf16_t));

        d_w.expert_gate[l] = (bf16_t **)malloc(c->n_routed_experts * sizeof(bf16_t *));
        d_w.expert_up[l]   = (bf16_t **)malloc(c->n_routed_experts * sizeof(bf16_t *));
        d_w.expert_down[l] = (bf16_t **)malloc(c->n_routed_experts * sizeof(bf16_t *));
        for (int e = 0; e < c->n_routed_experts; e++)
        {
            device_malloc_and_copy((void **)&d_w.expert_gate[l][e], w->expert_gate[l][e],
                                   (size_t)c->hidden_size * c->moe_inter_size * sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.expert_up[l][e], w->expert_up[l][e],
                                   (size_t)c->hidden_size * c->moe_inter_size * sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.expert_down[l][e], w->expert_down[l][e],
                                   (size_t)c->moe_inter_size * c->hidden_size * sizeof(bf16_t));
        }
    }
}

static void device_free_model_weights(Transformer *t)
{
    Config *c = &t->config;
    HIP_CHECK(hipFree(d_w.embed_tokens));
    HIP_CHECK(hipFree(d_w.norm));
    HIP_CHECK(hipFree(d_w.lm_head));

    for (int l = 0; l < c->n_layers; l++)
    {
        HIP_CHECK(hipFree(d_w.q_a_proj[l]));
        HIP_CHECK(hipFree(d_w.q_b_proj[l]));
        HIP_CHECK(hipFree(d_w.q_a_layernorm[l]));
        HIP_CHECK(hipFree(d_w.q_proj[l]));
        HIP_CHECK(hipFree(d_w.kv_a_proj[l]));
        HIP_CHECK(hipFree(d_w.W_UK[l])); /* W_UV[l] is an offset into the same alloc */
        HIP_CHECK(hipFree(d_w.o_proj[l]));
        HIP_CHECK(hipFree(d_w.input_layernorm[l]));
        HIP_CHECK(hipFree(d_w.post_attn_norm[l]));
        HIP_CHECK(hipFree(d_w.kv_a_layernorm[l]));

        if (l < c->first_k_dense)
        {
            HIP_CHECK(hipFree(d_w.dense_gate[l]));
            HIP_CHECK(hipFree(d_w.dense_up[l]));
            HIP_CHECK(hipFree(d_w.dense_down[l]));
            continue;
        }
        HIP_CHECK(hipFree(d_w.moe_gate[l]));
        HIP_CHECK(hipFree(d_w.shared_gate[l]));
        HIP_CHECK(hipFree(d_w.shared_up[l]));
        HIP_CHECK(hipFree(d_w.shared_down[l]));
        for (int e = 0; e < c->n_routed_experts; e++)
        {
            HIP_CHECK(hipFree(d_w.expert_gate[l][e]));
            HIP_CHECK(hipFree(d_w.expert_up[l][e]));
            HIP_CHECK(hipFree(d_w.expert_down[l][e]));
        }
        free(d_w.expert_gate[l]);
        free(d_w.expert_up[l]);
        free(d_w.expert_down[l]);
    }

    free(d_w.q_proj); free(d_w.q_a_proj); free(d_w.q_b_proj); free(d_w.kv_a_proj);
    free(d_w.W_UK); free(d_w.W_UV); free(d_w.o_proj);
    free(d_w.input_layernorm); free(d_w.post_attn_norm);
    free(d_w.q_a_layernorm); free(d_w.kv_a_layernorm);
    free(d_w.dense_gate); free(d_w.dense_up); free(d_w.dense_down);
    free(d_w.moe_gate); free(d_w.moe_gate_bias);
    free(d_w.shared_gate); free(d_w.shared_up); free(d_w.shared_down);
    free(d_w.expert_gate); free(d_w.expert_up); free(d_w.expert_down);
}

/* Upload the per-layer expert weight-pointer tables + router bias to the device,
 * so moe_route_batched_kernel can select and gather experts without a host
 * round-trip. */
static void build_moe_device_tables(Transformer *t)
{
    Config *c = &t->config;
    const int n_routed = c->n_routed_experts;
    d_ep_gate  = (bf16_t ***)malloc(c->n_layers * sizeof(bf16_t **));
    d_ep_up    = (bf16_t ***)malloc(c->n_layers * sizeof(bf16_t **));
    d_ep_down  = (bf16_t ***)malloc(c->n_layers * sizeof(bf16_t **));
    d_moe_bias = (float **)malloc(c->n_layers * sizeof(float *));

    for (int l = 0; l < c->n_layers; l++)
    {
        if (l < c->first_k_dense)
        {
            d_ep_gate[l] = d_ep_up[l] = d_ep_down[l] = NULL;
            d_moe_bias[l] = NULL;
            continue;
        }
        size_t sz = (size_t)n_routed * sizeof(bf16_t *);
        device_malloc_and_copy((void **)&d_ep_gate[l], d_w.expert_gate[l], sz);
        device_malloc_and_copy((void **)&d_ep_up[l],   d_w.expert_up[l],   sz);
        device_malloc_and_copy((void **)&d_ep_down[l], d_w.expert_down[l], sz);

        if (t->weights.moe_gate_bias[l])
            device_malloc_and_copy((void **)&d_moe_bias[l], t->weights.moe_gate_bias[l],
                                   (size_t)n_routed * sizeof(float));
        else
            d_moe_bias[l] = NULL;
    }
}

static void free_moe_device_tables(Transformer *t)
{
    Config *c = &t->config;
    if (!d_ep_gate) return;
    for (int l = c->first_k_dense; l < c->n_layers; l++)
    {
        HIP_CHECK(hipFree(d_ep_gate[l]));
        HIP_CHECK(hipFree(d_ep_up[l]));
        HIP_CHECK(hipFree(d_ep_down[l]));
        if (d_moe_bias[l]) HIP_CHECK(hipFree(d_moe_bias[l]));
    }
    free(d_ep_gate); free(d_ep_up); free(d_ep_down); free(d_moe_bias);
    d_ep_gate = d_ep_up = d_ep_down = NULL;
    d_moe_bias = NULL;
}

/* Quantize one bf16 weight [rows][cols] (device) into a fresh int8 buffer + fp32
 * per-row scale buffer (device), then free the bf16 source. */
static void quant_one_expert(bf16_t *src, int rows, int cols, int8_t **out_q, float **out_s)
{
    HIP_CHECK(hipMalloc((void **)out_q, (size_t)rows * cols * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)out_s, (size_t)rows * sizeof(float)));
    quantize_rows_i8_kernel<<<rows, 256>>>(src, *out_q, *out_s, rows, cols);
    HIP_CHECK(hipFree(src));
}





/* run20: quantize a bf16 weight [rows][cols] to int8 + per-row scale but KEEP the
 * bf16 (lm_head and moe_gate deliberately stay bf16). */
static void quant_keep(const bf16_t *src, int rows, int cols, int8_t **out_q, float **out_s)
{
    HIP_CHECK(hipMalloc((void **)out_q, (size_t)rows * cols * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)out_s, (size_t)rows * sizeof(float)));
    quantize_rows_i8_kernel<<<rows, 256>>>(src, *out_q, *out_s, rows, cols);
}



/* ==== W8A8 / INT8 QUANTIZATION TABLES (GLM-tuned optimizations) ============ */

/* run20: build int8 copies of the big per-layer dense projections (q_a/q_b/kv_a/
 * o_proj) so the W8A8 matmul can read 1-byte weights (L2-BW relief). */
static void build_dense_int8_tables(Transformer *t)
{
    const Config *c = &t->config;
    const int H = c->hidden_size, NH = c->n_heads;
    const int QHD = c->qk_nope_head_dim + c->qk_rope_head_dim;
    const int KVD = c->kv_lora_rank + c->qk_rope_head_dim;
    const int qlr = c->q_lora_rank, q_dim = NH * QHD, vdim = NH * c->v_head_dim;
    dq_qa=(int8_t**)calloc(c->n_layers,sizeof(int8_t*)); ds_qa=(float**)calloc(c->n_layers,sizeof(float*));
    dq_qb=(int8_t**)calloc(c->n_layers,sizeof(int8_t*)); ds_qb=(float**)calloc(c->n_layers,sizeof(float*));
    dq_kva=(int8_t**)calloc(c->n_layers,sizeof(int8_t*));ds_kva=(float**)calloc(c->n_layers,sizeof(float*));
    dq_op=(int8_t**)calloc(c->n_layers,sizeof(int8_t*)); ds_op=(float**)calloc(c->n_layers,sizeof(float*));
    /* run33: block-wise int4 (scale per K-group of 128) when g_dense_w4a8_bw; all
     * four projections' K (H, qlr=768, vdim=5120) are multiples of 128. */
    void (*qk)(const bf16_t *, int, int, int8_t **, float **) =
        quant_keep;
    for (int l = 0; l < c->n_layers; l++) {
        if (qlr > 0) {
            qk(d_w.q_a_proj[l], qlr,   H,   &dq_qa[l], &ds_qa[l]);
            qk(d_w.q_b_proj[l], q_dim, qlr, &dq_qb[l], &ds_qb[l]);
        }
        qk(d_w.kv_a_proj[l], KVD, H,    &dq_kva[l], &ds_kva[l]);
        qk(d_w.o_proj[l],    H,   vdim, &dq_op[l],  &ds_op[l]);
    }
    (void)KVD;
}

/* run22: int8 copies (per-output-row scale) of the shared-expert + dense-layer FFN
 * weights (gate/up/down). Same quant as the routed experts (already W8A8). Shared
 * runs every layer; dense only for l < first_k_dense. Keeps bf16 for fallback. */
static void build_ffn_int8_tables(Transformer *t)
{
    const Config *c = &t->config;
    const int H = c->hidden_size, L = c->n_layers;
    const int SI = c->n_shared_experts * c->moe_inter_size, DI = c->dense_inter_size;
    #define AZ(x) x = (int8_t **)calloc(L, sizeof(int8_t *))
    #define FZ(x) x = (float  **)calloc(L, sizeof(float  *))
    AZ(dq_sg); AZ(dq_su); AZ(dq_sd); FZ(ds_sg); FZ(ds_su); FZ(ds_sd);
    AZ(dq_dg); AZ(dq_du); AZ(dq_dd); FZ(ds_dg); FZ(ds_du); FZ(ds_dd);
    #undef AZ
    #undef FZ
    for (int l = 0; l < L; l++) {
        if (l < c->first_k_dense) {
            quant_keep(d_w.dense_gate[l], DI, H,  &dq_dg[l], &ds_dg[l]);
            quant_keep(d_w.dense_up[l],   DI, H,  &dq_du[l], &ds_du[l]);
            quant_keep(d_w.dense_down[l], H,  DI, &dq_dd[l], &ds_dd[l]);
        } else {
            quant_keep(d_w.shared_gate[l], SI, H,  &dq_sg[l], &ds_sg[l]);
            quant_keep(d_w.shared_up[l],   SI, H,  &dq_su[l], &ds_su[l]);
            quant_keep(d_w.shared_down[l], H,  SI, &dq_sd[l], &ds_sd[l]);
        }
    }
}



/* run4: quantize all routed-expert weights to int8 + per-output-row scale, free
 * the bf16, and build the device pointer tables consumed by ffn_batched. Gate/up
 * are [I][H] (rows=I), down is [H][I] (rows=H). Must run AFTER
 * device_malloc_model_weights (bf16 experts uploaded) and build_moe_device_tables
 * (which only needs d_moe_bias). After this, d_w.expert_*[l][e] are freed/NULL. */
static void build_moe_int8_tables(Transformer *t)
{
    Config *c = &t->config;
    const int NR = c->n_routed_experts, H = c->hidden_size, I = c->moe_inter_size;
    #define AL(x, T, n) x = (T *)malloc((size_t)(n) * sizeof(T))
    AL(hq_g, int8_t **, c->n_layers); AL(hq_u, int8_t **, c->n_layers); AL(hq_d, int8_t **, c->n_layers);
    AL(hs_g, float **,  c->n_layers); AL(hs_u, float **,  c->n_layers); AL(hs_d, float **,  c->n_layers);
    AL(d_epq_gate, int8_t **, c->n_layers); AL(d_epq_up, int8_t **, c->n_layers); AL(d_epq_down, int8_t **, c->n_layers);
    AL(d_eps_gate, float **,  c->n_layers); AL(d_eps_up, float **,  c->n_layers); AL(d_eps_down, float **,  c->n_layers);
    #undef AL

    for (int l = 0; l < c->n_layers; l++)
    {
        if (l < c->first_k_dense)
        {
            hq_g[l] = hq_u[l] = hq_d[l] = NULL; hs_g[l] = hs_u[l] = hs_d[l] = NULL;
            d_epq_gate[l] = d_epq_up[l] = d_epq_down[l] = NULL;
            d_eps_gate[l] = d_eps_up[l] = d_eps_down[l] = NULL;
            continue;
        }
        hq_g[l] = (int8_t **)malloc(NR * sizeof(int8_t *));
        hq_u[l] = (int8_t **)malloc(NR * sizeof(int8_t *));
        hq_d[l] = (int8_t **)malloc(NR * sizeof(int8_t *));
        hs_g[l] = (float **)malloc(NR * sizeof(float *));
        hs_u[l] = (float **)malloc(NR * sizeof(float *));
        hs_d[l] = (float **)malloc(NR * sizeof(float *));
        for (int e = 0; e < NR; e++)
        {
            void (*qexp)(bf16_t *, int, int, int8_t **, float **) =
                quant_one_expert;
            qexp(d_w.expert_gate[l][e], I, H, &hq_g[l][e], &hs_g[l][e]);
            qexp(d_w.expert_up[l][e],   I, H, &hq_u[l][e], &hs_u[l][e]);
            qexp(d_w.expert_down[l][e], H, I, &hq_d[l][e], &hs_d[l][e]);
            d_w.expert_gate[l][e] = d_w.expert_up[l][e] = d_w.expert_down[l][e] = NULL;
        }
        /* Upload per-expert pointer arrays to device tables (kernel indexes by e). */
        device_malloc_and_copy((void **)&d_epq_gate[l], hq_g[l], (size_t)NR * sizeof(int8_t *));
        device_malloc_and_copy((void **)&d_epq_up[l],   hq_u[l], (size_t)NR * sizeof(int8_t *));
        device_malloc_and_copy((void **)&d_epq_down[l], hq_d[l], (size_t)NR * sizeof(int8_t *));
        device_malloc_and_copy((void **)&d_eps_gate[l], hs_g[l], (size_t)NR * sizeof(float *));
        device_malloc_and_copy((void **)&d_eps_up[l],   hs_u[l], (size_t)NR * sizeof(float *));
        device_malloc_and_copy((void **)&d_eps_down[l], hs_d[l], (size_t)NR * sizeof(float *));
    }
    HIP_CHECK(hipDeviceSynchronize());
}

static void free_moe_int8_tables(Transformer *t)
{
    Config *c = &t->config;
    if (!hq_g) return;
    const int NR = c->n_routed_experts;
    for (int l = c->first_k_dense; l < c->n_layers; l++)
    {
        if (!hq_g[l]) continue;
        for (int e = 0; e < NR; e++)
        {
            HIP_CHECK(hipFree(hq_g[l][e])); HIP_CHECK(hipFree(hq_u[l][e])); HIP_CHECK(hipFree(hq_d[l][e]));
            HIP_CHECK(hipFree(hs_g[l][e])); HIP_CHECK(hipFree(hs_u[l][e])); HIP_CHECK(hipFree(hs_d[l][e]));
        }
        HIP_CHECK(hipFree(d_epq_gate[l])); HIP_CHECK(hipFree(d_epq_up[l])); HIP_CHECK(hipFree(d_epq_down[l]));
        HIP_CHECK(hipFree(d_eps_gate[l])); HIP_CHECK(hipFree(d_eps_up[l])); HIP_CHECK(hipFree(d_eps_down[l]));
        free(hq_g[l]); free(hq_u[l]); free(hq_d[l]); free(hs_g[l]); free(hs_u[l]); free(hs_d[l]);
    }
    free(hq_g); free(hq_u); free(hq_d); free(hs_g); free(hs_u); free(hs_d);
    free(d_epq_gate); free(d_epq_up); free(d_epq_down);
    free(d_eps_gate); free(d_eps_up); free(d_eps_down);
    hq_g = hq_u = hq_d = NULL; hs_g = hs_u = hs_d = NULL;
    d_epq_gate = d_epq_up = d_epq_down = NULL; d_eps_gate = d_eps_up = d_eps_down = NULL;
}

/* Build the transposed W_UK copies used by the absorbed q-fold (one per layer). */
static void build_wuk_transpose(Transformer *t)
{
    Config *c = &t->config;
    const int NH = c->n_heads, QKN = c->qk_nope_head_dim, KVL = c->kv_lora_rank;
    const int VHD = c->v_head_dim;
    const int kv_stride = (QKN + VHD) * KVL;
    d_wuk_t = (bf16_t **)malloc(c->n_layers * sizeof(bf16_t *));
    for (int l = 0; l < c->n_layers; l++)
    {
        HIP_CHECK(hipMalloc((void **)&d_wuk_t[l],
                            (size_t)NH * KVL * QKN * sizeof(bf16_t)));
        dim3 grid(KVL, NH);
        transpose_wuk_kernel<<<grid, 256>>>(d_wuk_t[l], d_w.W_UK[l], QKN, KVL, kv_stride);
    }
    HIP_CHECK(hipDeviceSynchronize());
}

static void free_wuk_transpose(Transformer *t)
{
    if (!d_wuk_t) return;
    for (int l = 0; l < t->config.n_layers; l++) HIP_CHECK(hipFree(d_wuk_t[l]));
    free(d_wuk_t);
    d_wuk_t = NULL;
}
