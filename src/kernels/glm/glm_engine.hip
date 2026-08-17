/* ==========================================================================
 * glm_engine.hip -- the three entry points the harness calls, and the scheduler.
 *
 *   glm_warm_up    upload + quantise + allocate + one tiny forward to JIT the
 *                  kernels and ramp the clocks, all outside the timed region
 *   glm_inference  group the requests, prefill them in chunks, then decode the
 *                  whole group in lockstep -- the only timed function
 *   glm_finish     release everything
 *
 * State lives in glm_state.hip, upload in glm_weights.hip, the forward passes in
 * glm_forward.hip; this file is the schedule and nothing else.
 * ==========================================================================*/
/* ==== LIFECYCLE: warm_up (upload + allocate) / finish (free) =============== */

/* run30: one tiny forward (prefill 4 tokens + 2 decode steps) at the end of
 * warm_up to JIT the hot kernels, touch their scratch, and ramp GPU clocks before
 * the timed inference. Defined after the forward-pass helpers it calls. */
static void glm_warmup_forward(Transformer *t);

void glm_warm_up(Transformer *t)
{
    device_malloc_model_weights(t);
    build_moe_device_tables(t);
    build_moe_int8_tables(t);     /* run4/19: experts -> int8 + per-row scale, free bf16 */
    build_wuk_transpose(t);       /* W_UK -> [KVL,QKN] so the absorbed q-fold coalesces */
    build_dense_int8_tables(t);   /* run20: int8 q_a/q_b/kv_a/o_proj */
    build_ffn_int8_tables(t);     /* run22: int8 shared-expert + dense-layer FFN */

    Config *c = &t->config;
    fprintf(stderr, "[getp] model = %s (%d layers, %d routed experts top-%d, %s router)\n",
            model_name(model_kind(c)), c->n_layers, c->n_routed_experts,
            c->n_experts_per_tok, c->router_sigmoid ? "sigmoid" : "softmax");
    int kv_dim = c->kv_lora_rank + c->qk_rope_head_dim;

    /* Batch size (concurrent requests per group) — tunable via MLA_BATCH. Cache
     * cap holds the longest prompt + generated tokens; keep it comfortably above
     * the dataset's 512-token prompts + 64 decode. */
    const char *env_b = getenv("MLA_BATCH");
    if (env_b) { int v = atoi(env_b); if (v > 0) g_batch = v; }
    const char *env_cap = getenv("MLA_CAP");
    if (env_cap) { int v = atoi(env_cap); if (v > 0) g_cap = v; }
    if (g_cap > CACHE_CAP_LIMIT) g_cap = CACHE_CAP_LIMIT;
    g_cache_cap = g_cap;

    /* MLA_PFCHUNK sets the token budget per batched-prefill forward; it also sizes
     * the shared MoE scratch (MR), so read it before allocating below. */
    const char *env_pfc = getenv("MLA_PFCHUNK");
    if (env_pfc) { int v = atoi(env_pfc); if (v > 0) g_pfchunk = v; }

    /* The decode attention kernel hard-codes the model geometry (FD_NH / FD_KVL /
     * FD_KVD in attention.hip) so its LDS strides can be bank-conflict-free
     * compile-time constants. Fail loudly rather than compute silently wrong
     * attention if this engine is ever pointed at a differently shaped model. */
    if (c->n_heads != FD_NH || c->kv_lora_rank != FD_KVL || kv_dim != FD_KVD)
    {
        fprintf(stderr, "glm engine: expected NH=%d KVL=%d KVD=%d, got %d/%d/%d\n",
                FD_NH, FD_KVL, FD_KVD, c->n_heads, c->kv_lora_rank, kv_dim);
        exit(EXIT_FAILURE);
    }
    /* The grouped MoE GEMM stages 32-K groups with 16 B loads, so both contracted
     * dimensions must be multiples of 32 (GLM: hidden 2048, moe_inter 1536). */
    if ((c->hidden_size & 31) || (c->moe_inter_size & 31))
    {
        fprintf(stderr, "glm engine: hidden_size (%d) and moe_inter_size (%d) must be "
                        "multiples of 32\n", c->hidden_size, c->moe_inter_size);
        exit(EXIT_FAILURE);
    }

    const int B = g_batch, H = c->hidden_size, NH = c->n_heads;
    const int QHD = c->qk_nope_head_dim + c->qk_rope_head_dim;
    const int KVL = c->kv_lora_rank, VHD = c->v_head_dim, KVD = kv_dim;
    const int VOC = c->vocab_size, NR = c->n_routed_experts, K = c->n_experts_per_tok;
    const int I = c->moe_inter_size, IM = inter_max_of(c);
    const int qlr = c->q_lora_rank > 0 ? c->q_lora_rank : 1;

    /* Per-slot latent KV cache: int8 [n_layers, B, g_cap, KVD] + per-row fp32 scale
     * [n_layers, B, g_cap] (run16). */
    HIP_CHECK(hipMalloc((void **)&d_kv_cache,
                        (size_t)c->n_layers * B * g_cap * KVD * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)&d_kv_scale,
                        (size_t)c->n_layers * B * g_cap * sizeof(float)));

    /* Batched decode scratch (allocated once). */
    #define BF(p, n) HIP_CHECK(hipMalloc((void **)&(p), (size_t)(n) * sizeof(float)))
    #define BI(p, n) HIP_CHECK(hipMalloc((void **)&(p), (size_t)(n) * sizeof(int)))
    BF(b_x, B * H);          BF(b_xb, B * H);        BF(b_qa, B * qlr);
    BF(b_q, B * NH * QHD);   BF(b_comp, B * KVD);    BF(b_qabs, B * NH * KVL);
    BF(b_score, B * NH * g_cap); BF(b_clat, B * NH * KVL); BF(b_ctx, B * NH * VHD);
    BF(b_ao, B * H);         BF(b_mo, B * H);
    BF(b_hb, B * IM);        BF(b_hb2, B * IM);
    /* MoE scratch is shared by prefill and decode (B rows). Per-request prefill
     * feeds up to g_cap rows; chunked prefill feeds up to g_pfchunk rows. Size for
     * the largest so ffn_batched never overflows these buffers. */
    int MR = g_cap;
    if (g_pfchunk > MR) MR = g_pfchunk;
    BF(b_shared, (size_t)MR * H);
    BF(b_moelog, (size_t)MR * NR);   BF(b_topk_wt, (size_t)MR * K);
    BF(b_sorted_wt, (size_t)MR * K);
    BF(b_hbk, (size_t)MR * K * I);   BF(b_hb2k, (size_t)MR * K * I);
    BF(b_downk, (size_t)MR * K * H);
    BF(b_logits, (size_t)B * VOC);
    /* run19 W8A8 MoE int8-activation scratch. */
    HIP_CHECK(hipMalloc((void **)&b_xn_i8,   (size_t)MR * H * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)&b_hbk_i8,  (size_t)MR * K * I * sizeof(int8_t)));
    BF(b_xn_scale, (size_t)MR);      BF(b_hbk_scale, (size_t)MR * K);
    /* run20 W8A8 dense int8-activation scratch (shared; sized for the widest K).
     * run22: also the W8A8 FFN down (K = dense/shared inter) and lm_head (K = H). */
    int actK = NH * VHD; if (H > actK) actK = H; if (qlr > actK) actK = qlr;
    { int im = inter_max_of(c); if (im > actK) actK = im; }
    HIP_CHECK(hipMalloc((void **)&b_act_i8, (size_t)MR * actK * sizeof(int8_t)));
    BF(b_act_sc, (size_t)MR);
    BI(b_topk_idx, (size_t)MR * K);  BI(b_ecount, NR);       BI(b_eoff, NR + 1);
    BI(b_cursor, NR);        BI(b_sorted_slot, (size_t)MR * K);
    BI(b_tokens, B);         BI(b_pos, B);           BI(b_argmax, B);
    #undef BF
    #undef BI

    int half = c->qk_rope_head_dim / 2;
    device_malloc_and_copy((void **)&d_inv_freq, t->rope_inv_freq, (size_t)half * sizeof(float));

    HIP_CHECK(hipMalloc((void **)&d_logits, (size_t)c->vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc((void **)&d_argmax, sizeof(int)));

    HIP_CHECK(hipMalloc((void **)&b_step, sizeof(int)));
    HIP_CHECK(hipStreamCreate(&g_str));

    glm_warmup_forward(t);   /* run30: prefill 4 + 2 decode steps to warm kernels */
}

void glm_finish(Transformer *t)
{
    if (g_dec_graph) { HIP_CHECK(hipGraphExecDestroy(g_dec_graph)); g_dec_graph = NULL; }
    if (g_str) { HIP_CHECK(hipStreamDestroy(g_str)); g_str = 0; }
    if (b_step) { HIP_CHECK(hipFree(b_step)); b_step = NULL; }

    free_moe_int8_tables(t);
    device_free_model_weights(t);
    free_moe_device_tables(t);
    free_wuk_transpose(t);
    free_scratch();
    HIP_CHECK(hipFree(d_kv_cache));
    HIP_CHECK(hipFree(d_kv_scale));
    HIP_CHECK(hipFree(d_inv_freq));
    HIP_CHECK(hipFree(d_logits));
    HIP_CHECK(hipFree(d_argmax));

    void *bufs[] = {b_x, b_xb, b_qa, b_q, b_comp, b_qabs, b_score, b_clat, b_ctx,
                    b_ao, b_mo, b_shared, b_hb, b_hb2, b_moelog, b_topk_wt,
                    b_sorted_wt, b_hbk, b_hb2k, b_downk, b_logits, b_topk_idx,
                    b_ecount, b_eoff, b_cursor, b_sorted_slot, b_tokens, b_pos,
                    b_argmax};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        if (bufs[i]) HIP_CHECK(hipFree(bufs[i]));
}


/* ==== INFERENCE DRIVER (batched prefill + lockstep decode) ================= */

/* run30: warm-up forward. Mirrors glm_inference's prefill + lockstep-decode path
 * on a tiny dummy request (prefill NP=4 tokens in slot 0, then NEW=2 decode steps)
 * so every hot kernel is JIT-compiled, its persistent scratch touched, and the GPU
 * clocked up BEFORE the timed run -- keeping cold-start (first-launch compile,
 * allocator faults, clock ramp) out of the measured throughput. Writes only slot 0
 * of the KV cache and the transient b_* scratch, all overwritten by the real
 * inference, so it changes no measured output. Uses the same forward helpers as
 * glm_inference, so it also triggers their one-time env-toggle reads. */
static void glm_warmup_forward(Transformer *t)
{
    enum { NP = 4, NEW = 2 };            /* prefill length, max_new_tokens */
    if (g_cap < NP + NEW) return;        /* need room in the slot-0 cache */

    int pf_tok[NP], pf_seq[NP], pf_pos[NP], pf_slot[NP], pf_last[1], pf_cslot[1] = {0};
    for (int j = 0; j < NP; j++) { pf_tok[j] = 0; pf_seq[j] = 0; pf_pos[j] = j; pf_slot[j] = 0; }
    pf_last[0] = NP - 1;
    device_forward_prefill_batched(t, NP, 1, pf_tok, pf_seq, pf_pos, pf_slot, pf_last,
                                  pf_cslot, NEW + 1);
    int h_pos = NP;
    HIP_CHECK(hipMemcpyAsync(b_pos, &h_pos, sizeof(int), hipMemcpyHostToDevice, g_str));
    reset_step_kernel<<<1, 1, 0, g_str>>>(b_step, 1);   /* column 0 = prefill token */
    for (int step = 0; step < NEW; step++)
        device_forward_absorbed_batched(t, 1, NEW + 1);   /* same path as the real loop */
    HIP_CHECK(hipDeviceSynchronize());
    fprintf(stderr, "[getp] warm-up forward done (prefill %d + %d decode)\n", NP, NEW);
}

long long glm_inference(Transformer *t, Requests *reqs)
{
    const int max_seq_len = t->config.max_seq_len;
    const int eos = t->tokenizer ? tokenizer_eos_id(t->tokenizer) : -1;
    const int max_steps = reqs->max_steps;
    const int N = reqs->num_reqs;

    if (!t->tokenizer)
    {
        fprintf(stderr, "inference: model_dir has no tokenizer.json\n");
        exit(EXIT_FAILURE);
    }

    /* Tokenize all prompts up front; sort request indices by length (bucketing). */
    int **ptok = (int **)malloc((size_t)N * sizeof(int *));
    int  *plen = (int *)malloc((size_t)N * sizeof(int));
    int  *order = (int *)malloc((size_t)N * sizeof(int));
    for (int r = 0; r < N; r++)
    {
        int *buf = (int *)malloc((size_t)max_seq_len * sizeof(int));
        int np = tokenizer_encode(t->tokenizer, reqs->prompts[r], 0, buf, max_seq_len);
        ptok[r] = buf;
        plen[r] = np;
        order[r] = r;
    }
    std::sort(order, order + N, [&](int a, int b) { return plen[a] < plen[b]; });

    /* Per-slot host state for the current group. */
    int  *slot_req  = (int *)malloc((size_t)g_batch * sizeof(int));
    int  *slot_st   = (int *)malloc((size_t)g_batch * sizeof(int));
    int  *slot_nout = (int *)malloc((size_t)g_batch * sizeof(int));
    int  *h_pos     = (int *)malloc((size_t)g_batch * sizeof(int));
    /* Empty prompts are never prefilled, so the packer must skip them; their
     * output length is 0 either way (slot_st = 0). */
    char *empty     = (char *)malloc((size_t)g_batch);

    /* Chunked-prefill host scratch: index arrays over the flat token axis (a chunk
     * holds whole prompts up to g_pfchunk tokens, but a single over-budget prompt
     * gets its own chunk, so size for that worst case). */
    const int TCAP = g_pfchunk + max_seq_len;
    int *pf_tok  = (int *)malloc((size_t)TCAP * sizeof(int));
    int *pf_seq  = (int *)malloc((size_t)TCAP * sizeof(int));
    int *pf_pos  = (int *)malloc((size_t)TCAP * sizeof(int));
    int *pf_slot = (int *)malloc((size_t)TCAP * sizeof(int));
    int *pf_last = (int *)malloc((size_t)g_batch * sizeof(int));
    int *pf_cslot= (int *)malloc((size_t)g_batch * sizeof(int));
    int *pf_cnp  = (int *)malloc((size_t)g_batch * sizeof(int));
    /* run36 fused decode-window scratch. */
    /* decode token history read back once per group: [slot][max_steps-1] */
    int *h_rec = (int *)malloc((size_t)g_batch * max_steps * sizeof(int));

    long long total = 0;

    for (int base = 0; base < N; base += g_batch)
    {
        int Bg = (N - base) < g_batch ? (N - base) : g_batch;

        /* Prefill each slot; capture its first token. Empty prompts finish
         * immediately; the rest are prefilled (chunked-batched or per-request). */
        for (int i = 0; i < Bg; i++)
        {
            int r = order[base + i];
            slot_req[i]  = r;
            slot_nout[i] = 0;
            if (plen[r] < 1)
            {
                empty[i] = 1; slot_st[i] = 0; h_pos[i] = 0;
                reqs->out_lens[r] = 0;
            }
            else
            {
                empty[i] = 0;
            }
        }

        /* Set a slot's post-prefill state from its first token. */
        /* Per-slot step budget and starting cache position. The first token itself
         * stays on the device (prefill scatters it into b_tokens[slot] and column 0
         * of the record), so the host only needs these two numbers. */
        auto set_slot = [&](int i, int np) {
            int budget = g_cap - np;
            int st = max_steps < budget ? max_steps : budget;
            slot_st[i] = st < 0 ? 0 : st;
            h_pos[i]   = np;
        };

        /* Pack whole prompts up to g_pfchunk tokens per batched forward, so the
         * weights are streamed once per chunk instead of once per request. */
        {
            int i = 0;
            while (i < Bg)
            {
                if (empty[i]) { i++; continue; }
                int T = 0, nseq = 0;
                while (i < Bg)
                {
                    if (empty[i]) { i++; continue; }
                    int np = plen[order[base + i]];
                    if (nseq > 0 && T + np > g_pfchunk) break;   /* budget (first prompt always fits) */
                    int *tk = ptok[order[base + i]];
                    for (int j = 0; j < np; j++)
                    {
                        pf_tok[T + j]  = tk[j];
                        pf_seq[T + j]  = T;      /* global start of this prompt */
                        pf_pos[T + j]  = j;      /* position within this prompt */
                        pf_slot[T + j] = i;      /* decode slot */
                    }
                    pf_last[nseq]  = T + np - 1;
                    pf_cslot[nseq] = i;
                    pf_cnp[nseq]   = np;
                    T += np; nseq++; i++;
                }
                device_forward_prefill_batched(t, T, nseq, pf_tok, pf_seq, pf_pos,
                                               pf_slot, pf_last, pf_cslot, max_steps);
                for (int c2 = 0; c2 < nseq; c2++)
                    set_slot(pf_cslot[c2], pf_cnp[c2]);
            }
        }

        /* Lockstep decode, fully asynchronous.
         *
         * The harness asks every request for the same max_steps, and the cache is
         * sized so every slot can hold all of them (g_cap = max prompt + steps),
         * so nothing in the loop depends on the host: the next input token is
         * produced by the argmax and consumed by the embedding lookup, both on
         * device. So upload the prefill token and position ONCE, let the argmax
         * record each step's token into b_tokrec and advance b_pos itself, and
         * read the whole [Bg][nrec] token matrix back in a single copy at the end.
         * That removes 3 blocking hipMemcpys per step (each of which drained the
         * pipeline between steps).
         *
         * Only max_steps-1 forwards run: the emitted sequence is t0 (from prefill)
         * followed by t1..t_{max_steps-1}, so the token a max_steps-th forward
         * would produce is never emitted -- that whole forward was wasted work.
         *
         * EOS is resolved on the host afterwards and is exactly equivalent to the
         * old per-step check: that stopped a slot when the token just produced was
         * EOS, without emitting it, i.e. out_len = index of the first EOS. Slots
         * that would have stopped early keep decoding here and their surplus
         * tokens are discarded; positions stay in range because the argmax clamps
         * to g_cap-1. */
        for (int i = 0; i < Bg; i++)
            if (h_pos[i] >= g_cap) h_pos[i] = g_cap - 1;
        HIP_CHECK(hipMemcpyAsync(b_pos, h_pos, (size_t)Bg * sizeof(int), hipMemcpyHostToDevice, g_str));

        /* One decode step is a fixed sequence of ~1900 launches; between steps it
         * differed only in which b_tokrec column the argmax writes, and that index
         * now lives on the device (b_step). So capture the step ONCE into a
         * hipGraph and replay it: ~1900 launches per step become one
         * hipGraphLaunch, which removes the per-launch dispatch gap.
         *
         * Step 0 runs the ordinary path first because it is what sizes b_tokrec
         * through sbuf(), and allocating during capture is illegal. Capture then
         * happens on step 1; capture itself enqueues nothing, so the graph is
         * launched straight afterwards to actually perform that step. */
        reset_step_kernel<<<1, 1, 0, g_str>>>(b_step, 1);   /* column 0 = prefill token */
        for (int step = 0; step + 1 < max_steps; step++)
        {
            if (step == 0)
            {
                device_forward_absorbed_batched(t, Bg, max_steps);
                continue;
            }
            if (!g_dec_graph)
            {
                hipGraph_t graph;
                HIP_CHECK(hipStreamBeginCapture(g_str, hipStreamCaptureModeGlobal));
                device_forward_absorbed_batched(t, Bg, max_steps);
                HIP_CHECK(hipStreamEndCapture(g_str, &graph));
                HIP_CHECK(hipGraphInstantiate(&g_dec_graph, graph, NULL, NULL, 0));
                HIP_CHECK(hipGraphDestroy(graph));
            }
            HIP_CHECK(hipGraphLaunch(g_dec_graph, g_str));
        }

        HIP_CHECK(hipMemcpyAsync(h_rec, sbuf(B_TOKREC, (size_t)g_batch * max_steps * sizeof(int)),
                            (size_t)Bg * max_steps * sizeof(int), hipMemcpyDeviceToHost, g_str));
        HIP_CHECK(hipStreamSynchronize(g_str));

        for (int i = 0; i < Bg; i++)
        {
            const int *rec = h_rec + (size_t)i * max_steps;
            int *dst = reqs->out_tokens[slot_req[i]];
            int nout = 0;
            while (nout < slot_st[i] && rec[nout] != eos) { dst[nout] = rec[nout]; nout++; }
            reqs->out_lens[slot_req[i]] = nout;
            total += nout;
        }
    }

    for (int r = 0; r < N; r++) free(ptok[r]);
    free(ptok); free(plen); free(order);
    free(slot_req); free(slot_st); free(slot_nout);
    free(h_pos); free(empty);
    free(pf_tok); free(pf_seq); free(pf_pos); free(pf_slot);
    free(pf_last); free(pf_cslot); free(pf_cnp);
    free(h_rec);
    return total;
}
