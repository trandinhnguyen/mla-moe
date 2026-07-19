/* MLA-MOE inference entry point; structure follows llama2.c.
 * forward_unabsorbed() (whole-prompt prefill) and forward_absorbed() (one-token
 * decode) are both oracle-validated for dsv2lite AND glm47 within tol=2e-3.
 * Config (incl. model family) is read from <model_dir>/config.json. */
#define _POSIX_C_SOURCE 199309L   /* clock_gettime / CLOCK_MONOTONIC (bench mode) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "model.h"   /* Config/ModelWeights/RunState/Transformer + loader */
#include "dump.h"    /* run_set_dump / DUMPING / DUMP_F32 / dump_prefill_layer0 */
#include "getp.h"    /* getp() batch-throughput harness (perf grading surface) */

/* ---------------------------------------------------------------------------
 * Unabsorbed (prefill) forward pass.
 * bf16 weights, fp32 compute; one stream, whole prompt in one call.
 * ------------------------------------------------------------------------- */

/* y = (x / rms(x)) * w, over n elements (RMSNorm). */
static void rmsnorm(float *y, const float *x, const bf16_t *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * bf16_to_f32(w[i]);
}

/* y[d] = sum_i x[i] * W[d,i], W is bf16 row-major [d_out, n_in] (== x @ W.T). */
static void matmul(float *y, const float *x, const bf16_t *w, int d_out, int n_in) {
    for (int d = 0; d < d_out; d++) {
        const bf16_t *row = w + (size_t)d * n_in;
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) acc += x[i] * bf16_to_f32(row[i]);
        y[d] = acc;
    }
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* In-place RoPE on a rope_dim slice at `pos`. Two layouts, selected per model:
 *   interleaved==0 (dsv2): complex adjacent-pair — rotate (v[2j], v[2j+1]) in place.
 *   interleaved==1 (GLM):  even/odd pairs, output split-half
 *     [v0'..v31' | v32'..v63'] where v_j'=even_j*c-odd_j*s, v_{half+j}'=odd_j*c+even_j*s.
 * Both rotate by angle pos*inv_freq[j]; q and k share the layout, so it is
 * self-consistent in the score either way. */
static void rope_apply(float *v, int pos, const float *inv_freq, int rope_dim,
                       int interleaved) {
    int half = rope_dim / 2;
    if (!interleaved) {
        for (int j = 0; j < half; j++) {
            float ang = (float)pos * inv_freq[j];
            float c = cosf(ang), s = sinf(ang);
            float x0 = v[2 * j], x1 = v[2 * j + 1];
            v[2 * j]     = x0 * c - x1 * s;
            v[2 * j + 1] = x0 * s + x1 * c;
        }
        return;
    }
    float out[256];   /* qk_rope_head_dim <= 256 */
    for (int j = 0; j < half; j++) {
        float ang = (float)pos * inv_freq[j];
        float c = cosf(ang), s = sinf(ang);
        float even = v[2 * j], odd = v[2 * j + 1];
        out[j]        = even * c - odd * s;
        out[half + j] = odd  * c + even * s;
    }
    memcpy(v, out, rope_dim * sizeof(float));
}

/* Query projection for one token. dsv2: a single q_proj. GLM: a query LoRA
 * q = q_b_proj(rmsnorm(q_a_proj(x))). Writes q[n_heads*q_head_dim]. */
static void project_q(const Config *c, ModelWeights *w, RunState *s, int l,
                      const float *xn, float *q) {
    int q_dim = c->n_heads * (c->qk_nope_head_dim + c->qk_rope_head_dim);
    if (c->q_lora_rank > 0) {
        matmul(s->q_a, xn, w->q_a_proj[l], c->q_lora_rank, c->hidden_size);
        rmsnorm(s->q_a, s->q_a, w->q_a_layernorm[l], c->q_lora_rank, c->mla_norm_eps);
        matmul(q, s->q_a, w->q_b_proj[l], q_dim, c->q_lora_rank);
    } else {
        matmul(q, xn, w->q_proj[l], q_dim, c->hidden_size);
    }
}

/* SwiGLU MLP: dst = down( silu(gate·x) ⊙ (up·x) ). hb/hb2 are [inter] scratch. */
static void swiglu(float *dst, const float *x, const bf16_t *gate,
                   const bf16_t *up, const bf16_t *down,
                   int inter, int hidden, float *hb, float *hb2) {
    matmul(hb,  x, gate, inter, hidden);
    matmul(hb2, x, up,   inter, hidden);
    for (int i = 0; i < inter; i++) hb[i] = silu(hb[i]) * hb2[i];
    matmul(dst, hb, down, hidden, inter);
}

/* numerically-stable softmax over n elements (treats -inf as 0). */
static void softmax(float *x, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = (x[i] == -INFINITY) ? 0.0f : expf(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* One token's FFN, shared by prefill and decode. xn is the post_attn_norm'd
 * hidden; writes out[hidden]. Dense SwiGLU for l < first_k_dense, else top-k
 * routed experts + shared expert. Router flavor is config-driven:
 *   dsv2: softmax scores, greedy top-k, weight = score, no norm, scale 1.
 *   GLM (noaux_tc): sigmoid scores, select by (score + e_score bias), weight =
 *     raw sigmoid score, normalize the top-k, then * routed_scaling.
 * (n_group/topk_group are 1 for GLM-4.7-Flash, so group-limited routing reduces
 * to a plain top-k — not implemented.) When topk_idx/topk_w are non-NULL the
 * chosen experts are returned; s->moe_logits is left holding the full scores. */
static void ffn_compute(const Config *c, ModelWeights *w, RunState *s, int l,
                        const float *xn, float *out, int *topk_idx, float *topk_w) {
    const int H = c->hidden_size;
    if (l < c->first_k_dense) {
        swiglu(out, xn, w->dense_gate[l], w->dense_up[l], w->dense_down[l],
               c->dense_inter_size, H, s->hb, s->hb2);
        return;
    }
    matmul(s->moe_logits, xn, w->moe_gate[l], c->n_routed_experts, H);
    if (c->router_sigmoid)
        for (int e = 0; e < c->n_routed_experts; e++)
            s->moe_logits[e] = 1.0f / (1.0f + expf(-s->moe_logits[e]));
    else
        softmax(s->moe_logits, c->n_routed_experts);

    const float *bias = w->moe_gate_bias[l];   /* GLM e_score_correction_bias (F32), else NULL */
    int  K = c->n_experts_per_tok;
    int  idx[16]; float wts[16];        /* K <= 16 */
    char used[1024] = {0};              /* n_routed <= 1024 */
    for (int r = 0; r < K; r++) {
        int best = -1; float bv = -INFINITY;
        for (int e = 0; e < c->n_routed_experts; e++) {
            if (used[e]) continue;
            float sel = s->moe_logits[e] + (bias ? bias[e] : 0.0f);
            if (sel > bv) { bv = sel; best = e; }
        }
        used[best] = 1; idx[r] = best; wts[r] = s->moe_logits[best];   /* raw score */
    }
    if (c->norm_topk) {
        float sum = 1e-20f;
        for (int r = 0; r < K; r++) sum += wts[r];
        for (int r = 0; r < K; r++) wts[r] /= sum;
    }
    for (int r = 0; r < K; r++) wts[r] *= c->routed_scaling;

    for (int i = 0; i < H; i++) out[i] = 0.0f;
    for (int r = 0; r < K; r++) {
        swiglu(s->expert_out, xn, w->expert_gate[l][idx[r]], w->expert_up[l][idx[r]],
               w->expert_down[l][idx[r]], c->moe_inter_size, H, s->hb, s->hb2);
        for (int i = 0; i < H; i++) out[i] += wts[r] * s->expert_out[i];
    }
    /* shared experts: one MLP of width n_shared * moe_inter */
    swiglu(s->expert_out, xn, w->shared_gate[l], w->shared_up[l], w->shared_down[l],
           c->n_shared_experts * c->moe_inter_size, H, s->hb, s->hb2);
    for (int i = 0; i < H; i++) out[i] += s->expert_out[i];
    if (topk_idx) for (int r = 0; r < K; r++) { topk_idx[r] = idx[r]; topk_w[r] = wts[r]; }
}

int sample(float *logits, int vocab_size);   /* greedy argmax; defined below */

/* Per-position teacher-forced top-1 capture (optional out of forward_unabsorbed).
 * For each scored position p in [0, n_prompt-1) (p predicts tokens[p+1]):
 *   argmax[p] = engine's argmax id;  gap[p] = logits[argmax] - logits[gold].
 * gap >= 0; gap == 0 iff the argmax matched the golden token (or an exact tie).
 * Both arrays are caller-allocated with n_prompt entries. */
typedef struct { int *argmax; float *gap; } TeacherForce;

/* Teacher-forced negative log-likelihood of `target` under one logit row:
 * -log softmax(logits)[target]. Double accumulation; never materializes probs. */
static double nll_at(const float *logits, int vocab, int target) {
    float m = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > m) m = logits[i];
    double se = 0.0;
    for (int i = 0; i < vocab; i++) se += exp((double)(logits[i] - m));
    return ((double)m + log(se)) - (double)logits[target];   /* logZ - logit[target] */
}

/* Whole-prompt prefill. Returns logits at the LAST position (RunState.logits).
 * If nll_out != NULL, also accumulates the teacher-forced sum-NLL over positions
 * [0, n_prompt-1): position p predicts tokens[p+1]. Writes the sum to *nll_out
 * (count = n_prompt-1); one vocab row at a time, no [seq,vocab] buffer.
 * When built with -DMLA_ENABLE_DUMP and a dump dir is set, also writes the
 * oracle-named intermediates for validation. */
float *forward_unabsorbed(Transformer *t, const int *tokens, int n_prompt,
                          double *nll_out, TeacherForce *tf) {
    const Config *c = &t->config;
    ModelWeights *w = &t->weights;
    RunState     *s = &t->state;
    const float  *inv_freq = t->rope_inv_freq;

    const int H   = c->hidden_size;
    const int NH  = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim;
    const int QHD = QKN + QKR;
    const int VHD = c->v_head_dim;
    const int KVL = c->kv_lora_rank;
    const int KVD = KVL + QKR;
    const int VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const size_t kv_stride = (size_t)(QKN + VHD) * KVL;   /* per-head kv_b stride */
    const size_t cache_row = (size_t)c->max_seq_len * KVD;

    /* residual stream for all prompt positions */
    float *xs = malloc((size_t)n_prompt * H * sizeof(float));
    /* per-layer scratch (reused across layers) */
    float *xb    = malloc(H * sizeof(float));
    float *comp  = malloc(KVD * sizeof(float));
    float *qall  = malloc((size_t)n_prompt * NH * QHD * sizeof(float));
    float *knope = malloc((size_t)NH * n_prompt * QKN * sizeof(float));   /* [h][k][d] */
    float *value = malloc((size_t)NH * n_prompt * VHD * sizeof(float));   /* [h][k][d] */
    float *scr   = malloc((size_t)NH * n_prompt * n_prompt * sizeof(float)); /* [h][q][k] scores */
    float *wgt   = malloc((size_t)NH * n_prompt * n_prompt * sizeof(float)); /* [h][q][k] weights */
    float *ctx   = malloc((size_t)NH * VHD * sizeof(float));
    float *ao    = malloc((size_t)n_prompt * H * sizeof(float));   /* attn module out */
    float *mo    = malloc((size_t)n_prompt * H * sizeof(float));   /* mlp module out */
    /* hidden_states snapshot [n_layers+1][n_prompt][H] (dump only) */
    float *hs = DUMPING() ? malloc((size_t)(c->n_layers + 1) * n_prompt * H * sizeof(float)) : NULL;

    /* embed */
    for (int p = 0; p < n_prompt; p++)
        for (int i = 0; i < H; i++)
            xs[p * H + i] = bf16_to_f32(w->embed_tokens[(size_t)tokens[p] * H + i]);
    if (hs) memcpy(hs, xs, (size_t)n_prompt * H * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        float *kv_l = &s->kv_cache[(size_t)l * cache_row];

        /* ---- per-position projections: q (roped), c_kv, k_pe (roped, cached) ---- */
        for (int p = 0; p < n_prompt; p++) {
            rmsnorm(xb, &xs[p * H], w->input_layernorm[l], H, eps);
            project_q(c, w, s, l, xb, &qall[(size_t)p * NH * QHD]);
            for (int h = 0; h < NH; h++)
                rope_apply(&qall[((size_t)p * NH + h) * QHD + QKN], p, inv_freq, QKR,
                           c->rope_interleaved);

            matmul(comp, xb, w->kv_a_proj[l], KVD, H);
            float *cache_p = &kv_l[(size_t)p * KVD];
            rmsnorm(cache_p, comp, w->kv_a_layernorm[l], KVL, c->mla_norm_eps);  /* c_kv */
            memcpy(cache_p + KVL, comp + KVL, QKR * sizeof(float));  /* k_pe (raw) */
            rope_apply(cache_p + KVL, p, inv_freq, QKR, c->rope_interleaved); /* roped */
        }

        /* ---- decompress per-head K_nope and V for every key ---- */
        for (int h = 0; h < NH; h++) {
            const bf16_t *WUK = w->W_UK[l] + (size_t)h * kv_stride;
            const bf16_t *WUV = w->W_UV[l] + (size_t)h * kv_stride;
            for (int k = 0; k < n_prompt; k++) {
                const float *ckv = &kv_l[(size_t)k * KVD];
                matmul(&knope[((size_t)h * n_prompt + k) * QKN], ckv, WUK, QKN, KVL);
                matmul(&value[((size_t)h * n_prompt + k) * VHD], ckv, WUV, VHD, KVL);
            }
        }

        /* ---- attention: per query, per head ---- */
        for (int q = 0; q < n_prompt; q++) {
            for (int h = 0; h < NH; h++) {
                const float *qnope = &qall[((size_t)q * NH + h) * QHD];
                const float *qpe   = qnope + QKN;
                float *row = &scr[((size_t)h * n_prompt + q) * n_prompt];
                for (int k = 0; k < n_prompt; k++) {
                    if (k > q) { row[k] = -INFINITY; continue; }   /* causal */
                    const float *knope_hk = &knope[((size_t)h * n_prompt + k) * QKN];
                    const float *kpe_k    = &kv_l[(size_t)k * KVD + KVL];
                    float dot = 0.0f;
                    for (int d = 0; d < QKN; d++) dot += qnope[d] * knope_hk[d];
                    for (int d = 0; d < QKR; d++) dot += qpe[d]   * kpe_k[d];
                    row[k] = dot * scale;
                }
                float *wrow = &wgt[((size_t)h * n_prompt + q) * n_prompt];
                memcpy(wrow, row, n_prompt * sizeof(float));
                softmax(wrow, n_prompt);

                float *ctx_h = &ctx[(size_t)h * VHD];
                for (int d = 0; d < VHD; d++) ctx_h[d] = 0.0f;
                for (int k = 0; k <= q; k++) {
                    float a = wrow[k];
                    const float *val_hk = &value[((size_t)h * n_prompt + k) * VHD];
                    for (int d = 0; d < VHD; d++) ctx_h[d] += a * val_hk[d];
                }
            }
            matmul(&ao[(size_t)q * H], ctx, w->o_proj[l], H, NH * VHD);
        }
        for (int p = 0; p < n_prompt; p++)
            for (int i = 0; i < H; i++) xs[p * H + i] += ao[p * H + i];

        /* ---- FFN: dense for l < first_k_dense, else MoE (shared helper) ---- */
        int dump_router = DUMPING() && (l == c->first_k_dense);
        for (int p = 0; p < n_prompt; p++) {
            rmsnorm(xb, &xs[p * H], w->post_attn_norm[l], H, eps);
            int ti_p[16]; float tw_p[16];
            ffn_compute(c, w, s, l, xb, &mo[(size_t)p * H],
                        dump_router ? ti_p : NULL, dump_router ? tw_p : NULL);
            /* dump first MoE layer's router (oracle: prefill_moe1_*) */
            if (dump_router) {
                static float rs[8 * 1024]; static int ti[8 * 16]; static float tw[8 * 16];
                int K = c->n_experts_per_tok;
                memcpy(&rs[(size_t)p * c->n_routed_experts], s->moe_logits,
                       c->n_routed_experts * sizeof(float));   /* router softmax */
                for (int r = 0; r < K; r++) { ti[p * K + r] = ti_p[r]; tw[p * K + r] = tw_p[r]; }
                if (p == n_prompt - 1) {
                    DUMP_F32("prefill_moe1_router_scores", rs, (size_t)n_prompt * c->n_routed_experts);
                    DUMP_I32("prefill_moe1_topk_idx", ti, (size_t)n_prompt * K);
                    DUMP_F32("prefill_moe1_topk_w",   tw, (size_t)n_prompt * K);
                }
            }
        }
        for (int p = 0; p < n_prompt; p++)
            for (int i = 0; i < H; i++) xs[p * H + i] += mo[p * H + i];

        /* ---- per-layer dumps ---- */
        if (DUMPING()) {
            char nm[64];
            snprintf(nm, sizeof(nm), "prefill_layer%02d_attn_out", l);
            DUMP_F32(nm, ao, (size_t)n_prompt * H);
            snprintf(nm, sizeof(nm), "prefill_layer%02d_mlp_out", l);
            DUMP_F32(nm, mo, (size_t)n_prompt * H);
            memcpy(&hs[(size_t)(l + 1) * n_prompt * H], xs, (size_t)n_prompt * H * sizeof(float));
            if (l == 0)
                dump_prefill_layer0(qall, kv_l, knope, value, scr, wgt,
                                    NH, n_prompt, QKN, QKR, KVL, VHD);
        }
    }

    /* final norm + lm_head */
    if (nll_out || tf) {
        /* teacher-forced scoring: lm_head per position into s->logits (reused as
         * scratch), score the next token; last row stays in s->logits for the
         * return, matching the non-scoring contract. One vocab row at a time —
         * never materializes a [seq, vocab] buffer (glm47 vocab is ~151k). */
        double sum_nll = 0.0;
        for (int p = 0; p < n_prompt; p++) {
            rmsnorm(xb, &xs[(size_t)p * H], w->norm, H, eps);
            matmul(s->logits, xb, w->lm_head, VOC, H);
            if (p < n_prompt - 1) {
                int gold = tokens[p + 1];
                if (nll_out) sum_nll += nll_at(s->logits, VOC, gold);
                if (tf) {
                    int am = sample(s->logits, VOC);
                    tf->argmax[p] = am;
                    tf->gap[p]    = s->logits[am] - s->logits[gold];
                }
            }
        }
        if (nll_out) *nll_out = sum_nll;
    } else if (DUMPING()) {
        float *logits_all = malloc((size_t)n_prompt * VOC * sizeof(float));
        for (int p = 0; p < n_prompt; p++) {
            rmsnorm(xb, &xs[p * H], w->norm, H, eps);
            /* HF's final hidden_states entry is post-norm: overwrite hs[n_layers] */
            memcpy(&hs[((size_t)c->n_layers * n_prompt + p) * H], xb, H * sizeof(float));
            matmul(&logits_all[(size_t)p * VOC], xb, w->lm_head, VOC, H);
        }
        DUMP_F32("prefill_logits", logits_all, (size_t)n_prompt * VOC);
        DUMP_F32("prefill_hidden_states", hs, (size_t)(c->n_layers + 1) * n_prompt * H);
        memcpy(s->logits, &logits_all[(size_t)(n_prompt - 1) * VOC], VOC * sizeof(float));
        free(logits_all);
    } else {
        rmsnorm(xb, &xs[(size_t)(n_prompt - 1) * H], w->norm, H, eps);
        matmul(s->logits, xb, w->lm_head, VOC, H);
    }

    free(xs); free(xb); free(comp); free(qall); free(knope); free(value);
    free(scr); free(wgt); free(ctx); free(ao); free(mo); free(hs);
    return s->logits;
}

/* Single-token decode via the ABSORBED MLA path. Reads the latent KV cache
 * filled by prefill (and prior decode steps), appends this position's c_kv/k_pe,
 * and attends in latent space: W_UK is folded into the query and W_UV into the
 * output, so no per-head K/V is ever materialized. Mathematically identical to
 * forward_unabsorbed for the same inputs; cheaper at q_len=1. Returns logits.
 * When built with -DMLA_ENABLE_DUMP and a dump dir is set, writes the layer-0
 * decode_* internals (oracle-named). */
float *forward_absorbed(Transformer *t, int token, int pos) {
    const Config *c = &t->config;
    ModelWeights *w = &t->weights;
    RunState     *s = &t->state;
    const float  *inv_freq = t->rope_inv_freq;

    const int H   = c->hidden_size;
    const int NH  = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim;
    const int QHD = QKN + QKR;
    const int VHD = c->v_head_dim;
    const int KVL = c->kv_lora_rank;
    const int KVD = KVL + QKR;
    const int VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const size_t kv_stride = (size_t)(QKN + VHD) * KVL;
    const size_t cache_row = (size_t)c->max_seq_len * KVD;
    const int kv_len = pos + 1;                /* keys: [0..pos] inclusive */

    float *x    = malloc(H * sizeof(float));
    float *xb   = malloc(H * sizeof(float));
    float *comp = malloc(KVD * sizeof(float));
    float *q    = malloc((size_t)NH * QHD * sizeof(float));
    float *qabs = malloc(KVL * sizeof(float));            /* q_absorbed (per head) */
    float *score= malloc(kv_len * sizeof(float));
    float *clat = malloc(KVL * sizeof(float));            /* latent context (per head) */
    float *ctx  = malloc((size_t)NH * VHD * sizeof(float));
    float *ao   = malloc(H * sizeof(float));
    float *mo   = malloc(H * sizeof(float));
    /* layer-0 decode dump scratch (heads-major, like the oracle) */
    float *d_qabs = DUMPING() ? malloc((size_t)NH * KVL * sizeof(float)) : NULL;
    float *d_clat = DUMPING() ? malloc((size_t)NH * KVL * sizeof(float)) : NULL;
    float *d_attn = DUMPING() ? malloc((size_t)NH * kv_len * sizeof(float)) : NULL;

    for (int i = 0; i < H; i++) x[i] = bf16_to_f32(w->embed_tokens[(size_t)token * H + i]);

    for (int l = 0; l < c->n_layers; l++) {
        float *kv_l = &s->kv_cache[(size_t)l * cache_row];

        rmsnorm(xb, x, w->input_layernorm[l], H, eps);
        project_q(c, w, s, l, xb, q);
        for (int h = 0; h < NH; h++)
            rope_apply(&q[(size_t)h * QHD + QKN], pos, inv_freq, QKR, c->rope_interleaved);

        /* append this position's latent KV */
        matmul(comp, xb, w->kv_a_proj[l], KVD, H);
        float *cache_p = &kv_l[(size_t)pos * KVD];
        rmsnorm(cache_p, comp, w->kv_a_layernorm[l], KVL, c->mla_norm_eps);
        memcpy(cache_p + KVL, comp + KVL, QKR * sizeof(float));
        rope_apply(cache_p + KVL, pos, inv_freq, QKR, c->rope_interleaved);

        for (int h = 0; h < NH; h++) {
            const bf16_t *WUK = w->W_UK[l] + (size_t)h * kv_stride;
            const bf16_t *WUV = w->W_UV[l] + (size_t)h * kv_stride;
            const float  *qnope = &q[(size_t)h * QHD];
            const float  *qpe   = qnope + QKN;

            /* q_absorbed[r] = sum_d q_nope[d] * W_UK[d,r]  (fold W_UK into query) */
            for (int r = 0; r < KVL; r++) qabs[r] = 0.0f;
            for (int d = 0; d < QKN; d++) {
                float qd = qnope[d];
                const bf16_t *row = WUK + (size_t)d * KVL;
                for (int r = 0; r < KVL; r++) qabs[r] += qd * bf16_to_f32(row[r]);
            }
            /* scores in latent space: q_absorbed·c_kv + q_pe·k_pe */
            for (int k = 0; k < kv_len; k++) {
                const float *ckv  = &kv_l[(size_t)k * KVD];
                const float *kpe  = ckv + KVL;
                float sn = 0.0f, sp = 0.0f;
                for (int r = 0; r < KVL; r++) sn += qabs[r] * ckv[r];
                for (int d = 0; d < QKR; d++) sp += qpe[d]  * kpe[d];
                score[k] = (sn + sp) * scale;
            }
            softmax(score, kv_len);
            /* latent context, then up-project with W_UV */
            for (int r = 0; r < KVL; r++) clat[r] = 0.0f;
            for (int k = 0; k < kv_len; k++) {
                float a = score[k];
                const float *ckv = &kv_l[(size_t)k * KVD];
                for (int r = 0; r < KVL; r++) clat[r] += a * ckv[r];
            }
            matmul(&ctx[(size_t)h * VHD], clat, WUV, VHD, KVL);

            if (DUMPING() && l == 0) {
                memcpy(&d_qabs[(size_t)h * KVL], qabs, KVL * sizeof(float));
                memcpy(&d_clat[(size_t)h * KVL], clat, KVL * sizeof(float));
                memcpy(&d_attn[(size_t)h * kv_len], score, kv_len * sizeof(float));
            }
        }
        matmul(ao, ctx, w->o_proj[l], H, NH * VHD);
        for (int i = 0; i < H; i++) x[i] += ao[i];

        rmsnorm(xb, x, w->post_attn_norm[l], H, eps);
        ffn_compute(c, w, s, l, xb, mo, NULL, NULL);
        for (int i = 0; i < H; i++) x[i] += mo[i];

        if (DUMPING() && l == 0) {
            DUMP_F32("decode_layer0_q_absorbed", d_qabs, (size_t)NH * KVL);
            DUMP_F32("decode_layer0_ctx_latent", d_clat, (size_t)NH * KVL);
            DUMP_F32("decode_layer0_attn_weights", d_attn, (size_t)NH * kv_len);
            DUMP_F32("decode_layer0_attn_out", ao, H);
            /* the full latent cache (prefill + this step), heads-shared */
            float *cc = malloc((size_t)kv_len * KVL * sizeof(float));
            float *kk = malloc((size_t)kv_len * QKR * sizeof(float));
            for (int k = 0; k < kv_len; k++) {
                memcpy(&cc[(size_t)k * KVL], &kv_l[(size_t)k * KVD], KVL * sizeof(float));
                memcpy(&kk[(size_t)k * QKR], &kv_l[(size_t)k * KVD + KVL], QKR * sizeof(float));
            }
            DUMP_F32("decode_layer0_c_kv_cache", cc, (size_t)kv_len * KVL);
            DUMP_F32("decode_layer0_k_pe_cache", kk, (size_t)kv_len * QKR);
            free(cc); free(kk);
        }
    }

    rmsnorm(xb, x, w->norm, H, eps);
    matmul(s->logits, xb, w->lm_head, VOC, H);
    if (DUMPING()) DUMP_F32("decode_logits", s->logits, VOC);

    free(x); free(xb); free(comp); free(q); free(qabs); free(score);
    free(clat); free(ctx); free(ao); free(mo);
    free(d_qabs); free(d_clat); free(d_attn);
    return s->logits;
}

/* greedy argmax over logits */
int sample(float *logits, int vocab_size) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

/* Read up to max int32 tokens from a raw little-endian .i32.bin file. */
static int read_tokens_i32(const char *path, int *out, int max) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open tokens file: %s\n", path); exit(1); }
    int32_t buf[4096];
    size_t n = fread(buf, sizeof(int32_t), max < 4096 ? max : 4096, f);
    fclose(f);
    for (size_t i = 0; i < n; i++) out[i] = buf[i];
    return (int)n;
}

/* Two-path generation: one unabsorbed prefill over the prompt, then a greedy
 * absorbed-decode loop. Prints generated token ids, and the decoded text when a
 * tokenizer is loaded. Stops early on EOS. */
static void generate(Transformer *t, const int *prompt, int n_prompt, int max_new) {
    int eos = t->tokenizer ? tokenizer_eos_id(t->tokenizer) : -1;
    float *logits = forward_unabsorbed(t, prompt, n_prompt, NULL, NULL);
    int tok = sample(logits, t->config.vocab_size);
    int pos = n_prompt;
    int *gen = malloc((size_t)max_new * sizeof(int)); int ng = 0;
    printf("generated:");
    for (int i = 0; i < max_new && tok != eos; i++) {
        printf(" %d", tok);
        gen[ng++] = tok;
        logits = forward_absorbed(t, tok, pos);
        tok = sample(logits, t->config.vocab_size);
        pos++;
    }
    printf("\n");
    if (t->tokenizer) {
        char *s = tokenizer_decode(t->tokenizer, gen, ng);
        printf("text:%s\n", s);
        free(s);
    }
    free(gen);
}

/* Monotonic wall-clock in ms — the timing primitive for `bench` mode. */
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

/* Median of n doubles (insertion sort on a scratch copy; n is tiny). */
static double median_double(const double *v, int n) {
    double *s = malloc((size_t)n * sizeof(double));
    memcpy(s, v, (size_t)n * sizeof(double));
    for (int i = 1; i < n; i++) {
        double key = s[i]; int j = i - 1;
        while (j >= 0 && s[j] > key) { s[j + 1] = s[j]; j--; }
        s[j + 1] = key;
    }
    double m = (n & 1) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    free(s);
    return m;
}

/* Device-agnostic prefill/decode benchmark. Wall-clock brackets the two forward
 * entry points; both return host logits, so any backend (incl. a future GPU
 * port) has synchronized by the time the clock stops. reps>1: rep 0 is warmup
 * (cold pages / allocs) and excluded from the median summary. prefill_ms is the
 * time-to-first-token (the first output token is sample() of prefill's logits). */
static void run_bench(Transformer *t, const int *tokens, int n_prompt,
                      int n_decode, int reps) {
    int VOC = t->config.vocab_size;
    if (n_prompt + n_decode > t->config.max_seq_len) {
        fprintf(stderr, "bench: n_prompt+n_decode (%d) > max_seq_len (%d)\n",
                n_prompt + n_decode, t->config.max_seq_len);
        return;
    }
    int keep = reps > 1 ? reps - 1 : reps;   /* summary excludes warmup rep 0 */
    double *pf_ms   = malloc((size_t)keep * sizeof(double));
    double *pf_toks = malloc((size_t)keep * sizeof(double));
    double *dc_toks = malloc((size_t)keep * sizeof(double));
    double *tpot    = malloc((size_t)keep * sizeof(double));
    int k = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        float *lg = forward_unabsorbed(t, tokens, n_prompt, NULL, NULL);
        double prefill_ms = now_ms() - t0;
        int tok = sample(lg, VOC);

        double d0 = now_ms();
        for (int pos = n_prompt; pos < n_prompt + n_decode; pos++) {
            lg = forward_absorbed(t, tok, pos);
            tok = sample(lg, VOC);
        }
        double decode_ms = now_ms() - d0;

        double pf_ts = (double)n_prompt / (prefill_ms / 1e3);
        double dc_ts = (double)n_decode / (decode_ms / 1e3);
        double tp    = decode_ms / (double)n_decode;
        printf("bench rep=%d prefill_tokens=%d prefill_ms=%.3f prefill_tok_s=%.2f "
               "decode_tokens=%d decode_ms=%.3f decode_tok_s=%.2f tpot_ms=%.3f\n",
               r, n_prompt, prefill_ms, pf_ts, n_decode, decode_ms, dc_ts, tp);
        if (reps > 1 && r == 0) continue;    /* warmup rep excluded from summary */
        pf_ms[k] = prefill_ms; pf_toks[k] = pf_ts;
        dc_toks[k] = dc_ts; tpot[k] = tp; k++;
    }
    printf("bench_summary prefill_tokens=%d decode_tokens=%d reps=%d "
           "prefill_ms_median=%.3f prefill_tok_s_median=%.2f "
           "decode_tok_s_median=%.2f tpot_ms_median=%.3f\n",
           n_prompt, n_decode, k,
           median_double(pf_ms, k), median_double(pf_toks, k),
           median_double(dc_toks, k), median_double(tpot, k));
    free(pf_ms); free(pf_toks); free(dc_toks); free(tpot);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <model_dir> [tokens.i32.bin | -p \"text\"] [MODE [ARG]]\n"
            "  model_dir: holds config.json, model.safetensors.index.json, shards\n"
            "             (model family auto-detected from config.json)\n"
            "  no input          -> weight-load smoke test\n"
            "  tokens file       -> prefill + greedy absorbed decode (prints token ids)\n"
            "  -p \"text\"         -> tokenize text (needs tokenizer.json), then decode as above\n"
            "  MODE = dump_dir   -> prefill + ONE decode step, writing oracle-named\n"
            "                       prefill_*/decode_* intermediates for validation\n"
            "  MODE = 'ppl'      -> teacher-forced perplexity over the token sequence\n"
            "                       (prints 'ppl <value>'); used by tests/oracle/compare_ppl.py\n"
            "  MODE = 'teacher' [prompt_len]\n"
            "                    -> teacher-forced top-1 over the sequence, BOTH engine\n"
            "                       paths: 'P' rows (prefill/unabsorbed) and 'D' rows\n"
            "                       (decode/absorbed, positions >= prompt_len). Each row:\n"
            "                       '<P|D> <pos> <gold> <argmax> <gap>'. Used by tests/eval/eval.py\n"
            "  MODE = 'gen' [max_new]\n"
            "                    -> greedy-decode max_new tokens; prints 'completion <ids...>'\n"
            "  MODE = 'bench' [n_decode] [reps]\n"
            "                    -> device-agnostic perf: time prefill (n_prompt tokens) and\n"
            "                       decode (n_decode steps), reps times (rep 0 = warmup); prints\n"
            "                       'bench ...' per rep + a 'bench_summary ...' median line\n"
            "  'getp' <requests.txt> <output.txt> [steps]\n"
            "                    -> batch-throughput grading: warm_up, then timed inference over\n"
            "                       the request set (line 0 = count, then one prompt/line); prints\n"
            "                       'achieved throughput TPS (tok/s)' and writes generated ids.\n"
            "                       The candidate's inference() lives in src/getp_run.hip\n",
            argv[0]);
        return 1;
    }
    const char *model_dir   = argv[1];
    const char *arg2        = (argc > 2) ? argv[2] : NULL;
    int   getp_mode         = arg2 && strcmp(arg2, "getp") == 0;
    int   prompt_mode       = arg2 && strcmp(arg2, "-p") == 0;
    const char *prompt_text = prompt_mode ? ((argc > 3) ? argv[3] : NULL) : NULL;
    const char *tokens_path = prompt_mode ? NULL : arg2;
    const char *third       = prompt_mode ? ((argc > 4) ? argv[4] : NULL)
                                          : ((argc > 3) ? argv[3] : NULL);
    const char *fourth      = prompt_mode ? ((argc > 5) ? argv[5] : NULL)
                                          : ((argc > 4) ? argv[4] : NULL);
    const char *fifth       = prompt_mode ? ((argc > 6) ? argv[6] : NULL)
                                          : ((argc > 5) ? argv[5] : NULL);
    int   ppl_mode          = third && strcmp(third, "ppl") == 0;
    int   teacher_mode      = third && strcmp(third, "teacher") == 0;
    int   gen_mode          = third && strcmp(third, "gen") == 0;
    int   bench_mode        = third && strcmp(third, "bench") == 0;
    const char *dump_dir    = (third && !ppl_mode && !teacher_mode && !gen_mode
                               && !bench_mode && !getp_mode) ? third : NULL;
#ifndef MLA_ENABLE_DUMP
    if (dump_dir)
        fprintf(stderr, "note: dump_dir given but dumps not compiled in "
                        "(rebuild with `make DUMP=1`); running without dumps.\n");
#endif

    Transformer t;
    build_transformer(&t, model_dir);
    printf("Loaded %zu tensors\n", st_count(t.store));
    printf("Config: n_layers=%d hidden=%d vocab=%d\n",
           t.config.n_layers, t.config.hidden_size, t.config.vocab_size);

    if (getp_mode) {
        const char *req_file = (argc > 3) ? argv[3] : NULL;
        const char *out_file = (argc > 4) ? argv[4] : NULL;
        int         steps    = (argc > 5) ? atoi(argv[5]) : 0;
        if (!req_file || !out_file) {
            fprintf(stderr, "getp: usage: %s <model_dir> getp <requests.txt> "
                            "<output.txt> [steps]\n", argv[0]);
            free_transformer(&t);
            return 1;
        }
        getp(&t, req_file, out_file, steps);
        free_transformer(&t);
        return 0;
    }

    if (!tokens_path && !prompt_mode) {
        printf("weights loaded; pass a tokens file or -p \"text\" to run prefill.\n");
        free_transformer(&t);
        return 0;
    }

    int tokens[4096];
    int n_prompt;
    if (prompt_mode) {
        if (!prompt_text) { fprintf(stderr, "-p needs a text argument\n"); free_transformer(&t); return 1; }
        if (!t.tokenizer) { fprintf(stderr, "no tokenizer.json in %s\n", model_dir); free_transformer(&t); return 1; }
        /* add_bos=0: matches the oracle's tokenization (gen_oracle adds no special
         * tokens) and the validated reference path. */
        n_prompt = tokenizer_encode(t.tokenizer, prompt_text, 0, tokens, 4096);
    } else {
        n_prompt = read_tokens_i32(tokens_path, tokens, 4096);
    }
    if (bench_mode) {
        printf("Prompt: %d tokens\n", n_prompt);   /* quiet: skip the id dump */
    } else {
        printf("Prompt: %d tokens [", n_prompt);
        for (int i = 0; i < n_prompt; i++) printf("%s%d", i ? ", " : "", tokens[i]);
        printf("]\n");
    }

    if (ppl_mode) {
        /* teacher-forced perplexity over the token sequence (n_prompt-1 targets) */
        if (n_prompt < 2) {
            fprintf(stderr, "ppl needs >= 2 tokens (got %d)\n", n_prompt);
            free_transformer(&t);
            return 1;
        }
        double nll = 0.0;
        forward_unabsorbed(&t, tokens, n_prompt, &nll, NULL);
        int ntok = n_prompt - 1;
        printf("ppl %.6f\nnll %.6f tokens %d\n", exp(nll / ntok), nll, ntok);
    } else if (teacher_mode) {
        /* teacher-forced top-1 over the sequence, via BOTH engine paths.
         * Feeds the GOLDEN token at every step (never the engine's own argmax),
         * so a single flip cannot cascade. eval.py filters to the completion
         * region (pos >= prompt_len) and applies the tie policy. */
        if (n_prompt < 2) {
            fprintf(stderr, "teacher needs >= 2 tokens (got %d)\n", n_prompt);
            free_transformer(&t); return 1;
        }
        int VOC = t.config.vocab_size;
        int prompt_len = fourth ? atoi(fourth) : 1;
        if (prompt_len < 1) prompt_len = 1;
        if (prompt_len >= n_prompt) prompt_len = n_prompt - 1;
        printf("teacher_begin n_prompt=%d prompt_len=%d\n", n_prompt, prompt_len);

        /* --- prefill / unabsorbed path: all positions in one forward --- */
        TeacherForce tf = { malloc((size_t)n_prompt * sizeof(int)),
                            malloc((size_t)n_prompt * sizeof(float)) };
        forward_unabsorbed(&t, tokens, n_prompt, NULL, &tf);
        for (int p = 0; p < n_prompt - 1; p++)   /* row keyed by predicted index p+1 */
            printf("P %d %d %d %.6f\n", p + 1, tokens[p + 1], tf.argmax[p], tf.gap[p]);
        free(tf.argmax); free(tf.gap);

        /* --- decode / absorbed path: prefill the prompt, then step the
         * completion region one GOLDEN token at a time through the KV cache.
         * This is the actual inference path and the one that accumulates
         * long-context / YaRN error over many steps. --- */
        float *lg = forward_unabsorbed(&t, tokens, prompt_len, NULL, NULL);
        for (int pos = prompt_len; pos < n_prompt; pos++) {
            int gold = tokens[pos];
            int am = sample(lg, VOC);
            printf("D %d %d %d %.6f\n", pos, gold, am, lg[am] - lg[gold]);
            if (pos + 1 < n_prompt) lg = forward_absorbed(&t, tokens[pos], pos);
        }
        printf("teacher_end\n");
    } else if (gen_mode) {
        /* greedy free-run for the fuzzy tier; emit the completion ids only. */
        int max_new = fourth ? atoi(fourth) : 256;
        int eos = t.tokenizer ? tokenizer_eos_id(t.tokenizer) : -1;
        float *logits = forward_unabsorbed(&t, tokens, n_prompt, NULL, NULL);
        int tok = sample(logits, t.config.vocab_size);
        int pos = n_prompt;
        printf("completion");
        for (int i = 0; i < max_new && tok != eos; i++) {
            printf(" %d", tok);
            logits = forward_absorbed(&t, tok, pos);
            tok = sample(logits, t.config.vocab_size);
            pos++;
        }
        printf("\n");
    } else if (bench_mode) {
        int n_decode = fourth ? atoi(fourth) : 64;
        int reps     = fifth  ? atoi(fifth)  : 5;
        if (n_decode < 1) n_decode = 1;
        if (reps < 1)     reps = 1;
        run_bench(&t, tokens, n_prompt, n_decode, reps);
    } else if (dump_dir) {
        /* validation scenario: prefill, then exactly one absorbed-decode step
         * (mirrors gen_oracle: feed the prefill argmax at position n_prompt). */
        run_set_dump(dump_dir);
        float *logits = forward_unabsorbed(&t, tokens, n_prompt, NULL, NULL);
        int next = sample(logits, t.config.vocab_size);
        printf("prefill argmax = %d  logit=%.6f\n", next, logits[next]);
        float *dlog = forward_absorbed(&t, next, n_prompt);
        int next2 = sample(dlog, t.config.vocab_size);
        printf("decode  argmax = %d  logit=%.6f\n", next2, dlog[next2]);
    } else {
        generate(&t, tokens, n_prompt, 16);
    }

    free_transformer(&t);
    return 0;
}
