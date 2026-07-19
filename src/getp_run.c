/* ============================================================================
 * THIS IS THE ONLY FILE YOU MAY MODIFY.
 *
 * Your task: maximize end-to-end throughput (tok/s) of inference() over the
 * request batch, WITHOUT breaking correctness (`make eval` must still pass its
 * thresholds). The intended path is to port the compute to the GPU: start from
 * this correct CPU reference, then replace the forward_* calls with your own
 * kernels — allocate/upload in warm_up(), free in finish().
 *
 * The frozen harness (src/getp_eval.c) times inference() and prints your score.
 * The frozen reference kernels are declared in include/engine.h.
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#include "engine.h"    /* forward_unabsorbed / forward_absorbed / sample */
#include "getp.h"      /* Requests + the warm_up/finish/inference contract */
#include "tokenizer.h" /* tokenizer_encode / tokenizer_eos_id */

// Wrap every HIP call so failures are loud instead of silent.
#define HIP_CHECK(cmd)                                                         \
    do                                                                         \
    {                                                                          \
        hipError_t err = (cmd);                                                \
        if (err != hipSuccess)                                                 \
        {                                                                      \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,       \
                    hipGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Global variable
ModelWeights d_w;
RunState d_s;

void device_malloc_and_copy(void **dst, void *src, size_t size)
{
    HIP_CHECK(hipMalloc(dst, size));
    HIP_CHECK(hipMemcpy(*dst, src, size, hipMemcpyHostToDevice));
}

void device_malloc_model_weights(Transformer *t)
{
    Config *c = &t->config;
    ModelWeights *w = &t->weights;

    /* --- embed / norm / lm_head (global, one tensor each) --- */
    device_malloc_and_copy((void **)&d_w.embed_tokens, w->embed_tokens,
                           c->vocab_size * c->hidden_size * sizeof(bf16_t));
    device_malloc_and_copy((void **)&d_w.norm, w->norm,
                           c->hidden_size * sizeof(bf16_t));
    device_malloc_and_copy((void **)&d_w.lm_head, w->lm_head,
                           c->hidden_size * c->vocab_size * sizeof(bf16_t));

    /* --- allocate per-layer pointer arrays --- */
    d_w.q_proj = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.q_a_proj = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.q_b_proj = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.kv_a_proj = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.W_UK = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.W_UV = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.o_proj = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.input_layernorm = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.post_attn_norm = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.q_a_layernorm = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.kv_a_layernorm = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.dense_gate = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.dense_up = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.dense_down = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.moe_gate = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.moe_gate_bias = malloc(c->n_layers * sizeof(float *));
    d_w.shared_gate = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.shared_up = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.shared_down = malloc(c->n_layers * sizeof(bf16_t *));
    d_w.expert_gate = malloc(c->n_layers * sizeof(bf16_t **));
    d_w.expert_up = malloc(c->n_layers * sizeof(bf16_t **));
    d_w.expert_down = malloc(c->n_layers * sizeof(bf16_t **));

    for (int l = 0; l < c->n_layers; l++)
    {
        if (c->q_lora_rank > 0)
        {
            device_malloc_and_copy((void **)&d_w.q_a_proj[l], w->q_a_proj[l],
                                   c->hidden_size * c->q_lora_rank *
                                       sizeof(bf16_t));
            device_malloc_and_copy(
                (void **)&d_w.q_b_proj[l], w->q_b_proj[l],
                c->q_lora_rank * c->n_heads *
                    (c->qk_nope_head_dim + c->qk_rope_head_dim) *
                    sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.q_a_layernorm[l],
                                   w->q_a_layernorm[l],
                                   c->q_lora_rank * sizeof(bf16_t));
            d_w.q_proj[l] = NULL;
        }
        else
        {
            d_w.q_a_proj[l] = d_w.q_b_proj[l] = d_w.q_a_layernorm[l] = NULL;
            device_malloc_and_copy(
                (void **)&d_w.q_proj[l], w->q_proj[l],
                c->hidden_size * c->n_heads *
                    (c->qk_nope_head_dim + c->qk_rope_head_dim) *
                    sizeof(bf16_t));
        }

        device_malloc_and_copy((void **)&d_w.kv_a_proj[l], w->kv_a_proj[l],
                               c->hidden_size *
                                   (c->kv_lora_rank + c->qk_rope_head_dim) *
                                   sizeof(bf16_t));

        device_malloc_and_copy((void **)&d_w.W_UK[l], w->W_UK[l],
                               c->kv_lora_rank * c->qk_nope_head_dim *
                                   sizeof(bf16_t));

        device_malloc_and_copy((void **)&d_w.W_UV[l], w->W_UV[l],
                               c->kv_lora_rank * c->v_head_dim *
                                   sizeof(bf16_t));

        device_malloc_and_copy((void **)&d_w.o_proj[l], w->o_proj[l],
                               c->n_heads * c->v_head_dim * c->hidden_size *
                                   sizeof(bf16_t));

        device_malloc_and_copy((void **)&d_w.input_layernorm[l],
                               w->input_layernorm[l],
                               c->hidden_size * sizeof(bf16_t));
        device_malloc_and_copy((void **)&d_w.post_attn_norm[l],
                               w->post_attn_norm[l],
                               c->hidden_size * sizeof(bf16_t));
        device_malloc_and_copy((void **)&d_w.kv_a_layernorm[l],
                               w->kv_a_layernorm[l],
                               c->kv_lora_rank * sizeof(bf16_t));

        /* FFN — dense for the first first_k_dense layers, MoE thereafter */
        if (l < c->first_k_dense)
        {
            device_malloc_and_copy(
                (void **)&d_w.dense_gate[l], w->dense_gate[l],
                c->hidden_size * c->dense_inter_size * sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.dense_up[l], w->dense_up[l],
                                   c->hidden_size * c->dense_inter_size *
                                       sizeof(bf16_t));
            device_malloc_and_copy(
                (void **)&d_w.dense_down[l], w->dense_down[l],
                c->dense_inter_size * c->hidden_size * sizeof(bf16_t));

            d_w.moe_gate[l] = NULL;
            d_w.moe_gate_bias[l] = NULL;
            d_w.shared_gate[l] = d_w.shared_up[l] = d_w.shared_down[l] = NULL;
            d_w.expert_gate[l] = d_w.expert_up[l] = d_w.expert_down[l] = NULL;
        }
        else
        {
            w->dense_gate[l] = w->dense_up[l] = w->dense_down[l] = NULL;

            /* MoE routing */
            device_malloc_and_copy((void **)&d_w.moe_gate[l], w->moe_gate[l],
                                   c->hidden_size * c->n_routed_experts *
                                       sizeof(bf16_t));

            /* optional router bias (GLM has it, DSV2 does not) */
            if (w->moe_gate_bias[l])
            {
                device_malloc_and_copy((void **)&d_w.moe_gate_bias[l],
                                       w->moe_gate_bias[l],
                                       c->n_routed_experts * sizeof(float));
            }
            else
            {
                d_w.moe_gate_bias[l] = NULL;
            }

            /* shared expert */
            device_malloc_and_copy((void **)&d_w.shared_gate[l],
                                   w->shared_gate[l],
                                   c->hidden_size * c->n_shared_experts *
                                       c->moe_inter_size * sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.shared_up[l], w->shared_up[l],
                                   c->hidden_size * c->n_shared_experts *
                                       c->moe_inter_size * sizeof(bf16_t));
            device_malloc_and_copy((void **)&d_w.shared_down[l],
                                   w->shared_down[l],
                                   c->n_shared_experts * c->moe_inter_size *
                                       c->hidden_size * sizeof(bf16_t));

            /* routed experts */
            d_w.expert_gate[l] = malloc(c->n_routed_experts * sizeof(bf16_t *));
            d_w.expert_up[l] = malloc(c->n_routed_experts * sizeof(bf16_t *));
            d_w.expert_down[l] = malloc(c->n_routed_experts * sizeof(bf16_t *));
            for (int e = 0; e < c->n_routed_experts; e++)
            {
                device_malloc_and_copy(
                    (void **)&d_w.expert_gate[l][e], w->expert_gate[l][e],
                    c->hidden_size * c->moe_inter_size * sizeof(bf16_t));
                device_malloc_and_copy(
                    (void **)&d_w.expert_up[l][e], w->expert_up[l][e],
                    c->hidden_size * c->moe_inter_size * sizeof(bf16_t));
                device_malloc_and_copy(
                    (void **)&d_w.expert_down[l][e], w->expert_down[l][e],
                    c->moe_inter_size * c->hidden_size * sizeof(bf16_t));
            }
        }
    }
}

void device_malloc_run_state(Transformer *t)
{
    Config *c = &t->config;
    RunState *s = &t->state;

    int h = c->hidden_size;
    int kv_dim = c->kv_lora_rank + c->qk_rope_head_dim;
    int q_dim = c->n_heads * (c->qk_nope_head_dim + c->qk_rope_head_dim);
    /* widest FFN:
    dense, the (n_shared * moe) shared-expert MLP, or one routed * expert */
    int shared = c->moe_inter_size * c->n_shared_experts;
    int inter = c->dense_inter_size;

    int max_seq_len = 32;

    if (shared > inter)
        inter = shared;
    if (c->moe_inter_size > inter)
        inter = c->moe_inter_size;

    device_malloc_and_copy((void **)&d_s.x, s->x, h * sizeof(float));
    device_malloc_and_copy((void **)&d_s.xb, s->xb, h * sizeof(float));
    device_malloc_and_copy((void **)&d_s.xb2, s->xb2, h * sizeof(float));
    device_malloc_and_copy((void **)&d_s.q, s->q, q_dim * sizeof(float));

    if (c->q_lora_rank > 0)
        device_malloc_and_copy((void **)&d_s.q_a, s->q_a,
                               c->q_lora_rank * sizeof(float));
    else
        d_s.q_a = NULL;

    device_malloc_and_copy((void **)&d_s.c_kv, s->c_kv, kv_dim * sizeof(float));
    device_malloc_and_copy((void **)&d_s.att, s->att,
                           c->n_heads * max_seq_len * sizeof(float));
    device_malloc_and_copy((void **)&d_s.kv_cache, s->kv_cache,
                           c->n_layers * max_seq_len * kv_dim * sizeof(float));

    device_malloc_and_copy((void **)&d_s.moe_logits, s->moe_logits,
                           c->n_routed_experts * sizeof(float));
    device_malloc_and_copy((void **)&d_s.expert_out, s->expert_out,
                           h * sizeof(float));
    device_malloc_and_copy((void **)&d_s.hb, s->hb, inter * sizeof(float));
    device_malloc_and_copy((void **)&d_s.hb2, s->hb2, inter * sizeof(float));
    device_malloc_and_copy((void **)&d_s.logits, s->logits,
                           c->vocab_size * sizeof(float));
}

void warm_up(Transformer *t)
{
    device_malloc_model_weights(t);
    device_malloc_run_state(t);
}

void device_free_model_weights(Transformer *t)
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
        HIP_CHECK(hipFree(d_w.W_UK[l]));
        HIP_CHECK(hipFree(d_w.W_UV[l]));
        HIP_CHECK(hipFree(d_w.o_proj[l]));
        HIP_CHECK(hipFree(d_w.input_layernorm[l]));
        HIP_CHECK(hipFree(d_w.post_attn_norm[l]));
        HIP_CHECK(hipFree(d_w.kv_a_layernorm[l]));

        if (l < c->first_k_dense)
        {
            HIP_CHECK(hipFree(d_w.dense_gate[l]));
            HIP_CHECK(hipFree(d_w.dense_up[l]));
            HIP_CHECK(hipFree(d_w.dense_down[l]));
        }
        else
        {
            HIP_CHECK(hipFree(d_w.moe_gate[l]));
            HIP_CHECK(hipFree(d_w.moe_gate_bias[l]));
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
    }

    free(d_w.q_a_proj);
    free(d_w.q_b_proj);
    free(d_w.q_a_layernorm);
    free(d_w.q_proj);
    free(d_w.kv_a_proj);
    free(d_w.W_UK);
    free(d_w.W_UV);
    free(d_w.o_proj);
    free(d_w.input_layernorm);
    free(d_w.post_attn_norm);
    free(d_w.kv_a_layernorm);
    free(d_w.dense_gate);
    free(d_w.dense_up);
    free(d_w.dense_down);
    free(d_w.moe_gate);
    free(d_w.moe_gate_bias);
    free(d_w.shared_gate);
    free(d_w.shared_up);
    free(d_w.shared_down);
}

void device_free_run_state()
{
    HIP_CHECK(hipFree(d_s.x));
    HIP_CHECK(hipFree(d_s.xb));
    HIP_CHECK(hipFree(d_s.xb2));
    HIP_CHECK(hipFree(d_s.q));
    HIP_CHECK(hipFree(d_s.q_a));
    HIP_CHECK(hipFree(d_s.c_kv));
    HIP_CHECK(hipFree(d_s.att));
    HIP_CHECK(hipFree(d_s.kv_cache));
    HIP_CHECK(hipFree(d_s.moe_logits));
    HIP_CHECK(hipFree(d_s.expert_out));
    HIP_CHECK(hipFree(d_s.hb));
    HIP_CHECK(hipFree(d_s.hb2));
    HIP_CHECK(hipFree(d_s.logits));
}

void finish(Transformer *t)
{
    device_free_model_weights(t);
    device_free_run_state();
}

/* Reference implementation: greedy generate each request independently.
 * Per request: prefill the whole prompt (unabsorbed), then absorbed decode
 * until EOS or max_steps. Returns the total number of tokens generated. */
long long inference(Transformer *t, Requests *reqs)
{
    const int vocab = t->config.vocab_size;
    const int max_seq_len = t->config.max_seq_len;
    const int eos = t->tokenizer ? tokenizer_eos_id(t->tokenizer) : -1;

    if (!t->tokenizer)
    {
        fprintf(stderr, "inference: model_dir has no tokenizer.json\n");
        exit(EXIT_FAILURE);
    }

    int *prompt = (int *)malloc((size_t)max_seq_len * sizeof(int));
    long long total = 0;

    for (int r = 0; r < reqs->num_reqs; r++)
    {
        /* add_bos=0 matches the oracle/reference tokenization path. */
        int n_prompt = tokenizer_encode(t->tokenizer, reqs->prompts[r], 0,
                                        prompt, max_seq_len);
        if (n_prompt < 1)
            continue;

        /* Cap generation so prompt + decode stays within the KV cache. */
        int budget = max_seq_len - n_prompt;
        int steps = reqs->max_steps < budget ? reqs->max_steps : budget;
        if (steps < 0)
            steps = 0;

        float *logits = forward_unabsorbed(t, prompt, n_prompt, NULL, NULL);
        int tok = sample(logits, vocab);
        int pos = n_prompt;
        int n_out = 0;

        while (n_out < steps && tok != eos)
        {
            reqs->out_tokens[r][n_out++] = tok;
            logits = forward_absorbed(t, tok, pos);
            tok = sample(logits, vocab);
            pos++;
        }
        reqs->out_lens[r] = n_out;
        total += n_out;
    }

    free(prompt);
    return total;
}
