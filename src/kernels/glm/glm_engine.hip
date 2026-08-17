/* ============================================================================
 * glm_engine.hip -- GLM-4.7-Flash host engine (a HIP/C++ translation unit).
 *
 * GPU port of the CPU reference in src/run.c (forward_unabsorbed /
 * forward_absorbed). Weights are uploaded once in glm_warm_up(); every layer op
 * (RMSNorm, the various Linear projections, RoPE, MLA attention -- both the
 * unabsorbed prefill path and the absorbed decode path -- and the dense/MoE
 * SwiGLU FFN) runs as a __global__ kernel on the GPU. The __global__ kernels
 * are the GLM set in src/kernels/glm/, declared in src/kernels/glm/kernels.h.
 * This file hosts the host-side orchestration: weight upload, per-layer
 * kernel-launch sequencing, and the forward passes.
 *
 * ENTRY POINTS: glm_warm_up / glm_finish / glm_inference (declared in
 * engines.h). The thin dispatcher in src/getp_run.hip implements the frozen
 * getp.h contract and forwards to these when it detects a GLM Transformer. The
 * optimization defaults (W8A8, 128-tile GEMM, int8 KV cache, B=1024) are tuned
 * for GLM-4.7-Flash. This engine, compiled into its own TU via glm.hip, is
 * fully independent of the DeepSeek-V2-Lite engine (src/kernels/dsv/).
 *
 * The only host-side compute left per step is the MoE top-k expert selection
 * (a handful of floats over <=64 experts -- negligible cost, identical to the
 * CPU reference's greedy top-k).
 * ==========================================================================*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include <hip/hip_runtime.h>

#include "getp.h"      /* Requests */
#include "tokenizer.h" /* tokenizer_encode / tokenizer_eos_id */
#include "kernels.h"  /* GLM kernel declarations */
#include "../engines.h"  /* ModelKind, model_name, glm_* entry points */

// Wrap every HIP call so failures are loud instead of silent.
#define HIP_CHECK(cmd)                                                   \
    do                                                                   \
    {                                                                    \
        hipError_t err = (cmd);                                          \
        if (err != hipSuccess)                                           \
        {                                                                \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err));                             \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* Cap on the device latent-KV cache length (positions), independent of the
 * model's (huge, YaRN-extended) max_position_embeddings -- real requests are a
 * few dozen prompt tokens plus <= a few hundred generated ones. Clamped
 * against the model's own max_seq_len at warm_up() time. */
#define CACHE_CAP_LIMIT 16384

/* Per-model differences (all Config-driven, handled inline where they matter):
 *                       DeepSeek-V2-Lite       GLM-4.7-Flash
 *   query projection    single q_proj          q_a/q_b LoRA (q_lora_rank > 0)
 *   MoE router          softmax                sigmoid + e_score bias
 *   routed_scaling      1.0                    1.8
 *   RoPE layout         complex adjacent-pair  split-half (rope_interleaved = 1)
 *   mla_norm_eps        1e-5                   1e-6
 * ModelKind / model_kind() / model_name() live in engines.h. This engine is
 * entered only for GLM Transformers, but the forward path stays config-driven. */

static ModelWeights d_w;         /* device pointers, same layout as host ModelWeights */
static int8_t *d_kv_cache = NULL; /* int8 [n_layers, g_batch, g_cap, KVD]; run16: int8 quarters
                                   * the ~24GB@B=384 fp32 cache (run15 was bf16). */
static float  *d_kv_scale = NULL; /* fp32 [n_layers, g_batch, g_cap]: per-(slot,pos) row scale. */
static float *d_inv_freq = NULL; /* [qk_rope_head_dim/2] */
static float *d_logits   = NULL; /* [vocab_size] device scratch for the final lm_head row */
static int   *d_argmax   = NULL; /* [1] device-computed greedy token id */
static int    g_cache_cap = 0;

/* ---- run2 grouped-batching state ------------------------------------------
 * The KV cache is now per-slot: d_kv_cache is [n_layers, g_batch, g_cap, KVD].
 * Prefill writes request i's prompt into slot i's region; batched decode then
 * advances all g_batch slots together (weights read once per batch). g_batch /
 * g_cap are set in warm_up; g_batch is overridable via MLA_BATCH. All decode
 * scratch below is allocated once (grow-free hot path) sized for g_batch. */
static int    g_batch = 1280;  /* concurrent requests per group (<= this); overridable
                                * via MLA_BATCH. run15 (bf16 KV) freed VRAM to raise the sweet spot
                                * 384->512; run16 made the KV cache int8 (~16GB@B=1024, quarter of the
                                * old fp32); run17 showed a larger CLEAN group still helps. run18
                                * (MFMA prefill flash-attention) makes prefill far cheaper. run34 chose
                                * B=1280 on the clean-group dataset (requests_dup2odd.txt, 512x2 + 256
                                * odd = one clean 256-multiple group); run37 pins it as the default with
                                * the VRAM-max prefill chunk (g_pfchunk=10240): B=1280 dup2odd = 836.65
                                * tok/s (acc PASS). NB near the VRAM ceiling on a 68GB-free GCD -- B>=1536
                                * or g_pfchunk>=12288 OOM at B=1280; lower MLA_BATCH / MLA_PFCHUNK if a
                                * busier shared GCD OOMs. int8 KV cache (~20GB@1280) is what makes it fit. */
static int    g_cap   = 576;   /* per-slot cache positions (max prompt 512 + 64 steps) */
static int    g_pfchunk = 10240; /* chunked-prefill token budget: pack whole prompts up
                                 * to this many tokens into one batched forward so weights
                                 * are read once per chunk (not once per request). Sizes
                                 * the MoE scratch (MR). Overridable via MLA_PFCHUNK; set 0
                                 * (or MLA_PFBATCH=0) to fall back to per-request prefill.
                                 * run35: prefill is ~85% per-token / L2-BW-bound, only ~15%
                                 * fixed weight-stream, so bumping the chunk amortizes that
                                 * fixed cost over fewer forwards (bit-safe: per-prompt causal,
                                 * atomic-noise only, acc PASS). run37: the chunk max is
                                 * VRAM-bound and shrinks as batch grows (KV cache competes) --
                                 * at B=1280 the largest chunk that fits a 68GB-free GCD is
                                 * ~10240 (12288 OOMs), giving 831->836.65 tok/s. Default 10240
                                 * pairs with the default B=1280; drop it if a busier GCD OOMs. */
static int    g_pfbatch = 1;    /* enable chunked/batched prefill (MLA_PFBATCH). */
static int    g_chunkfuse = 0;  /* run36: interleaved chunked prefill -- fuse each decode
                                 * step with a prefill chunk into ONE forward so the shared
                                 * GEMM weight-streams serve both phases. Opt-in via
                                 * MLA_CHUNKFUSE=1 (default off; run35 analysis predicts
                                 * ~neutral at constant completion length -- measured lever). */
static int    g_fusechunk = 4096; /* prefill tokens admitted per fused forward (MLA_FUSECHUNK).
                                 * Kept modest so M = g_batch + g_fusechunk bounds fused scratch. */

static float *b_x, *b_xb, *b_qa, *b_q, *b_comp, *b_qabs, *b_score, *b_clat;
static float *b_ctx, *b_ao, *b_mo, *b_hb, *b_hb2, *b_shared;
static float *b_moelog, *b_topk_wt, *b_sorted_wt, *b_hbk, *b_hb2k, *b_downk;
static float *b_logits;
/* run19 W8A8 MoE: int8 activations + per-row scale for gate/up (xn) and down (hbk). */
static int8_t *b_xn_i8, *b_hbk_i8;
static float  *b_xn_scale, *b_hbk_scale;
static int   *b_topk_idx, *b_ecount, *b_eoff, *b_cursor, *b_sorted_slot;
static int   *b_tokens, *b_pos, *b_argmax;

static int inter_max_of(const Config *c);   /* defined below; used in warm_up */

/* Per-layer device pointer tables for the routed experts (n_routed device
 * weight pointers each), uploaded once in warm_up so GPU-side routing can gather
 * the top-k experts without any host round-trip. NULL for dense layers. */
static bf16_t ***d_ep_gate  = NULL;
static bf16_t ***d_ep_up    = NULL;
static bf16_t ***d_ep_down  = NULL;
static float   **d_moe_bias = NULL; /* device copy of GLM e_score bias, or NULL */
static bf16_t  **d_wuk_t    = NULL; /* [n_layers] transposed W_UK: [NH, KVL, QKN] */

/* run4: int8-quantized MoE routed experts. Experts are ~95% of the 58GB weight
 * and in the active path (ffn_batched) are read only by the moe_grouped GEMM,
 * which is HBM-weight-bandwidth-bound. We quantize each expert's bf16 weight to
 * int8 + per-output-row fp32 scale at warm_up, free the bf16 (net ~27GB freed),
 * and feed the int8 tables to moe_grouped_mfma_i8_kernel. Per MoE layer we keep a
 * host mirror of the per-expert device buffers (hq/hs arrays, for freeing) and a
 * device pointer table (d_epq / d_eps) the kernel indexes by expert. */
static int8_t ***hq_g = NULL, ***hq_u = NULL, ***hq_d = NULL;   /* [l][e] device int8 buf  */
static float  ***hs_g = NULL, ***hs_u = NULL, ***hs_d = NULL;   /* [l][e] device scale buf */
static int8_t ***d_epq_gate = NULL, ***d_epq_up = NULL, ***d_epq_down = NULL; /* [l] -> device int8*[NR] */
static float  ***d_eps_gate = NULL, ***d_eps_up = NULL, ***d_eps_down = NULL; /* [l] -> device float*[NR] */
/* run20 W8A8 dense: int8 copies + per-row scale of the big per-layer projections
 * (q_a/q_b/kv_a/o_proj); lm_head + moe_gate stay bf16 (accuracy). g_dense_w8a8 gate. */
static int   g_dense_w8a8 = 1;
/* run21: route large W8A8 dense GEMMs through the 128x128-tile kernel (halves L2
 * traffic) when the coarser grid still fills the CUs. Opt-out via MLA_DENSE_T128=0. */
static int   g_dense_t128 = 1;
/* run33: BLOCK-WISE W4A8 dense attn projections (q_a/q_b/kv_a/o_proj): int4 weight
 * with scale per K-group of 128. When on, the dq/ds tables hold packed int4 +
 * [N][K/128] group scales and launch_matmul_w8a8(bw=1) routes to block-wise int4.
 * Opt-in MLA_DENSE_W4A8_BW=1 (occupancy risk from two-level acc; see run32). */
static int   g_dense_w4a8_bw = 0;
static int8_t **dq_qa=NULL, **dq_qb=NULL, **dq_kva=NULL, **dq_op=NULL;
static float  **ds_qa=NULL, **ds_qb=NULL, **ds_kva=NULL, **ds_op=NULL;
/* run22: extend W8A8 to the remaining large bf16 GEMMs. g_ffn_w8a8 = shared-expert
 * + dense-layer FFN (gate/up/down, structurally identical to the already-W8A8
 * routed experts -> acc −0.003, noise). g_lmhead_w8a8 = lm_head: measured a BAD
 * trade — perturbing argmax over int8-approx logits cost ~0.008 METEOR for only
 * +0.4% tok/s, so it is DEFAULT OFF (opt-in via MLA_LMHEAD_W8A8=1). moe_gate stays
 * bf16 always (tiny N=64 GEMM, ~0 throughput, chaotic top-K routing). FFN opt-out
 * via MLA_FFN_W8A8=0. */
static int   g_ffn_w8a8 = 1, g_lmhead_w8a8 = 0;
/* run29: W4A8 MoE experts -- int4 packed weight (per-row scale) x int8 activation.
 * Halves the MoE-GEMM weight L2 bytes (the L2-BW bound at B=1024). Opt-in via
 * MLA_MOE_W4A8=1 (accuracy risk: int4 is coarse; measured before spreading to dense).
 * When on, the MoE expert tables (d_epq_gate/...) hold packed int4 instead of int8. */
static int   g_moe_w4a8 = 0;
/* run32: BLOCK-WISE W4A8 MoE (int4 weight, scale per K-group of 128). Fine-grained
 * scale (DeepSeek/AWQ) to recover the accuracy per-row int4 lost (run29 FAILed
 * METEOR). Opt-in MLA_MOE_W4A8_BW=1; when on the expert scale tables hold
 * [rows][K/128] group scales and the block-wise GEMM/quant kernels are used. */
static int   g_moe_w4a8_bw = 0;
#define MOE_W4A8_G 128
/* run31: TBK=64 / 128-bit-load MoE GEMM (bit-identical; 4x fewer syncs). Opt-in
 * MLA_MOE_TBK64=1 to A/B the load-width on the L2-BW-bound MoE GEMM. */
static int   g_moe_tbk64 = 0;
static int8_t **dq_sg=NULL, **dq_su=NULL, **dq_sd=NULL;   /* shared-expert gate/up/down */
static float  **ds_sg=NULL, **ds_su=NULL, **ds_sd=NULL;
static int8_t **dq_dg=NULL, **dq_du=NULL, **dq_dd=NULL;   /* dense-layer FFN (l<first_k_dense) */
static float  **ds_dg=NULL, **ds_du=NULL, **ds_dd=NULL;
static int8_t *dq_lm=NULL;   /* lm_head [vocab][H] */
static float  *ds_lm=NULL;
static int8_t *b_act_i8=NULL;   /* shared int8 activation scratch [MR x maxK] */
static float  *b_act_sc=NULL;   /* per-row activation scale [MR] */

/* ---------------------------------------------------------------------------
 * Persistent device scratch. The two forward passes used to hipMalloc/hipFree
 * ~14 buffers *per call* (per decode token!). We instead keep one grow-only
 * buffer per logical scratch tensor: sbuf() reallocs only when a bigger prompt
 * arrives, so the decode hot path performs zero allocations. Sizes are a union
 * over both paths (prefill's n_prompt-scaled sizes dominate decode's), so a
 * buffer allocated during prefill is reused unchanged by the following decodes.
 * ------------------------------------------------------------------------- */
enum {
    B_TOKENS, B_XS, B_XB, B_QA, B_COMP, B_QALL, B_KNOPE, B_VALUE,
    B_QABS, B_SCORE, B_CLAT, B_CTX, B_AO, B_MO, B_HB, B_HB2,
    B_EXPERT, B_MOELOG,
    /* MoE expert-batching scratch */
    B_GATEPTR, B_UPPTR, B_DOWNPTR, B_WTSD, B_HBK, B_HB2K, B_DOWNOUT,
    /* chunked/batched-prefill scratch */
    B_KVLIN, B_PLASTX, B_PSEQ, B_PPOS, B_PSLOT, B_PLAST,
    /* MFMA prefill-attention query-tile work list (run18) */
    B_QTQ0, B_QTQS, B_QTQE,
    /* run36 interleaved chunked-prefill (fused decode+prefill) epilogue scratch */
    B_FGIDX, B_FLOGITS, B_FOUT,
    B_NBUF
};
typedef struct { void *ptr; size_t bytes; } DevBuf;
static DevBuf g_scratch[B_NBUF];

static void *sbuf(int i, size_t bytes)
{
    DevBuf *b = &g_scratch[i];
    if (b->bytes < bytes)
    {
        if (b->ptr) HIP_CHECK(hipFree(b->ptr));
        HIP_CHECK(hipMalloc(&b->ptr, bytes));
        b->bytes = bytes;
    }
    return b->ptr;
}

static void free_scratch(void)
{
    for (int i = 0; i < B_NBUF; i++)
    {
        if (g_scratch[i].ptr) HIP_CHECK(hipFree(g_scratch[i].ptr));
        g_scratch[i].ptr = NULL;
        g_scratch[i].bytes = 0;
    }
}

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

/* run29: quantize one bf16 expert [rows][cols] into PACKED int4 [rows][cols/2] +
 * per-row fp32 scale, then free the bf16 source (W4A8 MoE path). cols even. */
static void quant_one_expert_i4(bf16_t *src, int rows, int cols, int8_t **out_q, float **out_s)
{
    HIP_CHECK(hipMalloc((void **)out_q, (size_t)rows * (cols / 2) * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)out_s, (size_t)rows * sizeof(float)));
    quantize_rows_i4_kernel<<<rows, 256>>>(src, *out_q, *out_s, rows, cols);
    HIP_CHECK(hipFree(src));
}

/* run32: quantize one bf16 expert [rows][cols] into PACKED int4 [rows][cols/2] +
 * BLOCK-WISE fp32 scale [rows][cols/G] (G=MOE_W4A8_G), then free the bf16 source. */
static void quant_one_expert_i4_bw(bf16_t *src, int rows, int cols, int8_t **out_q, float **out_s)
{
    int ng = cols / MOE_W4A8_G;
    HIP_CHECK(hipMalloc((void **)out_q, (size_t)rows * (cols / 2) * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)out_s, (size_t)rows * ng * sizeof(float)));
    quantize_rows_i4_bw_kernel<<<rows, 256>>>(src, *out_q, *out_s, rows, cols, MOE_W4A8_G);
    HIP_CHECK(hipFree(src));
}

/* run20: quantize a bf16 weight [rows][cols] to int8 + per-row scale but KEEP the
 * bf16 (the W8A8 path is opt-out via MLA_DENSE_W8A8=0, and lm_head still needs bf16). */
static void quant_keep(const bf16_t *src, int rows, int cols, int8_t **out_q, float **out_s)
{
    HIP_CHECK(hipMalloc((void **)out_q, (size_t)rows * cols * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)out_s, (size_t)rows * sizeof(float)));
    quantize_rows_i8_kernel<<<rows, 256>>>(src, *out_q, *out_s, rows, cols);
}

/* run33: block-wise int4 copy [rows][cols/2] + [rows][cols/G] group scale, KEEP
 * bf16 (small-n / tiny-tile calls fall back to the bf16 GEMM). cols % G == 0. */
static void quant_keep_i4_bw(const bf16_t *src, int rows, int cols, int8_t **out_q, float **out_s)
{
    int ng = cols / MOE_W4A8_G;
    HIP_CHECK(hipMalloc((void **)out_q, (size_t)rows * (cols / 2) * sizeof(int8_t)));
    HIP_CHECK(hipMalloc((void **)out_s, (size_t)rows * ng * sizeof(float)));
    quantize_rows_i4_bw_kernel<<<rows, 256>>>(src, *out_q, *out_s, rows, cols, MOE_W4A8_G);
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
        g_dense_w4a8_bw ? quant_keep_i4_bw : quant_keep;
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

/* run22: int8 copy of lm_head [vocab][H] (+ per-vocab-row scale). Biggest bf16 GEMM. */
static void build_lmhead_int8_table(Transformer *t)
{
    const Config *c = &t->config;
    quant_keep(d_w.lm_head, c->vocab_size, c->hidden_size, &dq_lm, &ds_lm);
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
                g_moe_w4a8_bw ? quant_one_expert_i4_bw :
                g_moe_w4a8    ? quant_one_expert_i4 : quant_one_expert;
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

/* ==== LIFECYCLE: warm_up (upload + allocate) / finish (free) =============== */

/* run30: one tiny forward (prefill 4 tokens + 2 decode steps) at the end of
 * warm_up to JIT the hot kernels, touch their scratch, and ramp GPU clocks before
 * the timed inference. Defined after the forward-pass helpers it calls. */
static void glm_warmup_forward(Transformer *t);

void glm_warm_up(Transformer *t)
{
    device_malloc_model_weights(t);
    build_moe_device_tables(t);
    if (getenv("MLA_MOE_W4A8") && atoi(getenv("MLA_MOE_W4A8")) == 1) g_moe_w4a8 = 1;
    if (getenv("MLA_MOE_W4A8_BW") && atoi(getenv("MLA_MOE_W4A8_BW")) == 1) g_moe_w4a8_bw = 1;
    if (getenv("MLA_MOE_TBK64") && atoi(getenv("MLA_MOE_TBK64")) == 1) g_moe_tbk64 = 1;
    build_moe_int8_tables(t);   /* run4: quantize experts to int8 (or int4 if g_moe_w4a8), free bf16 */
    build_wuk_transpose(t);
    if (getenv("MLA_DENSE_W8A8") && atoi(getenv("MLA_DENSE_W8A8")) == 0) g_dense_w8a8 = 0;
    if (getenv("MLA_DENSE_T128") && atoi(getenv("MLA_DENSE_T128")) == 0) g_dense_t128 = 0;
    if (getenv("MLA_DENSE_W4A8_BW") && atoi(getenv("MLA_DENSE_W4A8_BW")) == 1) g_dense_w4a8_bw = 1;
    if (getenv("MLA_FFN_W8A8") && atoi(getenv("MLA_FFN_W8A8")) == 0) g_ffn_w8a8 = 0;
    if (getenv("MLA_LMHEAD_W8A8") && atoi(getenv("MLA_LMHEAD_W8A8")) != 0) g_lmhead_w8a8 = 1;
    if (g_dense_w8a8) build_dense_int8_tables(t);   /* run20: int8 dense projections */
    if (g_ffn_w8a8)    build_ffn_int8_tables(t);    /* run22: int8 shared/dense FFN */
    if (g_lmhead_w8a8) build_lmhead_int8_table(t);  /* run22: int8 lm_head */

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

    /* Chunked-prefill controls. MLA_PFBATCH=0 -> per-request prefill (old path).
     * MLA_PFCHUNK sets the token budget per batched-prefill forward; it also sizes
     * the shared MoE scratch (MR), so read it before allocating below. */
    const char *env_pfb = getenv("MLA_PFBATCH");
    if (env_pfb) g_pfbatch = atoi(env_pfb);
    const char *env_pfc = getenv("MLA_PFCHUNK");
    if (env_pfc) { int v = atoi(env_pfc); if (v > 0) g_pfchunk = v; }
    /* run36 interleaved chunked prefill (fused decode+prefill). */
    const char *env_cf = getenv("MLA_CHUNKFUSE");
    if (env_cf) g_chunkfuse = atoi(env_cf);
    const char *env_fc = getenv("MLA_FUSECHUNK");
    if (env_fc) { int v = atoi(env_fc); if (v > 0) g_fusechunk = v; }

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
    if (g_pfbatch && g_pfchunk > MR) MR = g_pfchunk;
    /* run36: a fused forward processes M = g_batch decode rows + g_fusechunk prefill
     * rows at once, so the shared MoE scratch must hold that union. */
    if (g_chunkfuse && g_batch + g_fusechunk > MR) MR = g_batch + g_fusechunk;
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

    glm_warmup_forward(t);   /* run30: prefill 4 + 2 decode steps to warm kernels */
}

void glm_finish(Transformer *t)
{
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

/* ---------------------------------------------------------------------------
 * Host-side launch helpers -- one kernel-launch-sequence per CPU reference op.
 * ------------------------------------------------------------------------- */

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static void launch_rmsnorm(float *y, const float *x, const bf16_t *w, int n_rows,
                           int n, float eps, int x_stride = -1, int y_stride = -1)
{
    if (x_stride < 0) x_stride = n;
    if (y_stride < 0) y_stride = n;
    rmsnorm_kernel<<<n_rows, 256>>>(y, x, w, n, eps, x_stride, y_stride);
}

/* ==== KERNEL LAUNCH WRAPPERS (shared; model differences flow through Config) = */

/* Y = X @ W.T, X is [n_rows, n_in], W is [d_out, n_in] (nn.Linear). For M>=4 the
 * tiled GEMM reuses each weight tile across the whole M batch (the batched-decode
 * / prefill win); for M<4 (single-row tails) the GEMV-per-row kernel is faster. */
static void launch_matmul(float *Y, const float *X, const bf16_t *W, int n_rows,
                          int d_out, int n_in, int x_stride = -1, int y_stride = -1)
{
    if (x_stride < 0) x_stride = n_in;
    if (y_stride < 0) y_stride = d_out;
    int b128 = ceil_div(d_out, 128) * ceil_div(n_rows, 128);
    if (g_dense_t128 && n_rows >= 128 && d_out >= 128 && b128 >= 96)
    {
        /* run21: coarse 128x128 tile halves the bf16 weight bandwidth for the large
         * bandwidth-bound GEMMs (lm_head/shared-expert/dense-layer); bit-identical. */
        dim3 grid(ceil_div(d_out, 128), ceil_div(n_rows, 128));
        matmul_mfma_128_kernel<<<grid, 256>>>(Y, X, W, n_rows, d_out, n_in, x_stride, y_stride);
    }
    else if (n_rows >= 4)
    {
        dim3 grid(ceil_div(d_out, MM_TILE), ceil_div(n_rows, MM_TILE));
        matmul_mfma_kernel<<<grid, 256>>>(Y, X, W, n_rows, d_out, n_in, x_stride, y_stride);
    }
    else
    {
        dim3 grid(d_out, n_rows);
        matmul_kernel<<<grid, 256>>>(Y, X, W, d_out, n_in, x_stride, y_stride);
    }
}

/* run20: W8A8 dense matmul. Routes to the int8 GEMM when g_dense_w8a8 && the int8
 * weight exists && n_rows >= 4 (MFMA path); else falls back to bf16 launch_matmul.
 * Quantizes the fp32 activation X (K = n_in, stride x_stride) into b_act_i8 first. */
static void launch_matmul_w8a8(float *Y, const float *X, const bf16_t *Wbf,
                               const int8_t *Wq, const float *Ws, int n_rows, int d_out,
                               int n_in, int x_stride = -1, int y_stride = -1, int bw = 0)
{
    if (x_stride < 0) x_stride = n_in;
    if (y_stride < 0) y_stride = d_out;
    /* Route on int8-table presence (Wq): each caller's table is built only when its
     * own W8A8 toggle is on, so a NULL Wq already means "use bf16" -- no need to
     * re-check g_dense_w8a8 here (build_dense_int8_tables is itself gated on it). */
    if (Wq && n_rows >= 4)
    {
        quant_act_rows_i8_kernel<<<n_rows, 256>>>(X, b_act_i8, b_act_sc, n_rows, n_in, x_stride);
        /* run33: block-wise int4 dense GEMM. Wq is packed int4, Ws is [d_out][n_in/128];
         * always the 128-tile kernel (the only one that unpacks int4). n_in % 128 == 0. */
        if (bw && g_dense_w4a8_bw)
        {
            dim3 grid(ceil_div(d_out, 128), ceil_div(n_rows, 128));
            matmul_mfma_i8i8_bw_128_kernel<<<grid, 256>>>(Y, b_act_i8, b_act_sc, Wq, Ws,
                                                          n_rows, d_out, n_in, n_in, y_stride);
            return;
        }
        /* run21: use the 128x128-tile kernel (half the L2 traffic) when its coarser
         * grid still fills the 104 CUs; otherwise the 64x64 kernel keeps occupancy. */
        int b128 = ceil_div(d_out, 128) * ceil_div(n_rows, 128);
        if (g_dense_t128 && n_rows >= 128 && d_out >= 128 && b128 >= 96)
        {
            dim3 grid(ceil_div(d_out, 128), ceil_div(n_rows, 128));
            matmul_mfma_i8i8_128_kernel<<<grid, 256>>>(Y, b_act_i8, b_act_sc, Wq, Ws,
                                                       n_rows, d_out, n_in, n_in, y_stride);
        }
        else
        {
            dim3 grid(ceil_div(d_out, MM_TILE), ceil_div(n_rows, MM_TILE));
            matmul_mfma_i8i8_kernel<<<grid, 256>>>(Y, b_act_i8, b_act_sc, Wq, Ws,
                                                   n_rows, d_out, n_in, n_in, y_stride);
        }
    }
    else
    {
        launch_matmul(Y, X, Wbf, n_rows, d_out, n_in, x_stride, y_stride);
    }
}

static void launch_rope(float *v, int n_rows, int n_heads, int row_stride,
                        int head_stride, int pos_base, int rope_dim, int interleaved)
{
    dim3 grid(n_rows, n_heads);
    int half = rope_dim / 2;
    size_t shmem = interleaved ? (size_t)rope_dim * sizeof(float) : 0;
    rope_kernel<<<grid, half, shmem>>>(v, n_heads, row_stride, head_stride, pos_base,
                                       rope_dim, interleaved, d_inv_freq);
}

/* W8A8 SwiGLU FFN (run22). gate/up/down go through the int8 GEMM when their int8
 * table exists (NULL -> bf16 fallback per projection, via launch_matmul_w8a8).
 * The silu*mul intermediate is re-quantized per-row inside launch_matmul_w8a8. */
static void launch_swiglu_w8a8(float *dst, const float *x,
                               const bf16_t *gate, const int8_t *gq, const float *gs,
                               const bf16_t *up,   const int8_t *uq, const float *us,
                               const bf16_t *down, const int8_t *dq, const float *ds,
                               int n_rows, int inter, int hidden, float *hb, float *hb2)
{
    launch_matmul_w8a8(hb,  x, gate, gq, gs, n_rows, inter, hidden);
    launch_matmul_w8a8(hb2, x, up,   uq, us, n_rows, inter, hidden);
    int total = n_rows * inter;
    silu_mul_kernel<<<ceil_div(total, 256), 256>>>(hb, hb2, total);
    launch_matmul_w8a8(dst, hb, down, dq, ds, n_rows, hidden, inter);
}

static void launch_project_q(const Config *c, int l, const float *xn, float *q,
                             int n_rows, float *qa_scratch)
{
    int q_dim = c->n_heads * (c->qk_nope_head_dim + c->qk_rope_head_dim);
    if (c->q_lora_rank > 0)
    {
        launch_matmul_w8a8(qa_scratch, xn, d_w.q_a_proj[l], dq_qa ? dq_qa[l] : NULL,
                           dq_qa ? ds_qa[l] : NULL, n_rows, c->q_lora_rank, c->hidden_size, -1, -1, 1);
        launch_rmsnorm(qa_scratch, qa_scratch, d_w.q_a_layernorm[l], n_rows,
                       c->q_lora_rank, c->mla_norm_eps);
        launch_matmul_w8a8(q, qa_scratch, d_w.q_b_proj[l], dq_qb ? dq_qb[l] : NULL,
                           dq_qb ? ds_qb[l] : NULL, n_rows, q_dim, c->q_lora_rank, -1, -1, 1);
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
                        int n_rows, float *hb, float *hb2)
{
    const int H = c->hidden_size;
    if (l < c->first_k_dense)
    {
        launch_swiglu_w8a8(out, xn,
            d_w.dense_gate[l], dq_dg?dq_dg[l]:NULL, dq_dg?ds_dg[l]:NULL,
            d_w.dense_up[l],   dq_du?dq_du[l]:NULL, dq_du?ds_du[l]:NULL,
            d_w.dense_down[l], dq_dd?dq_dd[l]:NULL, dq_dd?ds_dd[l]:NULL,
            n_rows, c->dense_inter_size, H, hb, hb2);
        return;
    }
    const int NR = c->n_routed_experts, K = c->n_experts_per_tok, I = c->moe_inter_size;
    const int T = n_rows * K;

    launch_matmul(b_moelog, xn, d_w.moe_gate[l], n_rows, NR, H);
    moe_route_batched_kernel<<<n_rows, 256>>>(b_moelog, NR, K, c->router_sigmoid,
                                             c->norm_topk, c->routed_scaling, d_moe_bias[l],
                                             b_topk_idx, b_topk_wt);
    HIP_CHECK(hipMemsetAsync(b_ecount, 0, (size_t)NR * sizeof(int)));
    moe_hist_kernel<<<ceil_div(T, 256), 256>>>(b_topk_idx, T, b_ecount);
    moe_scan_kernel<<<1, 1>>>(b_ecount, NR, b_eoff);
    HIP_CHECK(hipMemsetAsync(b_cursor, 0, (size_t)NR * sizeof(int)));
    moe_place_kernel<<<ceil_div(T, 256), 256>>>(b_topk_idx, b_topk_wt, T, K, b_eoff,
                                               b_cursor, b_sorted_slot, b_sorted_wt);
    /* NB: a fused gate+up GEMM (one A-tile load, two B-tiles) was tried but is
     * SLOWER -- doubling the per-block accumulators/LDS cuts occupancy more than
     * the shared-X reuse (already cheap via L2/LDS) saves. See report/run11. */
    /* run19: W8A8 (int8 activation + int8 weight, mfma_i32_16x16x16i8) removes the
     * dequant-VALU bottleneck of the weight-only-int8 GEMM. MLA_MOE_W8A8=0 reverts
     * to the bf16-MFMA weight-only path for A/B. Shared expert stays bf16. */
    static const int w8a8 = !(getenv("MLA_MOE_W8A8") && atoi(getenv("MLA_MOE_W8A8")) == 0);
    /* run24: 128x128-tile grouped MoE GEMM halves the L2 activation re-reads of the
     * 64x64 kernel (profile @ B=1024: moe_grouped_mfma_i8i8 L2-BW-bound, 133% peak).
     * The tile is bit-identical; the coarse grid still floods the CUs (NR=64 experts
     * x ceil(N/128) col-tiles). MLA_MOE_T128=0 reverts to the 64x64 kernel for A/B. */
    static const int t128 = !(getenv("MLA_MOE_T128") && atoi(getenv("MLA_MOE_T128")) == 0);
    dim3 gGU(ceil_div(I, MM_TILE), ceil_div(n_rows, MM_TILE), NR);
    dim3 gD(ceil_div(H, MM_TILE), ceil_div(n_rows, MM_TILE), NR);
    if (w8a8)
    {
        quant_act_rows_i8_kernel<<<n_rows, 256>>>(xn, b_xn_i8, b_xn_scale, n_rows, H, H);
        if (g_moe_w4a8 || g_moe_w4a8_bw)
        {
            /* run29/32: int4 packed weight -> must use a W4A8 kernel (128-tile).
             * g_moe_w4a8_bw = block-wise scale (run32); else per-row (run29). */
            dim3 gGU2(ceil_div(I, 128), ceil_div(n_rows, 128), NR);
            dim3 gD2(ceil_div(H, 128), ceil_div(n_rows, 128), NR);
            if (g_moe_w4a8_bw)
            {
                moe_grouped_mfma_w4a8_bw_128_kernel<<<gGU2, 256>>>(b_hbk, b_xn_i8, b_xn_scale, d_epq_gate[l], d_eps_gate[l], b_eoff, b_sorted_slot, I, H, H, 1);
                moe_grouped_mfma_w4a8_bw_128_kernel<<<gGU2, 256>>>(b_hb2k, b_xn_i8, b_xn_scale, d_epq_up[l], d_eps_up[l], b_eoff, b_sorted_slot, I, H, H, 1);
                silu_mul_kernel<<<ceil_div(T * I, 256), 256>>>(b_hbk, b_hb2k, T * I);
                quant_act_rows_i8_kernel<<<T, 256>>>(b_hbk, b_hbk_i8, b_hbk_scale, T, I, I);
                moe_grouped_mfma_w4a8_bw_128_kernel<<<gD2, 256>>>(b_downk, b_hbk_i8, b_hbk_scale, d_epq_down[l], d_eps_down[l], b_eoff, b_sorted_slot, H, I, I, 0);
            }
            else
            {
                moe_grouped_mfma_w4a8_128_kernel<<<gGU2, 256>>>(b_hbk, b_xn_i8, b_xn_scale, d_epq_gate[l], d_eps_gate[l], b_eoff, b_sorted_slot, I, H, H, 1);
                moe_grouped_mfma_w4a8_128_kernel<<<gGU2, 256>>>(b_hb2k, b_xn_i8, b_xn_scale, d_epq_up[l], d_eps_up[l], b_eoff, b_sorted_slot, I, H, H, 1);
                silu_mul_kernel<<<ceil_div(T * I, 256), 256>>>(b_hbk, b_hb2k, T * I);
                quant_act_rows_i8_kernel<<<T, 256>>>(b_hbk, b_hbk_i8, b_hbk_scale, T, I, I);
                moe_grouped_mfma_w4a8_128_kernel<<<gD2, 256>>>(b_downk, b_hbk_i8, b_hbk_scale, d_epq_down[l], d_eps_down[l], b_eoff, b_sorted_slot, H, I, I, 0);
            }
        }
        else if (t128)
        {
            dim3 gGU2(ceil_div(I, 128), ceil_div(n_rows, 128), NR);
            dim3 gD2(ceil_div(H, 128), ceil_div(n_rows, 128), NR);
            /* run31: g_moe_tbk64 picks the TBK=64/128-bit-load kernel (bit-identical). */
            if (g_moe_tbk64)
            {
                moe_grouped_mfma_i8i8_t64_kernel<<<gGU2, 256>>>(b_hbk, b_xn_i8, b_xn_scale, d_epq_gate[l], d_eps_gate[l], b_eoff, b_sorted_slot, I, H, H, 1);
                moe_grouped_mfma_i8i8_t64_kernel<<<gGU2, 256>>>(b_hb2k, b_xn_i8, b_xn_scale, d_epq_up[l], d_eps_up[l], b_eoff, b_sorted_slot, I, H, H, 1);
                silu_mul_kernel<<<ceil_div(T * I, 256), 256>>>(b_hbk, b_hb2k, T * I);
                quant_act_rows_i8_kernel<<<T, 256>>>(b_hbk, b_hbk_i8, b_hbk_scale, T, I, I);
                moe_grouped_mfma_i8i8_t64_kernel<<<gD2, 256>>>(b_downk, b_hbk_i8, b_hbk_scale, d_epq_down[l], d_eps_down[l], b_eoff, b_sorted_slot, H, I, I, 0);
            }
            else
            {
                moe_grouped_mfma_i8i8_128_kernel<<<gGU2, 256>>>(b_hbk, b_xn_i8, b_xn_scale, d_epq_gate[l], d_eps_gate[l], b_eoff, b_sorted_slot, I, H, H, 1);
                moe_grouped_mfma_i8i8_128_kernel<<<gGU2, 256>>>(b_hb2k, b_xn_i8, b_xn_scale, d_epq_up[l], d_eps_up[l], b_eoff, b_sorted_slot, I, H, H, 1);
                silu_mul_kernel<<<ceil_div(T * I, 256), 256>>>(b_hbk, b_hb2k, T * I);
                quant_act_rows_i8_kernel<<<T, 256>>>(b_hbk, b_hbk_i8, b_hbk_scale, T, I, I);
                moe_grouped_mfma_i8i8_128_kernel<<<gD2, 256>>>(b_downk, b_hbk_i8, b_hbk_scale, d_epq_down[l], d_eps_down[l], b_eoff, b_sorted_slot, H, I, I, 0);
            }
        }
        else
        {
            moe_grouped_mfma_i8i8_kernel<<<gGU, 256>>>(b_hbk, b_xn_i8, b_xn_scale, d_epq_gate[l], d_eps_gate[l], b_eoff, b_sorted_slot, I, H, H, 1);
            moe_grouped_mfma_i8i8_kernel<<<gGU, 256>>>(b_hb2k, b_xn_i8, b_xn_scale, d_epq_up[l], d_eps_up[l], b_eoff, b_sorted_slot, I, H, H, 1);
            silu_mul_kernel<<<ceil_div(T * I, 256), 256>>>(b_hbk, b_hb2k, T * I);
            quant_act_rows_i8_kernel<<<T, 256>>>(b_hbk, b_hbk_i8, b_hbk_scale, T, I, I);
            moe_grouped_mfma_i8i8_kernel<<<gD, 256>>>(b_downk, b_hbk_i8, b_hbk_scale, d_epq_down[l], d_eps_down[l], b_eoff, b_sorted_slot, H, I, I, 0);
        }
    }
    else
    {
        moe_grouped_mfma_i8_kernel<<<gGU, 256>>>(b_hbk, xn, d_epq_gate[l], d_eps_gate[l], b_eoff, b_sorted_slot, I, H, H, 1);
        moe_grouped_mfma_i8_kernel<<<gGU, 256>>>(b_hb2k, xn, d_epq_up[l], d_eps_up[l], b_eoff, b_sorted_slot, I, H, H, 1);
        silu_mul_kernel<<<ceil_div(T * I, 256), 256>>>(b_hbk, b_hb2k, T * I);
        moe_grouped_mfma_i8_kernel<<<gD, 256>>>(b_downk, b_hbk, d_epq_down[l], d_eps_down[l], b_eoff, b_sorted_slot, H, I, I, 0);
    }
    HIP_CHECK(hipMemsetAsync(out, 0, (size_t)n_rows * H * sizeof(float)));
    dim3 gSc(ceil_div(H, 256), T);
    moe_scatter_add_kernel<<<gSc, 256>>>(out, b_downk, b_sorted_slot, b_sorted_wt, T, H);
    launch_swiglu_w8a8(b_shared, xn,
        d_w.shared_gate[l], dq_sg?dq_sg[l]:NULL, dq_sg?ds_sg[l]:NULL,
        d_w.shared_up[l],   dq_su?dq_su[l]:NULL, dq_su?ds_su[l]:NULL,
        d_w.shared_down[l], dq_sd?dq_sd[l]:NULL, dq_sd?ds_sd[l]:NULL,
        n_rows, c->n_shared_experts * c->moe_inter_size, H, hb, hb2);
    weighted_add_kernel<<<ceil_div(n_rows * H, 256), 256>>>(out, b_shared, 1.0f, n_rows * H);
}

/* ==== FORWARD PASSES (shared by both models, config-driven) ================
 * GPU counterparts of forward_unabsorbed / forward_absorbed in src/run.c. Model
 * differences enter only through Config: launch_project_q picks the q LoRA vs
 * single-proj path (q_lora_rank), and ffn_batched's router honours router_sigmoid
 * / e_score bias / routed_scaling. Both return a HOST pointer (h_logits, valid
 * until the next forward call), so sample() can read it directly. ============ */

static int device_forward_unabsorbed(Transformer *t, const int *tokens, int n_prompt,
                                     int slot)
{
    const Config *c = &t->config;
    const int H = c->hidden_size, NH = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim, QHD = QKN + QKR;
    const int VHD = c->v_head_dim, KVL = c->kv_lora_rank, KVD = KVL + QKR, VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const int kv_stride = (QKN + VHD) * KVL;
    /* Per-slot cache: layer stride spans all B slots; this request's rows start
     * at slot_off within its layer (positions 0..n_prompt-1). */
    const size_t layer_stride = (size_t)g_batch * g_cap * KVD;
    const size_t slot_off = (size_t)slot * g_cap * KVD;
    const int inter_max = inter_max_of(c);

    int *d_tokens = (int *)sbuf(B_TOKENS, (size_t)n_prompt * sizeof(int));
    HIP_CHECK(hipMemcpy(d_tokens, tokens, (size_t)n_prompt * sizeof(int), hipMemcpyHostToDevice));

    float *qa_scratch = NULL;
    float *xs   = (float *)sbuf(B_XS,   (size_t)n_prompt * H * sizeof(float));
    float *xb   = (float *)sbuf(B_XB,   (size_t)n_prompt * H * sizeof(float));
    if (c->q_lora_rank > 0)
        qa_scratch = (float *)sbuf(B_QA, (size_t)n_prompt * c->q_lora_rank * sizeof(float));
    float *comp  = (float *)sbuf(B_COMP,  (size_t)n_prompt * KVD * sizeof(float));
    float *qall  = (float *)sbuf(B_QALL,  (size_t)n_prompt * NH * QHD * sizeof(float));
    float *knope = (float *)sbuf(B_KNOPE, (size_t)NH * n_prompt * QKN * sizeof(float));
    float *value = (float *)sbuf(B_VALUE, (size_t)NH * n_prompt * VHD * sizeof(float));
    float *scr   = (float *)sbuf(B_SCORE, (size_t)NH * n_prompt * n_prompt * sizeof(float));
    float *ctx   = (float *)sbuf(B_CTX,   (size_t)n_prompt * NH * VHD * sizeof(float));
    float *ao    = (float *)sbuf(B_AO,    (size_t)n_prompt * H * sizeof(float));
    float *mo    = (float *)sbuf(B_MO,    (size_t)n_prompt * H * sizeof(float));
    float *hb    = (float *)sbuf(B_HB,    (size_t)n_prompt * inter_max * sizeof(float));
    float *hb2   = (float *)sbuf(B_HB2,   (size_t)n_prompt * inter_max * sizeof(float));
    /* fp32 latent-KV scratch: the cache is now bf16, so build the latent KV here
     * in fp32 (used by the UK/UV matmuls + flash attn) then narrow-copy into the
     * bf16 cache. Mirrors the chunked path's kvlin+kv_scatter. */
    float *kvln  = (float *)sbuf(B_KVLIN, (size_t)n_prompt * KVD * sizeof(float));

    embed_lookup_kernel<<<n_prompt, 256>>>(xs, d_w.embed_tokens, d_tokens, H);

    for (int l = 0; l < c->n_layers; l++)
    {
        int8_t *kv_cache_row = d_kv_cache + (size_t)l * layer_stride + slot_off;
        float  *kv_scale_row = d_kv_scale + (size_t)l * g_batch * g_cap + (size_t)slot * g_cap;

        launch_rmsnorm(xb, xs, d_w.input_layernorm[l], n_prompt, H, eps);
        launch_project_q(c, l, xb, qall, n_prompt, qa_scratch);
        launch_rope(qall + QKN, n_prompt, NH, NH * QHD, QHD, 0, QKR, c->rope_interleaved);

        launch_matmul_w8a8(comp, xb, d_w.kv_a_proj[l], dq_kva?dq_kva[l]:NULL, dq_kva?ds_kva[l]:NULL, n_prompt, KVD, H, -1, -1, 1);
        launch_rmsnorm(kvln, comp, d_w.kv_a_layernorm[l], n_prompt, KVL, c->mla_norm_eps, KVD, KVD);
        copy_strided_kernel<<<n_prompt, 256>>>(kvln + KVL, comp + KVL, QKR, KVD, KVD);
        launch_rope(kvln + KVL, n_prompt, 1, KVD, 0, 0, QKR, c->rope_interleaved);
        /* Persist this request's prompt latent KV (positions 0..n_prompt-1 are a
         * contiguous slot region) into the int8 cache (per-row scale). */
        quant_kv_rows_kernel<<<n_prompt, 256>>>(kv_cache_row, kv_scale_row, kvln, KVD);

        for (int h = 0; h < NH; h++)
        {
            const bf16_t *WUK = d_w.W_UK[l] + (size_t)h * kv_stride;
            const bf16_t *WUV = d_w.W_UV[l] + (size_t)h * kv_stride;
            launch_matmul(knope + (size_t)h * n_prompt * QKN, kvln, WUK, n_prompt, QKN, KVL, KVD, QKN);
            launch_matmul(value + (size_t)h * n_prompt * VHD, kvln, WUV, n_prompt, VHD, KVL, KVD, VHD);
        }

        {
            /* Fused flash-attention: score+softmax+ctx in one pass (no O(n^2)
             * HBM scratch). Replaces attn_unabsorbed_score + softmax_row +
             * attn_unabsorbed_ctx. (void)scr keeps the buffer allocated but unused. */
            (void)scr;
            dim3 grid(n_prompt, NH);
            attn_unabsorbed_flash_kernel<<<grid, 256>>>(ctx, qall, knope, kvln, value,
                                                        n_prompt, NH, QHD, QKN, QKR,
                                                        KVD, VHD, scale);
        }
        launch_matmul_w8a8(ao, ctx, d_w.o_proj[l], dq_op?dq_op[l]:NULL, dq_op?ds_op[l]:NULL, n_prompt, H, NH * VHD, -1, -1, 1);
        residual_add_kernel<<<ceil_div(n_prompt * H, 256), 256>>>(xs, ao, n_prompt * H);

        launch_rmsnorm(xb, xs, d_w.post_attn_norm[l], n_prompt, H, eps);
        ffn_batched(c, l, xb, mo, n_prompt, hb, hb2);
        residual_add_kernel<<<ceil_div(n_prompt * H, 256), 256>>>(xs, mo, n_prompt * H);
    }

    launch_rmsnorm(xb, xs + (size_t)(n_prompt - 1) * H, d_w.norm, 1, H, eps);
    launch_matmul(d_logits, xb, d_w.lm_head, 1, VOC, H);
    argmax_kernel<<<1, 256>>>(d_logits, VOC, d_argmax);
    int tok;
    HIP_CHECK(hipMemcpy(&tok, d_argmax, sizeof(int), hipMemcpyDeviceToHost));

    return tok;
}

/* Chunked/batched prefill: prefill nseq whole prompts packed on one flat token
 * axis of length T in a SINGLE forward pass, so every layer's weights are read
 * once for the whole chunk instead of once per request. Everything but attention
 * is row-parallel over the T tokens (dense GEMMs, MoE, FFN); attention uses the
 * varlen flash kernel (each query attends only within its own prompt via the
 * per-token q_seqstart). The latent KV is built contiguously (kvlin) and scattered
 * into each prompt's slot cache so the absorbed decode path can read it. On return
 * out_tok[i] holds prompt i's first (greedy) token.
 *
 * Host inputs (all length T except last[nseq]): tokcat = concatenated prompt token
 * ids; seqstart[t] = global index of the first token of t's prompt; pos[t] = t's
 * position within its prompt (0-based, for RoPE + cache write); slot[t] = t's
 * prompt's decode slot; last[i] = global index of prompt i's final token. */
static void device_forward_prefill_batched(Transformer *t, int T, int nseq,
                                           const int *tokcat, const int *seqstart,
                                           const int *pos, const int *slot,
                                           const int *last, int *out_tok)
{
    const Config *c = &t->config;
    const int H = c->hidden_size, NH = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim, QHD = QKN + QKR;
    const int VHD = c->v_head_dim, KVL = c->kv_lora_rank, KVD = KVL + QKR, VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const int kv_stride = (QKN + VHD) * KVL;
    const size_t layer_stride = (size_t)g_batch * g_cap * KVD;
    const int slot_stride = g_cap * KVD;
    const int inter_max = inter_max_of(c);

    /* Upload the per-token index arrays. */
    int *d_tok = (int *)sbuf(B_TOKENS, (size_t)T * sizeof(int));
    int *d_seq = (int *)sbuf(B_PSEQ,   (size_t)T * sizeof(int));
    int *d_pos = (int *)sbuf(B_PPOS,   (size_t)T * sizeof(int));
    int *d_slt = (int *)sbuf(B_PSLOT,  (size_t)T * sizeof(int));
    int *d_lst = (int *)sbuf(B_PLAST,  (size_t)nseq * sizeof(int));
    HIP_CHECK(hipMemcpy(d_tok, tokcat,   (size_t)T * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seq, seqstart, (size_t)T * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pos, pos,      (size_t)T * sizeof(int), hipMemcpyHostToDevice));
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
    HIP_CHECK(hipMemcpy(d_qtq0, h_qtq0, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qtqs, h_qtqs, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qtqe, h_qtqe, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_slt, slot,     (size_t)T * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_lst, last,     (size_t)nseq * sizeof(int), hipMemcpyHostToDevice));

    float *qa    = c->q_lora_rank > 0 ? (float *)sbuf(B_QA, (size_t)T * c->q_lora_rank * sizeof(float)) : NULL;
    float *xs    = (float *)sbuf(B_XS,    (size_t)T * H * sizeof(float));
    float *xb    = (float *)sbuf(B_XB,    (size_t)T * H * sizeof(float));
    float *comp  = (float *)sbuf(B_COMP,  (size_t)T * KVD * sizeof(float));
    float *qall  = (float *)sbuf(B_QALL,  (size_t)T * NH * QHD * sizeof(float));
    float *kvlin = (float *)sbuf(B_KVLIN, (size_t)T * KVD * sizeof(float));
    float *knope = (float *)sbuf(B_KNOPE, (size_t)NH * T * QKN * sizeof(float));
    float *value = (float *)sbuf(B_VALUE, (size_t)NH * T * VHD * sizeof(float));
    float *ctx   = (float *)sbuf(B_CTX,   (size_t)T * NH * VHD * sizeof(float));
    float *ao    = (float *)sbuf(B_AO,    (size_t)T * H * sizeof(float));
    float *mo    = (float *)sbuf(B_MO,    (size_t)T * H * sizeof(float));
    float *hb    = (float *)sbuf(B_HB,    (size_t)T * inter_max * sizeof(float));
    float *hb2   = (float *)sbuf(B_HB2,   (size_t)T * inter_max * sizeof(float));
    float *lastx = (float *)sbuf(B_PLASTX,(size_t)nseq * H * sizeof(float));

    const size_t rope_sh = c->rope_interleaved ? (size_t)QKR * sizeof(float) : 0;

    embed_lookup_kernel<<<T, 256>>>(xs, d_w.embed_tokens, d_tok, H);

    for (int l = 0; l < c->n_layers; l++)
    {
        int8_t *cache_base = d_kv_cache + (size_t)l * layer_stride;
        float  *scale_base = d_kv_scale + (size_t)l * g_batch * g_cap;

        launch_rmsnorm(xb, xs, d_w.input_layernorm[l], T, H, eps);
        launch_project_q(c, l, xb, qall, T, qa);
        rope_pos_kernel<<<dim3(T, NH), QKR / 2, rope_sh>>>(qall + QKN, NH, NH * QHD, QHD,
                                                           d_pos, QKR, c->rope_interleaved, d_inv_freq);

        launch_matmul_w8a8(comp, xb, d_w.kv_a_proj[l], dq_kva?dq_kva[l]:NULL, dq_kva?ds_kva[l]:NULL, T, KVD, H, -1, -1, 1);
        /* Build the contiguous latent KV (rmsnorm the compressed part, copy + RoPE
         * the k_pe slice at each token's own position), then scatter into slots. */
        launch_rmsnorm(kvlin, comp, d_w.kv_a_layernorm[l], T, KVL, c->mla_norm_eps, KVD, KVD);
        copy_strided_kernel<<<T, 256>>>(kvlin + KVL, comp + KVL, QKR, KVD, KVD);
        rope_pos_kernel<<<dim3(T, 1), QKR / 2, rope_sh>>>(kvlin + KVL, 1, KVD, 0,
                                                          d_pos, QKR, c->rope_interleaved, d_inv_freq);
        kv_scatter_kernel<<<T, 256>>>(cache_base, scale_base, slot_stride, d_slt, d_pos, kvlin, KVD, T);

        /* Per-head knope[h] = kvlin @ W_UK[h]^T and value[h] = kvlin @ W_UV[h]^T
         * (kvlin shared across heads) folded into 2 head-batched MFMA GEMMs. */
        {
            dim3 gK(ceil_div(QKN, MM_TILE), ceil_div(T, MM_TILE), NH);
            matmul_headw_mfma_kernel<<<gK, 256>>>(knope, kvlin, d_w.W_UK[l], T, QKN, KVL,
                                                  KVD, kv_stride, T * QKN);
            dim3 gV(ceil_div(VHD, MM_TILE), ceil_div(T, MM_TILE), NH);
            matmul_headw_mfma_kernel<<<gV, 256>>>(value, kvlin, d_w.W_UV[l], T, VHD, KVL,
                                                  KVD, kv_stride, T * VHD);
        }

        /* run18: MFMA flash-attention (matrix cores) replaces the scalar varlen
         * kernel. MLA_PFSCALAR=1 falls back to the old path for A/B comparison. */
        static const int use_scalar_attn = (getenv("MLA_PFSCALAR") &&
                                            atoi(getenv("MLA_PFSCALAR")) != 0);
        if (use_scalar_attn)
            attn_unabsorbed_flash_varlen_kernel<<<dim3(T, NH), 256>>>(ctx, qall, knope, kvlin, value,
                                                                      d_seq, T, NH, QHD, QKN, QKR,
                                                                      KVD, VHD, scale);
        else
            attn_prefill_mfma_kernel<<<dim3(n_qtile, NH), 256>>>(ctx, qall, knope, kvlin, value,
                                                                 d_qtq0, d_qtqs, d_qtqe, T, NH,
                                                                 QHD, QKN, QKR, KVD, KVL, VHD, scale);
        launch_matmul_w8a8(ao, ctx, d_w.o_proj[l], dq_op?dq_op[l]:NULL, dq_op?ds_op[l]:NULL, T, H, NH * VHD, -1, -1, 1);
        residual_add_kernel<<<ceil_div(T * H, 256), 256>>>(xs, ao, T * H);

        launch_rmsnorm(xb, xs, d_w.post_attn_norm[l], T, H, eps);
        ffn_batched(c, l, xb, mo, T, hb, hb2);
        residual_add_kernel<<<ceil_div(T * H, 256), 256>>>(xs, mo, T * H);
    }

    /* First token of each prompt = argmax of its last position's logits. Gather
     * those rows, norm, lm_head (batched over nseq), argmax. */
    gather_rows_kernel<<<nseq, 256>>>(lastx, xs, d_lst, H);
    launch_rmsnorm(xb, lastx, d_w.norm, nseq, H, eps);
    launch_matmul_w8a8(b_logits, xb, d_w.lm_head, dq_lm, ds_lm, nseq, VOC, H);
    argmax_batched_kernel<<<nseq, 256>>>(b_logits, VOC, VOC, b_tokens);
    HIP_CHECK(hipMemcpy(out_tok, b_tokens, (size_t)nseq * sizeof(int), hipMemcpyDeviceToHost));
}

/* Batched absorbed decode: advance B slots by one token each. b_tokens holds the
 * per-slot input token; b_pos the position at which to write its KV latent. On
 * return b_tokens holds each slot's next greedy token. kv_max = max(pos)+1 over
 * the B slots (host-tracked) bounds the attention/softmax width. Every dense
 * matmul now reuses each weight tile across the whole B batch (matmul_tiled), and
 * the MoE experts are gathered/grouped so each expert's weights are read once. */
static void device_forward_absorbed_batched(Transformer *t, int B, int kv_max)
{
    const Config *c = &t->config;
    const int H = c->hidden_size, NH = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim, QHD = QKN + QKR;
    const int VHD = c->v_head_dim, KVL = c->kv_lora_rank, KVD = KVL + QKR, VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const int kv_stride = (QKN + VHD) * KVL;
    const size_t layer_stride = (size_t)g_batch * g_cap * KVD;
    const int slot_stride = g_cap * KVD;

    embed_lookup_kernel<<<B, 256>>>(b_x, d_w.embed_tokens, b_tokens, H);

    for (int l = 0; l < c->n_layers; l++)
    {
        int8_t *kv_base = d_kv_cache + (size_t)l * layer_stride;
        float  *kv_scale_base = d_kv_scale + (size_t)l * g_batch * g_cap;

        launch_rmsnorm(b_xb, b_x, d_w.input_layernorm[l], B, H, eps);
        launch_project_q(c, l, b_xb, b_q, B, b_qa);
        {
            dim3 grid(B, NH);
            size_t shmem = c->rope_interleaved ? (size_t)QKR * sizeof(float) : 0;
            rope_pos_kernel<<<grid, QKR / 2, shmem>>>(b_q + QKN, NH, NH * QHD, QHD,
                                                      b_pos, QKR, c->rope_interleaved, d_inv_freq);
        }
        launch_matmul_w8a8(b_comp, b_xb, d_w.kv_a_proj[l], dq_kva?dq_kva[l]:NULL, dq_kva?ds_kva[l]:NULL, B, KVD, H, -1, -1, 1);
        kv_write_batched_kernel<<<B, 256, (size_t)KVD * sizeof(float)>>>(
                                            kv_base, kv_scale_base, slot_stride, b_pos, b_comp,
                                            d_w.kv_a_layernorm[l], KVL, QKR, KVD,
                                            c->mla_norm_eps, d_inv_freq, c->rope_interleaved);

        /* qabs[s,h] = q_nope[s,h] @ W_UK[h]^T  (W_UK pre-transposed to [KVL,QKN]). */
        {
            dim3 grid(ceil_div(KVL, MM_TILE), ceil_div(B, MM_TILE), NH);
            matmul_bh_mfma_kernel<<<grid, 256>>>(b_qabs, b_q, d_wuk_t[l], B, KVL, QKN,
                                                 QHD, NH * QHD, KVL * QKN, KVL, NH * KVL);
        }
        /* Absorbed attention over each slot's own KV cache (masked to kv_len[s]):
         * one fused flash-decode pass per (slot,head) -> b_clat (no b_score HBM).
         * NB: a head-fused variant (1 block/slot, ckv read once for all heads) was
         * tried but is SLOWER -- grid=(NH,B) already schedules a slot's heads as
         * consecutive blocks so the "redundant" ckv reads hit L2, and fusing trades
         * that cheap reuse for worse thread utilization. See report/run8. */
        {
            (void)kv_max;
            /* run23: head-batched flash-decode (grid=(B,HG)) stages c_kv into LDS
             * once per slot per head group -> relieves the L2-read bound at B=1024.
             * A/B via MLA_FDHB=0 (revert to per-(head,slot) attn_flash_decode). */
            static const int fdhb = !(getenv("MLA_FDHB") && atoi(getenv("MLA_FDHB")) == 0);
            /* run27: MFMA-score head-batched decode (offloads the KVD dot to the
             * matrix cores). NEGATIVE at B=1024 (-2.4%: kernel is latency/occupancy-
             * bound, not compute-bound -- VALU only 43.6%; tk=32 -> 2 of 4 warps idle
             * during the MFMA, M=16 pads HPB=10). Kept opt-in (MLA_FDMFMA=1) for
             * reference; the scalar-score hb kernel stays default. See report/run27. */
            static const int fdmfma = (getenv("MLA_FDMFMA") && atoi(getenv("MLA_FDMFMA")) == 1);
            /* run38: ILP score/clat reductions (breaks the KVD=576 serial FMA chain).
             * NEGATIVE at B=1280 (-2.6%): the kernel is LDS-OCCUPANCY-limited (occ 11.5%,
             * "Insufficient CU LDS" 34%), not latency-throughput-bound, so the extra
             * accumulator registers cut resident waves more than the ILP hides latency.
             * Kept opt-in (MLA_FDILP=1) for reference; scalar hb stays default. See run38. */
            static const int fdilp = (getenv("MLA_FDILP") && atoi(getenv("MLA_FDILP")) == 1);
            if (fdhb && NH == 20 && KVL == 512 && KVD == 576) {
                dim3 gFlash(B, 2);   /* FD_HG = 2 head groups */
                if (fdmfma)
                    attn_flash_decode_hb_mfma_kernel<<<gFlash, 256>>>(b_clat, b_qabs, b_q, kv_base,
                                                                 kv_scale_base, b_pos, NH, QHD, QKN, QKR,
                                                                 KVL, KVD, slot_stride, scale);
                else if (fdilp)
                    attn_flash_decode_hb_ilp_kernel<<<gFlash, 256>>>(b_clat, b_qabs, b_q, kv_base,
                                                                 kv_scale_base, b_pos, NH, QHD, QKN, QKR,
                                                                 KVL, KVD, slot_stride, scale);
                else
                    attn_flash_decode_hb_kernel<<<gFlash, 256>>>(b_clat, b_qabs, b_q, kv_base,
                                                                 kv_scale_base, b_pos, NH, QHD, QKN, QKR,
                                                                 KVL, KVD, slot_stride, scale);
            } else {
                dim3 gFlash(NH, B);
                size_t shmem = (size_t)KVD * sizeof(float);
                attn_flash_decode_kernel<<<gFlash, 256, shmem>>>(b_clat, b_qabs, b_q, kv_base,
                                                                 kv_scale_base, b_pos, NH, QHD, QKN, QKR,
                                                                 KVL, KVD, slot_stride, scale);
            }
        }
        /* ctx[s,h] = clat[s,h] @ W_UV[h]^T. */
        {
            dim3 grid(ceil_div(VHD, MM_TILE), ceil_div(B, MM_TILE), NH);
            matmul_bh_mfma_kernel<<<grid, 256>>>(b_ctx, b_clat, d_w.W_UV[l], B, VHD, KVL,
                                                 KVL, NH * KVL, kv_stride, VHD, NH * VHD);
        }
        launch_matmul_w8a8(b_ao, b_ctx, d_w.o_proj[l], dq_op?dq_op[l]:NULL, dq_op?ds_op[l]:NULL, B, H, NH * VHD, -1, -1, 1);
        residual_add_kernel<<<ceil_div(B * H, 256), 256>>>(b_x, b_ao, B * H);

        launch_rmsnorm(b_xb, b_x, d_w.post_attn_norm[l], B, H, eps);
        ffn_batched(c, l, b_xb, b_mo, B, b_hb, b_hb2);
        residual_add_kernel<<<ceil_div(B * H, 256), 256>>>(b_x, b_mo, B * H);
    }

    launch_rmsnorm(b_xb, b_x, d_w.norm, B, H, eps);
    launch_matmul_w8a8(b_logits, b_xb, d_w.lm_head, dq_lm, ds_lm, B, VOC, H);
    argmax_batched_kernel<<<B, 256>>>(b_logits, VOC, VOC, b_tokens);
}

/* run36: interleaved chunked prefill (continuous batching). ONE fused forward
 * advances `nd` contiguous decode slots (window rows [0,nd) -> cache slots
 * [slot0, slot0+nd)) by one token AND prefills a chunk of `T` new tokens spanning
 * `nseq` whole prompts (rows [nd, nd+T)). Every per-row GEMM (q/kv/o_proj/FFN/MoE/
 * lm_head) runs once over the M = nd+T concatenated rows -> a single weight-stream
 * serves both phases instead of a separate decode pass plus a separate prefill pass.
 * Attention splits by row range: decode rows use the absorbed flash-decode over their
 * own KV cache; prefill rows use the causal MFMA attention over the freshly-scattered
 * latent KV. Per-slot greedy semantics are identical to the two-phase path (each slot:
 * prefill -> first token -> greedy decode); only the global schedule differs, so the
 * per-request output is unchanged (modulo moe_scatter atomic order). MLA_CHUNKFUSE=1. */
static void device_forward_fused(Transformer *t, int nd, int slot0,
                                 int T, int nseq,
                                 const int *pf_tokcat, const int *pf_seqstart,
                                 const int *pf_pos, const int *pf_slot,
                                 const int *pf_last, const int *dec_tok,
                                 const int *dec_pos, int *dec_out, int *pf_out)
{
    const Config *c = &t->config;
    const int H = c->hidden_size, NH = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim, QHD = QKN + QKR;
    const int VHD = c->v_head_dim, KVL = c->kv_lora_rank, KVD = KVL + QKR, VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const int kv_stride = (QKN + VHD) * KVL;
    const size_t layer_stride = (size_t)g_batch * g_cap * KVD;
    const int slot_stride = g_cap * KVD;
    const int inter_max = inter_max_of(c);
    const int M = nd + T;
    if (M <= 0) return;

    /* Combined per-row index arrays: rows [0,nd) decode, rows [nd,M) prefill. */
    static int *h_tokc = NULL, *h_posc = NULL, *h_sltc = NULL; static int h_ccap = 0;
    if (M > h_ccap) {
        h_tokc = (int *)realloc(h_tokc, (size_t)M * sizeof(int));
        h_posc = (int *)realloc(h_posc, (size_t)M * sizeof(int));
        h_sltc = (int *)realloc(h_sltc, (size_t)M * sizeof(int));
        h_ccap = M;
    }
    for (int r = 0; r < nd; r++) { h_tokc[r] = dec_tok[r]; h_posc[r] = dec_pos[r]; h_sltc[r] = slot0 + r; }
    for (int j = 0; j < T; j++)  { h_tokc[nd + j] = pf_tokcat[j]; h_posc[nd + j] = pf_pos[j]; h_sltc[nd + j] = pf_slot[j]; }

    int *d_tok = (int *)sbuf(B_TOKENS, (size_t)M * sizeof(int));
    int *d_pos = (int *)sbuf(B_PPOS,   (size_t)M * sizeof(int));
    int *d_slt = (int *)sbuf(B_PSLOT,  (size_t)M * sizeof(int));
    HIP_CHECK(hipMemcpy(d_tok, h_tokc, (size_t)M * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pos, h_posc, (size_t)M * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_slt, h_sltc, (size_t)M * sizeof(int), hipMemcpyHostToDevice));

    /* Prefill query-tile work list (global row = nd + prefill-local). */
    static int *h_qtq0 = NULL, *h_qtqs = NULL, *h_qtqe = NULL; static int h_qtcap = 0;
    int qt_bound = nseq + (T + 63) / 64 + 1;
    if (qt_bound > h_qtcap) {
        h_qtq0 = (int *)realloc(h_qtq0, (size_t)qt_bound * sizeof(int));
        h_qtqs = (int *)realloc(h_qtqs, (size_t)qt_bound * sizeof(int));
        h_qtqe = (int *)realloc(h_qtqe, (size_t)qt_bound * sizeof(int));
        h_qtcap = qt_bound;
    }
    int n_qtile = 0;
    for (int i = 0; i < nseq; i++) {
        int e = nd + pf_last[i] + 1, s = nd + pf_seqstart[pf_last[i]];
        for (int q0 = s; q0 < e; q0 += 64) { h_qtq0[n_qtile] = q0; h_qtqs[n_qtile] = s; h_qtqe[n_qtile] = e; n_qtile++; }
    }
    int *d_qtq0 = NULL, *d_qtqs = NULL, *d_qtqe = NULL;
    if (n_qtile > 0) {
        d_qtq0 = (int *)sbuf(B_QTQ0, (size_t)n_qtile * sizeof(int));
        d_qtqs = (int *)sbuf(B_QTQS, (size_t)n_qtile * sizeof(int));
        d_qtqe = (int *)sbuf(B_QTQE, (size_t)n_qtile * sizeof(int));
        HIP_CHECK(hipMemcpy(d_qtq0, h_qtq0, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qtqs, h_qtqs, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qtqe, h_qtqe, (size_t)n_qtile * sizeof(int), hipMemcpyHostToDevice));
    }

    /* Combined M-row activation buffers (sbuf auto-grows). */
    float *qa    = c->q_lora_rank > 0 ? (float *)sbuf(B_QA, (size_t)M * c->q_lora_rank * sizeof(float)) : NULL;
    float *xs    = (float *)sbuf(B_XS,    (size_t)M * H * sizeof(float));
    float *xb    = (float *)sbuf(B_XB,    (size_t)M * H * sizeof(float));
    float *comp  = (float *)sbuf(B_COMP,  (size_t)M * KVD * sizeof(float));
    float *qall  = (float *)sbuf(B_QALL,  (size_t)M * NH * QHD * sizeof(float));
    float *kvlin = (float *)sbuf(B_KVLIN, (size_t)M * KVD * sizeof(float));
    float *knope = (float *)sbuf(B_KNOPE, (size_t)NH * M * QKN * sizeof(float));
    float *value = (float *)sbuf(B_VALUE, (size_t)NH * M * VHD * sizeof(float));
    float *ctx   = (float *)sbuf(B_CTX,   (size_t)M * NH * VHD * sizeof(float));
    float *ao    = (float *)sbuf(B_AO,    (size_t)M * H * sizeof(float));
    float *mo    = (float *)sbuf(B_MO,    (size_t)M * H * sizeof(float));
    float *hb    = (float *)sbuf(B_HB,    (size_t)M * inter_max * sizeof(float));
    float *hb2   = (float *)sbuf(B_HB2,   (size_t)M * inter_max * sizeof(float));
    const size_t rope_sh = c->rope_interleaved ? (size_t)QKR * sizeof(float) : 0;

    embed_lookup_kernel<<<M, 256>>>(xs, d_w.embed_tokens, d_tok, H);

    for (int l = 0; l < c->n_layers; l++)
    {
        int8_t *cache_base = d_kv_cache + (size_t)l * layer_stride;
        float  *scale_base = d_kv_scale + (size_t)l * g_batch * g_cap;

        launch_rmsnorm(xb, xs, d_w.input_layernorm[l], M, H, eps);
        launch_project_q(c, l, xb, qall, M, qa);
        rope_pos_kernel<<<dim3(M, NH), QKR / 2, rope_sh>>>(qall + QKN, NH, NH * QHD, QHD,
                                                           d_pos, QKR, c->rope_interleaved, d_inv_freq);
        launch_matmul_w8a8(comp, xb, d_w.kv_a_proj[l], dq_kva?dq_kva[l]:NULL, dq_kva?ds_kva[l]:NULL, M, KVD, H, -1, -1, 1);
        launch_rmsnorm(kvlin, comp, d_w.kv_a_layernorm[l], M, KVL, c->mla_norm_eps, KVD, KVD);
        copy_strided_kernel<<<M, 256>>>(kvlin + KVL, comp + KVL, QKR, KVD, KVD);
        rope_pos_kernel<<<dim3(M, 1), QKR / 2, rope_sh>>>(kvlin + KVL, 1, KVD, 0,
                                                          d_pos, QKR, c->rope_interleaved, d_inv_freq);
        kv_scatter_kernel<<<M, 256>>>(cache_base, scale_base, slot_stride, d_slt, d_pos, kvlin, KVD, M);

        /* ---- decode rows [0,nd): absorbed flash-decode over cache slots [slot0,..) */
        if (nd > 0) {
            dim3 gQ(ceil_div(KVL, MM_TILE), ceil_div(nd, MM_TILE), NH);
            matmul_bh_mfma_kernel<<<gQ, 256>>>(b_qabs, qall, d_wuk_t[l], nd, KVL, QKN,
                                               QHD, NH * QHD, KVL * QKN, KVL, NH * KVL);
            int8_t *kvb = cache_base + (size_t)slot0 * slot_stride;
            float  *scb = scale_base + (size_t)slot0 * g_cap;
            dim3 gFlash(nd, 2);
            static const int fdilp_f = (getenv("MLA_FDILP") && atoi(getenv("MLA_FDILP")) == 1);
            if (fdilp_f)
                attn_flash_decode_hb_ilp_kernel<<<gFlash, 256>>>(b_clat, b_qabs, qall, kvb, scb,
                                                             d_pos, NH, QHD, QKN, QKR, KVL, KVD,
                                                             slot_stride, scale);
            else
                attn_flash_decode_hb_kernel<<<gFlash, 256>>>(b_clat, b_qabs, qall, kvb, scb,
                                                             d_pos, NH, QHD, QKN, QKR, KVL, KVD,
                                                             slot_stride, scale);
            dim3 gV(ceil_div(VHD, MM_TILE), ceil_div(nd, MM_TILE), NH);
            matmul_bh_mfma_kernel<<<gV, 256>>>(ctx, b_clat, d_w.W_UV[l], nd, VHD, KVL,
                                               KVL, NH * KVL, kv_stride, VHD, NH * VHD);
        }
        /* ---- prefill rows [nd,M): materialize per-head K/V then causal MFMA attn.
         * knope/value are [NH][M][.] (decode rows computed too but unused). */
        if (T > 0) {
            dim3 gK(ceil_div(QKN, MM_TILE), ceil_div(M, MM_TILE), NH);
            matmul_headw_mfma_kernel<<<gK, 256>>>(knope, kvlin, d_w.W_UK[l], M, QKN, KVL,
                                                  KVD, kv_stride, M * QKN);
            dim3 gVv(ceil_div(VHD, MM_TILE), ceil_div(M, MM_TILE), NH);
            matmul_headw_mfma_kernel<<<gVv, 256>>>(value, kvlin, d_w.W_UV[l], M, VHD, KVL,
                                                   KVD, kv_stride, M * VHD);
            attn_prefill_mfma_kernel<<<dim3(n_qtile, NH), 256>>>(ctx, qall, knope, kvlin, value,
                                                                 d_qtq0, d_qtqs, d_qtqe, M, NH,
                                                                 QHD, QKN, QKR, KVD, KVL, VHD, scale);
        }

        launch_matmul_w8a8(ao, ctx, d_w.o_proj[l], dq_op?dq_op[l]:NULL, dq_op?ds_op[l]:NULL, M, H, NH * VHD, -1, -1, 1);
        residual_add_kernel<<<ceil_div(M * H, 256), 256>>>(xs, ao, M * H);
        launch_rmsnorm(xb, xs, d_w.post_attn_norm[l], M, H, eps);
        ffn_batched(c, l, xb, mo, M, hb, hb2);
        residual_add_kernel<<<ceil_div(M * H, 256), 256>>>(xs, mo, M * H);
    }

    /* Epilogue: greedy token for every decode row [0,nd) + first token for each
     * prefill prompt (its last row = nd + pf_last[i]). One gathered lm_head. */
    int ng = nd + nseq;
    static int *h_gidx = NULL; static int h_gcap = 0;
    if (ng > h_gcap) { h_gidx = (int *)realloc(h_gidx, (size_t)ng * sizeof(int)); h_gcap = ng; }
    for (int r = 0; r < nd; r++) h_gidx[r] = r;
    for (int i = 0; i < nseq; i++) h_gidx[nd + i] = nd + pf_last[i];
    int *d_gidx = (int *)sbuf(B_FGIDX, (size_t)ng * sizeof(int));
    HIP_CHECK(hipMemcpy(d_gidx, h_gidx, (size_t)ng * sizeof(int), hipMemcpyHostToDevice));
    float *gbuf = (float *)sbuf(B_PLASTX, (size_t)ng * H * sizeof(float));
    float *glog = (float *)sbuf(B_FLOGITS, (size_t)ng * VOC * sizeof(float));
    int   *gout = (int *)sbuf(B_FOUT, (size_t)ng * sizeof(int));
    gather_rows_kernel<<<ng, 256>>>(gbuf, xs, d_gidx, H);
    launch_rmsnorm(xb, gbuf, d_w.norm, ng, H, eps);
    launch_matmul_w8a8(glog, xb, d_w.lm_head, dq_lm, ds_lm, ng, VOC, H);
    argmax_batched_kernel<<<ng, 256>>>(glog, VOC, VOC, gout);
    static int *h_gout = NULL; static int h_gocap = 0;
    if (ng > h_gocap) { h_gout = (int *)realloc(h_gout, (size_t)ng * sizeof(int)); h_gocap = ng; }
    HIP_CHECK(hipMemcpy(h_gout, gout, (size_t)ng * sizeof(int), hipMemcpyDeviceToHost));
    for (int r = 0; r < nd; r++)   dec_out[r] = h_gout[r];
    for (int i = 0; i < nseq; i++) pf_out[i]  = h_gout[nd + i];
}

/* ==== INFERENCE DRIVER (batched prefill + lockstep decode) ================= */

/* Grouped-batching driver: bucket requests by prompt length, then for each group
 * of <= g_batch requests prefill every prompt into its slot and decode all slots
 * in lockstep (weights read once per batch). Per-slot greedy semantics match the
 * single-request reference: emit the prompt's first token, then decode until EOS
 * or the step budget. */
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

    int first;
    if (g_pfbatch)
    {
        int pf_tok[NP], pf_seq[NP], pf_pos[NP], pf_slot[NP], pf_last[1], pf_out[1];
        for (int j = 0; j < NP; j++) { pf_tok[j] = 0; pf_seq[j] = 0; pf_pos[j] = j; pf_slot[j] = 0; }
        pf_last[0] = NP - 1;
        device_forward_prefill_batched(t, NP, 1, pf_tok, pf_seq, pf_pos, pf_slot, pf_last, pf_out);
        first = pf_out[0];
    }
    else
    {
        int pf_tok[NP];
        for (int j = 0; j < NP; j++) pf_tok[j] = 0;
        first = device_forward_unabsorbed(t, pf_tok, NP, 0);
    }

    int h_tok = first, h_pos = NP;
    for (int step = 0; step < NEW; step++)
    {
        HIP_CHECK(hipMemcpy(b_tokens, &h_tok, sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(b_pos,    &h_pos, sizeof(int), hipMemcpyHostToDevice));
        device_forward_absorbed_batched(t, 1, h_pos + 1);
        HIP_CHECK(hipMemcpy(&h_tok, b_tokens, sizeof(int), hipMemcpyDeviceToHost));
        h_pos++;
    }
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
    int  *h_tok     = (int *)malloc((size_t)g_batch * sizeof(int));
    int  *h_pos     = (int *)malloc((size_t)g_batch * sizeof(int));
    char *done      = (char *)malloc((size_t)g_batch);

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
    int *pf_out  = (int *)malloc((size_t)g_batch * sizeof(int));
    /* run36 fused decode-window scratch. */
    int *dec_tok = (int *)malloc((size_t)g_batch * sizeof(int));
    int *dec_pos = (int *)malloc((size_t)g_batch * sizeof(int));
    int *dec_out = (int *)malloc((size_t)g_batch * sizeof(int));

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
                done[i] = 1; slot_st[i] = 0; h_tok[i] = eos; h_pos[i] = 0;
                reqs->out_lens[r] = 0;
            }
            else
            {
                done[i] = 0;   /* active until set_slot decides after prefill */
            }
        }

        /* Set a slot's post-prefill state from its first token. */
        auto set_slot = [&](int i, int np, int tok0) {
            int budget = g_cap - np;
            int st = max_steps < budget ? max_steps : budget;
            if (st < 0) st = 0;
            slot_st[i] = st;
            h_tok[i]   = tok0;
            h_pos[i]   = np;
            done[i]    = (st <= 0 || tok0 == eos);
        };

        /* run36: interleaved chunked prefill. Admit slots in order; each super-step
         * runs ONE fused forward = decode(active window [Dwin,A)) + prefill(next chunk
         * [A,newA)). Cache slots used per step are contiguous [Dwin,newA). Per-slot
         * greedy semantics are identical to the two-phase path (prefill -> first token
         * -> decode); only the schedule differs. MLA_CHUNKFUSE=1. */
        if (g_chunkfuse)
        {
            int A = 0;   /* next slot to admit (group-local, in `order`) */
            while (true)
            {
                int Dwin = 0;
                while (Dwin < A && done[Dwin]) Dwin++;   /* window start = first non-done */
                int nd = A - Dwin;                        /* contiguous; done slots masked */

                /* Pack a prefill chunk of whole prompts up to g_fusechunk tokens. */
                int T = 0, nseq = 0, i = A;
                while (i < Bg)
                {
                    if (done[i]) { i++; continue; }       /* empty prompt already finished */
                    int np = plen[order[base + i]];
                    if (nseq > 0 && T + np > g_fusechunk) break;
                    int *tk = ptok[order[base + i]];
                    for (int j = 0; j < np; j++)
                    {
                        pf_tok[T + j]  = tk[j];
                        pf_seq[T + j]  = T;               /* prefill-local prompt start */
                        pf_pos[T + j]  = j;               /* within-prompt position     */
                        pf_slot[T + j] = i;               /* cache slot = group slot i  */
                    }
                    pf_last[nseq]  = T + np - 1;
                    pf_cslot[nseq] = i;
                    pf_cnp[nseq]   = np;
                    T += np; nseq++; i++;
                }
                int newA = i;

                if (nd == 0 && nseq == 0)
                {   /* nothing active and nothing packable: skip trailing empties */
                    A = newA;
                    if (A >= Bg) break;
                    continue;
                }

                /* Snapshot the decode window's current token/pos, then emit the current
                 * token for still-active slots (mirrors the lockstep emit-before-step). */
                for (int r = 0; r < nd; r++)
                {
                    int s = Dwin + r, p = h_pos[s];
                    if (p >= g_cap) p = g_cap - 1;        /* clamp done slots (KV OOB guard) */
                    dec_tok[r] = h_tok[s];
                    dec_pos[r] = p;
                }
                for (int r = 0; r < nd; r++)
                {
                    int s = Dwin + r;
                    if (!done[s]) reqs->out_tokens[slot_req[s]][slot_nout[s]++] = h_tok[s];
                }

                device_forward_fused(t, nd, Dwin, T, nseq,
                                     pf_tok, pf_seq, pf_pos, pf_slot, pf_last,
                                     dec_tok, dec_pos, dec_out, pf_out);

                for (int r = 0; r < nd; r++)
                {
                    int s = Dwin + r;
                    if (done[s]) continue;
                    h_tok[s] = dec_out[r];
                    h_pos[s]++;
                    if (slot_nout[s] >= slot_st[s] || h_tok[s] == eos) done[s] = 1;
                }
                for (int c2 = 0; c2 < nseq; c2++)
                    set_slot(pf_cslot[c2], pf_cnp[c2], pf_out[c2]);
                A = newA;

                if (A >= Bg)
                {
                    int any = 0;
                    for (int s = 0; s < Bg; s++) if (!done[s]) { any = 1; break; }
                    if (!any) break;
                }
            }
            for (int i2 = 0; i2 < Bg; i2++)
            {
                reqs->out_lens[slot_req[i2]] = slot_nout[i2];
                total += slot_nout[i2];
            }
            continue;   /* group handled entirely by the fused scheduler */
        }

        if (g_pfbatch)
        {
            /* Pack whole prompts up to g_pfchunk tokens per batched forward. */
            int i = 0;
            while (i < Bg)
            {
                if (done[i]) { i++; continue; }   /* empty prompt already finished */
                int T = 0, nseq = 0;
                while (i < Bg)
                {
                    if (done[i]) { i++; continue; }
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
                                               pf_slot, pf_last, pf_out);
                for (int c2 = 0; c2 < nseq; c2++)
                    set_slot(pf_cslot[c2], pf_cnp[c2], pf_out[c2]);
            }
        }
        else
        {
            for (int i = 0; i < Bg; i++)
            {
                if (done[i]) continue;
                int np = plen[order[base + i]];
                set_slot(i, np, device_forward_unabsorbed(t, ptok[order[base + i]], np, i));
            }
        }

        /* Lockstep decode. Each iteration mirrors the reference while-loop:
         * emit the current token for still-active slots, run one batched step,
         * then advance. */
        for (int step = 0; step < max_steps; step++)
        {
            int any = 0, kv_max = 1;
            for (int i = 0; i < Bg; i++)
            {
                /* Only active slots are decoded further; a *done* slot can sit at
                 * h_pos == g_cap (prompt_len + steps), and including it would push
                 * kv_max to g_cap+1 -> overflow b_score [B*NH*g_cap] and the attn
                 * grid (illegal access, previously masked at B<=96 by grouping /
                 * at B>96 by OOM). Active slots have h_pos < g_cap, so their real
                 * attention width <= g_cap; extra width is masked, so dropping done
                 * slots never changes an active slot's output. */
                if (done[i]) continue;
                any = 1;
                if (h_pos[i] + 1 > kv_max) kv_max = h_pos[i] + 1;
            }
            if (!any) break;
            if (kv_max > g_cap) kv_max = g_cap;   /* belt-and-suspenders clamp */

            for (int i = 0; i < Bg; i++)
                if (!done[i])
                    reqs->out_tokens[slot_req[i]][slot_nout[i]++] = h_tok[i];

            /* A done slot's frozen h_pos can be g_cap (== prompt_len + steps). Its
             * KV write is ignored, but the batched kv_write kernel still executes
             * for every slot, and pos == g_cap writes one slot-region past this
             * slot (pos*KVD == slot_stride) -> OOB at the last slot of the last
             * layer (illegal access), or silent cross-slot corruption otherwise
             * (which is why B<=96 "passed" while being subtly wrong). Active slots
             * never exceed g_cap-1, so clamping is a no-op for them. */
            for (int i = 0; i < Bg; i++)
                if (h_pos[i] >= g_cap) h_pos[i] = g_cap - 1;

            HIP_CHECK(hipMemcpy(b_tokens, h_tok, (size_t)Bg * sizeof(int), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(b_pos, h_pos, (size_t)Bg * sizeof(int), hipMemcpyHostToDevice));

            device_forward_absorbed_batched(t, Bg, kv_max);

            HIP_CHECK(hipMemcpy(h_tok, b_tokens, (size_t)Bg * sizeof(int), hipMemcpyDeviceToHost));

            for (int i = 0; i < Bg; i++)
            {
                if (done[i]) continue;
                h_pos[i]++;
                if (slot_nout[i] >= slot_st[i] || h_tok[i] == eos) done[i] = 1;
            }
        }

        for (int i = 0; i < Bg; i++)
        {
            reqs->out_lens[slot_req[i]] = slot_nout[i];
            total += slot_nout[i];
        }
    }

    for (int r = 0; r < N; r++) free(ptok[r]);
    free(ptok); free(plen); free(order);
    free(slot_req); free(slot_st); free(slot_nout);
    free(h_tok); free(h_pos); free(done);
    free(pf_tok); free(pf_seq); free(pf_pos); free(pf_slot);
    free(pf_last); free(pf_cslot); free(pf_cnp); free(pf_out);
    free(dec_tok); free(dec_pos); free(dec_out);
    return total;
}
