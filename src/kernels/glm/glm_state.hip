/* ==========================================================================
 * glm_state.hip -- the GLM engine's device-side state, and nothing else.
 *
 * Everything the engine owns on the device lives here: the uploaded weights, the
 * int8 quantisation tables, the latent-KV cache, the persistent activation
 * scratch, and the handful of configuration knobs. It is split out so the layers
 * above it read as pure logic:
 *
 *   glm_state.hip    <- this file: what exists on the device
 *   glm_weights.hip     how it gets there (upload, quantise, free)
 *   glm_forward.hip     what runs on it (launch wrappers + the two forward passes)
 *   glm_engine.hip      warm_up / finish / inference (the entry points)
 *
 * All four are #included into the single glm.hip translation unit, so these
 * statics are shared without any of them being externally visible.
 * ==========================================================================*/
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

/* Row block for the dense-layer FFN. Only layer 0 is dense and it is row-parallel,
 * so processing it in blocks keeps the SwiGLU scratch (hb/hb2) sized by this
 * instead of by the whole prefill chunk -- worth ~1 GB of VRAM at T=16384, on an
 * engine that otherwise runs with 0.6 GB to spare. */
#define DENSE_ROWS 4096

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
/* run65: per-layer slabs backing the int8 expert tables (one allocation each
 * instead of 3*NR per layer -- see quant_from_host_into). */
static int8_t **d_eslab_q = NULL;
static float  **d_eslab_s = NULL;

/* run67: slot -> KV slot. Requests with identical prompts share one KV slot, so the
 * cache is sized by DISTINCT prompts (g_kvslots) instead of by batch. NULL until the
 * first group builds it. */
static int   *b_kvslot = NULL;

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
static int    g_batch = 32768;  /* concurrent requests per group (<= this); overridable
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
static int    g_kvslots = 0;   /* KV slots allocated (<= g_batch); set in warm_up */
static int    g_cap   = 576;   /* per-slot cache positions (max prompt 512 + 64 steps) */
static int    g_pfchunk = 16384; /* chunked-prefill token budget: pack whole prompts up
                                 * to this many tokens into one batched forward so weights
                                 * are read once per chunk (not once per request). Sizes
                                 * the MoE scratch (MR). Overridable via MLA_PFCHUNK; set 0
                                 * run35: prefill is ~85% per-token / L2-BW-bound, only ~15%
                                 * fixed weight-stream, so bumping the chunk amortizes that
                                 * fixed cost over fewer forwards (bit-safe: per-prompt causal,
                                 * atomic-noise only, acc PASS). run37: the chunk max is
                                 * VRAM-bound and shrinks as batch grows (KV cache competes) --
                                 * at B=1280 the largest chunk that fits a 68GB-free GCD is
                                 * ~10240 (12288 OOMs), giving 831->836.65 tok/s. Default 10240
                                 * pairs with the default B=1280; drop it if a busier GCD OOMs. */

static float *b_x, *b_xb, *b_qa, *b_q, *b_comp, *b_qabs, *b_clat;
static float *b_ctx, *b_ao, *b_mo;
static bf16_t *b_shared;       /* run65: only moe_gather_add reads it */
static bf16_t *b_hb, *b_hb2;   /* run65: bf16 FFN intermediates (see launch_swiglu_w8a8) */
static float *b_moelog, *b_topk_wt, *b_sorted_wt;
/* run65: the three grouped-GEMM outputs are bf16 -- gate/up feed silu+quantise and
 * down feeds a weighted fp32 sum, so nothing read the extra mantissa. Halves both
 * their traffic and their 1.34 GB of VRAM. */
static bf16_t *b_hbk, *b_hb2k, *b_downk;
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
/* run20/21/22: the attention projections (q_a/q_b/kv_a/o_proj), the shared-expert
 * FFN and the dense-layer FFN all run W8A8 -- int8 weight with a per-output-row
 * fp32 scale, int8 activation with a per-row scale, straight into
 * mfma_i32_16x16x16i8. dq_/ds_ hold those tables, built once in warm_up.
 *
 * Two GEMMs deliberately stay bf16: lm_head (int8 there perturbs the greedy
 * argmax -- it cost ~0.008 METEOR for +0.4% tok/s) and moe_gate (N=64, ~0% of
 * runtime, and top-K routing is chaotic under perturbation). int4/W4A8 was tried
 * for both the experts and the dense projections and lost on every axis; see
 * report/run29..33 and report/run39_bankconflict_widegemm/. */
static int8_t **dq_qa=NULL, **dq_qb=NULL, **dq_kva=NULL, **dq_op=NULL;
static float  **ds_qa=NULL, **ds_qb=NULL, **ds_kva=NULL, **ds_op=NULL;
static int8_t **dq_sg=NULL, **dq_su=NULL, **dq_sd=NULL;   /* shared-expert gate/up/down */
static float  **ds_sg=NULL, **ds_su=NULL, **ds_sd=NULL;
static int8_t **dq_dg=NULL, **dq_du=NULL, **dq_dd=NULL;   /* dense-layer FFN (l<first_k_dense) */
static float  **ds_dg=NULL, **ds_du=NULL, **ds_dd=NULL;
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
    /* decode token history [g_batch][max_steps-1], read back once per group */
    B_TOKREC, B_PCSLOT,
    /* inverse of the MoE expert sort, keyed by row*K + k */
    B_MOEINV,
    /* lm_head activation narrowed to bf16 */
    B_XBF16,
    /* latent KV narrowed to bf16 for the per-head W_UK/W_UV GEMMs */
    B_KVBF,
    /* fused-argmax lm_head: per-(row, 128-col tile) max and index */
    B_LMPV, B_LMPI,
    /* duplicate-prompt prefill sharing: [3][nfol] = leader slot, follower slot, len */
    B_DUPIDX,
    B_NBUF
};
typedef struct { void *ptr; size_t bytes; } DevBuf;
static DevBuf g_scratch[B_NBUF];

/* Every launch and every copy in the forward path goes to this stream. It exists
 * so one decode step can be captured into a hipGraph (capture requires a
 * non-default stream and forbids blocking copies -- the decode loop has none).
 * Created in warm_up; while it is still 0 everything runs on the null stream,
 * which is exactly the previous behaviour. */
static hipStream_t g_str = 0;

/* Device-side decode step counter (the column of b_tokrec the argmax writes) and
 * the captured decode step. Because the counter is on the device, every decode
 * step issues an identical sequence of launches, so ONE captured graph replays
 * for all of them -- ~1900 launches per step become one hipGraphLaunch. */
static int *b_step = NULL;
static int g_graph_bg = -1;   /* group size the captured graph was built for */
static hipGraphExec_t g_dec_graph = NULL;

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
