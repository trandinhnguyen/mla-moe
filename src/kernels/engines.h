/* ============================================================================
 * engines.h -- per-model engine dispatch surface.
 *
 * src/getp_run.hip is a thin dispatcher: it implements the frozen getp.h
 * contract (warm_up / finish / inference) and forwards each call to the engine
 * for the detected model. Each model's host orchestration lives in its own
 * translation unit so their file-static state never collides:
 *   GLM-4.7-Flash    -> src/kernels/glm/glm_engine.hip   (glm_*)
 *   DeepSeek-V2-Lite -> src/kernels/dsv/dsv_engine.hip   (dsv_*)
 *
 * model_kind() is the single source of truth for which model a Transformer is;
 * both models are distinguishable from Config alone (GLM has a q LoRA and a
 * sigmoid+bias router, DSV2-Lite has neither).
 * ========================================================================== */
#ifndef MLA_ENGINES_H
#define MLA_ENGINES_H

#include "model.h"   /* Transformer, Config */
#include "getp.h"    /* Requests */

typedef enum { MODEL_DSV2, MODEL_GLM } ModelKind;

static inline ModelKind model_kind(const Config *c) {
    /* q_lora_rank is the structural discriminator: >0 only for GLM-4.7-Flash. */
    return (c->q_lora_rank > 0) ? MODEL_GLM : MODEL_DSV2;
}

static inline const char *model_name(ModelKind k) {
    return k == MODEL_GLM ? "GLM-4.7-Flash" : "DeepSeek-V2-Lite";
}

/* GLM-4.7-Flash engine (src/kernels/glm/glm_engine.hip). */
void      glm_warm_up(Transformer *t);
void      glm_finish(Transformer *t);
long long glm_inference(Transformer *t, Requests *reqs);

/* DeepSeek-V2-Lite engine (src/kernels/dsv/dsv_engine.hip). */
void      dsv_warm_up(Transformer *t);
void      dsv_finish(Transformer *t);
long long dsv_inference(Transformer *t, Requests *reqs);

#endif /* MLA_ENGINES_H */
