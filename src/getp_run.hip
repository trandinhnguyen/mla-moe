/* ============================================================================
 * getp_run.hip -- model dispatcher (the file the grading harness links against).
 *
 * Implements the frozen getp.h contract (warm_up / finish / inference) and
 * forwards each call to the engine for the detected model. The engines live in
 * their own translation units so their file-static state never collides:
 *   GLM-4.7-Flash    -> src/kernels/glm/glm_engine.hip  (glm_*)   [glm.hip TU]
 *   DeepSeek-V2-Lite -> src/kernels/dsv/dsv_engine.hip  (dsv_*)   [dsv.hip TU]
 *
 * The model is detected once, in warm_up(), from Config (model_kind) and cached
 * for the finish()/inference() calls that follow on the same Transformer.
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>

#include "getp.h"             /* Requests + the warm_up/finish/inference contract */
#include "kernels/engines.h"  /* ModelKind, model_kind, per-model entry points */

/* Model chosen in warm_up(), reused by the following finish()/inference(). */
static ModelKind g_engine;

extern "C" void warm_up(Transformer *t)
{
    g_engine = model_kind(&t->config);
    fprintf(stderr, "[getp] dispatch -> %s engine\n", model_name(g_engine));
    switch (g_engine) {
        case MODEL_GLM:  glm_warm_up(t); break;
        case MODEL_DSV2: dsv_warm_up(t); break;
    }
}

extern "C" void finish(Transformer *t)
{
    switch (g_engine) {
        case MODEL_GLM:  glm_finish(t); break;
        case MODEL_DSV2: dsv_finish(t); break;
    }
}

extern "C" long long inference(Transformer *t, Requests *reqs)
{
    switch (g_engine) {
        case MODEL_GLM:  return glm_inference(t, reqs);
        case MODEL_DSV2: return dsv_inference(t, reqs);
    }
    return 0;
}
