/* ============================================================================
 * glm.hip -- aggregator translation unit for the GLM-4.7-Flash kernel set.
 *
 * The frozen Makefile compiles every "src/kernels" top-level .hip (a
 * NON-recursive wildcard) plus src/getp_run.hip. Files under src/kernels/glm
 * are therefore NOT picked up on their own; this top-level TU pulls them in so
 * they compile exactly once, as one translation unit. Device code generation is
 * per-__global__ kernel, so grouping them here changes no kernel's codegen/perf.
 *
 * Everything GLM-owned lives under src/kernels/glm (so the DeepSeek optimizer
 * only ever touches src/kernels/dsv, and vice versa):
 *   src/kernels/glm/kernels.h            -- GLM device helpers + kernel decls
 *   src/kernels/glm/<category>.hip       -- GLM kernel impls (included below)
 *   src/kernels/glm/glm_engine.hip       -- GLM host engine (glm_* entry points)
 *   src/kernels/glm/unused_reference.hip -- retired kernels, NOT included/compiled
 * The only shared files are src/getp_run.hip (the model dispatcher) and
 * src/kernels/engines.h (the dispatch declarations); neither is model-specific.
 * ========================================================================== */
#include "glm/elementwise.hip"
#include "glm/linear.hip"
#include "glm/rope.hip"
#include "glm/attention.hip"
#include "glm/batched.hip"
/* GLM host engine (glm_warm_up / glm_finish / glm_inference) -- shares this TU
 * with the kernels above so its launches resolve in-unit. */
#include "glm/glm_engine.hip"
