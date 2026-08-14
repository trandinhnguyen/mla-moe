/* ============================================================================
 * glm.hip -- aggregator translation unit for the GLM-4.7-Flash kernel set.
 *
 * The frozen Makefile compiles every "src/kernels" top-level .hip (a
 * NON-recursive wildcard) plus src/getp_run.hip. Files under src/kernels/glm
 * are therefore NOT picked up on their own; this top-level TU pulls them in so
 * they compile exactly once, as one translation unit. Device code generation is
 * per-__global__ kernel, so grouping them here changes no kernel's codegen/perf.
 *
 * Layout (prepared for merging with the DeepSeek-only optimization branch):
 *   src/kernels/kernels.h                -- shared device helpers + kernel decls
 *   src/kernels/glm/<category>.hip       -- GLM kernel impls (included below)
 *   src/kernels/glm/unused_reference.hip -- retired kernels, NOT included/compiled
 * A DeepSeek branch can add a src/kernels/dsv folder + a sibling
 * src/kernels/dsv.hip aggregator without touching any of the files above.
 * ========================================================================== */
#include "glm/elementwise.hip"
#include "glm/linear.hip"
#include "glm/rope.hip"
#include "glm/attention.hip"
#include "glm/batched.hip"
