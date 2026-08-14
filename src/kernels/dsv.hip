/* ============================================================================
 * dsv.hip -- aggregator translation unit for the DeepSeek-V2-Lite engine.
 *
 * Mirrors glm.hip: the frozen Makefile compiles top-level src/kernels/*.hip
 * only (non-recursive), so this file pulls the dsv/ sources into one TU that
 * the Makefile picks up. Both files below come from the deepseekv2lite branch:
 *   dsv2_primitives.hip -- __global__ kernels (anonymous namespace = internal
 *                          linkage, so their generic names like rmsnorm_kernel
 *                          never clash with the GLM kernels) + the extern "C"
 *                          dsv2_launch_* host API.
 *   dsv_engine.hip      -- host orchestration (dsv_warm_up/finish/inference)
 *                          that composes those dsv2_launch_* primitives.
 * Sharing one TU lets dsv_engine's launch calls resolve in-unit.
 * ========================================================================== */
#include "dsv/dsv2_primitives.hip"
#include "dsv/dsv_engine.hip"
