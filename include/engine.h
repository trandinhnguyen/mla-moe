/* Frozen engine entry points the candidate driver (src/getp_run.hip) may call.
 * These are the oracle-validated CPU reference kernels defined in src/run.c.
 * A candidate porting to GPU starts by calling them (correct-by-construction),
 * then replaces them incrementally with their own kernels inside getp_run.hip. */
#ifndef MLA_ENGINE_H
#define MLA_ENGINE_H

#include "model.h"   /* Transformer */

#ifdef __cplusplus
extern "C" {   /* C linkage so the C++/HIP getp_run TU resolves these unmangled */
#endif

/* Teacher-forced top-1 capture — opaque to the driver; always pass NULL. */
typedef struct TeacherForce TeacherForce;

/* Whole-prompt prefill. Returns logits at the LAST position (points into
 * RunState; valid until the next forward call). Pass nll_out=NULL, tf=NULL. */
float *forward_unabsorbed(Transformer *t, const int *tokens, int n_prompt,
                          double *nll_out, TeacherForce *tf);

/* One-token absorbed decode at absolute position `pos`. Returns logits. */
float *forward_absorbed(Transformer *t, int token, int pos);

/* Greedy argmax over `vocab_size` logits. */
int sample(float *logits, int vocab_size);

#ifdef __cplusplus
}
#endif

#endif /* MLA_ENGINE_H */
