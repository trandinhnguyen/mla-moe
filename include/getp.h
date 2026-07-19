/* Batch-throughput harness — the performance grading surface.
 *
 * `getp()` (frozen, src/getp_eval.c) reads a fixed request set, brackets the
 * candidate's inference() with a wall clock, and prints one end-to-end number:
 *   achieved throughput TPS (tok/s)
 * The candidate implements the three functions below in src/getp_run.hip (the
 * file they modify). */
#ifndef MLA_GETP_H
#define MLA_GETP_H

#include "model.h"   /* Transformer */

#ifdef __cplusplus
extern "C" {   /* C linkage: getp_run.hip *defines* these; the C harness calls them */
#endif

/* One batch of text prompts + their generated token ids. Owned/managed by the
 * frozen harness; the driver reads prompts[] and fills out_tokens[]/out_lens[]. */
typedef struct {
    int    num_reqs;      /* number of requests */
    int    max_steps;     /* cap on generated tokens per request */
    char **prompts;       /* [num_reqs] request prompt text (NUL-terminated) */
    int  **out_tokens;    /* [num_reqs][max_steps] generated ids (driver writes) */
    int   *out_lens;      /* [num_reqs] number of ids written per request */
} Requests;

/* Implemented by the candidate in src/getp_run.hip: */
void      warm_up(Transformer *t);                    /* allocate / upload weights / … */
void      finish(Transformer *t);                     /* free / tear down */
long long inference(Transformer *t, Requests *reqs);  /* returns total tokens generated */

/* Implemented by the frozen harness (src/getp_eval.c). Reads `req_file`
 * (line 0 = count, then one prompt per line), runs warm_up → timed inference →
 * finish, prints throughput, and writes generated ids to `out_file`. steps<=0
 * uses the built-in default. */
void getp(Transformer *t, const char *req_file, const char *out_file, int steps);

#ifdef __cplusplus
}
#endif

#endif /* MLA_GETP_H */
