#ifndef TOKENIZER_H
#define TOKENIZER_H

/* Byte-level BPE tokenizer (GPT-2 family) for DeepSeek-V2-Lite and
 * GLM-4.7-Flash (pre-tokenizer selected by config.json model_type).
 * Faithful to HuggingFace for ASCII/English text; the pre-tokenizer's
 * non-ASCII letter/punctuation/CJK handling is intentionally not implemented
 * (such input still yields valid tokens, but may split differently). */

#ifdef __cplusplus
extern "C" {   /* C linkage so the C++/HIP getp_run TU resolves these unmangled */
#endif

typedef struct Tokenizer Tokenizer;

/* Load <model_dir>/tokenizer.json (+ config.json for bos/eos). Returns NULL if
 * tokenizer.json is absent — callers fall back to raw token-id I/O. */
Tokenizer *tokenizer_load(const char *model_dir);

/* Encode UTF-8 `text` into token ids (writes up to `max_ids`, returns count).
 * Prepends the BOS id when add_bos is nonzero and a bos token is defined. */
int tokenizer_encode(const Tokenizer *t, const char *text, int add_bos,
                     int *out_ids, int max_ids);

/* Decode `n` ids to a freshly malloc'd UTF-8 string (caller frees). Special
 * tokens (bos/eos/added) contribute no text. */
char *tokenizer_decode(const Tokenizer *t, const int *ids, int n);

int  tokenizer_bos_id(const Tokenizer *t);   /* -1 if none */
int  tokenizer_eos_id(const Tokenizer *t);   /* -1 if none */
void tokenizer_free(Tokenizer *t);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_H */
