# Plan — build the candidate "kickoff" scaffolding

## Goal
Bring `mla-moe` to a candidate-ready starting state: a runnable CPU engine +
**a single editable file** + **one command that produces a single throughput
number (tok/s) over a fixed workload** (the perf score), while correctness is
gated by the existing harness (`make eval`). The task shape: a CPU baseline
that candidates port to GPU kernels themselves.

## Reference layout → mla-moe mapping

| Reference layout | Role | mla-moe — status | Action |
|---|---|---|---|
| `run.cpp` (frozen) with `forward()` | engine + reference oracle | `src/run.c` `forward_unabsorbed`/`forward_absorbed` (oracle-validated) | KEEP frozen, expose prototypes via `include/engine.h` |
| `getp-csrc/getp_eval.cpp` (frozen) | read batch, time, print `achieved throughput TPS (tok/s)`, write output | — MISSING | ADD `src/getp_eval.c` + `include/getp.h` |
| `getp-csrc/getp_run.cpp` (**editable**) `warm_up/finish/inference` | file candidate edits / where GPU kernels go | — MISSING | ADD `src/getp_run.c` (editable surface) |
| `./run model -m getp -i in -o out` | perf grading command | — MISSING | ADD `getp` mode to `src/run.c main()` |
| `data/input.txt` (count + lines) | fixed workload | `tests/eval/<model>/requests.txt` (same format) | REUSE, no new data |
| `evaluation/` METEOR+BERTScore + threshold | correctness gate | `tests/eval/eval.py` (teacher top-1 + ppl + FUZZY) — richer | REUSE as correctness gate |
| `decode.cpp` | view output tokens → text | `run -p` + tokenizer already detokenizes | no addition needed |
| README "Do not modify …" + Quick start | exam framing | README has "Future: candidate hand-off" (undecided) | ADD Quick start + Task + do-not-modify |

## Frozen / editable split (design decision)
- **Editable (the submission): `src/getp_run.c` only.** Candidates can (a) call
  the reference CPU `forward_*` to be correct immediately, then (b) replace them
  incrementally with GPU kernels written in this same file. The frozen CPU
  forward is the oracle; the candidate writes the fast path in their own file.
- **Frozen:** `run.c`, `model_load.c`, `safetensors_loader.c`, `tokenizer.c`,
  `dump.c`, `main.c`, `getp_eval.c`, all of `tests/`, and `include/*`.
- **Point to confirm (judgment call):** a GPU port may require a different
  compiler/flags in `Makefile` than the current `clang` default. I left
  `Makefile` **off** the hard frozen list and documented it, rather than
  unilaterally freezing it.

## How throughput is measured (frozen harness)
`getp()` in `getp_eval.c`:
1. read `requests.txt` (line 0 = request count, then N prompt-text lines);
2. `warm_up(t)` — timed separately (excluded from throughput);
3. time `inference(t, &reqs)` → total generated tokens / elapsed →
   print `achieved throughput TPS (tok/s)`;
4. write the output file: one line per request, space-separated generated token
   ids (same shape as `completions.i32.txt`, so output can be re-scored);
5. `finish(t)` — timed separately.

## Editable-file contract (`getp_run.c`)
```c
void      warm_up(Transformer *t);                    // allocate, upload weights to GPU…
void      finish(Transformer *t);                     // free
long long inference(Transformer *t, Requests *reqs);  // returns total tokens generated
```
Reference impl: per request → tokenize (add_bos=0) → `forward_unabsorbed` prefill
→ greedy `forward_absorbed` until EOS or `max_steps` → store tokens in `reqs`.

## Work items
1. `include/engine.h` — prototypes for `forward_unabsorbed/forward_absorbed/sample`.
2. `include/getp.h` — `Requests` + the `getp/warm_up/finish/inference` API.
3. `src/getp_eval.c` (frozen) — read/time/write/print-throughput harness.
4. `src/getp_run.c` (editable) — CPU reference implementation.
5. `src/run.c` — add the `getp` mode dispatch in `main()` (minimal change).
6. `Makefile` — add the two sources to `run`, add a `make getp MODEL=…` target.
7. `README.md` — Quick start / Task / do-not-modify / how it is graded.

## Verification
- `make` builds cleanly (with the two new files).
- `./run "$DSV" getp tests/eval/dsv2lite/requests.txt /tmp/out.txt` prints
  `achieved throughput TPS (tok/s): …` and writes `/tmp/out.txt`.
- `make eval MODEL=dsv2lite` still passes (the correctness path is untouched).

## Amendment — `make getp-eval` grades the candidate's engine

The Goal above says correctness is "gated by the existing harness (`make eval`)".
That is not right, and it misled candidates. `make eval` drives `run`'s
single-sequence `teacher`/`ppl` modes, which call the **frozen** `forward_*`
kernels in `src/run.c` — it never calls `inference()`. It therefore says nothing
about the candidate's engine, and its one-request-at-a-time shape read as a
requirement: the only structure known to reproduce the golden ids was the
reference's per-request prefill + absorbed-decode loop. That is the opposite of a
throughput task.

`make getp-eval` fixes this. Work item 3 step 4 already had the harness write one
line of generated ids per request, "so output can be re-scored" — nothing in the
frozen harness needed to change. `tests/eval/eval.py --tokens` scores that file, so
`inference()` is graded on its own output and the scoring is blind to how the batch
was scheduled. Batched prefill, continuous batching, and several requests in flight
are all fair game.

The gate is the announced accuracy gate (METEOR + BERTScore-F1, `threshold.json`).
Free-run prefix agreement prints as a diagnostic only: greedy decoding is a hard
argmax and a free-run sequence never recovers from one flip, so an engine using
bf16/fp8 weights or a bf16 KV cache diverges from the fp32 reference while still
being correct. Sameness cannot gate.

`DATA=` points the eval targets at any dataset dir of the same shape, including the
published request set fetched by `make eval-fetch`.
