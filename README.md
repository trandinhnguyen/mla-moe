# MLA-MoE

An educational, from-scratch C inference engine for transformer models that use
Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE) — the
DeepSeek-V2 architecture family. In the spirit of Andrej Karpathy's llama2.c:
one readable C codebase where every operation is explicit and auditable.

**Targets:** DeepSeek-V2-Lite (`deepseek_v2`) and GLM-4.7-Flash (`glm4_moe_lite`).

## Environment setup

- **Toolchain**: the Makefile is **frozen** to `hipcc --offload-arch=gfx90a`
  (MI250 / CDNA2). The CPU reference builds and runs as ordinary host code under
  it; candidates add HIP device kernels against this same target. (For CPU-only
  dev off an MI250 box you can `make CC=clang` locally, but the frozen Makefile
  is what grades.)
- **Python tooling**: `uv sync` installs the pinned deps from `pyproject.toml`
  (numpy, safetensors, transformers, accelerate, torch-cpu) — run this once
  before any `uv run python ...` command below. The METEOR/BERTScore accuracy
  tier needs the `fuzzy` extra (`uv sync --extra fuzzy`); `make getp-eval` and
  `make eval FUZZY=1` pull it in for you. That tier also downloads metric scripts,
  nltk data and roberta-large on first use. Run `make eval-warm` once on a
  networked machine to fill those caches; after that `HF_HUB_OFFLINE=1 make
  getp-eval ...` works offline (verified — `HF_DATASETS_OFFLINE` is not needed).
  Skipping `eval-warm` on an offline box makes the accuracy tier exit with a
  message naming the target, rather than a traceback.
- **Model weights**: checkpoints are plain HF directories (`config.json` +
  `model.safetensors.index.json` + shards), mmap'd directly — no download or
  conversion step. On the shared cluster they live at
  `/remote/vast0/share-mv/<hf-org>/<hf-model>`, e.g.
  `/remote/vast0/share-mv/deepseek-ai/DeepSeek-V2-Lite` and
  `/remote/vast0/share-mv/zai-org/GLM-4.7-Flash` (same convention used by
  `tests/eval/*/reference.json` and `tests/oracle/gen_oracle.py`). Export
  `DSV=`/`GLM=` pointing at those paths, or at your own local copy of the same
  layout.

## Build

```sh
make                 # optimized; builds `run` (engine) and `mla-moe` (weight inspector)
make DUMP=1          # + -DMLA_ENABLE_DUMP: writes oracle-named intermediates for validation
```

Default builds carry no dump code (it is `#ifdef`-gated). **Run `make clean`
when toggling `DUMP=1`** — make does not rebuild on a CFLAGS-only change.

Debug builds (ASan + UBSan): `make debug-run` / `make debug-tool`.

## Run

```
run <model_dir> [tokens.i32.bin | -p "text"] [dump_dir|ppl]
```

`<model_dir>` holds `config.json`, `model.safetensors.index.json`, and the
shards; the model family is auto-detected from `config.json`. `tokens.i32.bin`
is raw little-endian int32 token ids (the oracle's `input_ids.i32.bin` files are
exactly this). Alternatively, `-p "text"` tokenizes a prompt with the in-C
tokenizer (reads `tokenizer.json`; pre-tokenizer chosen from `config.json`), and
generated ids are decoded back to text.

See "Environment setup" above for where `DSV`/`GLM` actually point on this cluster.

```sh
DSV=/path/to/deepseek-ai/DeepSeek-V2-Lite
GLM=/path/to/zai-org/GLM-4.7-Flash

# weight-load smoke test (no tokens)
./run "$DSV"

# prefill + greedy absorbed-decode; prints generated token ids
./run "$DSV" tests/oracle/dumps/dsv2lite/input_ids.i32.bin

# same, but from a text prompt (tokenized + detokenized in C)
./run "$DSV" -p "The capital of France is"
./run "$GLM" -p "The capital of France is"

# (DUMP=1 build) prefill + one decode step, writing intermediates to <dump_dir>
./run "$GLM" tests/oracle/dumps/glm47/input_ids.i32.bin /tmp/cdump
```

## Tokenizer

`-p "text"` drives an in-C byte-level BPE tokenizer (GPT-2 family) that reads
the model's `tokenizer.json` (vocab + merges + added tokens) and `config.json`
(bos/eos, model family).  Prompts are encoded with **no BOS**, matching the
oracle's tokenization and the validated reference path.

It is **faithful to HuggingFace for ASCII/English text**. The pre-tokenizer's
non-ASCII letter/punctuation/CJK handling is intentionally omitted, so non-ASCII
input still yields valid tokens but may split differently from HF.

Cross-check the C tokenizer against HuggingFace on an ASCII panel (no model
weights loaded):

```sh
make tok-cli   # builds tests/tokenizer/tok_cli
uv run python tests/tokenizer/compare_hf.py "$DSV" ./tests/tokenizer/tok_cli
uv run python tests/tokenizer/compare_hf.py "$GLM" ./tests/tokenizer/tok_cli
```

## Validate against the oracle

The oracle (`tests/oracle/`) dumps ground-truth activations from transformers in
fp32. Run a `DUMP=1` build, then diff:

```sh
make clean && make DUMP=1
./run "$DSV" tests/oracle/dumps/dsv2lite/input_ids.i32.bin /tmp/cdump
cd tests/oracle
uv run python compare_prefill.py /tmp/cdump dsv2lite   # 0 failures; argmax 8913 (" Paris")
uv run python compare_decode.py  /tmp/cdump dsv2lite
```

Use `glm47` + `$GLM` for the other model (argmax 12089). Regenerate the dumps
with `uv run python tests/oracle/gen_oracle.py <dsv2lite|glm47>` (see
`tests/oracle/README.md`; glm47 needs ~125 GB RAM).

## Correctness eval

The primary correctness gate scores the engine against a **frozen golden dataset**
of greedy sequences generated once from HF (fp32), not per-tensor dumps. Three
device-neutral metrics: teacher-forced top-1 agreement, perplexity relative error,
and (optional) free-run METEOR/BERTScore.

```sh
# 1. generate the golden dataset from HF (greedy, fp32); run once per model
make eval-gen MODEL=dsv2lite        # writes tests/eval/dsv2lite/{prompts,completions}.i32.txt, reference.json

# 2. score the C engine (teacher-forced top-1 both paths + ppl rel-err)
make eval MODEL=dsv2lite            # add FUZZY=1 for the METEOR/BERTScore tier
```

The engine exposes the eval modes directly:

```sh
./run "$DSV" seq.i32.bin teacher 5  # teacher-forced top-1: 'P'/'D' rows, decode region pos>=5
./run "$DSV" prompt.i32.bin gen 64  # greedy free-run; prints 'completion <ids>'
```

For the GPU/HIP port, freeze the validated CPU engine as a **tagged, buildable**
reference (not the mutating `run.c`): `git tag cpu-oracle-v1 && make ref-binary`
builds `run-ref` from that tag to diff kernels against. The per-tensor oracle
(`tests/oracle/`, below) is retained as the bring-up microscope for localizing a
failing kernel.

## Benchmark

Performance is measured as **two separate regimes** — they have opposite compute
profiles: prefill is compute-bound dense GEMM, decode is memory-bound with
per-token MoE routing. The engine's `bench` mode owns the wall-clock timing;
`tests/bench/bench.py` sweeps prefill lengths and reports both.

```sh
make bench MODEL=dsv2lite                       # sweep default prefill lengths
make bench MODEL=glm47 PREFILL=256,1024 DECODE=32 REPS=7 OUT=/tmp/glm.json
```

Reported per prefill length: prefill tok/s (`prefill_ms` **is** time-to-first-token,
since the first output token is the argmax of prefill's logits), decode tok/s, and
TPOT (time-per-output-token). rep 0 is a warmup and excluded from the medians.

The timing is **device-agnostic**: it brackets the forward calls inside `run`,
whose host-side logits force any backend to synchronize at the clock boundary.
So the same harness measures a CPU build, `run-ref`, or a future GPU/HIP build —
just point `-r`/`RUN` at the binary. `--compare` prints per-regime speedup, the
perf analogue of the `run` vs `run-ref` correctness diff:

```sh
uv run python tests/bench/bench.py dsv2lite -r ./run-ref -o /tmp/base.json
uv run python tests/bench/bench.py dsv2lite -r ./run     -o /tmp/cur.json --compare /tmp/base.json
```

## Candidate task & throughput grading (`getp`)

This is the intern exam surface: a working CPU baseline that you optimize by
porting the compute to the GPU.

**You modify exactly one file: `src/getp_run.hip`.** It implements `warm_up()`,
`finish()`, and `inference()` (contract in `include/getp.h`). It compiles as a
HIP/C++ translation unit, so you write `__global__` kernels and launch them with
`<<<>>>` directly in it; to split kernels across files, add `src/kernels/*.hip`
(picked up automatically, no Makefile edit). Everything else is
frozen — `src/run.c` and its forward kernels, `src/getp_eval.c` (the timing
harness), `model_load.c`, `tokenizer.c`, `main.c`, `tests/`, and `include/*`.
The frozen CPU kernels are declared in `include/engine.h`; the reference
`inference()` calls them, so it is correct on day one. Replace those calls with
your own GPU kernels incrementally.

The request set you are scored on is published as
[`thanhnx12/mla-moe-dataset-public`](https://huggingface.co/datasets/thanhnx12/mla-moe-dataset-public)
— 512 prompts per model, prompt lengths 64–512, 64-token greedy fp32 references.
Grading uses a held-out private set of the same shape, so tune against the public
one, not against individual prompts:

```sh
make eval-fetch      # -> tests/eval/fetched/mla-moe-dataset-public/{dsv2lite,glm47}
make getp-eval MODEL=dsv2lite MODELDIR="$DSV" STEPS=64 \
  DATA=tests/eval/fetched/mla-moe-dataset-public/dsv2lite
```

`DATA` selects the dataset dir for `eval`/`getp`/`getp-eval`; it defaults to the
small in-repo dev set, which is for smoke-testing, not for tuning. The fetch
directory is named after the set, so two sets never overwrite each other.

`getp` mode runs a fixed request set (`requests.txt`: line 0 = count, then one
prompt per line) through your `inference()` and prints one end-to-end number —
the perf score:

```sh
DSV=/path/to/deepseek-ai/DeepSeek-V2-Lite
./run "$DSV" getp tests/eval/dsv2lite/requests.txt /tmp/out.txt 128
# -> "achieved throughput TPS (tok/s): <score>"; writes generated ids to /tmp/out.txt

make getp MODEL=dsv2lite MODELDIR="$DSV"     # convenience wrapper (STEPS/OUT overridable)
```

`warm_up()`/`finish()` are timed separately and excluded from the throughput
number — do allocation and weight upload there, not inside `inference()`.

**`inference()` owns the whole batch.** It is handed all `num_reqs` prompts at
once and is free to schedule them however it likes — one at a time, batched
prefill, continuous batching, paged KV, several requests in flight. The only
contract is that `out_tokens[r][0..out_lens[r]-1]` ends up holding request `r`'s
greedy continuation. The reference implementation loops one request at a time
because that is the simplest correct start, *not* because the grader requires it;
note that the frozen `forward_unabsorbed`/`forward_absorbed` kernels drive a
single shared `RunState` KV cache, so batching means writing your own kernels and
your own cache — which is the point of the exercise.

**Correctness gate:** raising throughput must not change the output.

```sh
make getp-eval MODEL=dsv2lite MODELDIR="$DSV"    # the gate on YOUR engine
make eval      MODEL=dsv2lite                    # the gate on the frozen reference paths
```

`make getp-eval` runs the timed `getp` batch and then scores the token-ids file it
wrote against the golden completions (`<dataset>/completions.i32.txt`), so **your
`inference()` is graded on its own output** and the scoring is blind to how you
scheduled the batch.

The **gate** is the accuracy gate: `meteor >= 0.25` and `bertscore_f1 >= 0.90`
(`tests/eval/threshold.json`). Free-run **prefix agreement** — how many tokens each
request emits before diverging from the reference continuation — prints alongside as
a **diagnostic** and does not decide the verdict. It cannot: one flipped argmax
derails every token after it, so an engine using bf16/fp8 weights or a bf16 KV cache
diverges from the fp32 reference while still being correct. Note though that the 0.90
raw BERTScore limit gives the gate a partial prefix-agreement effect anyway: an engine
that diverges inside the first half of a completion can fail it. The exam accepts that
as a property of the chosen value. `QUICK=1` prints the
diagnostics and skips the accuracy tier's heavy deps (it grades nothing, and exits 2).

**Exit codes are `tests/eval/eval.py`'s, not `make`'s.** GNU make reports its own
status 2 for any failed recipe, so a script that drives `make getp-eval` cannot tell
a gate failure from an environment fault. A grading script should run the timed
batch with `make` and then call the scorer directly:

```sh
OUT="$PWD/getp_ids.txt"                  # one path, used by both commands
STEPS=$(make -s print-getp-steps)        # the harness default, or set your own
make getp MODEL=dsv2lite MODELDIR="$DSV" DATA="$DATA" OUT="$OUT" STEPS="$STEPS"
uv run --extra fuzzy python tests/eval/eval.py dsv2lite \
  -d "$DATA" --tokens "$OUT" --model-dir "$DSV" --steps "$STEPS"   # exit code below
```

Set `OUT` explicitly rather than letting it default — `make` derives the ids path from
`MODEL` and `DATA` and never prints it. Read `STEPS` from `make print-getp-steps`
rather than writing a number here: the default lives in `src/getp_eval.c`, and the
scorer needs the same value the run used or its cap check describes a run that did
not happen.

**0** ok · **1** the gate failed · **2** not graded (nothing was scored — `--quick`, a
misconfiguration, or a generation capped below the reference length) · **3**
environment fault (missing deps, cold cache, no network, no engine binary). Only
**1** is a candidate failure.

`STEPS`/`OUT`/`RUN` are overridable as with `make getp` (`RUN=./run-ref` or
`RUN=./submission` scores another binary); generating past the golden completion
length is fine, the surplus is ignored, so one timed run yields both numbers.

Read the prefix diagnostics even though they do not gate. A correct fp32 engine
matches the reference token-for-token; a reduced-precision engine diverges but stays
fluent and on-topic. Prefix agreement that collapses while your throughput jumps is
the signal that something broke, and the accuracy gate alone will not tell you.
Submissions are additionally reviewed by hand.

`make eval` still runs the teacher-forced top-1 + perplexity ladder, but note what
it covers: it drives `run`'s single-sequence `teacher`/`ppl` modes, i.e. the
**frozen** kernels in `src/run.c` — it never calls your `inference()`. Keep it
passing as the reference-path regression; `make getp-eval` is what grades you.

## Limitations

The reference engine in `src/run.c` is single-stream (batch=1) and greedy, with an
ASCII/English tokenizer (see above); its performance work (blocked matmuls,
threading) is deliberately absent — correctness first. Those are properties of the
reference, not limits on the `getp` engine you are asked to build.

## Candidate hand-off — status

The hand-off scaffolding now exists with a frozen/editable split:
`src/getp_eval.c` is the frozen timing harness and `src/getp_run.hip` (+ optional
`src/kernels/*.hip`) is the editable surface. See **Candidate task & throughput
grading** above.

The `Makefile` is **frozen**, pre-set to the GPU toolchain `hipcc
--offload-arch=gfx90a` (MI250 / CDNA2). The CPU reference compiles and runs as
host code under hipcc (verified), so it is correct on day one; candidates write
HIP `__global__` kernels directly in `src/getp_run.hip` (a HIP/C++ TU) against
the same target. The frozen headers `engine.h`/`getp.h`/`tokenizer.h` carry
`extern "C"` guards so the C++ TU links against the C reference. Frozen list:
`Makefile`, `src/run.c` and its kernels, `src/getp_eval.c`, `model_load.c`,
`tokenizer.c`, `main.c`, `tests/`, `include/*` — **editable: `src/getp_run.hip`
(+ `src/kernels/*.hip`).**
