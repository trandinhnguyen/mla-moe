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
  before any `uv run python ...` command below.
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

**Correctness gate:** raising throughput must not change the output. `make eval
MODEL=...` (teacher-forced top-1 + perplexity; `--fuzzy` adds METEOR/BERTScore)
is the correctness gate and must keep passing its thresholds.

## Limitations

Single stream (batch=1), greedy sampling, ASCII/English tokenizer (see above).
Performance work (blocked matmuls, threading) is deferred — correctness first.

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
