# FROZEN exam toolchain: hipcc targeting MI250 (gfx90a / CDNA2). The CPU
# reference builds and runs as ordinary host code under hipcc on day one;
# candidates add HIP device kernels in src/getp_run.hip against this same target.
# Do not edit — the Makefile is on the frozen list (see README).
CC      = hipcc
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wpedantic --offload-arch=gfx90a \
          -Iinclude -Ivendor
LDFLAGS = -lm

# Build with `make DUMP=1` to compile in the oracle-validation dumps
# (forward_*'s intermediate writes). Off by default: no dump code in the binary.
ifdef DUMP
CFLAGS += -DMLA_ENABLE_DUMP
endif

# HIP device flags: compile the candidate's getp_run TU as C++/HIP. NO -std=c11
# (fatal in HIP mode: "invalid argument '-std=c11' not allowed with 'HIP'") and
# no -Wpedantic (warns on HIP extensions). -ffp-contract=off keeps kernel fp32
# rounding aligned with the CPU reference so `make eval` stays within tolerance;
# a candidate may drop it for speed at their own correctness risk.
HIPFLAGS = -O2 -Wall -Wextra -ffp-contract=off --offload-arch=gfx90a \
           -Iinclude -Ivendor
ifdef DUMP
HIPFLAGS += -DMLA_ENABLE_DUMP
endif

LIB_SRCS = src/safetensors_loader.c \
           src/tensor.c

# Two binaries:
#   run      — llama2.c-style inference entry point (src/run.c)
#   mla-moe  — weight inspection CLI (src/main.c)
# Frozen C reference/host TUs (compiled with -std=c11). getp_run is NOT here — it
# is the candidate's HIP/C++ TU so __global__ kernels can live in it.
RUN_C_SRCS = $(LIB_SRCS) vendor/cJSON.c src/tokenizer.c src/model_load.c src/dump.c \
             src/getp_eval.c src/run.c
# Candidate-editable HIP TU (+ optional extra kernel files under src/kernels/).
# Drop the $(wildcard ...) if you want strictly one editable file.
HIP_SRCS   = src/getp_run.hip $(wildcard src/kernels/*.hip)
RUN_C_OBJS = $(RUN_C_SRCS:.c=.o)
HIP_OBJS   = $(HIP_SRCS:.hip=.o)
TOOL_SRCS  = $(LIB_SRCS) src/main.c

.PHONY: all clean tok-cli eval eval-gen eval-fetch eval-warm ref-binary bench getp getp-eval print-getp-steps

# For grading scripts: the steps value the targets would use, so no third copy.
print-getp-steps:
	@test "$(GETP_STEPS)" -ge 1 2>/dev/null || { echo "STEPS must be an integer >= 1 (got '$(GETP_STEPS)') -- or src/getp_eval.c is unreadable" >&2; exit 1; }
	@echo $(GETP_STEPS)

all: run mla-moe

# --- correctness eval -----------------------------------------------------
# MODEL selects the dataset under tests/eval/<MODEL>/.
MODEL ?= dsv2lite

# Regenerate the frozen golden dataset from HF (greedy, fp32). Heavy: needs the
# HF model + RAM. Run once per model; commit the resulting *.i32.txt/reference.json.
eval-gen: run
	uv run python tests/eval/gen_reference.py $(MODEL)

# RUN points at the engine binary, so a grader can score a submitted or
# alternately-built binary (RUN=./run-ref, RUN=./submission) with any target
# below. Building the working tree is only a prerequisite when RUN is that build:
# a submitted binary must be gradeable on a tree that does not compile.
# This block must stay ABOVE every target that uses GETP_DEPS -- make expands
# prerequisites when it reads the rule, so a reference from higher up is empty.
RUN      ?= ./run
# `test -x run` passes (the shell resolves it against cwd) but `run ...` as a
# command searches PATH, which does not hold cwd -- exit 127 at run time, which
# `make -n` cannot show. Make the path explicit for every bare-word spelling.
RUN_BIN   = $(if $(findstring /,$(RUN)),$(RUN),./$(RUN))
# Match by name AND by resolved path, so `run`, `./run`, `$(CURDIR)/run` and any
# symlinked alias of the same file all rebuild. realpath is empty before the first
# build, which is why the textual filter stays as the fallback.
GETP_DEPS = $(if $(filter run ./run $(CURDIR)/run,$(RUN))$(filter $(realpath ./run),$(realpath $(RUN))),run,)
# The steps count has ONE definition, read from the C harness so the two cannot
# drift: it is passed to the binary and to eval.py --steps. The path is anchored
# to THIS makefile's directory, not $(CURDIR) -- make sets CURDIR to the working
# directory. That anchor covers THIS READ ONLY: DATA and the scorer path stay
# relative to the working directory, so run make from the repo root.
# Known limit: make variables are space-separated lists, so $(firstword
# $(MAKEFILE_LIST)) truncates if this file's own path contains a space, and the
# read then yields nothing. Quoting cannot fix that. The guard below turns it
# into a clean stop, so no timed run ever gets a wrong count.
# The recipes assert the result is a usable integer before spending a timed run.
GETP_MK_DIR = $(dir $(firstword $(MAKEFILE_LIST)))
GETP_DEFAULT_STEPS = $(shell sed -n 's/^\#define GETP_DEFAULT_STEPS *\([0-9]*\).*/\1/p' \
                       "$(GETP_MK_DIR)src/getp_eval.c")
GETP_STEPS = $(if $(STEPS),$(STEPS),$(GETP_DEFAULT_STEPS))

# DATA selects the dataset dir. Defaults to the small in-repo dev set; point it at
# a fetched public/private set (see eval-fetch) to grade on the real request mix.
DATA ?= tests/eval/$(MODEL)

# Fetch the participant-facing 512-prompt set from the Hub into
# tests/eval/fetched/<set> (gitignored — a published artifact, not repo state).
# Same file layout as the in-repo dev set, so every target below takes it via
# DATA=. The directory is derived from the set name, so a private set never lands
# on top of the public one: the path on disk always says which set it holds.
PUBLIC_SET ?= thanhnx12/mla-moe-dataset-public
FETCH_DIR   = tests/eval/fetched/$(notdir $(PUBLIC_SET))
eval-fetch:
	@for m in dsv2lite glm47; do mkdir -p "$(FETCH_DIR)/$$m"; \
	  for f in requests.txt prompts.i32.txt completions.i32.txt reference.json manifest.json; do \
	    curl -sSLf -o "$(FETCH_DIR)/$$m/$$f" \
	      "https://huggingface.co/datasets/$(PUBLIC_SET)/resolve/main/$$m/$$f" \
	      || { echo "fetch failed: $$m/$$f"; exit 1; }; \
	  done; done
	@echo "fetched $(PUBLIC_SET) (split: $$(sed -n 's/.*\"split\": *\"\([^\"]*\)\".*/\1/p' \
	  "$(FETCH_DIR)/dsv2lite/manifest.json")) -> $(FETCH_DIR)/"
	@echo "use DATA=$(FETCH_DIR)/<model>"

# Pre-fill the Hub and nltk caches that the accuracy tier needs (metric scripts,
# roberta-large, wordnet/punkt). Run once on a machine that has network; grading
# can then run with HF_HUB_OFFLINE=1. Without this the gate needs the network.
eval-warm:
	uv run --extra fuzzy python -c "import evaluate; \
	  evaluate.load('meteor').compute(predictions=['a b c'], references=['a b d']); \
	  evaluate.load('bertscore').compute(predictions=['a b c'], references=['a b d'], \
	    lang='en'); print('accuracy-tier caches warm')"

# Score the C engine against the frozen dataset: teacher-forced top-1 (both
# paths) + perplexity rel-err. Add FUZZY=1 for the METEOR/BERTScore free-run tier.
# MODELDIR is needed when the dataset's reference.json records a Hub repo id
# rather than a local path (the public set does).
eval: $(GETP_DEPS)
	uv run $(if $(FUZZY),--extra fuzzy,) python tests/eval/eval.py $(MODEL) -d "$(DATA)" \
	  $(if $(RUN),-r "$(RUN_BIN)",) \
	  $(if $(MODELDIR),--model-dir "$(MODELDIR)",) $(if $(FUZZY),--fuzzy,)

# --- performance benchmark ------------------------------------------------
# Device-agnostic prefill/decode perf. Point -r/RUN at any engine build (CPU,
# run-ref, or a future GPU binary). Override PREFILL/DECODE/REPS/OUT as needed.
bench: $(GETP_DEPS)
	uv run python tests/bench/bench.py $(MODEL) \
	  $(if $(RUN),-r "$(RUN_BIN)",) $(if $(PREFILL),--prefill "$(PREFILL)",) \
	  $(if $(DECODE),--decode "$(DECODE)",) $(if $(REPS),--reps "$(REPS)",) \
	  $(if $(OUT),-o "$(OUT)",) $(if $(COMPARE),--compare "$(COMPARE)",)

# Batch-throughput grading (the perf score). Runs the fixed request set through
# the candidate's inference() (src/getp_run.hip) and prints one tok/s number.
# MODEL selects tests/eval/<MODEL>/requests.txt; MODELDIR points at the weights
# (defaults to $DSV / $GLM per model). Override STEPS/OUT as needed.
MODELDIR ?= $(if $(filter glm47,$(MODEL)),$(GLM),$(DSV))
# Output path is per (model, dataset) and inside the tree, not a fixed /tmp name:
# a shared machine gives EACCES on another user's file, and two runs of one user
# would otherwise overwrite the ids that getp-eval then grades.
GETP_OUT  = $(if $(OUT),$(OUT),$(CURDIR)/getp_$(MODEL)_$(subst /,_,$(DATA)).txt)
getp: $(GETP_DEPS)
	@test -n "$(RUN)" -a -x "$(RUN_BIN)" || { echo "no engine binary at RUN=$(RUN)"; exit 1; }
	@test "$(GETP_STEPS)" -ge 1 2>/dev/null || { echo "STEPS must be an integer >= 1 (got '$(GETP_STEPS)') -- or src/getp_eval.c is unreadable"; exit 1; }
	@test -n "$(MODELDIR)" || { echo "set MODELDIR=<model_dir> (or DSV=/GLM=)"; exit 1; }
	"$(RUN_BIN)" "$(MODELDIR)" getp "$(DATA)/requests.txt" \
	  "$(GETP_OUT)" $(GETP_STEPS)

# Correctness gate for the CANDIDATE'S engine: run the timed getp batch, then
# score the token ids it wrote against the frozen golden completions. `make eval`
# only exercises the frozen single-sequence paths in run.c, so it says nothing
# about inference(); this scores inference()'s OWN output and is agnostic to how
# it produced it -- batched, continuous-batched, or one request at a time.
# GETP_STEPS is the single source for the steps value: it is passed to the binary
# AND to eval.py --steps, so the hint can never disagree with what actually ran.
# Its default comes from GETP_DEFAULT_STEPS in src/getp_eval.c (read above), so
# this file holds no copy of the number.
# The gate is the announced accuracy gate (METEOR + BERTScore-F1); prefix agreement
# prints as a diagnostic and does not decide the verdict, because a bf16/fp8 engine
# legitimately diverges from the fp32 reference. (The 0.90 BERTScore limit still gives
# the gate a partial prefix-agreement effect -- see the README accuracy-gate paragraph.)
# QUICK=1 skips the accuracy tier and
# its heavy deps, printing diagnostics only (exit 2 -- it grades nothing).
getp-eval: $(GETP_DEPS)
	@test -n "$(RUN)" -a -x "$(RUN_BIN)" || { echo "no engine binary at RUN=$(RUN)"; exit 1; }
	@test "$(GETP_STEPS)" -ge 1 2>/dev/null || { echo "STEPS must be an integer >= 1 (got '$(GETP_STEPS)') -- or src/getp_eval.c is unreadable"; exit 1; }
	@test -n "$(MODELDIR)" || { echo "set MODELDIR=<model_dir> (or DSV=/GLM=)"; exit 1; }
	"$(RUN_BIN)" "$(MODELDIR)" getp "$(DATA)/requests.txt" \
	  "$(GETP_OUT)" $(GETP_STEPS)
	uv run $(if $(QUICK),,--extra fuzzy) python tests/eval/eval.py $(MODEL) \
	  -d "$(DATA)" --tokens "$(GETP_OUT)" --model-dir "$(MODELDIR)" \
	  --steps $(GETP_STEPS) $(if $(QUICK),--quick,)

# Build the golden CPU reference binary `run-ref` from a TAGGED commit, isolated
# from working-tree edits, so the GPU/HIP port always has a fixed, buildable
# oracle to diff against: the golden reference is a tagged, buildable binary,
# not the current run.c. Tag first: git tag $(REF).
REF ?= cpu-oracle-v1
ref-binary:
	@git rev-parse --verify "$(REF)^{commit}" >/dev/null 2>&1 || \
	  { echo "ref '$(REF)' not found -- tag the validated commit first: git tag $(REF)"; exit 1; }
	rm -rf .ref-build && mkdir -p .ref-build && git archive "$(REF)" | tar -x -C .ref-build
	$(MAKE) -C .ref-build run CC="$(CC)"
	cp .ref-build/run run-ref && rm -rf .ref-build
	@echo "built run-ref from $(REF)"

# Standalone tokenizer harness (no model weights) for tests/tokenizer/compare_hf.py
tok-cli: src/tokenizer.c vendor/cJSON.c tests/tokenizer/tok_cli.c
	$(CC) $(CFLAGS) $^ -o tests/tokenizer/tok_cli $(LDFLAGS)

# Separate compilation: C TUs keep -std=c11; the HIP TU compiles without it (it
# is fatal in HIP mode). Never put .c sources and .o objects on the same hipcc
# line — its injected -x c sticks to the trailing .o and breaks the build; the
# link recipes below use objects only.
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.hip
	$(CC) $(HIPFLAGS) -c $< -o $@

run: $(RUN_C_OBJS) $(HIP_OBJS)
	$(CC) $(RUN_C_OBJS) $(HIP_OBJS) -o $@ $(LDFLAGS)

mla-moe: $(TOOL_SRCS)
	$(CC) $(CFLAGS) $(TOOL_SRCS) -o $@ $(LDFLAGS)

# ASan/UBSan debug build. Distinct .dbg.o object names so the -O2 `run` objects
# are not silently reused with sanitizer flags (make would see them up-to-date).
SANFLAGS = -O0 -g -fsanitize=address,undefined
debug-run: CFLAGS   += $(SANFLAGS)
debug-run: HIPFLAGS += $(SANFLAGS)
debug-run: $(RUN_C_OBJS:.o=.dbg.o) $(HIP_OBJS:.o=.dbg.o)
	$(CC) $(SANFLAGS) $(RUN_C_OBJS:.o=.dbg.o) $(HIP_OBJS:.o=.dbg.o) -o run $(LDFLAGS)

%.dbg.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.dbg.o: %.hip
	$(CC) $(HIPFLAGS) -c $< -o $@

debug-tool: CFLAGS += -O0 -g -fsanitize=address,undefined
debug-tool: $(TOOL_SRCS)
	$(CC) $(CFLAGS) $(TOOL_SRCS) -o mla-moe $(LDFLAGS)

clean:
	rm -f run mla-moe $(RUN_C_OBJS) $(HIP_OBJS) \
	      $(RUN_C_OBJS:.o=.dbg.o) $(HIP_OBJS:.o=.dbg.o)
