# FROZEN exam toolchain: hipcc targeting MI250 (gfx90a / CDNA2). The CPU
# reference builds and runs as ordinary host code under hipcc on day one;
# candidates add HIP device kernels in src/getp_run.hip against this same target.
# Do not edit — the Makefile is on the frozen list (see README).
CC      = hipcc
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wpedantic --offload-arch=gfx90a \
          -Iinclude -Ivendor -I/opt/rocm-7.2.2/include
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

.PHONY: all clean tok-cli eval eval-gen ref-binary bench getp

all: run mla-moe

# --- correctness eval -----------------------------------------------------
# MODEL selects the dataset under tests/eval/<MODEL>/.
MODEL ?= dsv2lite

# Regenerate the frozen golden dataset from HF (greedy, fp32). Heavy: needs the
# HF model + RAM. Run once per model; commit the resulting *.i32.txt/reference.json.
eval-gen: run
	uv run python tests/eval/gen_reference.py $(MODEL)

# Score the C engine against the frozen dataset: teacher-forced top-1 (both
# paths) + perplexity rel-err. Add FUZZY=1 for the METEOR/BERTScore free-run tier.
eval: run
	uv run python tests/eval/eval.py $(MODEL) $(if $(FUZZY),--fuzzy,)

# --- performance benchmark ------------------------------------------------
# Device-agnostic prefill/decode perf. Point -r/RUN at any engine build (CPU,
# run-ref, or a future GPU binary). Override PREFILL/DECODE/REPS/OUT as needed.
bench: run
	uv run python tests/bench/bench.py $(MODEL) \
	  $(if $(RUN),-r $(RUN),) $(if $(PREFILL),--prefill $(PREFILL),) \
	  $(if $(DECODE),--decode $(DECODE),) $(if $(REPS),--reps $(REPS),) \
	  $(if $(OUT),-o $(OUT),) $(if $(COMPARE),--compare $(COMPARE),)

# Batch-throughput grading (the perf score). Runs the fixed request set through
# the candidate's inference() (src/getp_run.hip) and prints one tok/s number.
# MODEL selects tests/eval/<MODEL>/requests.txt; MODELDIR points at the weights
# (defaults to $DSV / $GLM per model). Override STEPS/OUT as needed.
MODELDIR ?= $(if $(filter glm47,$(MODEL)),$(GLM),$(DSV))
getp: run
	@test -n "$(MODELDIR)" || { echo "set MODELDIR=<model_dir> (or DSV=/GLM=)"; exit 1; }
	./run "$(MODELDIR)" getp tests/eval/$(MODEL)/requests.txt \
	  $(if $(OUT),$(OUT),/tmp/getp_$(MODEL).txt) $(if $(STEPS),$(STEPS),)

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
