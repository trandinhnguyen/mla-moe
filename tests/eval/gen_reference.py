"""Generate the frozen golden eval dataset from HF (greedy, fp32).

    uv run python gen_reference.py [dsv2lite|glm47] [options]

The golden reference for the correctness eval gate. For each prompt in
`requests.txt` it:
  1. tokenizes with the model's HF tokenizer (add_special_tokens=False, matching
     the C engine's no-BOS contract and the oracle),
  2. greedy-generates `--max-new` tokens (do_sample=False, fp32, eager),
  3. records the HF teacher-forced NLL over the FULL sequence (prompt+completion),

then freezes everything device-neutrally under `<model>/`:

  requests.txt          human-readable prompts (first line = count, then N lines)
  prompts.i32.txt       canonical prompt token ids, one space-separated line each
  completions.i32.txt   greedy completion token ids, one space-separated line each
  reference.json        per-request {prompt_len, completion_len, hf_nll, hf_ntok}
                        + provenance (model_dir, max_new, transformers version)

Because the completion is HF's own greedy free-run, completion token c[i] IS HF's
argmax given everything before it. So an engine's teacher-forced argmax over the
completion region agreeing with these ids == agreeing with HF's greedy choice --
which is why eval.py needs no live HF. The frozen hf_nll does the same for PPL.
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "oracle"))
import load_oracle  # noqa: E402  (model_dir single-sourced from the oracle manifest)

import transformers  # noqa: E402


_RUN_C_MAX_IDS = 4096      # src/run.c read_tokens_i32 reads at most this many


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="dsv2lite", choices=["dsv2lite", "glm47"])
    p.add_argument("-r", "--requests", default=None,
                   help="requests file (default <model>/requests.txt)")
    p.add_argument("-o", "--out", default=None, help="output dir (default <model>/)")
    p.add_argument("--max-new", type=int, default=64, help="tokens to greedy-generate per prompt")
    p.add_argument("--max-tokens", type=int, default=_RUN_C_MAX_IDS,
                   help="C-engine buffer cap (full seq)")
    args = p.parse_args()
    # src/run.c reads at most _RUN_C_MAX_IDS ids. tests/eval/eval.py refuses a
    # dataset above min(that limit, the model's KV cache), so this bound is the
    # looser of the two -- a set inside it can still be refused by a model with a
    # smaller max_position_embeddings.
    if not 2 <= args.max_tokens <= _RUN_C_MAX_IDS:
        p.error(f"--max-tokens must be in [2, {_RUN_C_MAX_IDS}] "
                f"({_RUN_C_MAX_IDS} is the read limit in src/run.c; "
                f"below 2 no prompt fits)")
    if args.max_new < 1:
        p.error("--max-new must be >= 1")
    return args


def read_requests(path):
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    if not lines:
        sys.exit(f"empty requests file: {path}")
    n = int(lines[0].strip())
    prompts = lines[1:1 + n]
    if len(prompts) != n:
        sys.exit(f"requests count {n} != {len(prompts)} prompt lines in {path}")
    return prompts


def hf_full_nll(model, ids):
    """Teacher-forced sum-NLL over positions [0, len-1): p predicts ids[p+1]."""
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    nll = torch.nn.functional.cross_entropy(
        logits[:-1, :], ids[0, 1:], reduction="sum")
    return float(nll), int(ids.shape[1] - 1)


def main():
    args = parse_args()
    model_dir = load_oracle.manifest(os.path.join(_HERE, "..", "oracle", "dumps", args.model))["model_dir"]
    out_dir = args.out or os.path.join(_HERE, args.model)
    req_path = args.requests or os.path.join(out_dir, "requests.txt")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(req_path):
        sys.exit(f"no requests file: {req_path} (create one; first line = count)")

    prompts = read_requests(req_path)
    print(f"[{args.model}] {model_dir}  ({len(prompts)} prompts, max_new={args.max_new})", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    print("  loading HF model (fp32, eager) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True)
    model.eval()

    records, prompt_lines, completion_lines = [], [], []
    for i, text in enumerate(prompts):
        pids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        plen = pids.shape[1]
        room = args.max_tokens - plen
        if room < 2:
            sys.exit(f"prompt {i} already fills the {args.max_tokens}-token buffer")
        max_new = min(args.max_new, room)
        with torch.no_grad():
            out = model.generate(pids, do_sample=False, max_new_tokens=max_new,
                                 use_cache=True, pad_token_id=tok.eos_token_id)
        cids = out[0, plen:].tolist()
        full = torch.cat([pids, torch.tensor([cids])], dim=1)
        nll, ntok = hf_full_nll(model, full)
        # The dataset is the frozen artefact of the exam: a non-finite or
        # unusable nll here would be baked in, and every later `make eval` would
        # report it as a misconfiguration of the grader's machine. Fail now.
        # The 709.0 must equal _EXP_MAX in tests/eval/eval.py: past that mean,
        # math.exp overflows and the reader rejects the record as a dataset
        # defect -- which is exactly what this test exists to prevent.
        if (not (nll == nll and abs(nll) != float("inf")) or ntok < 1 or nll < 0
                or nll > 709.0 * ntok):
            sys.exit(f"prompt {i}: unusable teacher-forced nll {nll!r} over {ntok} "
                     f"tokens -- refusing to freeze it into reference.json")
        records.append({"prompt_len": plen, "completion_len": len(cids),
                        "hf_nll": nll, "hf_ntok": ntok})
        prompt_lines.append(" ".join(map(str, pids[0].tolist())))
        completion_lines.append(" ".join(map(str, cids)))
        print(f"  [{i}] plen={plen} clen={len(cids)} first_completion_ids={cids[:8]}", flush=True)

    with open(os.path.join(out_dir, "prompts.i32.txt"), "w") as f:
        f.write("\n".join(prompt_lines) + "\n")
    with open(os.path.join(out_dir, "completions.i32.txt"), "w") as f:
        f.write("\n".join(completion_lines) + "\n")
    with open(os.path.join(out_dir, "reference.json"), "w") as f:
        json.dump({"model": args.model, "model_dir": model_dir,
                   "max_new": args.max_new,
                   "transformers_version": transformers.__version__,
                   "requests": records}, f, indent=2)
    print(f"  wrote prompts.i32.txt, completions.i32.txt, reference.json -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
