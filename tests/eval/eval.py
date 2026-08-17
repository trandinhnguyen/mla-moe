"""Score an engine against the frozen golden dataset.

    uv run python eval.py [dsv2lite|glm47] [options]           # drive `run` directly
    uv run python eval.py [dsv2lite|glm47] --tokens out.txt    # score a token-ids file

Two entry points onto the same frozen `<model>/` dataset (built by gen_reference.py):

A. DEFAULT -- drives the C `run` binary through its single-sequence eval modes:

  1. teacher-forced top-1 agreement  (headline "accuracy") -- needs no HF
  2. perplexity relative error       (C ppl vs frozen HF nll) -- needs no HF
  3. free-run fuzzy (METEOR/BERTScore, optional --fuzzy) -- lexical/semantic

  Top-1 is scored over the COMPLETION region only (pos >= prompt_len): prompt
  positions measure natural-language unpredictability, not engine correctness.
  Both engine paths are scored: 'P' (prefill/unabsorbed) and 'D' (decode/absorbed),
  and BOTH gate on top1_strict. Measured on the frozen CPU build (2026-08-14,
  dsv2lite, full 5-request dev set): decode 100.000% (160/160), prefill 100.000%
  (160/160), worst ppl rel-err 2.695e-05. Same on glm47: decode 100.000%
  (160/160), prefill 100.000% (160/160), worst ppl rel-err 3.030e-05, ppl
  token-count mismatch 0/5. "160/160" is the NUMBER of scored rows, equal on
  both paths and enforced per request; the position values are not compared.
  The 0.99 threshold was calibrated against the decode kernel; those runs are
  the evidence for applying it to the prefill kernel too, which uses a different
  forward and accumulates different rounding.
  A mismatch with logit gap <= --tie is a numerical tie, not an error (tie-tolerant
  column); the strict column gates.

B. --tokens FILE -- scores a generated-token-ids file (one line of space-separated
  ids per request) against completions.i32.txt. This is the getp gate: the batch
  harness (src/getp_eval.c) writes exactly this file after timing inference(), so
  the graded engine is scored on ITS OWN output. It makes no assumption about how
  those ids were produced -- batched, continuous-batched, or one request at a time.

  The GATE is the announced accuracy gate: METEOR + BERTScore-F1. Free-run prefix
  agreement (tokens before the first divergence from the golden continuation) is
  printed alongside as a DIAGNOSTIC and does not affect the verdict -- an engine
  using bf16/fp8 weights or a bf16 KV cache legitimately diverges from the fp32
  reference, so sameness cannot gate. Read the prefix numbers anyway: agreement
  that collapses while throughput jumps is the sign that something broke.

  Caveat: the 0.90 raw BERTScore limit gives the gate a partial prefix-agreement
  effect anyway -- an engine that diverges inside the first half of a completion
  can fail it. The exam accepts that as a property of the chosen value.

Thresholds for both come from threshold.json.

Exit codes: 0 = ok, 1 = the gate failed, 2 = not graded (nothing was scored),
3 = environment fault (missing deps, cold cache, no network). A grading script
must not record 3 as a candidate failure.

Options:
  -r, --run PATH   C run binary (default <repo>/run)
  --tokens PATH    score this generated-ids file instead of driving `run`
  --quick          --tokens: diagnostics only, skip the accuracy tier (exits 2)
  --tie FLOAT      logit-gap tie threshold (default 2e-3, the oracle budget)
  --thresholds P   threshold.json (default <here>/threshold.json)
  --fuzzy          path A only: also run the METEOR/BERTScore tier (heavy deps)
  --max-new INT    tokens to free-run in the fuzzy tier (default = dataset max_new)
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="dsv2lite", choices=["dsv2lite", "glm47"])
    p.add_argument("-d", "--dir", default=None, help="dataset dir (default <here>/<model>)")
    p.add_argument("-r", "--run", default=os.path.join(_REPO, "run"))
    p.add_argument("--tokens", default=None,
                   help="score this generated-ids file (getp output) instead of "
                        "driving the run binary")
    p.add_argument("--model-dir", default=None,
                   help="override the model dir (reference.json's is provenance; "
                        "its absolute path won't exist on another machine)")
    p.add_argument("--steps", type=int, default=None,
                   help="--tokens: the STEPS the run requested, so a generation "
                        "capped below the reference length is reported as a "
                        "misconfiguration rather than graded as a failure")
    p.add_argument("--quick", action="store_true",
                   help="--tokens: print the prefix diagnostics and skip the accuracy "
                        "tier (heavy deps). Grades nothing; exits 2.")
    p.add_argument("--tie", type=float, default=2e-3)
    p.add_argument("--thresholds", default=os.path.join(_HERE, "threshold.json"))
    p.add_argument("--fuzzy", action="store_true")
    p.add_argument("--max-new", type=int, default=None)
    args = p.parse_args()
    # Both belong to --tokens. Path A ignores them, so a command line that lost
    # its --tokens would otherwise print path A's verdict and a grading script
    # would read it as the getp verdict.
    for opt, val in (("--quick", args.quick), ("--steps", args.steps is not None)):
        if val and args.tokens is None:
            p.error(f"{opt} requires --tokens")
    # ...and the reverse: path B takes its lengths from the ids file, so these
    # would be silently ignored there.
    for opt, val in (("--max-new", args.max_new is not None), ("--fuzzy", args.fuzzy)):
        if val and args.tokens is not None:
            p.error(f"{opt} does not apply with --tokens")
    # ...and --max-new is read only by run_fuzzy, so without --fuzzy it is inert.
    if args.max_new is not None and not args.fuzzy:
        p.error("--max-new requires --fuzzy")
    if args.max_new is not None and args.max_new < 1:
        p.error("--max-new must be >= 1")
    if args.steps is not None and args.steps < 1:
        # getp_eval.c raises steps<=0 to GETP_DEFAULT_STEPS, so the hint would
        # describe a run that never happened.
        p.error("--steps must be >= 1 (the harness raises 0 to its own default)")
    return args


def read_id_lines(path):
    with open(path) as f:
        return [[int(x) for x in ln.split()] for ln in f if ln.strip()]


def write_i32(ids):
    import struct
    f = tempfile.NamedTemporaryFile(suffix=".i32.bin", delete=False)
    f.write(struct.pack("<%di" % len(ids), *ids))
    f.close()
    return f.name


def run_c(run_bin, model_dir, ids, *mode):
    path = write_i32(ids)
    try:
        return subprocess.run([run_bin, model_dir, path, *map(str, mode)],
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError) as e:
        # CalledProcessError is a SubprocessError, not an OSError: a binary that
        # never starts (no execute bit, a directory) raises the latter and would
        # otherwise exit 1, which the README calls a candidate failure.
        print_warnings()
        rc = getattr(e, "returncode", None)
        print(f"engine failed: {run_bin} "
              + (f"exited {rc}" if rc is not None
                 else f"could not start ({type(e).__name__}: {e})")
              + f" (mode {' '.join(map(str, mode))}). A wrong model dir is the usual "
                f"cause.\n{(getattr(e, 'stderr', '') or '').strip()[:400]}",
              file=sys.stderr)
        sys.exit(3)                      # environment fault, not a gate failure
    finally:
        os.unlink(path)


def parse_teacher(out, prompt_len):
    """-> {'P': (strict, tie_tol, n, misses), 'D': (...)} over completion region."""
    rows = {"P": [], "D": []}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in ("P", "D"):
            tag, pos, gold, am, gap = parts
            rows[tag].append((int(pos), int(gold), int(am), float(gap)))
    return rows


def score_rows(rows, prompt_len, tie):
    comp = [(pos, gold, am, gap) for (pos, gold, am, gap) in rows if pos >= prompt_len]
    if not comp:
        return None
    strict = sum(am == gold for _, gold, am, _ in comp)
    tol = sum(am == gold or gap <= tie for _, gold, am, gap in comp)
    misses = [(pos, gold, am, gap) for (pos, gold, am, gap) in comp if am != gold]
    return strict / len(comp), tol / len(comp), len(comp), misses


def parse_ppl(out):
    c_nll = c_ntok = None
    for line in out.splitlines():
        if line.startswith("nll "):
            _, nll, _, ntok = line.split()
            c_nll, c_ntok = float(nll), int(ntok)
    return c_nll, c_ntok


_MODEL_HINT = {"dsv2lite": "deepseek", "glm47": "glm"}

# Warnings are re-printed next to RESULT: a caution that scrolls past 512
# diagnostic lines is a caution nobody reads, and the cases that raise one
# (wrong --model-dir, generation shorter than the reference) produce a
# plausible-looking verdict rather than an obvious failure.
_WARNINGS = []


def warn(msg):
    _WARNINGS.append(msg)
    print(f"  WARNING: {msg}", flush=True)


# math.exp overflows above ~709.78; past that a perplexity is not representable.
_EXP_MAX = 709.0


def _ppl(nll, ntok):
    """exp(nll/ntok), or inf when that overflows -- never raises."""
    try:
        return math.exp(nll / ntok)
    except OverflowError:
        return float("inf")


# src/run.c reads at most this many token ids; a longer sequence is silently
# truncated there, which shows up as a ppl token-count mismatch and would
# otherwise be reported as an engine defect.
_MAX_SEQ = 4096


_KV_CACHE_CAP = 163840     # src/model_load.c KV_CACHE_CAP
def _read_const(path, name, default):
    """Read `name = <int>` out of a sibling script without importing it.

    gen_reference.py owns _RUN_C_MAX_IDS, but importing it would pull torch and
    the oracle manifest into every eval run (~4s and a hard dependency chain for
    one integer). The Makefile reads GETP_DEFAULT_STEPS out of the C source the
    same way.
    """
    full = os.path.join(_HERE, path)
    try:
        with open(full) as f:
            # Anchored at both ends: an unanchored (\d+) matches a prefix, so a
            # constant that grew a suffix or an expression would silently yield a
            # wrong number instead of missing.
            m = re.search(rf"^{name}\s*=\s*(\d+)\s*(?:#.*)?$", f.read(), re.M)
    except OSError:
        return default                   # sibling absent: the mirror is expected
    if m is None:
        # Present but changed shape -- that is drift, and drift must not be quiet.
        print(f"WARNING: {path} no longer defines {name} as a plain integer; "
              f"falling back to {default}", file=sys.stderr)
        return default
    return int(m.group(1))


_RUN_C_MAX_IDS = _read_const("gen_reference.py", "_RUN_C_MAX_IDS", 4096)


def _kv_capacity(model_dir):
    """-> (cap, shown). cap is max_seq_len as src/model_load.c computes it:
    min(max_position_embeddings, KV_CACHE_CAP). cap <= 0 means the engine cannot
    load the model; `shown` is the config value AS WRITTEN IN THE FILE, so the
    caller can quote what an operator will actually find there -- and can tell
    the literal `Infinity` (which cJSON rejects) from `1e400` (valid JSON that
    cJSON parses to inf), which the parsed value cannot. `shown` is None when
    the file holds no top-level token for the key: quote the file, or quote
    nothing.

    cfg_int() takes the default unless the value is a JSON number, so a numeric
    *string* must not be accepted here either -- int("2048") would make the two
    disagree. A number below 2 IS kept, because C keeps it: the caller refuses
    such a model rather than silently grading against a cache the engine cannot
    allocate.
    """
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            text = f.read()
        # The parsed value cannot distinguish the literal token `Infinity` (which
        # cJSON rejects outright) from `1e400` (valid JSON that cJSON parses to
        # inf), and both arrive here as Python inf. So keep the source token
        # beside the value: the number hooks see the exact text json.loads used,
        # which a search over the file cannot promise -- a nested config (say a
        # multimodal text_config) holds the key too, and a duplicate top-level
        # key leaves only the last one standing.
        pair = lambda tok: (tok, float(tok))                        # noqa: E731
        cfg = json.loads(text, parse_int=lambda tok: (tok, int(tok)),
                         parse_float=pair, parse_constant=pair)
        if not isinstance(cfg, dict):
            # cJSON_Parse accepts a top-level array, and cJSON_GetObjectItem on
            # one returns NULL, so cfg_int takes its default -- same as an
            # absent key. (The engine still dies on n_layers = 0.)
            return _KV_CACHE_CAP, None
        entry = cfg.get("max_position_embeddings")
        if not isinstance(entry, tuple):
            # Absent, or present but not a JSON number (a string, bool, list or
            # object never reaches the number hooks): cfg_int() takes its
            # KV_CACHE_CAP default, and the file holds no token to quote.
            return _KV_CACHE_CAP, None
        raw, maxpos = entry
        if (isinstance(maxpos, float) and not math.isfinite(maxpos)) \
                or not -2**31 <= int(maxpos) < 2**31:
            # cfg_int casts cJSON's valuedouble to a C int, so the value is lost
            # and max_seq_len is whatever that cast produced -- which is not 0,
            # and is not something to guess at. (NaN/Infinity are not valid JSON
            # either: cJSON_Parse fails and model_load.c exits before the cache
            # is allocated.) cap None means "unrepresentable", which 0 cannot
            # signal -- 0 is itself a legal (and unloadable) cache length.
            return None, raw
        return min(int(maxpos), _KV_CACHE_CAP), raw
    except (OSError, ValueError, TypeError, AttributeError, OverflowError):
        # Unreadable or not JSON. cJSON_Parse rejects it too, so the engine
        # exits at the parse and reports it in its own words -- eval.py only has
        # to reach that point without crashing. Do NOT return _MAX_SEQ: it would
        # tie with src/run.c's read limit, and the too_long message would then
        # name a KV cache read from a file that has none.
        return _KV_CACHE_CAP, None


def _usable_len(rec):
    """completion_len usable as a generation count: a whole number >= 1."""
    if not isinstance(rec, dict):
        return False
    v = rec.get("completion_len")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def print_warnings():
    """Re-print next to the verdict. Must run before EVERY exit path, including
    the early ones -- a warning 500 lines up is a warning nobody reads."""
    for w in _WARNINGS:
        print(f"  WARNING: {w}")


def check_dataset_model(args, ref, data):
    """Fail fast when MODEL, DATA and --model-dir do not describe the same model.

    They are independent variables in the Makefile, so `MODEL=glm47
    DATA=<dsv2lite dir>` silently scores GLM output against DeepSeek ids with the
    GLM tokenizer and reports a plain FAIL. Not every dataset carries the same
    provenance keys -- the in-repo dev sets have "model", the published set has
    only "model_dir" and a manifest -- so cross-check whatever is present.
    """
    stated = ref.get("model")
    if stated is None:
        man = os.path.join(data, "manifest.json")
        if os.path.exists(man):
            # A damaged manifest is a misconfiguration like the rest of the
            # dataset reads; it must not surface as a traceback at exit 1.
            try:
                man_obj = json.load(open(man))
            except (OSError, ValueError) as e:
                print_warnings()
                print(f"cannot read {man} ({type(e).__name__}: {e})", file=sys.stderr)
                sys.exit(2)
            if not isinstance(man_obj, dict):
                print_warnings()
                print(f"{man} does not hold a JSON object", file=sys.stderr)
                sys.exit(2)
            stated = man_obj.get("model")
    if stated and stated != args.model:
        print_warnings()
        print(f"dataset {data} is for model '{stated}', but MODEL={args.model}",
              file=sys.stderr)
        sys.exit(2)                      # misconfiguration, not a gate failure

    # model_dir is provenance -- the weight-directory name on the machine that
    # generated the set -- so a rename there must not block a consistent dataset.
    # Heuristic, therefore a warning; the authoritative check above stays fatal.
    # Note this branch cannot fire for either real dataset kind: the in-repo dev
    # sets carry "model" in reference.json and the published set carries it in
    # manifest.json, so `stated` is always set. It is a last resort for a
    # hand-made directory holding reference.json alone -- do not rely on it to
    # catch a wrong MODEL.
    ref_dir = os.path.basename(str(ref.get("model_dir", "")).rstrip("/")).lower()
    hint = _MODEL_HINT.get(args.model)
    if ref_dir and hint and hint not in ref_dir and not stated:
        warn(f"dataset {data} was generated from '{ref['model_dir']}', which does "
             f"not look like a {args.model} model")
    if args.model_dir and ref_dir:
        got = os.path.basename(args.model_dir.rstrip("/")).lower()
        if got != ref_dir:
            warn(f"--model-dir is '{got}' but the dataset was generated from "
                 f"'{ref_dir}' -- the engine may be running the wrong weights")


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def score_tokens(args, model_dir, comps, thr):
    """Score a generated-ids file (getp output) against the golden completions.

    One line of space-separated ids per request, in request order. The engine may
    generate more than the golden `max_new` (e.g. the getp default steps=128); the
    surplus is ignored, so one timed getp run scores without a second, shorter run.
    """
    # Not read_id_lines(): a request that generated nothing writes an empty line,
    # and dropping it would misreport the failure as a line-count mismatch.
    try:
        raw = open(args.tokens).read().split("\n")
        if raw and raw[-1] == "":
            raw.pop()
        gen = [[int(x) for x in ln.split()] for ln in raw]
    except (OSError, ValueError) as e:
        print_warnings()
        print(f"cannot read {args.tokens} ({type(e).__name__}: {e})", file=sys.stderr)
        sys.exit(2)                      # misconfiguration, not a gate failure
    
    if len(gen) != len(comps):
        print_warnings()
        print(f"{args.tokens}: {len(gen)} lines != {len(comps)} requests in the dataset",
              file=sys.stderr)
        sys.exit(2)                      # misconfiguration, not a gate failure

    print("  gate: " + ("(none -- --quick skips the accuracy tier)" if args.quick else
                        f"meteor>={thr['meteor']}  bertscore_f1>={thr['bertscore_f1']}"),
          flush=True)

    fracs, n_exact = [], 0
    for i, (g, c) in enumerate(zip(gen, comps)):
        pre = common_prefix(g, c)
        frac = pre / len(c)
        fracs.append(frac)
        exact = pre == len(c)
        n_exact += exact
        div = "" if exact else f"  first diff @{pre}: gold {c[pre]} != gen " + (
            str(g[pre]) if pre < len(g) else "<end>")
        print(f"  [{i}] gen={len(g):4d} gold={len(c):4d}  "
              f"prefix={pre:4d}/{len(c)} ({frac*100:6.2f}%){div}", flush=True)

    # Prefix statistics are DIAGNOSTIC, not a gate: the announced accuracy gate is
    # METEOR/BERTScore, and an engine may legitimately diverge from the fp32
    # reference (bf16/fp8 weights or KV cache) while still being correct. Read
    # these to tell a numerically-different engine from a broken one; the
    # thresholds they cite are advisory reference points.
    mean_prefix = sum(fracs) / len(fracs)
    bad = [i for i, f in enumerate(fracs) if f < thr["getp_min_prefix"]]
    print()
    print(f"  requests matching exactly = {n_exact}/{len(comps)}")
    print(f"  mean prefix agreement     = {mean_prefix*100:.3f}%        [diagnostic, "
          f"advisory floor {thr['getp_prefix']*100:.0f}%]")
    print(f"  worst request             = {min(fracs)*100:.3f}%        [diagnostic]")
    print(f"  below {thr['getp_min_prefix']*100:.0f}% floor           = {len(bad)}/{len(fracs)}"
          f" ({len(bad)/len(fracs)*100:.2f}%)        [diagnostic, advisory max "
          f"{thr.get('getp_bad_frac', 0.05)*100:.0f}%]"
          + (f"  reqs {bad[:8]}{'...' if len(bad) > 8 else ''}" if bad else ""))

    # Truncating over-long generations is free, but a generation SHORTER than the
    # reference silently costs recall on every metric. The usual cause is STEPS
    # capped below the reference completion length, which fails a correct engine.
    short = [i for i, (g, c) in enumerate(zip(gen, comps)) if len(g) < len(c)]
    if short:
        worst = min((len(gen[i]) - len(comps[i]), i) for i in short)[1]
        print(f"  shorter than reference    = {len(short)}/{len(comps)}"
              f"        [diagnostic]  worst: req{worst} "
              f"{len(gen[worst])} vs {len(comps[worst])} tokens")
        # Every short generation the same length is a hard cap (STEPS below the
        # reference length), not an engine defect. Grading that would report FAIL
        # for a correct engine, so refuse to grade instead of scoring it.
        # An engine that stops on EOS gives mixed short lengths, so "all equal"
        # was too strict -- one early stop let a capped run through to a FAIL.
        # Prefer the cap the caller actually asked for; fall back to the most
        # common short length when --steps was not passed.
        from collections import Counter
        counts = Counter(len(gen[i]) for i in short)
        # --steps is a HINT, not a switch: an engine whose own loop stops one token
        # early would otherwise match nothing and escape the check entirely.
        cap, n_at_cap = counts.most_common(1)[0]
        # Prefer the requested cap only when it explains at least as many requests
        # as the measured one; a couple of requests ending exactly at STEPS must
        # not hide a larger cap somewhere else.
        if args.steps is not None and counts.get(args.steps, 0) >= n_at_cap:
            cap, n_at_cap = args.steps, counts[args.steps]
        longest = max(len(c) for c in comps)
        # cap 0 is not reachable (getp_eval.c floors steps<=0 at GETP_DEFAULT_STEPS),
        # so an engine emitting nothing is a real failure, not a misconfiguration.
        # If STEPS is at least the longest reference, no cap can explain a short
        # generation: the engine truncated on its own, which is a real defect and
        # must reach the gate rather than be excused as a misconfiguration.
        capped = (args.steps if args.steps is not None else 0) < longest
        # A STEPS cap stops generation AT the requested count, so a stop far below
        # it is the engine truncating itself. One token of tolerance keeps an
        # off-by-one engine loop inside the check.
        explained = cap >= args.steps - 1 if args.steps is not None else True
        if capped and explained and cap > 0 \
                and n_at_cap >= max(2, thr.get("getp_cap_quorum", 0.05) * len(comps)) \
                and cap < longest:
            print_warnings()
            # exit 2 = "not graded", same as --quick; 1 is reserved for a real FAIL
            print(f"\n  RESULT: not graded -- {n_at_cap}/{len(comps)} generations "
                  f"stop at exactly {cap} tokens, below the reference "
                  f"(max {longest}).\n"
                  f"  That is a generation cap, not an engine defect: re-run with "
                  f"STEPS >= {longest}.", flush=True)
            sys.exit(2)
        if capped and explained:
            warn(f"{len(short)}/{len(comps)} requests generated fewer tokens than the "
                 f"reference; if STEPS caps generation below the reference length "
                 f"(max {longest}), the gate will fail a correct engine")
        elif capped:
            # Only n_at_cap requests stop at `cap`; a mixed run can also hold
            # requests that stop exactly at STEPS, and those ARE capped.
            n_at_steps = counts.get(args.steps, 0)
            warn(f"{n_at_cap} of the {len(short)} short requests stop at "
                 f"{cap}, far below STEPS={args.steps} -- the engine stopped early itself"
                 + (f"; {n_at_steps} other request(s) stop at exactly STEPS="
                    f"{args.steps}, which IS a cap below the reference (max {longest})"
                    if n_at_steps else ""))
        else:
            warn(f"{len(short)}/{len(comps)} requests generated fewer tokens than the "
                 f"reference, and STEPS={args.steps} cannot be the cause (>= the "
                 f"longest reference, {longest}) -- the engine stopped early itself")

    if args.quick:
        return None                      # nothing was gated
    preds = [g[:len(c)] for g, c in zip(gen, comps)]
    return score_fuzzy(model_dir, preds, comps, thr)


def main():
    args = parse_args()
    data = args.dir or os.path.join(_HERE, args.model)
    try:
        ref = json.load(open(os.path.join(data, "reference.json")))
        thr = json.load(open(args.thresholds))
        prompts = read_id_lines(os.path.join(data, "prompts.i32.txt"))
        comps = read_id_lines(os.path.join(data, "completions.i32.txt"))
        recs = ref["requests"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"cannot read the dataset or thresholds ({type(e).__name__}: {e})",
              file=sys.stderr)
        sys.exit(2)                      # misconfiguration, not a gate failure
    if not isinstance(thr, dict):
        print(f"{args.thresholds} does not hold a JSON object", file=sys.stderr)
        sys.exit(2)
    if not isinstance(recs, list) or not recs:
        print(f"{data}/reference.json: 'requests' is not a non-empty list",
              file=sys.stderr)
        sys.exit(2)
    # Checked up front, not at first use: both reads happen after the engine has
    # already run, so a malformed record would cost timed passes first -- and the
    # --fuzzy read would discard a complete top-1/ppl result. Only the fields THIS
    # run will actually read are required: --tokens reads neither pair.
    def _num(v):                          # finite real; bool subclasses int
        # Compared, not converted: json.load builds unbounded ints, and both
        # math.isfinite and float() raise OverflowError on one -- turning the
        # value this test must reject into a traceback at exit 1.
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and v == v and -math.inf < v < math.inf)

    def _count(v):                        # a token count: whole number in [1, 1e9]
        # The upper bound is load-bearing, and not for the reason it once was:
        # the magnitude test below multiplies (_EXP_MAX * hf_ntok), and that
        # conversion raises OverflowError on an unbounded int. Do not drop it.
        return (_num(v) and (isinstance(v, int) or v.is_integer())
                and 1 <= v <= 10**9)

    if not args.model_dir and "model_dir" not in ref:
        print(f"{data}/reference.json has no 'model_dir': pass --model-dir",
              file=sys.stderr)
        sys.exit(2)
    model_dir = args.model_dir or ref["model_dir"]
    n = len(recs)
    # Not an assert: `python -O` would drop it and zip() would then hide the
    # difference silently. A malformed dataset is a misconfiguration, not a FAIL.
    if not (len(prompts) == len(comps) == n):
        print(f"dataset {data} is inconsistent: {len(prompts)} prompts, "
              f"{len(comps)} completions, {n} reference records", file=sys.stderr)
        sys.exit(2)
    check_dataset_model(args, ref, data)


    if args.tokens is None:              # path A: the only path that drives run.c
        # Two limits, and the smaller binds: src/run.c reads at most _MAX_SEQ ids
        # (a longer sequence is truncated, and the ppl token counts then
        # disagree), and teacher/ppl both hand the whole sequence to
        # forward_unabsorbed, which writes kv_l[p * KVD] for every position.
        kv_cap, kv_shown = _kv_capacity(model_dir)
        cap = _MAX_SEQ if kv_cap is None else min(_MAX_SEQ, kv_cap)
        if kv_cap is None or kv_cap < 1:
            # Only a non-positive length is unloadable. A length of 0 makes
            # calloc return a zero-size block, and every layer then writes past
            # it. A negative length converts to a huge size_t at
            # src/model_load.c:168-169, so calloc returns NULL and neither line
            # tests the pointer. A length of 1 allocates fine and the engine
            # runs a one-token sequence, so that case belongs to the too_long
            # test below, which names the dataset instead.
            print_warnings()
            # NaN/Infinity are literals JSON does not define: cJSON rejects the
            # file. 1e400 is valid JSON that cJSON parses to inf, so the cast is
            # what loses it. Only the source token separates the two.
            if kv_shown in ("NaN", "Infinity", "-Infinity"):
                why = (f"holds max_position_embeddings {kv_shown}, which is not "
                       f"valid JSON: cJSON_Parse fails and src/model_load.c exits "
                       f"at the parse")
            elif kv_cap is None and kv_shown is not None:
                why = (f"holds a max_position_embeddings ({kv_shown}) that "
                       f"src/model_load.c cannot represent (cfg_int casts it to "
                       f"a C int)")
            else:
                why = (f"gives a KV cache length of {kv_cap}"
                       + ("" if kv_shown is None
                          else f" (max_position_embeddings = {kv_shown})"))
            print(f"{model_dir}/config.json {why}: the engine cannot load this "
                  f"model", file=sys.stderr)
            sys.exit(3)                  # environment fault, not a gate failure
        too_long = [i for i, (p, c) in enumerate(zip(prompts, comps))
                    if len(p) + len(c) > cap]
        if too_long:
            print_warnings()
            tie = cap == _MAX_SEQ == kv_cap
            _READ = "src/run.c's id-read limit, mirrored at tests/eval/eval.py:_MAX_SEQ"
            src = (f"both {_READ} and the KV cache from {model_dir}/config.json"
                   if tie else _READ if cap == _MAX_SEQ
                   else f"the KV cache sized from {model_dir}/config.json")
            # --max-tokens caps the COMPLETION and cannot shorten a prompt, and
            # gen_reference.py refuses any prompt with fewer than 2 free
            # positions -- so regeneration can only work when the LONGEST prompt
            # plus 2 fits under the limit.
            need = max(len(p) for p in prompts) + 2
            _RAISE = ("raise the read limit in src/run.c AND _MAX_SEQ in "
                      "tests/eval/eval.py (the harness keeps its own copy)")
            raise_ = (f"{_RAISE}, and use a model with a larger "
                      "max_position_embeddings (both bind)" if tie else
                      _RAISE if cap == _MAX_SEQ
                      else "use a model with a larger max_position_embeddings")
            longest = need - 2               # the longest prompt itself
            if cap >= need:
                fix = (f"regenerate the dataset with --max-tokens {cap} or less "
                       f"(the longest prompt is {longest} tokens)")
            elif cap == longest + 1 and need <= _RUN_C_MAX_IDS:
                # A 1-token completion still fits, but gen_reference needs two
                # free positions above the prompt -- and it refuses a
                # --max-tokens above the read limit, so `need` must fit too.
                fix = (f"regenerate with --max-tokens {need} --max-new 1 (the "
                       f"longest prompt is {longest} tokens, so only a 1-token "
                       f"completion fits under {cap})")
            elif cap >= longest:
                # The prompt fits; what does not is the 2 free positions
                # gen_reference.py keeps above every prompt -- and when `need` is
                # over the generator's ceiling, that ceiling too. The engine limit
                # is still worth naming as the second remedy, since raising it
                # also raises the ceiling the generator checks against.
                blocker = ("gen_reference.py needs 2 free positions above it, and "
                           f"its --max-tokens ceiling is {_RUN_C_MAX_IDS}"
                           if need > _RUN_C_MAX_IDS else
                           "gen_reference.py needs 2 free positions above it")
                fix = (f"no flags help: the longest prompt is {longest} tokens and "
                       f"{blocker} -- shorten the prompts, or {raise_}")
            else:
                fix = (f"no flags help: the longest prompt is {longest} tokens and "
                       f"does not fit under {cap} on its own -- {raise_}, or "
                       f"shorten the prompts")
            print(f"{data}: request(s) {too_long[:8]} exceed the {cap}-token limit "
                  f"({src}) -- {fix}", file=sys.stderr)
            sys.exit(2)                  # a harness limit, not a gate failure

        # path A reads the ppl pair
        # hf_nll is a sum of negative log-likelihoods (>= 0); hf_ntok is a token
        # count. NaN/Infinity parse fine from JSON and would survive to make the
        # ppl aggregate silently meaningless, so _num rejects them.
        bad = [i for i, r in enumerate(recs)
               if not isinstance(r, dict) or not _num(r.get("hf_nll"))
               or r["hf_nll"] < 0 or not _count(r.get("hf_ntok"))
               or r["hf_nll"] > _EXP_MAX * r["hf_ntok"]]
        if bad:
            print_warnings()
            print(f"{data}/reference.json: record(s) {bad[:8]} miss a usable "
                  f"'hf_nll' (finite, >= 0, mean <= {_EXP_MAX}) / 'hf_ntok' "
                  f"(whole, >= 1) pair",
                  file=sys.stderr)
            sys.exit(2)
        if args.fuzzy:
            if not args.max_new:                 # run_fuzzy reads completion_len
                bad = [i for i, r in enumerate(recs) if not _usable_len(r)]
                if bad:
                    print_warnings()
                    print(f"{data}/reference.json: record(s) {bad[:8]} have no usable "
                          f"'completion_len' -- pass --max-new", file=sys.stderr)
                    sys.exit(2)
            # `gen` mode in src/run.c raises pos once per step and tests nothing,
            # so an oversized length writes past the end of the KV cache. bench
            # mode has this test; gen mode does not.
            # The write bound is the KV cache, sized from the model config, not
            # the id-read limit; they coincide for the exam models but not by
            # construction. Take whichever is smaller.
            over = [i for i, (p, r) in enumerate(zip(prompts, recs))
                    if len(p) + (args.max_new or int(r["completion_len"])) > cap]
            if over:
                cause = ("--max-new" if args.max_new
                         else f"'completion_len' in {data}/reference.json")
                print_warnings()
                print(f"prompt + generation exceeds the {cap}-token buffer for "
                      f"request(s) {over[:8]} -- lower {cause}", file=sys.stderr)
                sys.exit(2)
    missing = [k for k in ("meteor", "bertscore_f1", "getp_min_prefix",
                           "getp_prefix", "top1_strict", "ppl_rel") if k not in thr]
    if missing:
        print_warnings()
        print(f"{args.thresholds} is missing required key(s): {', '.join(missing)}",
              file=sys.stderr)
        sys.exit(2)

    if args.tokens is not None:
        if not args.tokens:
            print("--tokens needs a path; an empty value scores nothing",
                  file=sys.stderr)
            sys.exit(2)
        print(f"[{args.model}] {args.tokens}  ({n} requests)", flush=True)
        ok = score_tokens(args, model_dir, comps, thr)
        print()
        print_warnings()
        if ok is None:      # --quick: diagnostics only, so say so rather than pass
            print("  RESULT: not graded (--quick skipped the accuracy gate)")
            sys.exit(2)
        print("  RESULT:", "ok" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    if not args.run:
        print_warnings()
        print("--run needs a path; an empty value names no binary", file=sys.stderr)
        sys.exit(3)                      # environment fault, not a gate failure
    # A bare word goes to PATH in execvp but to the working directory in
    # os.access, so the guard and the launch would test different files. The
    # Makefile makes the same correction for RUN.
    if os.sep not in args.run:
        args.run = os.path.join(".", args.run)
    if not os.path.exists(args.run) or not os.access(args.run, os.X_OK):
        print_warnings()
        print(f"C run binary not usable: {args.run} "
              f"({'not found' if not os.path.exists(args.run) else 'not executable'}) "
              f"-- build with `make run`", file=sys.stderr)
        sys.exit(3)                      # environment fault, not a gate failure
    print(f"[{args.model}] {model_dir}  ({n} requests)  tie={args.tie:.1e}", flush=True)
    print(f"  gates: top1_strict>={thr['top1_strict']} (both paths)  "
          f"ppl_rel<={thr['ppl_rel']}  ppl_ntok must match the reference"
          + (f"  meteor>={thr['meteor']}  bertscore_f1>={thr['bertscore_f1']}" if args.fuzzy else ""),
          flush=True)

    tot = {"P_ok": 0, "D_ok": 0, "cmp": 0}          # top-1 aggregates
    worst_ppl = 0.0
    all_misses = []
    nonfinite, mismatched = [], []
    for i, (pids, cids, rec) in enumerate(zip(prompts, comps, recs)):
        full = pids + cids
        plen = len(pids)
        # --- Tier 1+2: teacher-forced top-1, both paths ---
        out = run_c(args.run, model_dir, full, "teacher", plen)
        rows = parse_teacher(out, plen)
        sp = score_rows(rows["P"], plen, args.tie)
        sd = score_rows(rows["D"], plen, args.tie)
        if sp is not None and sd is not None and sp[2] != sd[2]:
            # Per request, not on the totals: a build that is short on one
            # request and long on another cancels in the sum.
            print_warnings()
            print(f"{args.run} printed {sp[2]} prefill rows against {sd[2]} decode "
                  f"rows for request {i} -- the two top-1 numbers are not "
                  f"comparable", file=sys.stderr)
            sys.exit(3)                  # environment fault, not a gate failure
        if sp is None or sd is None:
            print_warnings()
            print(f"{args.run} printed no teacher row for the completion region of "
                  f"request {i} (mode teacher {plen}) -- wrong binary or a build "
                  f"without the eval modes", file=sys.stderr)
            sys.exit(3)                  # environment fault, not a gate failure
        # --- Tier 2: perplexity rel-err vs frozen HF nll ---
        c_nll, c_ntok = parse_ppl(run_c(args.run, model_dir, full, "ppl"))
        if c_nll is not None and c_ntok and c_ntok != rec["hf_ntok"]:
            # Both averages divide by their own count, so they are comparable
            # only over the same targets. A mismatch reads as rel=0 otherwise.
            mismatched.append((i, c_ntok, rec["hf_ntok"]))
        if c_nll is None or not c_ntok:
            print_warnings()
            print(f"{args.run} printed no 'nll' line in ppl mode -- wrong binary or "
                  f"a build without the eval modes", file=sys.stderr)
            sys.exit(3)                  # environment fault, not a gate failure
        hf_ppl = _ppl(rec["hf_nll"], rec["hf_ntok"])
        c_ppl = _ppl(c_nll, c_ntok)
        rel = abs(c_ppl - hf_ppl) / hf_ppl
        if not (rel == rel and rel < math.inf):
            # max() compares with >, and every comparison with NaN is false, so a
            # NaN would leave the aggregate at 0.0 and pass the gate. The dataset
            # side is validated above, so this came from the engine. Collected,
            # not warned per request: one per request would push the summary
            # block above N copies of the same line.
            nonfinite.append(i)
            rel = float("inf")
        worst_ppl = max(worst_ppl, rel)
        # Both are 4-tuples here: the None case exited above.
        tot["D_ok"] += round(sd[0] * sd[2]); tot["cmp"] += sd[2]
        tot["P_ok"] += round(sp[0] * sp[2])
        all_misses += [(i, "D", *m) for m in sd[3]]
        all_misses += [(i, "P", *m) for m in sp[3]]
        print(f"  [{i}] comp={sd[2]:4d}  "
              f"P_top1={sp[0]*100:6.2f}% (tie {sp[1]*100:6.2f}%)  "
              f"D_top1={sd[0]*100:6.2f}% (tie {sd[1]*100:6.2f}%)  "
              f"ppl C={c_ppl:.4f} HF={hf_ppl:.4f} rel={rel:.2e}", flush=True)

    if nonfinite:
        warn(f"the engine's perplexity is not a finite number for {len(nonfinite)}/"
             f"{n} request(s) {nonfinite[:8]} -- scored as maximally wrong")
    if mismatched:
        i, got, want = mismatched[0]
        warn(f"{len(mismatched)}/{n} request(s) scored a different number of ppl "
             f"targets than the reference (e.g. request {i}: {got} vs {want}) -- the "
             f"two perplexities are not comparable")
    cmp = tot["cmp"] or 1
    d_strict = tot["D_ok"] / cmp
    p_strict = tot["P_ok"] / cmp
    print()
    print(f"  decode-path  top1_strict = {d_strict*100:.3f}%  ({tot['D_ok']}/{cmp})")
    print(f"  prefill-path top1_strict = {p_strict*100:.3f}%  ({tot['P_ok']}/{cmp})")
    print(f"  worst ppl rel-err        = {worst_ppl:.3e}")
    print(f"  ppl token-count mismatch = {len(mismatched)}/{n}")
    if all_misses:
        # Ranked per path: the two kernels have different gap scales, so one
        # combined top-10 lets the larger-gap path hide the other entirely.
        for path in ("D", "P"):
            rows = [m for m in all_misses if m[1] == path]
            if not rows:
                continue
            print(f"  {'decode' if path == 'D' else 'prefill'} misses "
                  f"({len(rows)} total; pos, gold, argmax, gap), worst first:")
            for req, _p, pos, gold, am, gap in sorted(rows, key=lambda m: -m[5])[:5]:
                print(f"    req{req} pos{pos}: gold {gold} != argmax {am}  "
                      f"gap={gap:.4f}" + ("  [tie]" if gap <= args.tie else ""))

    # Both paths gate: a prefill number printed beside RESULT: ok while reaching
    # no condition is the same asymmetry as a condition the gate line omits.
    ok = (d_strict >= thr["top1_strict"] and p_strict >= thr["top1_strict"]
          and worst_ppl <= thr["ppl_rel"] and not mismatched)

    if args.fuzzy:
        ok = run_fuzzy(args, model_dir, prompts, comps, recs, thr) and ok

    print()
    print_warnings()
    print("  RESULT:", "ok" if ok else "FAIL")
    sys.exit(0 if ok else 1)


_WARM = ("run `make eval-warm` once on a networked machine to fill the metric, "
         "nltk and roberta-large caches; HF_HUB_OFFLINE=1 works after that")


def score_fuzzy(model_dir, pred_ids, ref_ids, thr):
    """Detokenize both sides and score METEOR + BERTScore. Heavy deps."""
    from transformers import AutoTokenizer
    try:
        import evaluate  # noqa
    except ModuleNotFoundError:
        print_warnings()
        print("the accuracy tier needs the `fuzzy` extra: `uv sync --extra fuzzy`",
              file=sys.stderr)
        sys.exit(3)
    # model_dir may be a Hub repo id (the published reference.json records one),
    # so this load can hit the network too -- same guard, same hint.
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print_warnings()
        print(f"could not load the tokenizer for '{model_dir}' "
              f"({type(e).__name__}: {e})\n  If this machine is offline or the "
              f"caches are cold, {_WARM}.", file=sys.stderr)
        sys.exit(3)
    preds = [tok.decode(ids) for ids in pred_ids]
    refs = [tok.decode(ids) for ids in ref_ids]
    try:
        meteor = evaluate.load("meteor").compute(predictions=preds, references=refs)["meteor"]
    except Exception as e:
        print_warnings()
        print(f"could not load or run the METEOR metric ({type(e).__name__}: {e})\n"
              f"  If this machine is offline or the caches are cold, {_WARM}.",
              file=sys.stderr)
        sys.exit(3)
    # BERTScore's tokenizer raises on an empty/whitespace-only prediction, which a
    # request that generated nothing produces. Score those 0 (maximally wrong) and
    # keep them out of the compute call, rather than crashing the gate.
    live = [i for i, p in enumerate(preds) if p.strip()]
    f1s = [0.0] * len(preds)
    if live:
        try:
            bs = evaluate.load("bertscore").compute(
                predictions=[preds[i] for i in live],
                references=[refs[i] for i in live], lang="en")
        except Exception as e:
            print_warnings()
            print(f"could not load or run BERTScore ({type(e).__name__}: {e})\n"
                  f"  If this machine is offline or the caches are cold, {_WARM}.",
                  file=sys.stderr)
            sys.exit(3)
        for i, v in zip(live, bs["f1"]):
            f1s[i] = v
    f1 = sum(f1s) / len(f1s)
    empty = len(preds) - len(live)
    print(f"  METEOR = {meteor:.4f}  BERTScore-F1 = {f1:.4f}"
          + (f"  ({empty} empty prediction(s) scored 0)" if empty else ""))
    return meteor >= thr["meteor"] and f1 >= thr["bertscore_f1"]


def run_fuzzy(args, model_dir, prompts, comps, recs, thr):
    """Free-run greedy generation vs golden completion, scored METEOR+BERTScore."""
    preds = []
    for pids, rec in zip(prompts, recs):
        # Validated up front (see main), so this only normalises JSON's 4.0.
        max_new = args.max_new or int(rec["completion_len"])
        out = run_c(args.run, model_dir, pids, "gen", max_new)
        gen_ids = []
        for line in out.splitlines():
            if line.startswith("completion"):
                gen_ids = [int(x) for x in line.split()[1:]]
        preds.append(gen_ids)
    return score_fuzzy(model_dir, preds, comps, thr)


if __name__ == "__main__":
    main()
