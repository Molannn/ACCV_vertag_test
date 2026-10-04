"""LETITBE: per-point explanation-coverage scoring for trademark similarity.

Two tasks, one engine layer.

    python letitbe.py coverage --input examples/coverage.example.jsonl --engine qwen2.5:7b
    python letitbe.py detector --input examples/detector.example.jsonl --engine qwen2.5:7b
    python letitbe.py qualify  --input examples/coverage.example.jsonl --engine qwen2.5:7b

`coverage` scores candidate explanations: for each gold reason point it asks the engine one
yes/no question, and a case's coverage is the fraction of its points the candidate entails.
`detector` evaluates the engine itself on labelled candidate-claim pairs and reports
precision, recall and F1, broken down by how each negative was constructed.

`qualify` decides whether an engine may be used at all, by scoring five candidates of known
relative quality per case and checking the result against a published gate.

All three ask exactly the same question, built by `qa_text` below. That prompt is part of the
metric's specification, not an implementation detail: a reworded or reordered prompt is a
different metric.

Backends
    --backend api   an OpenAI-compatible endpoint. Local Ollama by default; OpenAI for
                    gpt-* and ft:* model ids. Override with --base-url.
    --backend hf    a local Hugging Face model in 4-bit, optionally with a LoRA adapter.

See examples/SCHEMA.md for the input formats.
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from collections import Counter

# --------------------------------------------------------------------------------- prompt
# The judging rule. Zero-shot and fine-tuned detectors are asked the identical question, so
# that training and inference never drift apart. Aspect definitions mirror those used when
# the gold checklists were decomposed.

_RULE = """你是商標近似論述的命中判定器。判斷「候選解釋」有沒有做出指定的「理由論斷」，輸出 是/否。
標籤：是 = 候選有做出該論斷（語意蘊含即可，用字不必相同）；否 = 沒提到、或講相反／矛盾。

該論斷屬於下列 aspect 之一（用語依商標近似審查基準 §5.2，括號為基準原文範例）：
- 主要識別部分：消費者關注、事後留印象中較顯著的部分；含主要字詞與形容字詞之分。指名兩商標共同的字／外文／圖形。（例：「泰山」與「小泰山」主要字詞均為「泰山」）
- 差異：些微／細微差異；不可機械式比對。指名差在哪。（例：「house」與「horse」可能近似，「house」與「mouse」則否）
- 外觀：文字商標就外觀觀察是否相像（常同時斷言「在外觀」）。
- 近似程度結論：構成近似與否及其近似程度（高／不低／低）。
※ 主要識別部分與差異常同句出現，例：「『功夫』二字外觀近似」→ 主要識別部分「共同含『功夫』」＋外觀「外觀近似」。
範例命中判定：論斷「兩商標共同含『鍋寶』」、解釋「二者都有鍋寶二字」→ 是；論斷「外觀近似」、解釋「外觀並不近似」→ 否。"""

_FIELDS = "【aspect】{aspect}\n【理由論斷】{claim}\n【候選解釋】{cand}"

LABELS = {True: "是", False: "否"}

ASPECTS = ("主要識別部分", "差異", "外觀", "近似程度結論")

# The boilerplate candidate used by `qualify`. It recites how similarity is assessed and
# makes no case-specific claim, so a faithful metric should score it near zero. This exact text
# is what the paper's boilerplate column was measured with; changing it changes the gate.
BOILER = ("商標近似係指二商標予人之整體印象有其相近之處，其判斷得就二商標外觀、觀念及讀音為觀察，"
          "以具有普通知識經驗之消費者，於購買時施以普通之注意，異時異地隔離觀察，"
          "可能會誤認二商品來自同一來源或雖不相同但有關聯之來源。")

# The gate an engine must pass to be usable: boilerplate coverage no higher than 0.05, and a
# strictly monotone decline across full, ablate-1, ablate-half and wrong. Engines that fail
# do not merely score lower; they score high on boilerplate, crediting text that says nothing
# about the case.
GATE_BOILERPLATE_MAX = 0.05
GATE_CHAIN = ("full", "ablate1", "ablate_half", "wrong")


def qa_text(cand, claim, aspect):
    """(candidate explanation, one reason point, its aspect) -> the judge's input string."""
    return f"{_RULE}\n\n" + _FIELDS.format(cand=cand, claim=claim, aspect=aspect)


def _check_prompts_doc(fragments):
    """Assert PROMPTS.md still quotes these verbatim, so the doc and the code cannot drift.

    A prompt reproduced in prose is a prompt that will one day disagree with the one that
    runs, and a reworded prompt is a different metric. Silently skipped when the doc is absent,
    so a vendored copy of a single script still checks out.
    """
    doc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROMPTS.md")
    if not os.path.exists(doc):
        return
    with open(doc) as f:
        text = f.read()
    for name, fragment in fragments:
        assert fragment in text, f"PROMPTS.md no longer quotes {name} verbatim; regenerate it"


def _parse(text):
    """Read 是/否 out of a completion: the first verdict wins, 不是 counts as 否, and a
    reply with no verdict at all counts as 否."""
    m = re.search(r"不是|[是否]", text or "")
    return bool(m) and m.group() == "是"


# -------------------------------------------------------------------------------- backends
# Both engines expose the same interface: judge(prompts) yields (index, bool) as answers
# arrive, so callers can write results incrementally and resume after a crash.

_OLLAMA_DEFAULT = "http://localhost:11434/v1"


def _rejects_temperature(exc):
    """Whether this error is the endpoint refusing our temperature rather than a real fault.

    Some reasoning models accept only their own default temperature and answer a request that
    pins it with a 400. That is a statement about the model, not a bug, so the caller drops the
    parameter and carries on rather than failing a whole run over it.
    """
    m = str(exc).lower()
    return "temperature" in m and ("does not support" in m or "unsupported" in m)


def _is_permanent(exc):
    """Whether retrying could ever help. A malformed or unauthorised request will not fix itself."""
    return getattr(exc, "status_code", None) in (400, 401, 403, 404)



def _env(name, default=None):
    """Read an environment variable, treating an empty value as unset.

    `set -a; source .env; set +a` over a file containing `OLLAMA_BASE=` exports an empty
    string, not nothing, so a plain `os.environ.get(name, default)` would hand back `""`
    and never reach the default.
    """
    return os.environ.get(name) or default


class ApiEngine:
    """An OpenAI-compatible chat endpoint. Covers both OpenAI and a local Ollama server."""

    def __init__(self, model, base_url=None, workers=8, retries=5):
        from openai import OpenAI
        self.model, self.workers, self.retries = model, workers, retries
        self.temperature = 0          # dropped on the first refusal; see _one
        if base_url:
            self.client = OpenAI(base_url=base_url, api_key=_env("OPENAI_API_KEY", "ollama"))
        elif model.startswith(("gpt", "ft:", "o1", "o3")):
            key = _env("OPENAI_API_KEY")
            if not key:
                sys.exit("OPENAI_API_KEY is not set; pass --base-url for a local endpoint instead")
            self.client = OpenAI(api_key=key)
        else:                                       # anything else is taken to be a local Ollama server; override with --base-url
            self.client = OpenAI(base_url=_env("OLLAMA_BASE", _OLLAMA_DEFAULT),
                                 api_key="ollama")

    def _one(self, prompt):
        for i in range(self.retries):
            # Greedy, to match the hf backend's do_sample=False. A metric that returns a
            # different score for the same input on a second run is not a metric.
            kw = {} if self.temperature is None else {"temperature": self.temperature}
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}], **kw)
                return _parse(r.choices[0].message.content)
            except Exception as e:
                if _rejects_temperature(e):
                    # Decided from the error alone, never from self.temperature: the workers
                    # run in parallel and all of them see this refusal, but only the first
                    # finds the attribute still set.
                    if self.temperature is not None:
                        self.temperature = None
                        print(f"  note: {self.model} accepts only its default temperature, so "
                              f"this run is not greedy and may not repeat exactly", flush=True)
                    continue                     # retry without the parameter
                if _is_permanent(e) or i == self.retries - 1:
                    raise
                time.sleep(2 * (i + 1))

    def judge(self, prompts):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self._one, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                yield futs[fut], fut.result()


class HfEngine:
    """A local Hugging Face causal LM in 4-bit, optionally with a LoRA adapter on top.

    Used so that a base model and its fine-tuned adapter are compared over the identical
    inference path, rather than one through a server and the other locally.
    """

    def __init__(self, model, adapter=None, batch=32, max_length=2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch, self.batch, self.max_length = torch, batch, max_length
        token = _env("HF_TOKEN")                     # gated bases such as gemma need this
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(model, token=token, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=bnb, dtype=torch.bfloat16,
            device_map="auto", token=token)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def judge(self, prompts):
        for s in range(0, len(prompts), self.batch):
            chunk = prompts[s:s + self.batch]
            texts = [self.tok.apply_chat_template([{"role": "user", "content": c}],
                                                  tokenize=False, add_generation_prompt=True)
                     for c in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            if enc["input_ids"].shape[1] > self.max_length:    # truncation would cut the answer cue off the end
                sys.exit(f"a prompt among items {s}-{s + len(chunk) - 1} is {enc['input_ids'].shape[1]} "
                         f"tokens, over the {self.max_length}-token limit; shorten that candidate")
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=4, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for j, (g, inp) in enumerate(zip(gen, enc["input_ids"])):
                yield s + j, _parse(self.tok.decode(g[len(inp):], skip_special_tokens=True))


def make_engine(args):
    if args.backend == "hf":
        return HfEngine(args.engine, adapter=args.adapter, batch=args.batch)
    if args.adapter:
        sys.exit("--adapter only applies to --backend hf")
    return ApiEngine(args.engine, base_url=args.base_url, workers=args.workers)


# ----------------------------------------------------------------------------------- tasks


def _read(path):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}, line {n}: not valid JSON ({e.msg})")
    if not rows:
        sys.exit(f"{path}: no records")
    return rows


def _require(rows, fields, path):
    """Check every record before spending anything on an engine.

    Without this a missing field surfaces as a KeyError several hundred judgments into a paid
    or hour-long run, with nothing to say which record was wrong.
    """
    for i, r in enumerate(rows, 1):
        for key, kind in fields.items():
            if key not in r:
                sys.exit(f"{path}, record {i}: missing field {key!r}")
            if not isinstance(r[key], kind):
                sys.exit(f"{path}, record {i}: field {key!r} should be "
                         f"{kind.__name__}, found {type(r[key]).__name__}")
    for i, r in enumerate(rows, 1):
        for j, p in enumerate(r.get("reason_points", []), 1):
            for key in ("aspect", "claim"):
                if key not in p:
                    sys.exit(f"{path}, record {i}, reason point {j}: missing field {key!r}")


def _resume(path, rows):
    """Record indices already written to an output file, verified against the current input.

    Resume keys on position, so pointing it at an output written from a different or reordered
    file would mix two runs into one aggregate and look entirely normal. Every output row also
    carries its `id`, so that mixing is detectable: compare them and refuse rather than average
    across two datasets.
    """
    done = set()
    if not (path and os.path.exists(path)):
        return done
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue                             # skip a corrupt line rather than let resume crash
            i = rec.get("i")
            if i is None:
                continue
            if i >= len(rows):
                sys.exit(f"{path} holds record {i} but the input has only {len(rows)}; "
                         f"it was written from a different file. Use a new --output.")
            if rec.get("id") is not None and rec["id"] != rows[i].get("id"):
                sys.exit(f"{path} records {rec['id']!r} at position {i}, but the input has "
                         f"{rows[i].get('id')!r} there; it was written from a different or "
                         f"reordered file. Use a new --output.")
            done.add(i)
    return done


def _validate_aspect(aspect, where):
    if aspect not in ASPECTS:
        sys.exit(f"{where}: unknown aspect {aspect!r}; expected one of {'/'.join(ASPECTS)}")


def cmd_coverage(args):
    rows = _read(args.input)
    _require(rows, {"cand": str, "reason_points": list}, args.input)
    done = _resume(args.output, rows)
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]
    for i, r in todo:
        for p in r["reason_points"]:
            _validate_aspect(p["aspect"], f"record {i + 1} ({r.get('id')})")

    prompts, span = [], []                           # span[k] = (start, end) of case k in prompts
    for _, r in todo:
        start = len(prompts)
        prompts += [qa_text(r["cand"], p["claim"], p["aspect"]) for p in r["reason_points"]]
        span.append((start, len(prompts)))
    print(f"{len(rows)} cases, {len(done)} already scored, "
          f"{len(todo)} to go ({len(prompts)} point judgments)")

    flags = [None] * len(prompts)
    engine = make_engine(args)
    out = open(args.output, "a") if args.output else None
    written, n = 0, 0
    for idx, ans in engine.judge(prompts):
        flags[idx] = ans
        n += 1
        if n % 100 == 0:
            print(f"  ...{n}/{len(prompts)} point judgments", flush=True)
        while written < len(todo):                   # write a case only once every one of its points is judged, so no half-scored row lands
            s, e = span[written]
            if any(f is None for f in flags[s:e]):
                break
            i, r = todo[written]
            _emit(out, {"i": i, "id": r.get("id"), **_score(flags[s:e])})
            written += 1
    if out:
        out.close()

    scored = [_score(flags[s:e]) for s, e in span]
    _summarise_coverage(rows, done, args.output, scored)


def _score(flags):
    return {"score": (sum(flags) / len(flags) if flags else None),
            "n_points": len(flags), "hits": [int(f) for f in flags]}


def _emit(out, rec):
    if out:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()


def _summarise_coverage(rows, done, output, scored):
    """Aggregate over this run plus anything a previous run already wrote."""
    vals = [s["score"] for s in scored if s["score"] is not None]
    if done and output:
        for l in open(output):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r["i"] in done and r.get("score") is not None:
                vals.append(r["score"])
    if not vals:
        print("no cases scored")
        return
    print(f"=== coverage over {len(vals)} cases ===")
    print(f"  mean {sum(vals) / len(vals):.3f}")
    full = sum(1 for v in vals if v == 1.0)
    zero = sum(1 for v in vals if v == 0.0)
    print(f"  fully covered {full}/{len(vals)}   nothing covered {zero}/{len(vals)}")


def _variants(rows, i):
    """Five candidates with an expected ordering by quality for one case.

    full         every reason the examiner gave
    ablate1      one reason dropped
    ablate_half  half the reasons dropped
    wrong        another case's reasons; recurring conclusions may still overlap with this case's reference reasons
    floor        pure boilerplate, which asserts nothing about this case

    The two ablations need at least two points to be meaningful and are omitted otherwise.
    `wrong` borrows from the next case rather than a random one, so the set is reproducible.
    """
    claims = [p["claim"] for p in rows[i]["reason_points"]]
    other = [p["claim"] for p in rows[(i + 1) % len(rows)]["reason_points"]]
    v = {"full": list(claims), "wrong": list(other), "floor": [BOILER]}
    if len(claims) >= 2:
        v["ablate1"] = claims[:-1]
        v["ablate_half"] = claims[:max(1, len(claims) // 2)]
    return v


def cmd_qualify(args):
    """Decide whether an engine may be used to produce coverage scores.

    The engine is part of this metric's specification, so a coverage number is only
    comparable when the engine passes this gate. An engine that fails typically scores high
    on pure boilerplate rather than low across the board.
    """
    rows = _read(args.input)
    _require(rows, {"reason_points": list}, args.input)
    if len(rows) < 2:
        sys.exit("qualify needs at least two cases, so that `wrong` can borrow from another")
    for i, r in enumerate(rows):
        for p in r["reason_points"]:
            _validate_aspect(p["aspect"], f"record {i + 1} ({r.get('id')})")

    prompts, index = [], []                          # index[k] = (variant, case, span)
    for i, r in enumerate(rows):
        pts = r["reason_points"]
        for name, sents in _variants(rows, i).items():
            cand = "；".join(sents)
            start = len(prompts)
            prompts += [qa_text(cand, p["claim"], p["aspect"]) for p in pts]
            index.append((name, i, start, len(prompts)))
    print(f"{len(rows)} cases, {len(index)} candidates, {len(prompts)} point judgments")

    flags = [None] * len(prompts)
    engine = make_engine(args)
    n = 0
    for k, ans in engine.judge(prompts):
        flags[k] = ans
        n += 1
        if n % 100 == 0:
            print(f"  ...{n}/{len(prompts)}", flush=True)

    scored = {}
    for name, _, s, e in index:
        if e > s:
            scored.setdefault(name, []).append(sum(flags[s:e]) / (e - s))
    means = {k: sum(v) / len(v) for k, v in scored.items()}
    if not _report_gate(args.engine, means, scored):
        sys.exit(1)


def _report_gate(engine, means, scored):
    order = ("full", "ablate1", "ablate_half", "wrong", "floor")
    label = {"floor": "boilerplate"}
    print(f"\n=== engine qualification: {engine} ===")
    for name in order:
        if name in means:
            arrow = " (lower is better)" if name in ("wrong", "floor") else ""
            print(f"  {label.get(name, name):<12} {means[name]:.3f}  (n={len(scored[name])}){arrow}")
        else:
            print(f"  {label.get(name, name):<12} not measured, too few reason points")

    reasons = []
    if "floor" not in means:
        reasons.append("boilerplate was not measured")
    elif means["floor"] > GATE_BOILERPLATE_MAX:
        reasons.append(f"boilerplate {means['floor']:.3f} exceeds {GATE_BOILERPLATE_MAX}, "
                       f"so the engine credits text that makes no claim about the case")
    chain = [c for c in GATE_CHAIN if c in means]
    if len(chain) < len(GATE_CHAIN):
        reasons.append("the decline could not be checked in full: "
                       + ", ".join(c for c in GATE_CHAIN if c not in means) + " missing")
    for a, b in zip(chain, chain[1:]):
        if means[a] <= means[b]:
            reasons.append(f"{a} {means[a]:.3f} does not exceed {b} {means[b]:.3f}, "
                           f"so the metric does not register the removed reasons")

    if reasons:
        print("\n  FAIL")
        for r in reasons:
            print(f"    - {r}")
        print("\n  Coverage numbers from this engine are not comparable with the paper's.")
    else:
        print(f"\n  PASS: boilerplate {means['floor']:.3f} <= {GATE_BOILERPLATE_MAX}, "
              "and the decline is monotone.")
    return not reasons


def cmd_detector(args):
    rows = _read(args.input)
    _require(rows, {"cand": str, "claim": str, "aspect": str}, args.input)
    done = _resume(args.output, rows)
    todo = [(i, r) for i, r in enumerate(rows) if i not in done]
    for i, r in todo:
        _validate_aspect(r["aspect"], f"record {i + 1} ({r.get('id')})")
    prompts = [qa_text(r["cand"], r["claim"], r["aspect"]) for _, r in todo]
    print(f"{len(rows)} pairs, {len(done)} already judged, {len(todo)} to go")

    engine = make_engine(args)
    out = open(args.output, "a") if args.output else None
    preds, n = {}, 0
    for k, ans in engine.judge(prompts):
        i, r = todo[k]
        preds[i] = ans
        _emit(out, {"i": i, "id": r.get("id"), "type": r.get("type"),
                    "label": r.get("label"), "pred": ans})
        n += 1
        if n % 50 == 0:
            print(f"  ...{n}/{len(prompts)}", flush=True)
    if out:
        out.close()

    judged = [(r, preds[i]) for i, r in todo]
    if done and args.output:                         # fold in whatever an earlier run already wrote
        by_i = {i: r for i, r in enumerate(rows)}
        for l in open(args.output):
            try:
                rec = json.loads(l)
            except Exception:
                continue
            if rec["i"] in done and rec["i"] in by_i:
                judged.append((by_i[rec["i"]], rec["pred"]))
    _summarise_detector(judged)


def _summarise_detector(judged):
    if not judged:
        print("nothing judged")
        return
    if any(r.get("label") is None for r, _ in judged):
        print(f"{len(judged)} pairs judged; no `label` field, so no scores to report")
        return
    tp = fp = tn = fn = 0
    by_type = {}
    for r, p in judged:
        if r["label"]:
            tp += p
            fn += not p
        else:
            fp += p
            tn += not p
        d = by_type.setdefault(r.get("type", "?"), [0, 0])
        d[0] += p == r["label"]
        d[1] += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    print(f"=== detector over {tp + tn + fp + fn} pairs ===")
    print(f"  accuracy {acc:.3f} | precision {prec:.3f} | recall {rec:.3f} | F1 {f1:.3f}")
    print(f"  TP {tp}  FP {fp}  TN {tn}  FN {fn}")
    for t, (c, n) in sorted(by_type.items()):        # a low hard_neg means the detector is riding on surface overlap
        print(f"  [{t:9}] accuracy {c / n:.3f} ({c}/{n})")


# ------------------------------------------------------------------------------- selfcheck


def selfcheck():
    """The prompt contract and the aggregation, without touching an engine."""
    p = qa_text("二者都有鍋寶二字", "兩商標共同含『鍋寶』", "主要識別部分")
    assert p.startswith(_RULE), "the rule must lead the prompt"
    assert "【aspect】主要識別部分" in p and "【理由論斷】兩商標共同含『鍋寶』" in p
    assert p.index("【aspect】") < p.index("【理由論斷】") < p.index("【候選解釋】"), "field order"

    _check_prompts_doc([("_RULE", _RULE), ("_FIELDS", _FIELDS)])

    class _Err(Exception):                                   # duck-typed like the OpenAI SDK's
        def __init__(self, msg, status=None):
            super().__init__(msg)
            self.status_code = status
    temp_400 = _Err("Unsupported value: 'temperature' does not support 0 with this model.", 400)
    assert _rejects_temperature(temp_400), "a refused temperature must be recognised, not retried"
    assert not _rejects_temperature(_Err("rate limit exceeded", 429))
    assert _is_permanent(temp_400) and not _is_permanent(_Err("server error", 500)), \
        "only a 4xx is hopeless; a 5xx is worth retrying"

    assert _parse("是") is True and _parse("否") is False
    assert _parse("") is False and _parse("maybe") is False, "no verdict means 否"
    assert _parse("否，候選並未提及") is False, "the first verdict character wins"
    assert _parse("不是") is False and _parse("這不是命中") is False, "不是 is a 否"

    assert _score([True, False])["score"] == 0.5
    assert _score([True, True])["hits"] == [1, 1]
    assert _score([])["score"] is None, "a case with no points has no score"

    def _exits(fn):                                          # the guards below use sys.exit
        try:
            fn()
        except SystemExit:
            return True
        return False

    rows_ok = [{"id": "A", "cand": "x", "claim": "c", "aspect": "外觀"}]
    assert not _exits(lambda: _require(rows_ok, {"cand": str, "claim": str}, "f")), "valid rows pass"
    assert _exits(lambda: _require([{"cand": "x"}], {"cand": str, "claim": str}, "f")), \
        "a missing field must stop the run before it reaches an engine"
    assert _exits(lambda: _require([{"cand": 7, "claim": "c"}], {"cand": str, "claim": str}, "f")), \
        "a field of the wrong type must stop the run too"
    assert _exits(lambda: _require([{"reason_points": [{"aspect": "外觀"}]}],
                                   {"reason_points": list}, "f")), "a point missing claim is caught"
    assert _exits(lambda: _validate_aspect("外觀觀念", "f")), "an unknown aspect is an error"

    os.environ["LETITBE_SELFCHECK_EMPTY"] = ""      # an empty value must count as unset; see _env
    assert _env("LETITBE_SELFCHECK_EMPTY", "fallback") == "fallback"
    assert _env("LETITBE_SELFCHECK_ABSENT_VAR", "fallback") == "fallback"
    os.environ["LETITBE_SELFCHECK_EMPTY"] = "set"
    assert _env("LETITBE_SELFCHECK_EMPTY", "fallback") == "set"
    del os.environ["LETITBE_SELFCHECK_EMPTY"]

    class _Fixed:                                    # a stub engine with fixed answers: exercises the aggregation, not a model
        def __init__(self, answers):
            self.answers = answers

        def judge(self, prompts):
            for i, _ in enumerate(prompts):
                yield i, self.answers[i]

    cases = [{"id": "A", "reason_points": [{"aspect": "外觀", "claim": "c1"},
                                           {"aspect": "差異", "claim": "c2"},
                                           {"aspect": "外觀", "claim": "c3"},
                                           {"aspect": "差異", "claim": "c4"}]},
             {"id": "B", "reason_points": [{"aspect": "外觀", "claim": "d1"}]}]
    v = _variants(cases, 0)
    assert v["full"] == ["c1", "c2", "c3", "c4"]
    assert v["ablate1"] == ["c1", "c2", "c3"], "ablate1 drops exactly one reason"
    assert v["ablate_half"] == ["c1", "c2"]
    assert v["wrong"] == ["d1"], "wrong borrows from the next case"
    assert v["floor"] == [BOILER]
    assert "ablate1" not in _variants(cases, 1), "a single-point case cannot be ablated"

    # One engine that passes, one that fails on boilerplate, one that fails on the decline.
    passes = {"full": .8, "ablate1": .6, "ablate_half": .4, "wrong": .3, "floor": .01}
    loose = {**passes, "floor": .2}
    flat = {**passes, "ablate1": .9}
    ns = {k: [0] for k in passes}
    with contextlib.redirect_stdout(io.StringIO()):          # the verdict matters here, not the table
        assert _report_gate("passes", passes, ns) is True, "a monotone decline with boilerplate under the cap passes"
        assert _report_gate("loose", loose, ns) is False, "boilerplate .2 exceeds the cap"
        assert _report_gate("flat", flat, ns) is False, "ablate1 above full breaks the decline"

    rows = [{"cand": "x", "claim": "c", "aspect": "外觀", "label": True, "type": "pos"},
            {"cand": "x", "claim": "c", "aspect": "外觀", "label": False, "type": "hard_neg"}]
    judged = list(zip(rows, [True, True]))           # the second pair is a false positive -> precision 0.5, recall 1.0
    with contextlib.redirect_stdout(io.StringIO()):
        _summarise_detector(judged)
    print("selfcheck ok")


# ------------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_engine_args(p):
        p.add_argument("--engine", required=True,
                       help="model id: an Ollama or OpenAI model for --backend api, "
                            "or a Hugging Face repo for --backend hf")
        p.add_argument("--backend", choices=("api", "hf"), default="api")
        p.add_argument("--base-url", help="override the OpenAI-compatible endpoint")
        p.add_argument("--adapter", help="LoRA adapter path or repo, --backend hf only")
        p.add_argument("--workers", type=int, default=8, help="parallel requests, api only")
        p.add_argument("--batch", type=int, default=32,
                       # Batched 4-bit inference takes a different numerical path than an
                       # unbatched one, so a borderline judgment can differ between batch
                       # sizes, and resuming through --output regroups the remaining rows.
                       # Fix the batch size, and start from an empty output, when comparing scores.
                       help="batch size, hf only; see the note in the source on reproducibility")
        p.add_argument("--input", required=True, help="input JSONL")
        p.add_argument("--output", help="per-row JSONL; if it exists, finished rows are skipped")

    p = sub.add_parser("coverage", help="score candidate explanations against gold reason points")
    add_engine_args(p)
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("detector", help="evaluate the per-point detector on labelled pairs")
    add_engine_args(p)
    p.set_defaults(func=cmd_detector)

    p = sub.add_parser("qualify", help="check whether an engine may be used for coverage")
    add_engine_args(p)
    p.set_defaults(func=cmd_qualify)

    p = sub.add_parser("selfcheck", help="run the internal consistency checks and exit")
    p.set_defaults(func=lambda args: selfcheck())

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
