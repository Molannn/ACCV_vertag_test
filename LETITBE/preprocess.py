"""LETITBE corpus preprocessing: decompose office actions into reason points, split by
year, and build the per-point detector's training pairs.

Four steps, run in order. Each writes into --data-dir (default ./data).

    python preprocess.py decompose   --corpus <corpus.jsonl>   # needs GEMINI_API_KEY
    python preprocess.py split       --corpus <corpus.jsonl>
    python preprocess.py build-pairs
    python preprocess.py build-500

`decompose` is the only step that calls an LLM, and the only one that needs
`google-genai`; the import is deferred so the other steps run without it.

The source corpus is a JSONL with one object per case:

    {"id": "T0398530", "similarity_text": "<the appearance paragraph>", "date_pub": "2024/01/05"}

It is not distributed with this repository. See README.md for how to obtain it.

Every step is deterministic: no random sampling, no shuffling. Re-running a step on the
same input reproduces the same output, except `decompose`, which calls an LLM at
temperature 0 and is therefore stable but not byte-identical across model versions.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict

from letitbe import BOILER, _check_prompts_doc

ASPECTS = ["主要識別部分", "差異", "外觀", "近似程度結論"]

# ----------------------------------------------------------------------------- decompose

MODEL = "gemini-2.5-flash"

PROMPT = """你是商標近似論述的結構化標註員。輸入一段 TIPO 核駁處分書的「商標近似」論述，拆成「審查官判斷此案為何近似的理由點」清單。
【不要抽（每案都有的套語，不帶此案資訊）】
- 抽象定義開場：「商標近似係指…整體印象有相近之處」
- 觀察方法：以普通知識經驗消費者、施以普通注意、異時異地隔離觀察、連貫唱呼
- 制式混淆結尾：「可能誤認…來自同一來源或雖不相同但有關聯之來源」← 每案都有的套語、是混淆「結果」非近似「理由」，跳過不抽
（但：該案構成近似/近似程度高之判斷要抽，見下。）
【aspect 只能選下列 4 個之一】（用語依審查基準 §5.2，括號為基準原文範例）
A. 指名「哪個字/部分」，回答「兩商標共同的是哪個元素」
   - 主要識別部分：「主要部分」＝消費者關注、事後留印象中較顯著的部分；含主要字詞與形容字詞之分。抽：指名共同的字/外文/圖形。（例：「泰山」與「小泰山」主要字詞均為「泰山」）
	商標混淆誤認審查基準：判斷商標近似，應以商標圖樣整體為觀察。此乃係由於商標呈現在商品/服務之消費者眼前的是整體圖樣，而非割裂為各部分後
分別呈現。所謂「主要部分」觀察，則係商標雖然以整體圖樣呈現，然而商品/服務之消費者關注或者事後留在其印象中的，可能是其中較為顯著的部分，此一顯著的部分即屬主要部分。
B. 指名「哪個字/部分」，回答「兩商標相異的是哪個元素」
   - 差異：些微差異/細微差異；不可機械式比對。抽：指名差在哪。（例：「house」與「horse」可能近似，「house」與「mouse」則否）
	商標混淆誤認審查基準：判斷近似另一重要原則為異時異地、隔離觀察原則。所謂一般實際購買行為態樣，係指一般消費者都是憑著對商標未必清晰完整的印象，在不同的時間或地點，來作重覆選購的行為，而不是拿著商標以併列比對的方式來選購，所以細微部分的差異，在消費者的印象中難以發揮區辨的功能，判斷商標是否近似時，得無庸納入考量。這樣一個設身處地的思考原則，在判斷商標近似時，應注意切實掌握。
C.  判斷「在外觀面向是否相像」：
   - 外觀：文字商標就外觀觀察。常同時斷言（「在外觀」）。
商標混淆誤認審查基準：傳統商標的構成要素，包括文字、圖形、記號或其聯合式等，其書寫方式、組合態樣、構圖設色、相對位置、比例大小等情形，均可能影響消費者對於該商標的注意程度，判斷商標是否構成近似，原則上以商標註冊申請之商標圖樣為準據，就商標整體外觀方面，判斷是否已達到可能誤認的近似程度，而非單純機械性比對。
D. 綜合關係
   - 近似程度結論：構成近似與否及其近似程度（高/不低/低）。
	商標混淆誤認審查基準：商標近似程度的認定，必須考量構成商標要素的識別性，及商標予消費者之整體商業印象加以判斷。商標在外觀、觀念或讀音等方面，若存在相似之處越多，近似程度就越高，反之越低。商標雖同為構近似，然僅讀音近似者，或讀音、外觀及觀念均近似；其近似之程度即有不同。商標為單純之文字商標，或為文字結合圖形、數字等其他要素之聯合式商標，在近似程度之判斷上，可能因比對的主要部分或予人整體印象有所不同，而影響考量因素之比重，必須依個案具體情況而定。
※ A、B 常在同一句出現，兩者都講就各抽一點。例：「『功夫』二字外觀近似」→ 主要識別部分「共同含『功夫』」＋ 外觀「外觀近似」。
【粒度規則】
1. claim 要含案件具體內容（指名共同字、具體差異）。原文無細節就照原文，不要自行編造。
2. 綁在一起的多面向各抽一點。
3. claim 精簡成一句理由：不含註冊號（如「註冊第00799710號」）、不複製「本件商標與據以核駁…相較」之類框架，只留理由本身（如「兩商標共同含『天春』」）。
【範例1（真實案 T0398530；文字／差異／外觀／結論）】
輸入：商標是否近似暨其近似之程度商標近似係指二商標予人之整體印象有其相近之處，其判斷得就二商標外觀、觀念及讀音為觀察，查本件商標圖樣與據以核駁註冊第00799710號「鍋寶」商標、第00998744號「鍋寶」商標相較，皆有文字「鍋寶」，僅字體之些微差異，且皆予人單純「文字」商標之整體印象，在外觀、觀念、讀音上相雷同，以具有普通知識經驗之消費者，於購買時施以普通之注意，可能會誤認二商品來自同一來源或雖不相同但有關聯之來源，應屬構成近似之商標，且其近似程度高。
輸出：{"reason_points":[{"aspect":"主要識別部分","claim":"兩商標共同含文字『鍋寶』"},{"aspect":"差異","claim":"僅字體之些微差異"},{"aspect":"外觀","claim":"外觀相雷同"} ,{"aspect":"近似程度結論","claim":"構成近似之商標、近似程度高"}]}

【範例2（真實案 T0357231；主要識別部分/近似程度結論）】
輸入：商標是否近似暨其近似之程度商標近似係指二商標予人之整體印象有其相近之處，其判斷得就二商標外觀、觀念及讀音為觀察。經查，本件「馥蘭姬雪」商標圖樣，與據以核駁註冊第593422號「馥蘭皙兒」商標圖樣、註冊第1379455號「馥蘭皙兒深層極致」商標圖樣相較，二者分別以「馥蘭姬雪」、「馥蘭皙兒」為其商標主要識別部分，予人寓目印象均有醒目且深刻之「馥蘭」為字首；另與據以核駁註冊第1582957號「姬雪」商標圖樣、註冊第1501274號「姬雪GISELLE」商標圖樣相較，二者均有相同之「姬雪」二中文字，以具有普通知識經驗之消費者，於購買時施以普通之注意，仍有可能會誤認二商品來自同一來源或雖不相同但有關聯之來源，或誤認二商標之使用人間存在關係企業、授權關係、加盟關係或其他類似關係，應屬構成近似之商標，且其近似程度高。
輸出：{"reason_points":[{"aspect":"主要識別部分","claim":"與『馥蘭皙兒』商標均以醒目字首『馥蘭』為主要識別部分"},{"aspect":"主要識別部分","claim":"與『姬雪』商標均有相同之『姬雪』二字"} ,{"aspect":"近似程度結論","claim":"應屬構成近似之商標、近似程度高"}]}

【待拆解輸入】
"""


def _decompose_one(client, types, errors, schema, text, retries=6):
    for i in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=PROMPT + text,
                config=types.GenerateContentConfig(
                    temperature=0, response_mime_type="application/json",
                    response_schema=schema),
            )
            return [p.model_dump() for p in resp.parsed.reason_points]
        except errors.APIError as e:
            if getattr(e, "code", None) in (429, 500, 503) and i < retries - 1:
                time.sleep(3 * (i + 1))
                continue
            raise


def cmd_decompose(args):
    from typing import Literal                      # deferred import: this is the only step that needs google-genai
    from google import genai
    from google.genai import errors, types
    from pydantic import BaseModel

    class Point(BaseModel):                         # response_schema pins aspect to the enum, so no invented labels
        aspect: Literal[tuple(ASPECTS)]             # type: ignore[valid-type]
        claim: str

    class Decomp(BaseModel):
        reason_points: list[Point]

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key)

    out = os.path.join(args.data_dir, "all_decomposed.jsonl")
    rows = [json.loads(l) for l in open(args.corpus)]
    if args.limit:                                  # even stride when sampling for a quality check, to keep it varied
        step = max(1, len(rows) // args.limit)
        rows = rows[::step][:args.limit]

    done = set()                                    # resume: skip ids already written
    if os.path.exists(out):
        for l in open(out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [o for o in rows if o["id"] not in done]
    print(f"{len(rows)} cases in corpus, {len(done)} already done, {len(todo)} to go, "
          f"{args.workers} workers")

    lock, cnt = threading.Lock(), [0]
    f = open(out, "a")                              # append as we go, so an interrupted run resumes

    def work(o):
        pts = _decompose_one(client, types, errors, Decomp, o["similarity_text"])
        rec = {"id": o["id"], "similarity_text": o["similarity_text"], "reason_points": pts}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            cnt[0] += 1
            if cnt[0] % 100 == 0:
                print(f"  ...{cnt[0]}/{len(todo)}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, o): o["id"] for o in todo}
        for fut in as_completed(futs):
            try:
                fut.result()                     # never skip this: without it a parse error,
            except Exception as e:               # a schema change or a bad record vanishes and
                failed.append((futs[fut], e))    # the run still exits 0 with a short corpus
    f.close()
    print(f"added {cnt[0]} cases -> {out} ({len(done) + cnt[0]} total)")
    if failed:
        print(f"\n{len(failed)} cases raised and were not written:")
        for cid, e in failed[:10]:
            print(f"  {cid}: {type(e).__name__}: {e}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        print("Re-running skips what was written, so this command retries only these.")
        sys.exit(1)


# --------------------------------------------------------------------------------- split


def _year(date_pub):
    return int(str(date_pub)[:4]) if date_pub else None


def cmd_split(args):
    """Split by publication year. The frozen all_decomposed.jsonl is never rewritten.

    date_pub is joined back in from the source corpus by id, since the decomposition output
    does not carry it. Cases published up to 2023 go to train and val, later ones to test, so
    no case leaks across the boundary. Val is a deterministic 10% of the train pool: sort by
    id and take every tenth.
    """
    gold = os.path.join(args.data_dir, "all_decomposed.jsonl")
    id2date = {}
    for l in open(args.corpus):
        r = json.loads(l)
        id2date[r["id"]] = r.get("date_pub")

    rows = [json.loads(l) for l in open(gold)]      # read-only
    for r in rows:
        r["date_pub"] = id2date.get(r["id"])        # in memory only; the frozen gold file is never rewritten

    miss = [r["id"] for r in rows if _year(r["date_pub"]) is None]
    if miss:
        # Refusing is the only honest option. A case with no publication date cannot be placed
        # on either side of the year line, and silently treating it as old would put an unknown
        # case in the training split, which is the one thing the split exists to prevent.
        sys.exit(f"{len(miss)} cases have no date_pub (e.g. {miss[:3]}); the split is defined by "
                 f"publication year, so these cannot be placed. Fix them in the corpus or drop "
                 f"them before splitting.")

    test = [r for r in rows if _year(r["date_pub"]) > 2023]
    pool = sorted((r for r in rows if _year(r["date_pub"]) <= 2023), key=lambda r: r["id"])
    val = [r for i, r in enumerate(pool) if i % 10 == 0]
    val_ids = {r["id"] for r in val}
    train = [r for r in pool if r["id"] not in val_ids]

    splits = {"train": train, "val": val, "test": test}
    allids = [r["id"] for s in splits.values() for r in s]
    assert len(allids) == len(set(allids)) == len(rows), "splits overlap or drop cases"
    assert all(_year(r["date_pub"]) > 2023 for r in test), "a test case is not after the line"
    assert all(_year(r["date_pub"]) <= 2023 for r in train + val), "a train case is after the line"


    for name, recs in splits.items():
        path = os.path.join(args.data_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in recs:
                r["split"] = name
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        yrs = sorted({_year(r["date_pub"]) for r in recs if _year(r["date_pub"])})
        span = f"{yrs[0]}-{yrs[-1]}" if yrs else "-"
        print(f"{name:5} {len(recs):6} cases  {span} -> {os.path.basename(path)}")


# ---------------------------------------------------------------------------- build-pairs

ELEM_ASPECTS = {"主要識別部分", "差異"}              # case-specific aspects whose elements can be borrowed or swapped
ELEM_RE = re.compile(r"[『「](.+?)[』」]")
# Polarity and degree flips: near-identical wording, opposite answer, which forces the
# detector to read meaning rather than match strings.
FLIP_CONC = [("構成近似", "不構成近似"), ("近似程度高", "近似程度低"),
             ("近似程度不低", "近似程度低"), ("高度近似", "不近似")]
FLIP_APP = [("相彷彿", "不相彷彿"), ("相雷同", "不相雷同"), ("極為近似", "不近似"),
            ("高度近似", "不近似"), ("近似", "不近似")]

# Boilerplate negatives: the candidate only recites the standard framing about how similarity
# is assessed and makes no judgment about the given claim, so the label is 否. This teaches the
# detector that reciting procedure is not asserting a reason. Variants are cycled deterministically.
BOILER_VARS = [
    BOILER,                                          # the same text `letitbe.py qualify` scores
    "商標是否近似暨其近似之程度，應就商標圖樣整體為觀察，並考量商標之主要識別部分，以具有普通知識經驗之消費者於購買時施以普通之注意為準，本於客觀事實依經驗法則綜合判斷之。",
    "按判斷商標是否近似，應就商標之外觀、觀念及讀音，以異時異地、隔離通體觀察及主要部分觀察之方式為之，並斟酌個案之具體情形審認之。",
    "商標圖樣近似與否，須就其設色、圖形、文字設計及整體構圖意匠等因素通盤斟酌，非得將商標割裂觀察或僅憑局部之異同即遽下斷語。",
]


def elems(claim):
    return ELEM_RE.findall(claim)


def eligible(p):
    return p["aspect"] in ELEM_ASPECTS and bool(elems(p["claim"]))


def _flip(claim, rules):
    for s, d in rules:
        if s in claim:
            return claim.replace(s, d, 1)
    return None


def hard_neg(p, R, pool, start=0):
    """Perturb one point into a false claim, returning (claim, kind) or (None, None).

    kind is either swap, which replaces the named element, or flip, which reverses polarity.
    """
    a, claim = p["aspect"], p["claim"]
    if a == "主要識別部分":                          # swap the element, only for a claim naming exactly one
        es = elems(claim)
        if len(es) == 1 and pool:
            X, m = es[0], len(pool)
            for j in range(m):
                Y = pool[(start + j) % m]
                if Y != X and Y not in R and X in claim:
                    return claim.replace(X, Y, 1), "swap"
    if a == "近似程度結論":                          # flip the conclusion
        h = _flip(claim, FLIP_CONC)
        if h:
            return h, "flip"
    if a == "外觀":                                  # flip the appearance judgment
        h = _flip(claim, FLIP_APP)
        if h:
            return h, "flip"
    return None, None


def build(rows):
    """Build candidate-claim pairs whose labels follow from the construction, not annotation.

    positive      passage R paired with one of R's own points               -> 是
    easy negative R paired with a claim from another case, sharing no
                  element with R, preferring cases that share characters
                  with this one so the pair is not trivially separable      -> 否
    hard negative one of R's own points perturbed, by swapping the named
                  element or reversing polarity                             -> 否

    Roughly 2:1:1 across the three, with each kind of negative capped at half the positives.
    """
    rows = sorted(rows, key=lambda r: r["id"])
    pool = sorted({e for r in rows for p in r["reason_points"] if eligible(p) for e in elems(p["claim"])
                   if len(e) >= 2 and re.search(r"[一-鿿A-Za-z]", e)})   # drop junk elements such as a bare "+"
    borrow = [(p["claim"], p["aspect"], r["id"]) for r in rows for p in r["reason_points"] if eligible(p)]
    nb = len(borrow)
    char2b = defaultdict(list)                       # character -> indices into borrow, for finding lookalike cases
    for bi, (claim, asp, src) in enumerate(borrow):
        for ch in {ch for e in elems(claim) for ch in e}:
            char2b[ch].append(bi)

    pos, easy, hard = [], [], []
    for i, r in enumerate(rows):
        R = r["similarity_text"]
        for p in r["reason_points"]:
            pos.append({"id": r["id"], "cand": R, "claim": p["claim"], "aspect": p["aspect"],
                        "label": True, "type": "pos"})
            h, kind = hard_neg(p, R, pool, start=i)
            if h and all(e not in R for e in elems(h)):   # guard: discard if a real element survived the perturbation
                hard.append({"id": r["id"], "cand": R, "claim": h, "aspect": p["aspect"],
                             "label": False, "type": "hard_neg", "rule": kind})

        my_chars = {ch for p in r["reason_points"] if eligible(p) for e in elems(p["claim"]) for ch in e}
        got, used, seen = 0, set(), set()

        def take(bi):
            nonlocal got
            if bi in seen:
                return
            seen.add(bi)
            claim, asp, src = borrow[bi]
            if src == r["id"] or claim in used:
                return
            if all(e not in R for e in elems(claim)):
                easy.append({"id": r["id"], "cand": R, "claim": claim, "aspect": asp,
                             "label": False, "type": "easy_neg"})
                used.add(claim)
                got += 1

        for ch in sorted(my_chars):                  # confusable cases first
            for bi in char2b[ch]:
                take(bi)
                if got >= 2:
                    break
            if got >= 2:
                break
        for k in range(1, nb):                       # top up with (i+k) if still short
            if got >= 2:
                break
            take((i + k) % nb)

    def dedup(lst):                                  # deduplicate by (case, claim)
        s, out = set(), []
        for r in lst:
            k = (r["id"], r["claim"])
            if k not in s:
                s.add(k)
                out.append(r)
        return out

    pos, easy, hard = dedup(pos), dedup(easy), dedup(hard)
    hard.sort(key=lambda r: 0 if r.get("rule") == "flip" else 1)   # flips survive the cap ahead of swaps
    cap = len(pos) // 2
    return pos + easy[:cap] + hard[:cap]


def boiler_negs(rows):
    """Pair every true claim with a stretch of boilerplate as the candidate, labelled 否.

    BOILER_VARS is cycled deterministically, so the output does not depend on a seed.
    """
    out = []
    for i, r in enumerate(sorted(rows, key=lambda r: r["id"])):
        for j, p in enumerate(r["reason_points"]):
            out.append({"id": r["id"], "cand": BOILER_VARS[(i + j) % len(BOILER_VARS)],
                        "claim": p["claim"], "aspect": p["aspect"],
                        "label": False, "type": "boiler_neg"})
    return out


def cmd_build_pairs(args):
    selfcheck()
    suffix = "_hits_boiler" if args.boiler else "_hits"
    for sp in ("train", "val", "test"):
        src = os.path.join(args.data_dir, f"{sp}.jsonl")
        if not os.path.exists(src):
            print(f"skipping {sp}: {src} not found")
            continue
        rows = [json.loads(l) for l in open(src)]
        recs = build(rows)
        if args.boiler:
            recs = recs + boiler_negs(rows)
        out = os.path.join(args.data_dir, f"{sp}{suffix}.jsonl")
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter(r["type"] for r in recs)
        hc = Counter(r.get("rule") for r in recs if r["type"] == "hard_neg")
        print(f"{sp:5} {len(recs):7} pairs  pos {c['pos']} / easy {c['easy_neg']} / "
              f"hard {c['hard_neg']} (flip {hc['flip']} / swap {hc['swap']}) / "
              f"boiler {c['boiler_neg']} -> {os.path.basename(out)}")


# ----------------------------------------------------------------------------- build-500


def cmd_build_500(args):
    """Take a fixed balanced subset of test_hits.jsonl, half positive and half negative.

    Sampling is an even stride with no randomness, so every machine produces the same file.
    """
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "test_hits.jsonl"))]
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    half = args.n // 2
    samp = pos[::max(1, len(pos) // half)][:half] + neg[::max(1, len(neg) // half)][:half]

    out = os.path.join(args.data_dir, f"test_hits_{args.n}.jsonl")
    with open(out, "w") as f:
        for r in samp:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{os.path.basename(out)}: {len(samp)} pairs  "
          f"pos {sum(r['label'] for r in samp)} / neg {sum(not r['label'] for r in samp)}")
    print("negatives by type:", dict(Counter(r["type"] for r in samp if not r["label"])))


# ----------------------------------------------------------------------------- selfcheck


def selfcheck():
    """The three hard-negative rules and build()'s invariants. Runs before every build-pairs."""
    _check_prompts_doc([("the decomposition PROMPT", PROMPT)])
    R = "本件商標與據以核駁商標均有相同之『鍋寶』，外觀近似，應屬構成近似之商標、近似程度高。"
    assert hard_neg({"aspect": "主要識別部分", "claim": "兩商標共同含『鍋寶』"},
                    R, ["天春"])[0] == "兩商標共同含『天春』", "swap"
    assert hard_neg({"aspect": "近似程度結論", "claim": "應屬構成近似之商標"},
                    R, [])[0] == "應屬不構成近似之商標", "flip conclusion"
    assert hard_neg({"aspect": "外觀", "claim": "外觀近似"}, R, [])[0] == "外觀不近似", "flip appearance"

    rows = [
        {"id": "A", "similarity_text": R, "reason_points": [
            {"aspect": "主要識別部分", "claim": "兩商標共同含『鍋寶』"},
            {"aspect": "外觀", "claim": "外觀近似"},
            {"aspect": "近似程度結論", "claim": "應屬構成近似之商標、近似程度高"}]},
        {"id": "B", "similarity_text": "別案原文，含『天春』。",
         "reason_points": [{"aspect": "主要識別部分", "claim": "兩商標共同含『天春』"}]},
    ]
    recs = build(rows)
    pos = [r for r in recs if r["type"] == "pos"]
    easy = [r for r in recs if r["type"] == "easy_neg"]
    hard = [r for r in recs if r["type"] == "hard_neg"]
    assert all(r["label"] for r in pos), "positives must be labelled true"
    assert not any(r["label"] for r in easy + hard), "negatives must be labelled false"
    assert all(all(e not in r["cand"] for e in elems(r["claim"])) for r in easy), \
        "an easy negative must not share an element with its candidate"
    print(f"selfcheck ok (pos={len(pos)} easy={len(easy)} hard={len(hard)})")


# ---------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="where the intermediate files live")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("decompose", help="office action paragraphs -> reason points (needs an LLM)")
    p.add_argument("--corpus", required=True, help="source corpus JSONL")
    p.add_argument("-n", "--limit", type=int, help="only process N cases, sampled evenly")
    p.add_argument("--workers", type=int, default=10)
    p.set_defaults(func=cmd_decompose)

    p = sub.add_parser("split", help="split by publication year into train/val/test")
    p.add_argument("--corpus", required=True, help="source corpus JSONL, for date_pub")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("build-pairs", help="build the detector's candidate-claim pairs")
    p.add_argument("--boiler", action="store_true", help="also emit boilerplate negatives")
    p.set_defaults(func=cmd_build_pairs)

    p = sub.add_parser("build-500", help="fixed balanced subset of test_hits.jsonl")
    p.add_argument("-n", type=int, default=500)
    p.set_defaults(func=cmd_build_500)

    p = sub.add_parser("selfcheck", help="run the internal consistency checks and exit")
    p.set_defaults(func=lambda args: selfcheck())

    args = ap.parse_args()
    if args.cmd != "selfcheck":          # a pure check should have no side effects
        os.makedirs(args.data_dir, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
