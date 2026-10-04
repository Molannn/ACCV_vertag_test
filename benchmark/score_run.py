"""Score your own ranking on the examiner-confusion benchmark, whatever model produced it.

Input: a CSV with a header and the columns `case_id, rank, regno`: for each query (case_id in
qrels_test.csv), the gallery marks your system returns, by rank (1 = most similar). A list may be
truncated (e.g. the top 100 or 1000).

Scoring follows the paper: a mark the examiner did not cite is unjudged, never a negative, so only the
positions of the cited marks matter; a cited mark missing from your list counts as ranked beyond every
cutoff; a query missing from your file scores as a miss. Reported: Recall@K (>= 1 cited mark in the top
K), PRES@K and mAP@K, averaged over all queries (K = 100 by default).

    python score_run.py --run my_run.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FADE"))
from fade.evaluation import map_at_k_per_query, pres_at, recall_at_k_per_query  # noqa: E402

NOT_RETRIEVED = 10 ** 9                      # rank of a cited mark that is not in the submitted list


def load_qrels(bdir: Path):
    with open(bdir / "qrels_test.csv", newline="", encoding="utf-8") as fh:
        qrels = {r["case_id"]: [x for x in r["cited_regnos"].split(";") if x] for r in csv.DictReader(fh)}
    gallery = {ln.strip() for ln in (bdir / "gallery_regnos.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
    return qrels, gallery


def load_run(path: Path):
    """{case_id: [regno, ...] in rank order}; ties keep file order, repeated marks keep the first rank."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing_cols = {"case_id", "rank", "regno"} - set(reader.fieldnames or [])
        if missing_cols:
            raise SystemExit(f"{path}: missing column(s) {sorted(missing_cols)}; expected case_id, rank, regno")
        for n, r in enumerate(reader):
            rows.setdefault(r["case_id"].strip(), []).append((float(r["rank"]), n, r["regno"].strip()))
    run = {}
    for cid, items in rows.items():
        ranked = [regno for _, _, regno in sorted(items)]
        run[cid] = list(dict.fromkeys(ranked))
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="CSV with columns case_id, rank, regno")
    ap.add_argument("--benchmark", type=Path, default=HERE, help="dir with qrels_test.csv and gallery_regnos.txt")
    ap.add_argument("--k", type=int, default=100, help="cutoff K")
    args = ap.parse_args()

    qrels, gallery = load_qrels(args.benchmark)
    run = load_run(args.run)
    unknown_q = [c for c in run if c not in qrels]
    absent_q = [c for c in qrels if c not in run]
    outside = sum(1 for c in run for r in run[c] if r not in gallery)

    rec, pres, ap_k = [], [], []
    for cid, rels in qrels.items():
        pos = {r: i + 1 for i, r in enumerate(run.get(cid, []))}
        ranks = torch.tensor([pos.get(r, NOT_RETRIEVED) for r in rels], dtype=torch.long)
        rec.append(recall_at_k_per_query(ranks, args.k))
        pres.append(pres_at(ranks, args.k))
        ap_k.append(map_at_k_per_query(ranks, args.k))

    k = args.k
    print(f"{len(qrels)} queries scored; {len(qrels) - len(absent_q)} present in the run, {len(absent_q)} absent "
          f"(scored as misses)")
    if unknown_q:
        print(f"  {len(unknown_q)} case_id(s) in the run are not benchmark queries and were ignored "
              f"(e.g. {unknown_q[:3]})")
    if outside:
        print(f"  {outside} ranked mark(s) are not in gallery_regnos.txt (fine for G2 distractors; "
              f"for G1 the gallery is gallery_regnos.txt)")
    print(f"R@{k} {np.mean(rec):.4f}   PRES@{k} {np.mean(pres):.4f}   mAP@{k} {np.mean(ap_k):.4f}")


if __name__ == "__main__":
    main()
