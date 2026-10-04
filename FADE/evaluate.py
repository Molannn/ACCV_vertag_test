"""Evaluate on the examiner-confusion benchmark (../benchmark): G1, and optionally G2.

For each held-out query (an applied mark) the gallery is ranked by whitened-descriptor cosine and the
ranks of the prior marks the examiner cited are scored; gallery marks the examiner did not cite are
unjudged, never negatives. Primary metrics: Recall@100 and PRES@100; mAP@100 and NAR are secondary.

  * G1: the 83,336 cited prior marks (the TIPO register).
  * G2: G1 injected into the METU-v2 gallery (922,926 images), 1,006,262 in total (--g2-metu).

A run always scores the frozen GeM baseline of --backbone first and then the FADE checkpoint, so the
paired delta comes from one run; this is also the order (and the seeded random whitening samples)
of the runs reported in the paper.

Images: put the applied marks in <images>/queries/<case_id>.<ext> and the prior marks in
<images>/gallery/<registration number>.<ext> (see ../benchmark/README.md for obtaining them).

    python evaluate.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
        --images /path/to/tipo_images --output outputs/eval_g1_dinov2.json
    python evaluate.py --checkpoint checkpoints/fade_siglip_so400m.safetensors \
        --backbone siglip_so400m_224 --images /path/to/tipo_images \
        --g2-metu /path/to/METU/930k_logo_v3 --output outputs/eval_g2_siglip.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fade import build_transform, load_backbone, load_fade
from fade.evaluation import encode, gem_pool, score_confusion
from fade.io_utils import index_by_stem

REPO = Path(__file__).resolve().parents[1]


def load_benchmark(bdir: Path):
    """(queries [(case_id, [cited regnos])], gallery [regno]) from the released label files."""
    with open(bdir / "qrels_test.csv", newline="", encoding="utf-8") as fh:
        queries = [(r["case_id"], [x for x in r["cited_regnos"].split(";") if x]) for r in csv.DictReader(fh)]
    gallery = [ln.strip() for ln in (bdir / "gallery_regnos.txt").read_text(encoding="utf-8").splitlines()
               if ln.strip()]
    return queries, gallery


def resolve(ids, directory: Path, what: str, skip_missing: bool):
    """Image path per id (None if missing and --skip-missing)."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{what} image directory not found: {directory}")
    index = index_by_stem(directory)
    paths = [index.get(i) for i in ids]
    missing = [i for i, p in zip(ids, paths) if p is None]
    if missing and not skip_missing:
        raise FileNotFoundError(f"{len(missing)} of {len(ids)} {what} images missing in {directory} "
                                f"(e.g. {missing[:5]}); pass --skip-missing to evaluate on the rest")
    if missing:
        print(f"!! {len(missing)} {what} images missing: results are NOT comparable with the paper", flush=True)
    return paths


def per_query_arrays(recs):
    keys = [k for k, v in recs[0].items() if isinstance(v, (int, float))] if recs else []
    return {k: [float(r[k]) for r in recs] for k in keys}


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None, help="FADE checkpoint; omit to score the "
                                                                   "frozen baseline only")
    ap.add_argument("--benchmark", type=Path, default=REPO / "benchmark", help="dir with the label files")
    ap.add_argument("--images", type=Path, required=True, help="dir with queries/ and gallery/ images")
    ap.add_argument("--backbone", type=str, default="dinov2_vitl14_reg",
                    help="frozen baseline backbone (GeM pooling); use the checkpoint's backbone for the paired delta")
    ap.add_argument("--ar-mode", type=str, default="letterbox-gray", help="preprocessing of the frozen baseline")
    ap.add_argument("--g2-metu", type=Path, default=None,
                    help="METU-v2 gallery dir (930k_logo_v3, *.jpg): also score G2 (needs hours)")
    ap.add_argument("--metu-pool", type=int, default=0, help="METU distractors for G2 (0 = all)")
    ap.add_argument("--skip-missing", action="store_true", help="drop queries / gallery marks without an image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path("outputs/benchmark_eval.json"))
    ap.add_argument("--dump-per-query", action="store_true",
                    help="also save per-query metrics (aligned across models) for paired bootstrap CIs")
    args = ap.parse_args()
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)

    queries, gallery = load_benchmark(args.benchmark)
    gal_paths = resolve(gallery, args.images / "gallery", "gallery", args.skip_missing)
    keep = [i for i, p in enumerate(gal_paths) if p is not None]
    gallery, gal_paths = [gallery[i] for i in keep], [gal_paths[i] for i in keep]
    reg2idx = {r: i for i, r in enumerate(gallery)}
    q_found = resolve([cid for cid, _ in queries], args.images / "queries", "query", args.skip_missing)
    q_ids, q_paths, rel_idx = [], [], []
    for (cid, rels), p in zip(queries, q_found):
        rel = torch.tensor([reg2idx[r] for r in rels if r in reg2idx], dtype=torch.long)
        if p is None or (args.skip_missing and rel.numel() == 0):
            continue
        q_ids.append(cid); q_paths.append(p); rel_idx.append(rel)
    mean_rel = sum(r.numel() for r in rel_idx) / max(1, len(rel_idx))
    print(f"Benchmark: {len(q_paths)} queries, {len(gal_paths)} gallery marks "
          f"(mean {mean_rel:.3f} relevants/q)", flush=True)

    # descriptor functions: the frozen-GeM baseline first, then the FADE checkpoint
    fns = {}
    base_model, base_spec = load_backbone(args.backbone, device); base_model.eval()
    tf_base = build_transform(base_spec, args.ar_mode)
    fns["frozen-GeM"] = (lambda imgs: gem_pool(
        F.normalize(base_model.forward_features(imgs)["x_norm_patchtokens"].float(), p=2, dim=-1)), tf_base)
    if args.checkpoint is not None:
        enc, cfg = load_fade(args.checkpoint, device)
        if cfg["backbone"] != args.backbone:
            print(f"  note: the checkpoint's backbone is {cfg['backbone']}, the frozen baseline is "
                  f"{args.backbone}", flush=True)
        fns[f"FADE:{args.checkpoint.stem}"] = (lambda imgs: enc(imgs)[0], build_transform(enc.spec, cfg["ar_mode"]))

    metu_paths = []
    if args.g2_metu:
        metu_paths = sorted(args.g2_metu.glob("*.jpg"))
        if not metu_paths:
            raise FileNotFoundError(f"no .jpg under {args.g2_metu}")
        random.shuffle(metu_paths)
        if args.metu_pool:
            metu_paths = metu_paths[:args.metu_pool]
        print(f"G2: injecting {len(gallery)} TIPO marks into {len(metu_paths)} METU distractors", flush=True)

    results = {"benchmark": {"n_queries": len(q_paths), "n_gallery": len(gal_paths),
                             "mean_relevants_per_query": round(mean_rel, 3), "skip_missing": args.skip_missing,
                             "frozen_backbone": args.backbone, "seed": args.seed},
               "G1": {}, "G2": {} if metu_paths else None}
    perq = {"query_ids": [cid for cid, rel in zip(q_ids, rel_idx) if rel.numel() > 0],   # = scored queries
            "G1": {}, "G2": {}}
    enc_kw = dict(bs=args.batch_size, nw=args.num_workers)
    for name, (fn, tf) in fns.items():
        t0 = time.time()
        g_emb = encode(gal_paths, fn, tf, device, **enc_kw)
        q_emb = encode(q_paths, fn, tf, device, **enc_kw)
        g1, n1, recs1 = score_confusion(q_emb, g_emb, rel_idx, device)
        results["G1"][name] = {**g1, "n_queries": n1}
        perq["G1"][name] = per_query_arrays(recs1)
        print(f"\n[{name}] G1 ({len(gal_paths)} gallery, {(time.time()-t0)/60:.1f} min):"
              f"  [primary] R@100 {g1['R@100']:.4f}  PRES@100 {g1['PRES@100']:.4f}  |  "
              f"mAP@100 {g1['mAP@100']:.4f}  R@1 {g1['R@1']:.4f}  R@10 {g1['R@10']:.4f}  NAR {g1['NAR']:.4f}",
              flush=True)
        if metu_paths:
            md_emb = encode(metu_paths, fn, tf, device, **enc_kw)
            g2_emb = torch.cat([g_emb, md_emb])                        # TIPO marks first, then distractors
            g2, n2, recs2 = score_confusion(q_emb, g2_emb, rel_idx, device)
            results["G2"][name] = {**g2, "n_queries": n2, "gallery_size": g2_emb.shape[0]}
            perq["G2"][name] = per_query_arrays(recs2)
            print(f"[{name}] G2 ({g2_emb.shape[0]} gallery):  [primary] R@100 {g2['R@100']:.4f}  "
                  f"PRES@100 {g2['PRES@100']:.4f}  |  mAP@100 {g2['mAP@100']:.4f}  R@1 {g2['R@1']:.4f}  "
                  f"NAR {g2['NAR']:.4f}", flush=True)

    if args.checkpoint is not None:
        for gal in ("G1", "G2") if metu_paths else ("G1",):
            fade_name = next(k for k in results[gal] if k.startswith("FADE:"))
            results[f"delta_{gal}"] = {m: results[gal][fade_name][m] - results[gal]["frozen-GeM"][m]
                                       for m in ("R@100", "PRES@100", "mAP@100")}
            print(f">>> {gal} FADE - frozen: " + "  ".join(f"{m} {v:+.4f}" for m, v in results[f'delta_{gal}'].items()))

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.output}", flush=True)
    if args.dump_per_query:
        pq = args.output.with_name(args.output.stem + "_perquery.json")
        pq.write_text(json.dumps(perq, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote per-query arrays {pq}", flush=True)


if __name__ == "__main__":
    main()
