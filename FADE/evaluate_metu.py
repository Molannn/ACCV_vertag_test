"""Evaluate on the METU-v2 near-duplicate benchmark (the paper's "near-duplicate ruler").

The 417 METU-v2 queries are injected after the gallery; the relevant set of a query is the other
queries of its group (file name <instance>-<group>.jpg); the query itself is excluded. Descriptors
are PCA-whitened on up to 20,000 gallery images. A run scores the FADE checkpoint and the frozen GeM
baseline of the same backbone on the same gallery, in the order of the paper's runs.

    python evaluate_metu.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
        --gallery-dir /path/to/METU/930k_logo_v3 \
        --query-dir /path/to/METU/new_queryset_allinone --output outputs/eval_metu_dinov2.json

--pool 0 (default) uses all 922,926 gallery images (the paper's Tab. 3 setting, a few GPU hours);
--pool 50000 evaluates on a seeded 50k subset. --content-types adds a per-type breakdown.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fade import build_transform, load_backbone, load_fade
from fade.evaluation import encode, eval_metu, gem_pool

QUERY_NAME_RE = re.compile(r"^(\d+)-(\d+)\.jpg$")          # <instance>-<group>.jpg; group = 2nd number


def load_queries(query_dir: Path):
    out = []
    for jpg in sorted(query_dir.glob("*.jpg")):
        m = QUERY_NAME_RE.match(jpg.name)
        if m is None:
            raise ValueError(f"Bad query filename: {jpg.name}")
        out.append((jpg, int(m.group(2))))
    if not out:
        raise FileNotFoundError(f"no METU queries under {query_dir}")
    return out


def load_content_types(path: Path):
    """{image stem: TEXT | SHAPE | TEXT-SHAPE | ...} from METU's content-type list."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("<"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].rsplit("/", 1)[-1]
        out[name[:-4] if name.endswith(".jpg") else name] = parts[1]
    return out


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--gallery-dir", type=Path, required=True, help="METU-v2 gallery (930k_logo_v3, *.jpg)")
    ap.add_argument("--query-dir", type=Path, required=True, help="METU-v2 queries (new_queryset_allinone)")
    ap.add_argument("--content-types", type=Path, default=None, help="optional logo_content_types.txt")
    ap.add_argument("--pool", type=int, default=0, help="gallery images to use (0 = all 922,926)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=Path("outputs/metu_eval.json"))
    args = ap.parse_args()
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)

    enc, cfg = load_fade(args.checkpoint, device)
    tf = build_transform(enc.spec, cfg["ar_mode"])
    gal = sorted(args.gallery_dir.glob("*.jpg"))
    if not gal:
        raise FileNotFoundError(f"no .jpg under {args.gallery_dir}")
    qrs = load_queries(args.query_dir)
    random.shuffle(gal)
    pool = gal[:args.pool] if args.pool else gal
    qpaths = [p for p, _ in qrs]; qgroups = [g for _, g in qrs]
    types = load_content_types(args.content_types) if args.content_types else {}
    qctypes = [types.get(p.stem, "UNKNOWN") for p in qpaths]
    print(f"METU-v2: pool={len(pool)} + {len(qpaths)} queries in {len(set(qgroups))} groups", flush=True)
    enc_kw = dict(bs=args.batch_size, nw=args.num_workers)

    # frozen GeM baseline: a fresh copy of the same backbone, no training
    base_model, _ = load_backbone(cfg["backbone"], device); base_model.eval()

    def gem_fn(imgs):
        pt = F.normalize(base_model.forward_features(imgs)["x_norm_patchtokens"].float(), p=2, dim=-1)
        return gem_pool(pt)

    t0 = time.time()
    base = eval_metu(encode(pool, gem_fn, tf, device, **enc_kw), encode(qpaths, gem_fn, tf, device, **enc_kw),
                     qgroups, qctypes, device)
    print(f"  frozen-GeM  mAP@100 {base['overall']['mAP@100']:.4f}  NAR {base['overall']['NAR']:.4f}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t0 = time.time()
    fade_fn = lambda imgs: enc(imgs)[0]
    res = eval_metu(encode(pool, fade_fn, tf, device, **enc_kw), encode(qpaths, fade_fn, tf, device, **enc_kw),
                    qgroups, qctypes, device)
    m, bm = res["overall"]["mAP@100"], base["overall"]["mAP@100"]
    print(f"  FADE        mAP@100 {m:.4f}  NAR {res['overall']['NAR']:.4f}  (delta {m - bm:+.4f}, "
          f"{(time.time()-t0)/60:.1f} min)", flush=True)
    for t in sorted({*res} - {"overall", "per_query"}):
        if t in base:
            print(f"    {t:<12} frozen {base[t]['mAP@100']:.4f} -> FADE {res[t]['mAP@100']:.4f}  (n={res[t]['n']})")

    args.output.write_text(json.dumps(
        {"pool": len(pool), "n_queries": len(qpaths), "seed": args.seed, "backbone": cfg["backbone"],
         "frozen_map100": bm, "fade_map100": m, "delta_map100": m - bm, "fade": res, "frozen": base},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
