"""Rank a gallery of marks for one or more query marks with FADE, and explain the top hits.

    python retrieve.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
        --gallery /path/to/gallery_dir --query query.jpg --topk 10 --explain 3

--gallery is a directory of images or a .txt file listing image paths; --query takes images and/or
directories. Ranking uses cosine over PCA-whitened descriptors, as in the paper, once the gallery is
large enough to fit the whitening (see --whiten); the C_ij explanation decomposes the unwhitened cosine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from fade import Correspondence, build_transform, load_fade
from fade.evaluation import encode, fit_whiten
from fade.io_utils import list_images
from fade.visualize import render_correspondence


def gallery_descriptors(paths, enc, tf, args, device) -> torch.Tensor:
    """Encode the gallery, reusing --cache when it holds descriptors for exactly these paths."""
    key = [str(p) for p in paths]
    if args.cache and args.cache.exists():
        blob = torch.load(args.cache, map_location="cpu", weights_only=True)
        if blob.get("paths") == key and blob.get("checkpoint") == str(args.checkpoint):
            print(f"Loaded {len(key)} gallery descriptors from {args.cache}", flush=True)
            return blob["emb"]
    emb = encode(paths, lambda x: enc(x)[0], tf, device, args.batch_size, args.num_workers)
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"paths": key, "checkpoint": str(args.checkpoint), "emb": emb}, args.cache)
    return emb


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--gallery", type=Path, required=True, help="image directory or .txt list of paths")
    ap.add_argument("--query", type=Path, nargs="+", required=True, help="query image(s) or directories")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--whiten", choices=["auto", "on", "off"], default="auto",
                    help="PCA whitening fitted on the gallery; auto = on when the gallery has at least "
                         "2x as many images as the descriptor has dimensions")
    ap.add_argument("--whiten-sample", type=int, default=20000, help="gallery descriptors used to fit it")
    ap.add_argument("--seed", type=int, default=0, help="seed of the whitening sample")
    ap.add_argument("--cache", type=Path, default=None, help="cache file for the gallery descriptors")
    ap.add_argument("--explain", type=int, default=0, help="render C_ij figures for the top-N hits")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/retrieve"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    enc, cfg = load_fade(args.checkpoint, device)
    tf = build_transform(enc.spec, cfg["ar_mode"])
    gal = list_images(args.gallery)
    queries = [q for src in args.query for q in list_images(src)]
    print(f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'} vs {len(gal)} gallery images", flush=True)

    g_emb = gallery_descriptors(gal, enc, tf, args, device)
    q_emb = encode(queries, lambda x: enc(x)[0], tf, device, args.batch_size, args.num_workers)

    use_whiten = args.whiten == "on" or (args.whiten == "auto" and len(gal) >= 2 * g_emb.shape[1])
    if use_whiten:
        gen = torch.Generator().manual_seed(args.seed)
        samp = g_emb[torch.randperm(len(gal), generator=gen)[:min(args.whiten_sample, len(gal))]]
        mu, W = fit_whiten(samp.to(device))
        g = F.normalize((g_emb.to(device) - mu) @ W, p=2, dim=1)
        q = F.normalize((q_emb.to(device) - mu) @ W, p=2, dim=1)
    else:
        if args.whiten == "auto":
            print(f"  gallery too small to fit a {g_emb.shape[1]}-d whitening; ranking by plain cosine", flush=True)
        g, q = g_emb.to(device), q_emb.to(device)

    corr = Correspondence(enc, cfg["ar_mode"]) if args.explain else None
    results = []
    for qi, qpath in enumerate(queries):
        sims = q[qi] @ g.T
        vals, idx = sims.topk(min(args.topk, len(gal)))
        hits = [{"rank": r + 1, "path": str(gal[j]), "score": float(v)}
                for r, (v, j) in enumerate(zip(vals.tolist(), idx.tolist()))]
        print(f"\nquery {qpath}  ({'whitened' if use_whiten else 'plain'} cosine)")
        for h in hits:
            print(f"  {h['rank']:>3}  {h['score']:.4f}  {h['path']}")
        for h in hits[:args.explain]:
            out = corr.cij(qpath, h["path"])
            fig, drawn = render_correspondence(corr, out)
            name = f"{qpath.stem}__rank{h['rank']:02d}__{Path(h['path']).stem}.png"
            fig.save(args.out_dir / name)
            h.update({"explanation": name, "cosine_sum_cij": out["score"], **drawn})
        results.append({"query": str(qpath), "whitened": use_whiten, "hits": hits})

    (args.out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\nWrote {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
