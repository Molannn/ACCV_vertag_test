"""Explain why two marks are similar under FADE: the exact patch-pair decomposition C_ij of their score.

For each (applied, cited) pair this writes a figure (default: the region style of the paper's Fig. 3),
a JSON record (score, completeness residual, top patch pairs, region boxes, evidence text), and, with
--evidence-out, the region evidence (text + crops) consumed by the grounded explainer (../explainer).

One pair:
    python explain.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
        --applied applied.jpg --cited cited.jpg
Many pairs (JSONL lines with "id", "applied", "cited"):
    python explain.py --checkpoint checkpoints/fade_dinov2_vitl14_reg.safetensors \
        --pairs examples/pairs.example.jsonl --image-root /path/to/images \
        --evidence-out outputs/explain/evidence.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from fade import Correspondence
from fade.io_utils import read_pairs
from fade.visualize import render_correspondence


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="FADE checkpoint (.safetensors or .pt)")
    ap.add_argument("--applied", type=Path, help="applied (query) mark image")
    ap.add_argument("--cited", type=Path, help="cited (prior) mark image")
    ap.add_argument("--pairs", type=Path, help="JSONL of pairs: {\"id\", \"applied\", \"cited\"}")
    ap.add_argument("--image-root", type=Path, default=None, help="base dir for relative paths in --pairs")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/explain"))
    ap.add_argument("--style", choices=["region", "heatmap"], default="region",
                    help="region = one box per mark bounding --mass-q of its contribution (paper Fig. 3); "
                         "heatmap = contribution heatmaps + the top --draw-k patch pairs")
    ap.add_argument("--mass-q", type=float, default=0.7, help="contribution mass bounded by the region box")
    ap.add_argument("--no-fg", action="store_true",
                    help="do not restrict the drawing to mark content (letterbox padding / blank margins "
                         "carry part of the score and stay in C_ij either way)")
    ap.add_argument("--fg-thr", type=float, default=18.0, help="foreground threshold (0-255)")
    ap.add_argument("--draw-k", type=int, default=3, help="patch pairs drawn in the heatmap style")
    ap.add_argument("--disp", type=int, default=672, help="display size of each mark in pixels")
    ap.add_argument("--top-k", type=int, default=5, help="patch pairs in the JSON record and the evidence")
    ap.add_argument("--ctx", type=int, default=3, help="evidence crop context, in patches around each pair")
    ap.add_argument("--evidence-out", type=Path, default=None,
                    help="write the explainer's evidence index here (crops go next to it)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.pairs:
        pairs = read_pairs(args.pairs, args.image_root)
    elif args.applied and args.cited:
        pairs = [{"id": f"{args.applied.stem}__{args.cited.stem}", "applied": args.applied, "cited": args.cited}]
    else:
        ap.error("give --applied and --cited, or --pairs")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corr = Correspondence.from_checkpoint(args.checkpoint, args.device)
    evidence, ev_dir = {}, None
    if args.evidence_out:
        ev_dir = args.evidence_out.parent / "evidence_crops"
        ev_dir.mkdir(parents=True, exist_ok=True)

    for p in pairs:
        pid = str(p["id"])
        out = corr.cij(p["applied"], p["cited"])
        fig, drawn = render_correspondence(corr, out, style=args.style, mass_q=args.mass_q,
                                           fg_only=not args.no_fg, fg_thr=args.fg_thr,
                                           draw_k=args.draw_k, disp=args.disp)
        fig.save(args.out_dir / f"{pid}.png")
        ev = corr.evidence_from(out, top_k=args.top_k, ctx=args.ctx)
        record = {"id": pid, "applied": str(p["applied"]), "cited": str(p["cited"]),
                  "score": out["score"], "residual": out["residual"], "grid": corr.grid,
                  "top_pairs": ev["pairs"], "evidence_text": ev["text"], **drawn}
        (args.out_dir / f"{pid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        if ev_dir is not None:
            crops_a, crops_c = [], []
            cdir = ev_dir / pid
            cdir.mkdir(exist_ok=True)
            for k, (ac, cc) in enumerate(zip(ev["applied_crops"], ev["cited_crops"])):
                p1, p2 = cdir / f"applied_{k}.png", cdir / f"cited_{k}.png"
                ac.save(p1); cc.save(p2)
                # relative to the evidence file, so the index works from any working directory
                crops_a.append(str(p1.relative_to(args.evidence_out.parent)))
                crops_c.append(str(p2.relative_to(args.evidence_out.parent)))
            evidence[pid] = {"text": ev["text"], "n_pairs": len(ev["pairs"]),
                             "applied_crops": crops_a, "cited_crops": crops_c,
                             "applied": str(p["applied"]), "cited": str(p["cited"])}
        print(f"[{pid}] cosine {out['score']:.4f} = sum C_ij (residual {out['residual']:.1e})\n{ev['text']}\n",
              flush=True)

    if args.evidence_out:
        args.evidence_out.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote evidence for {len(evidence)} pair(s) to {args.evidence_out}")
    print(f"Wrote {len(pairs)} figure(s) and record(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
