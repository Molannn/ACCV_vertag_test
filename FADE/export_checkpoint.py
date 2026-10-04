"""Convert a FADE training checkpoint (.pt) into a release file (.safetensors).

By default the release file is "slim": it stores only the tensors that training changed (the last
`n_unfreeze` backbone blocks, the final norm and the head) and restores everything else from the public
pretrained backbone at load time. Every tensor left out is first checked to be bit-identical to that
pretrained backbone, and the written file is reloaded with `load_fade` and compared tensor by tensor
with the source checkpoint. --full stores every tensor instead.

    python export_checkpoint.py --checkpoint fade_dinov2_training.pt \
        --output checkpoints/fade_dinov2_vitl14_reg.safetensors
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from fade.model import CONFIG_DEFAULTS, CONFIG_KEYS, FADE, load_fade, read_checkpoint, trainable_keys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="training checkpoint (.pt)")
    ap.add_argument("--output", type=Path, required=True, help="release file (.safetensors)")
    ap.add_argument("--full", action="store_true", help="store every tensor (no pretrained-backbone reuse)")
    args = ap.parse_args()
    if args.output.suffix != ".safetensors":
        ap.error("--output must end in .safetensors")

    cfg, state, _ = read_checkpoint(args.checkpoint)
    cfg = {**CONFIG_DEFAULTS, **cfg}
    if cfg["head_type"] != "attn":
        raise SystemExit(f"only the additive attention head is supported, got head_type={cfg['head_type']!r}")
    cpu = torch.device("cpu")
    base = FADE(cfg["backbone"], cpu, n_heads=cfg["n_heads"], dual_gate=cfg["dual_gate"])
    base_sd = base.state_dict()
    if set(base_sd) != set(state):
        raise SystemExit(f"layout mismatch with {cfg['backbone']}: "
                         f"{sorted(set(base_sd) ^ set(state))[:10]}")
    trained = trainable_keys(base, cfg["n_unfreeze"])
    changed_frozen = [k for k in base_sd if k not in trained and not torch.equal(base_sd[k], state[k])]
    if changed_frozen and not args.full:
        raise SystemExit(f"{len(changed_frozen)} frozen tensors differ from the pretrained backbone "
                         f"(e.g. {changed_frozen[:5]}); export with --full")

    keep = sorted(state) if args.full else sorted(trained)
    tensors = {k: state[k].detach().clone().contiguous() for k in keep}
    meta = {"fade_config": json.dumps({k: cfg[k] for k in CONFIG_KEYS if k in cfg}),
            "fade_format": "full" if args.full else "slim",
            "source_checkpoint": args.checkpoint.name}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=meta)
    del base, base_sd

    enc, _ = load_fade(args.output, cpu)
    reloaded = enc.state_dict()
    bad = [k for k in state if not torch.equal(reloaded[k], state[k])]
    if bad:
        args.output.unlink()
        raise SystemExit(f"round trip FAILED on {len(bad)} tensors (e.g. {bad[:5]}); nothing written")
    n_params = sum(t.numel() for t in tensors.values())
    print(f"Wrote {args.output} ({meta['fade_format']}: {len(tensors)} tensors, {n_params/1e6:.1f}M parameters, "
          f"{args.output.stat().st_size/2**20:.1f} MiB)")
    print(f"config: {meta['fade_config']}")
    print(f"round trip: all {len(state)} tensors identical to {args.checkpoint.name}")
    print(f"sha256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
