"""FADE: a light additive attention-pooling head on a mostly frozen backbone.

The descriptor is d = sum_i a_i v_i over the L2-normalised patch tokens v_i, with a_i a learned softmax
attention. Because d is additive, the cosine between two descriptors decomposes exactly into
patch-pair contributions C_ij (see `correspondence.py`).

Checkpoints come in two formats, both read by `load_fade`:
  * a training checkpoint (`.pt`): {"state_dict", "backbone", "n_unfreeze", "n_heads", "dual_gate",
    "head_type", "ar_mode", ...} holding the full encoder state;
  * a released checkpoint (`.safetensors`): the trained tensors plus the config in the file metadata.
    A "slim" file stores only what training changed (the last `n_unfreeze` backbone blocks, the final
    norm and the head); the frozen backbone weights are taken from the public pretrained backbone.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import load_backbone

CONFIG_KEYS = ("backbone", "n_unfreeze", "n_heads", "dual_gate", "head_type", "ar_mode", "embed_dim")
CONFIG_DEFAULTS = {"n_heads": 1, "dual_gate": False, "head_type": "attn", "ar_mode": "letterbox-gray"}


class AttnPool(nn.Module):
    """Learned attention pooling -> additive descriptor + saliency map.

    With n_heads > 1 the descriptor is the concatenation of per-head attention-pooled, L2-normalised
    sub-vectors; every released model uses n_heads = 1. `dual_gate` adds a separate saliency head `sal`
    (the decoupled, examiner-supervised saliency model); it is never mixed into the descriptor.
    """

    def __init__(self, d: int, n_heads: int = 1, dual_gate: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, n_heads))
        self.dual_gate = dual_gate
        if dual_gate:
            self.sal = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, patches: torch.Tensor):
        a = torch.softmax(self.score(patches), dim=1)            # (B, N, H) attention per head over patches
        pooled = torch.einsum("bnh,bnd->bhd", a, patches)        # (B, H, d) per-head weighted sum
        pooled = F.normalize(pooled, p=2, dim=-1)                # per-head L2
        desc = F.normalize(pooled.reshape(patches.shape[0], -1), p=2, dim=-1)   # (B, H*d) descriptor
        return desc, a.mean(dim=2)                               # descriptor, saliency (B, N)


class FADE(nn.Module):
    """Backbone (`self.model`) + additive attention-pooling head (`self.head`).

    `forward(x)` returns (descriptor (B, D), attention saliency (B, N)). Descriptors are L2-normalised;
    retrieval compares them by cosine after PCA whitening fitted on the gallery (`evaluation.fit_whiten`).
    """

    def __init__(self, backbone_name: str, device: torch.device, n_heads: int = 1, dual_gate: bool = False):
        super().__init__()
        self.model, self.spec = load_backbone(backbone_name, device)
        self.head = AttnPool(self.spec.embed_dim, n_heads, dual_gate).to(device)

    def forward(self, x: torch.Tensor):
        patches = self.model.forward_features(x)["x_norm_patchtokens"]
        patches = F.normalize(patches.float(), p=2, dim=-1)
        return self.head(patches)


def trainable_keys(enc: FADE, n_unfreeze: int) -> Set[str]:
    """State-dict keys of what the FADE recipe trains: the last `n_unfreeze` transformer blocks, the
    backbone's final norm, and the head. Everything else is the frozen pretrained backbone."""
    m = enc.model
    if hasattr(m, "blocks"):                                     # DINOv2 (torch.hub ViT)
        blocks, final_norm = m.blocks, getattr(m, "norm", None)
    elif hasattr(m, "vm"):                                       # SigLIP adapter
        blocks, final_norm = m.vm.encoder.layers, getattr(m.vm, "post_layernorm", None)
    else:
        raise ValueError(f"cannot locate transformer blocks on {type(m).__name__}")
    modules = list(blocks[-n_unfreeze:]) if n_unfreeze > 0 else []
    if final_norm is not None:
        modules.append(final_norm)
    modules.append(enc.head)
    ids = {id(t) for mod in modules for t in [*mod.parameters(), *mod.buffers()]}
    return {k for k, v in enc.state_dict(keep_vars=True).items() if id(v) in ids}


def _torch_load(path: Path) -> dict:
    """Load a training checkpoint, without arbitrary unpickling when the file allows it."""
    for kwargs in ({"weights_only": True, "mmap": True}, {"weights_only": True}):
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except Exception:
            continue
    warnings.warn(f"{path.name}: weights-only loading failed, falling back to full unpickling. "
                  f"Only load checkpoints you trust.")
    return torch.load(path, map_location="cpu", weights_only=False)


def read_checkpoint(path) -> Tuple[dict, Dict[str, torch.Tensor], str]:
    """(config, state_dict, format) of a FADE checkpoint; format is "full" or "slim"."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            state = {k: f.get_tensor(k) for k in f.keys()}
        if "fade_config" not in meta:
            raise ValueError(f"{path} has no 'fade_config' metadata; is it a FADE checkpoint?")
        return json.loads(meta["fade_config"]), state, meta.get("fade_format", "full")
    blob = _torch_load(path)
    cfg = {k: blob[k] for k in CONFIG_KEYS if k in blob}
    return cfg, blob["state_dict"], "full"


def load_fade(path, device=None) -> Tuple[FADE, dict]:
    """Build a FADE encoder from a checkpoint. Returns (encoder in eval mode, config)."""
    cfg, state, fmt = read_checkpoint(path)
    cfg = {**CONFIG_DEFAULTS, **cfg}
    if cfg["head_type"] != "attn":
        raise ValueError(f"only the additive attention head is supported, got head_type={cfg['head_type']!r}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = FADE(cfg["backbone"], device, n_heads=cfg["n_heads"], dual_gate=cfg["dual_gate"])
    if fmt == "slim":
        missing, unexpected = enc.load_state_dict(state, strict=False)
        need = trainable_keys(enc, cfg["n_unfreeze"])
        if unexpected or not need <= set(state) or need & set(missing):
            raise RuntimeError(f"{path}: slim checkpoint does not match the {cfg['backbone']} FADE layout "
                               f"(unexpected={list(unexpected)[:5]}, missing trained={sorted(need - set(state))[:5]})")
    elif fmt == "full":
        enc.load_state_dict(state)
    else:
        raise ValueError(f"unknown checkpoint format {fmt!r}")
    enc.eval()
    return enc, cfg
