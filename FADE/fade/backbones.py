"""Backbone factory.

Every backbone is returned frozen and in eval mode, wrapped so that `forward_features(x)` yields a dict
with `x_norm_patchtokens` (B, N, D) and `x_norm_clstoken` (B, D), the DINOv2 convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    embed_dim: int
    resolution: int
    patch_size: int
    num_patches: int  # = (resolution // patch_size)^2
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

REGISTRY: dict[str, BackboneSpec] = {
    # backbone of the main FADE model (Tab. 3)
    "dinov2_vitl14_reg": BackboneSpec(
        name="dinov2_vitl14_reg", embed_dim=1024, resolution=224, patch_size=14,
        num_patches=256, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ),
    # backbone of the SigLIP FADE model (Tab. 4)
    "siglip_so400m_224": BackboneSpec(
        name="siglip_so400m_224", embed_dim=1152, resolution=224, patch_size=14,
        num_patches=256, mean=SIGLIP_MEAN, std=SIGLIP_STD,
    ),
    # frozen baselines of Tab. 4 only
    "dinov2_vitb14": BackboneSpec(
        name="dinov2_vitb14", embed_dim=768, resolution=224, patch_size=14,
        num_patches=256, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ),
    "eva02_clip_l14": BackboneSpec(
        name="eva02_clip_l14", embed_dim=768, resolution=224, patch_size=14,
        num_patches=256, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ),
}


def load_backbone(name: str, device: torch.device):
    """Returns (model, spec). The model is frozen, in eval mode and on `device`."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown backbone {name}. Known: {list(REGISTRY.keys())}")
    spec = REGISTRY[name]

    if name in ("dinov2_vitb14", "dinov2_vitl14_reg"):
        model = torch.hub.load("facebookresearch/dinov2", name)
    elif name == "siglip_so400m_224":
        from transformers import AutoModel
        siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-224")
        model = _SigLIPVisionAdapter(siglip.vision_model, spec)
    elif name == "eva02_clip_l14":
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("EVA02-L-14", pretrained="merged2b_s4b_b131k")
        model = _OpenCLIPVisionAdapter(clip_model.visual, spec)
    else:
        raise ValueError(f"No loader for backbone {name}")

    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, spec


class _SigLIPVisionAdapter(torch.nn.Module):
    """DINOv2-style `forward_features` for the SigLIP vision tower.

    SigLIP has no CLS token; a mean-pooled token stands in for it, but no FADE code path reads it.
    """

    def __init__(self, vision_model, spec: BackboneSpec):
        super().__init__()
        self.vm = vision_model
        self.spec = spec

    def forward_features(self, x: torch.Tensor) -> dict:
        out = self.vm(pixel_values=x)
        patches = out.last_hidden_state  # (B, N, D)
        cls_like = patches.mean(dim=1)
        return {"x_norm_patchtokens": patches, "x_norm_clstoken": cls_like}


class _OpenCLIPVisionAdapter(torch.nn.Module):
    """DINOv2-style `forward_features` for open_clip ViTs (patch tokens before the projection head)."""

    def __init__(self, visual, spec: BackboneSpec):
        super().__init__()
        self.visual = visual
        self.spec = spec

    def forward_features(self, x: torch.Tensor) -> dict:
        if hasattr(self.visual, "trunk"):
            tokens = self.visual.trunk.forward_features(x)
        else:
            tokens = self.visual.forward_features(x)
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected (B, N, D) from open_clip, got {tokens.shape}")
        if tokens.shape[1] == self.spec.num_patches + 1:
            cls = tokens[:, 0]
            patches = tokens[:, 1:]
        else:
            patches = tokens
            cls = patches.mean(dim=1)
        return {"x_norm_patchtokens": patches, "x_norm_clstoken": cls}
