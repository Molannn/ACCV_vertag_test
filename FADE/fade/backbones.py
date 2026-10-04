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
    # backbone of fade_dinov2_vitl14_reg (Tab. 3, Fig. 3)
    "dinov2_vitl14_reg": BackboneSpec(
        name="dinov2_vitl14_reg", embed_dim=1024, resolution=224, patch_size=14,
        num_patches=256, mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ),
    # backbone of fade_siglip_so400m (Tab. 4)
    "siglip_so400m_224": BackboneSpec(
        name="siglip_so400m_224", embed_dim=1152, resolution=224, patch_size=14,
        num_patches=256, mean=SIGLIP_MEAN, std=SIGLIP_STD,
    ),
}


def load_backbone(name: str, device: torch.device):
    """Returns (model, spec). The model is frozen, in eval mode and on `device`."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown backbone {name}. Known: {list(REGISTRY.keys())}")
    spec = REGISTRY[name]

    if name == "dinov2_vitl14_reg":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    elif name == "siglip_so400m_224":
        from transformers import AutoModel
        siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-224")
        model = _SigLIPVisionAdapter(siglip.vision_model, spec)
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
