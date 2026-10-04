"""FADE (faithful additive descriptor): the examiner-confusion retriever of VERTAG (ACCV 2026).

Inference release: backbones, the additive attention-pooling head, the exact patch-pair decomposition
C_ij of the retrieval score, retrieval / evaluation utilities and visualisation.

    from fade import Correspondence
    corr = Correspondence.from_checkpoint("checkpoints/fade_dinov2_vitl14_reg.safetensors")
    out = corr.cij("applied.jpg", "cited.jpg")      # out["C"].sum() == out["score"] up to float eps
"""
from .backbones import REGISTRY, BackboneSpec, load_backbone
from .correspondence import Correspondence, region_name
from .model import FADE, AttnPool, load_fade, read_checkpoint, trainable_keys
from .transforms import build_ar_preprocessor, build_transform

__all__ = [
    "REGISTRY", "BackboneSpec", "load_backbone",
    "FADE", "AttnPool", "load_fade", "read_checkpoint", "trainable_keys",
    "Correspondence", "region_name",
    "build_ar_preprocessor", "build_transform",
]
