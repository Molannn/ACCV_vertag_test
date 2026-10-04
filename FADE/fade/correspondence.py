"""Faithful patch-pair correspondence C_ij of a FADE model.

For the additive descriptor d = sum_i a_i v_i the deployed (unwhitened) cosine decomposes exactly:

    s(A,B) = d_A . d_B / (|d_A| |d_B|) = sum_{i,j} a_i^A a_j^B (v_i^A . v_j^B) / (|d_A| |d_B|) = sum_{i,j} C_ij

C_ij is the contribution of the patch pair (i of the applied mark, j of the cited mark). Properties:
  * completeness: sum_ij C_ij equals the score exactly (the returned `residual` is ~float eps);
  * no argmax or assignment step, and no separate matcher: the explanation is the score itself;
  * background is suppressed by the learned attention a_i, not by a hand-made mask.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .model import AttnPool, FADE, load_fade
from .transforms import build_ar_preprocessor, build_transform

# coarse 3x3 region names used in the evidence text given to the explainer: top/bottom, left/right
_V = ["上", "", "下"]
_H = ["左", "", "右"]


def region_name(idx: int, grid: int) -> str:
    r, c = idx // grid, idx % grid
    name = _V[min(int(r * 3 / grid), 2)] + _H[min(int(c * 3 / grid), 2)]
    return name or "中央"


def _open(img) -> Image.Image:
    return img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")


class Correspondence:
    """C_ij readout of a FADE encoder."""

    def __init__(self, encoder: FADE, ar_mode: str = "letterbox-gray"):
        if not isinstance(encoder.head, AttnPool) or encoder.head.n_heads != 1:
            raise ValueError("C_ij is implemented for the single-head additive attention head")
        self.enc = encoder.eval()
        self.device = next(encoder.head.parameters()).device
        self.spec = encoder.spec
        self.tf = build_transform(self.spec, ar_mode)
        self.letterbox = build_ar_preprocessor(ar_mode, self.spec.resolution)
        self.grid = int(round(self.spec.num_patches ** 0.5))             # 256 patches -> 16 x 16

    @classmethod
    def from_checkpoint(cls, path, device=None) -> "Correspondence":
        enc, cfg = load_fade(path, device)
        return cls(enc, cfg["ar_mode"])

    # ---- tokens v_i and attention a_i exactly as the trained head computes them ----
    @torch.no_grad()
    def tokens_attn(self, pil: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tf(pil).unsqueeze(0).to(self.device)
        patches = self.enc.model.forward_features(x)["x_norm_patchtokens"]
        v = F.normalize(patches.float(), p=2, dim=-1)                     # (1, N, D) unit patch tokens v_i
        a = torch.softmax(self.enc.head.score(v), dim=1)[0, :, 0]         # (N,) attention a_i
        return v[0], a

    @staticmethod
    def _descriptor(v: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        D = (a.unsqueeze(1) * v).sum(0)                                  # sum_i a_i v_i (un-normalised)
        return D, D.norm().clamp(min=1e-8)

    @property
    def has_sal(self) -> bool:
        """True for the decoupled saliency-head model (a separate examiner-supervised saliency head)."""
        return getattr(self.enc.head, "dual_gate", False)

    @torch.no_grad()
    def sal_map(self, img) -> torch.Tensor:
        """(N,) saliency of the decoupled saliency head (only for models with `has_sal`)."""
        x = self.tf(_open(img)).unsqueeze(0).to(self.device)
        patches = self.enc.model.forward_features(x)["x_norm_patchtokens"]
        v = F.normalize(patches.float(), p=2, dim=-1)
        return torch.softmax(self.enc.head.sal(v), dim=1)[0, :, 0]

    # ---- the contribution matrix ----
    @torch.no_grad()
    def cij(self, applied, cited) -> dict:
        """C (Na, Nb), the score it sums to, the completeness residual, and the inputs it was built from."""
        a_pil, c_pil = _open(applied), _open(cited)
        vA, aA = self.tokens_attn(a_pil)
        vB, aB = self.tokens_attn(c_pil)
        DA, nA = self._descriptor(vA, aA)
        DB, nB = self._descriptor(vB, aB)
        sim = vA @ vB.T                                                  # (Na, Nb) v_i . v_j
        C = (aA.unsqueeze(1) * aB.unsqueeze(0) * sim) / (nA * nB)        # (Na, Nb) contributions
        score = float(F.normalize(DA, dim=0) @ F.normalize(DB, dim=0))
        residual = abs(float(C.sum()) - score)                          # completeness check, ~0
        return {"C": C, "score": score, "residual": residual,
                "vA": vA, "vB": vB, "aA": aA, "aB": aB, "DB": DB, "nB": nB,
                "applied": a_pil, "cited": c_pil}

    # ---- readouts ----
    @staticmethod
    def top_pairs(C: torch.Tensor, top_k: int = 5) -> List[Tuple[int, int, float]]:
        """Top-k (applied patch, cited patch, contribution) by C_ij."""
        Nb = C.shape[1]
        vals, idx = C.flatten().topk(min(top_k, C.numel()))
        return [(int(p // Nb), int(p % Nb), float(v)) for p, v in zip(idx.tolist(), vals.tolist())]

    @staticmethod
    def contrib_applied(C: torch.Tensor) -> torch.Tensor:
        """Per applied-patch contribution sum_j C_ij."""
        return C.sum(1)

    @staticmethod
    def contrib_cited(C: torch.Tensor) -> torch.Tensor:
        """Per cited-patch contribution sum_i C_ij."""
        return C.sum(0)

    def _crop(self, pil: Image.Image, idx: int, ctx: int = 3) -> Image.Image:
        """Crop of the letterboxed image around patch `idx`, +-`ctx` patches of context."""
        lb = self.letterbox(pil); W, H = lb.size; g = self.grid
        r, c = idx // g, idx % g; cw, ch = W / g, H / g
        return lb.crop((max(0, (c - ctx) * cw), max(0, (r - ctx) * ch),
                        min(W, (c + ctx + 1) * cw), min(H, (r + ctx + 1) * ch)))

    def evidence_from(self, out: dict, top_k: int = 5, ctx: int = 3) -> dict:
        """Region evidence for the explainer from a `cij` result: text + matched region crops.

        The text is the exact Traditional Chinese wording the explainer was run with in the paper.
        """
        pairs = self.top_pairs(out["C"], top_k)
        g = self.grid
        lines = [f"({k+1}) 申請商標「{region_name(i, g)}」區 ↔ 據以核駁商標「{region_name(j, g)}」區"
                 f"（貢獻度 {s:.3f}）" for k, (i, j, s) in enumerate(pairs)]
        text = ("C1（faithful 分解）偵測到對兩商標相似度貢獻最大的對應區域：\n" + "\n".join(lines)
                ) if pairs else "C1 未偵測到顯著的對應視覺區域。"
        return {"text": text, "pairs": pairs, "score": out["score"], "residual": out["residual"],
                "applied_crops": [self._crop(out["applied"], i, ctx) for i, _, _ in pairs],
                "cited_crops": [self._crop(out["cited"], j, ctx) for _, j, _ in pairs]}

    def evidence(self, applied, cited, top_k: int = 5, ctx: int = 3) -> dict:
        return self.evidence_from(self.cij(applied, cited), top_k, ctx)
