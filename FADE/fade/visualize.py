"""Rendering of the C_ij correspondence (PIL only).

Two styles:
  * "region" (Fig. 3 of the paper): on each mark, ONE box bounding the smallest set of patches that
    carries `mass_q` of that mark's total contribution, restricted to mark content (`fg_only`), and a
    line joining the two boxes;
  * "heatmap": per-patch contribution heatmaps with the top-k patch pairs drawn as colour-matched boxes
    joined by lines.
Display choices (foreground restriction, box extent) never change C_ij or the score.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .correspondence import Correspondence

# distinct hues for the top-k correspondences (rank 1 = largest contribution)
PALETTE = [(255, 60, 60), (60, 130, 255), (60, 210, 90), (255, 190, 40),
           (200, 80, 255), (0, 210, 210), (255, 120, 200), (150, 150, 60)]
WIN_COL = (30, 150, 70)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def heat_overlay(lb: Image.Image, contrib: torch.Tensor, grid: int, disp: int, alpha: float = 0.5,
                 colour=(255, 0, 0), gamma: float = 1.0) -> Image.Image:
    """Blend a per-patch weight map (grid*grid,) over the letterboxed image, up-scaled to `disp` px.
    gamma > 1 sharpens the display only."""
    c = contrib.reshape(grid, grid).float()
    c = (c - c.min()) / (c.max() - c.min() + 1e-8)
    if gamma != 1.0:
        c = c.clamp(min=0.0) ** gamma
    hm = Image.fromarray((c.cpu().numpy() * 255).astype("uint8")).resize((disp, disp), Image.BICUBIC)
    hm = np.asarray(hm, np.float32) / 255.0
    base = np.asarray(lb.convert("RGB").resize((disp, disp), Image.BICUBIC), np.float32)
    tint = np.zeros_like(base)
    for ch, val in enumerate(colour):
        tint[..., ch] = float(val)
    a = (alpha * hm)[..., None]
    return Image.fromarray((base * (1 - a) + tint * a).clip(0, 255).astype("uint8"))


def fg_mask(lb: Image.Image, orig_size, grid: int, thr: float = 18.0) -> torch.Tensor:
    """(grid^2,) bool, True where the patch overlaps mark content: not letterbox padding and not the
    mark's own blank margin (mean |pixel - border median| or in-patch std above `thr`, 0-255 scale)."""
    S = lb.size[0]
    w, h = orig_size
    s = max(w, h)
    cx0, cy0 = round((s - w) / 2 / s * S), round((s - h) / 2 / s * S)
    cx1, cy1 = cx0 + round(w / s * S), cy0 + round(h / s * S)
    g_arr = np.asarray(lb.convert("L"), np.float32)
    content = g_arr[cy0:cy1, cx0:cx1]
    border = np.concatenate([content[0], content[-1], content[:, 0], content[:, -1]])
    bg = float(np.median(border))
    pad = np.ones_like(g_arr, bool)
    pad[cy0:cy1, cx0:cx1] = False
    dev = np.abs(g_arr - bg)
    dev[pad] = 0.0                                                       # padding is never foreground
    ps = S / grid
    m = np.zeros(grid * grid, bool)
    for r in range(grid):
        for c in range(grid):
            cell = dev[int(r * ps):int((r + 1) * ps), int(c * ps):int((c + 1) * ps)]
            m[r * grid + c] = (cell.mean() > thr) or (cell.std() > thr)
    return torch.from_numpy(m)


def content_patch_bounds(orig_size, grid: int):
    """Inclusive (r0, c0, r1, c1) patch range covered by image content (excludes letterbox padding)."""
    w, h = orig_size
    s = max(w, h)
    py, px = int((s - h) / 2 / s * grid), int((s - w) / 2 / s * grid)
    return py, px, grid - 1 - py, grid - 1 - px


def mass_bbox(contrib: torch.Tensor, grid: int, q: float, mask, bounds, pad: int = 1):
    """(r0, c0, r1, c1) box of the smallest patch set carrying >= q of the total per-patch contribution
    mass, restricted to `mask` patches (None = all), padded by `pad` patches and clipped to `bounds`."""
    c = contrib.detach().float().cpu().reshape(-1).clone()
    if mask is not None:
        c[~mask] = 0.0
    tot = float(c.sum())
    if tot <= 0:
        return None
    vals, idx = torch.sort(c, descending=True)
    keep, acc = [], 0.0
    for v, i in zip(vals.tolist(), idx.tolist()):
        if v <= 0:
            break
        keep.append(i)
        acc += v
        if acc >= q * tot:
            break
    rs = [i // grid for i in keep]
    cs = [i % grid for i in keep]
    r0b, c0b, r1b, c1b = bounds
    return (max(r0b, min(rs) - pad), max(c0b, min(cs) - pad),
            min(r1b, max(rs) + pad), min(c1b, max(cs) + pad))


def render_region_pair(applied_disp: Image.Image, cited_disp: Image.Image, box_a, box_b, grid: int,
                       disp: int, gap: int = 28) -> Image.Image:
    """applied | gap | cited, one box per mark and one line joining the box centres."""
    bw = max(4, disp // 64)
    lw = max(2, disp // 112)
    fs = max(15, disp // 16)
    hdr = fs + 7
    row = Image.new("RGB", (2 * disp + gap, disp + hdr), (255, 255, 255))
    row.paste(applied_disp, (0, hdr))
    row.paste(cited_disp, (disp + gap, hdr))
    d = ImageDraw.Draw(row)
    ps = disp / grid
    ft = _font(fs)
    d.text((4, 4), "applied", fill=(0, 0, 0), font=ft)
    d.text((disp + gap + 4, 4), "cited", fill=(0, 0, 0), font=ft)
    centers = []
    for off, box in ((0, box_a), (disp + gap, box_b)):
        if box is None:
            continue
        r0, c0, r1, c1 = box
        x0, y0 = off + c0 * ps, r0 * ps + hdr
        x1, y1 = off + (c1 + 1) * ps, (r1 + 1) * ps + hdr
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=WIN_COL, width=bw)
        centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
    if len(centers) == 2:
        d.line([*centers[0], *centers[1]], fill=WIN_COL, width=lw)
    return row


def render_pair(applied_disp: Image.Image, cited_disp: Image.Image, pairs, grid: int, disp: int,
                gap: int = 28) -> Image.Image:
    """applied | gap | cited with the top-k correspondences: colour-matched boxes, a line and a rank."""
    bw = max(3, disp // 84)
    lw = max(2, disp // 112)
    fs = max(15, disp // 16)
    hdr = fs + 7
    row = Image.new("RGB", (2 * disp + gap, disp + hdr), (255, 255, 255))
    row.paste(applied_disp, (0, hdr)); row.paste(cited_disp, (disp + gap, hdr))
    d = ImageDraw.Draw(row); ps = disp / grid; ft = _font(fs)
    d.text((4, 4), "applied", fill=(0, 0, 0), font=ft)
    d.text((disp + gap + 4, 4), "cited", fill=(0, 0, 0), font=ft)
    for rank, (i, j, s) in enumerate(pairs):
        col = PALETTE[rank % len(PALETTE)]
        ar, ac = i // grid, i % grid
        br, bc = j // grid, j % grid
        ax0, ay0 = ac * ps, ar * ps + hdr
        bx0, by0 = (disp + gap) + bc * ps, br * ps + hdr
        d.rectangle([ax0, ay0, ax0 + ps, ay0 + ps], outline=col, width=bw)
        d.rectangle([bx0, by0, bx0 + ps, by0 + ps], outline=col, width=bw)
        d.line([ax0 + ps / 2, ay0 + ps / 2, bx0 + ps / 2, by0 + ps / 2], fill=col, width=lw)
        d.text((ax0 + 1, max(hdr, ay0 - fs)), str(rank + 1), fill=col, font=ft)
        d.text((bx0 + 1, max(hdr, by0 - fs)), str(rank + 1), fill=col, font=ft)
    return row


def render_correspondence(corr: Correspondence, out: dict, style: str = "region", mass_q: float = 0.7,
                          fg_only: bool = True, fg_thr: float = 18.0, draw_k: int = 3, disp: int = 672,
                          gamma: float = 1.0):
    """Render one `Correspondence.cij` result. Returns (PIL image, info dict with the drawn boxes/pairs)."""
    C, g = out["C"], corr.grid
    lb_a, lb_c = corr.letterbox(out["applied"]), corr.letterbox(out["cited"])
    mA = fg_mask(lb_a, out["applied"].size, g, fg_thr) if fg_only else None
    mB = fg_mask(lb_c, out["cited"].size, g, fg_thr) if fg_only else None
    if style == "region":
        la = lb_a.convert("RGB").resize((disp, disp), Image.BICUBIC)
        lc = lb_c.convert("RGB").resize((disp, disp), Image.BICUBIC)
        box_a = mass_bbox(corr.contrib_applied(C), g, mass_q, mA, content_patch_bounds(out["applied"].size, g))
        box_b = mass_bbox(corr.contrib_cited(C), g, mass_q, mB, content_patch_bounds(out["cited"].size, g))
        return render_region_pair(la, lc, box_a, box_b, g, disp), {"region_applied": box_a, "region_cited": box_b}
    if style == "heatmap":
        Cm = C
        if fg_only and int(mA.sum()) >= 4 and int(mB.sum()) >= 4:     # near-empty mask: draw unmasked
            Cm = C.clone()
            Cm[~mA.to(C.device)] = float("-inf")
            Cm[:, ~mB.to(C.device)] = float("-inf")
        pairs = [p for p in corr.top_pairs(Cm, draw_k) if p[2] != float("-inf")]
        la = heat_overlay(lb_a, corr.contrib_applied(C), g, disp, gamma=gamma)
        lc = heat_overlay(lb_c, corr.contrib_cited(C), g, disp, gamma=gamma)
        return render_pair(la, lc, pairs, g, disp), {"drawn_pairs": pairs}
    raise ValueError(f"unknown style {style!r} (use 'region' or 'heatmap')")
