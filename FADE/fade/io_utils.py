"""Small file helpers shared by the command-line scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def list_images(src) -> List[Path]:
    """Images of a directory (sorted, non-recursive), a single image, or the paths listed in a .txt file."""
    src = Path(src)
    if src.is_dir():
        return sorted(p for p in src.iterdir() if p.is_file() and is_image(p))
    if src.suffix.lower() == ".txt":
        base = src.parent
        lines = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines()]
        return [Path(ln) if Path(ln).is_absolute() else base / ln for ln in lines if ln]
    if src.is_file():
        return [src]
    raise FileNotFoundError(src)


def index_by_stem(directory) -> Dict[str, Path]:
    """{file stem: path} for the images in `directory`; a .jpg wins over other extensions of the same stem."""
    out: Dict[str, Path] = {}
    with os.scandir(directory) as it:
        for entry in it:
            p = Path(entry.path)
            if not entry.is_file() or not is_image(p):
                continue
            if p.stem not in out or p.suffix.lower() == ".jpg":
                out[p.stem] = p
    return out


def read_pairs(jsonl, image_root: Optional[Path] = None) -> List[dict]:
    """Read a JSONL of {"id", "applied", "cited", ...}; relative image paths resolve against `image_root`."""
    root = Path(image_root) if image_root else Path(".")
    pairs = []
    with open(jsonl, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("applied", "cited"):
                if key not in rec:
                    raise ValueError(f"{jsonl}:{n}: missing '{key}'")
                p = Path(rec[key])
                rec[key] = p if p.is_absolute() else root / p
            rec.setdefault("id", f"{Path(rec['applied']).stem}__{Path(rec['cited']).stem}")
            pairs.append(rec)
    return pairs
