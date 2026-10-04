"""Image preprocessing. Released FADE models use `letterbox-gray`: pad to a square with gray (128,128,128)
so the mark's aspect ratio is kept, then resize to the backbone resolution."""
from __future__ import annotations

from typing import Tuple

from PIL import Image
from torchvision import transforms

from .backbones import BackboneSpec


class LetterboxResize:
    """Pad to square with `fill`, then resize. Preserves the aspect ratio."""

    def __init__(self, size: int, fill: Tuple[int, int, int] = (128, 128, 128)) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        s = max(w, h)
        padded = Image.new("RGB", (s, s), self.fill)
        padded.paste(img, ((s - w) // 2, (s - h) // 2))
        return padded.resize((self.size, self.size), Image.BICUBIC)


class CenterCropAfterResize:
    """Resize the short side, then center-crop (standard ViT preprocessing; crops elongated marks)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.transform = transforms.Compose(
            [transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.CenterCrop(size)]
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)


class DirectResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize((self.size, self.size), Image.BICUBIC)


def build_ar_preprocessor(mode: str, size: int):
    """PIL -> PIL aspect-ratio handling.

    Modes: "letterbox-gray" (default for every released model), "letterbox-white", "resize",
    "center-crop".
    """
    if mode == "resize":
        return DirectResize(size)
    if mode == "letterbox-white":
        return LetterboxResize(size, fill=(255, 255, 255))
    if mode == "letterbox-gray":
        return LetterboxResize(size, fill=(128, 128, 128))
    if mode == "center-crop":
        return CenterCropAfterResize(size)
    raise ValueError(f"Unknown aspect-ratio mode: {mode}")


def build_transform(spec: BackboneSpec, ar_mode: str) -> transforms.Compose:
    return transforms.Compose(
        [
            build_ar_preprocessor(ar_mode, spec.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=spec.mean, std=spec.std),
        ]
    )
