"""Image-format normalization helpers used by the ViStoryBench exporter."""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageOps


def normalize_image_to_png(image_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Decode an image by content and write a real PNG to the requested path."""
    source = Path(image_path)
    destination = Path(output_path) if output_path is not None else source
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.png-normalizing.tmp")

    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
            if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            image.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
