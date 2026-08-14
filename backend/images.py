"""Pixel lookup for the datasets whose rows are images.

The frontend draws the pixels onto a canvas itself, so this hands back plain
numbers instead of an encoded image — no image library on the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Dataset key -> (width, height). Every entry here stores its image as `px_0..px_n-1`
# columns in row-major order, greyscale, already scaled to 0..1 by the loader in
# src/datasets.py — so a fixed *255 is the right conversion for all of them.
IMAGE_SPECS: dict[str, tuple[int, int]] = {
    "Digits (Low)": (8, 8),
    "Olivetti faces (Medium)": (64, 64),
    "Fashion-MNIST (High)": (28, 28),
    "MNIST (High)": (28, 28),
}


def spec(key: str) -> dict[str, int] | None:
    """Image dimensions of a dataset, or None if its rows aren't images."""
    size = IMAGE_SPECS.get(key)
    return None if size is None else {"width": size[0], "height": size[1]}


def pixels(df: pd.DataFrame, key: str, row_id: int) -> dict[str, Any]:
    """One row as 0..255 greyscale values, row-major."""
    width, height = IMAGE_SPECS[key]
    cols = [f"px_{i}" for i in range(width * height)]
    # Slice the single row first: `df[cols]` on MNIST would materialize 784x70k floats.
    values = df.iloc[[row_id]][cols].to_numpy(dtype=np.float64)[0]
    grey = np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)
    return {"width": width, "height": height, "pixels": grey.tolist()}
