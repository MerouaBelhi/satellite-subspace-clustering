"""Per-pixel feature extractors.

Every extractor takes an (H, W, 3) uint8 RGB image and returns an
(H, W, d) float array. `build_features` stacks several of them and
min-max scales each column so that no feature dominates the distances.

Available kinds:
    thesis   R, G, B, grey, local mean grey   (5)   - Master's thesis baseline
    lab      CIE L*a*b* + local mean of each   (6)
    hsv      H (as sin/cos), S, V              (4)
    lbp      local histogram of uniform LBP    (10)
    gabor    local Gabor energy, 4 orient. x 2 frequencies (8)
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import local_binary_pattern
from skimage.filters import gabor


def _local_mean(channel: np.ndarray, size: int) -> np.ndarray:
    return uniform_filter(channel, size=size, mode="reflect")


def thesis(img: np.ndarray, roi: int = 5) -> np.ndarray:
    rgb = img.astype(float) / 255.0
    grey = rgb @ np.array([0.299, 0.587, 0.114])
    return np.dstack([rgb, grey, _local_mean(grey, roi)])


def lab(img: np.ndarray, roi: int = 5) -> np.ndarray:
    L = rgb2lab(img)
    smooth = np.dstack([_local_mean(L[..., c], roi) for c in range(3)])
    return np.dstack([L, smooth])


def hsv(img: np.ndarray, roi: int = 5) -> np.ndarray:
    h, s, v = np.moveaxis(rgb2hsv(img), -1, 0)
    angle = 2 * np.pi * h                       # hue is circular
    return np.dstack([np.sin(angle), np.cos(angle), s, v])


def lbp(img: np.ndarray, roi: int = 15, points: int = 8, radius: int = 1) -> np.ndarray:
    """Local histogram of uniform LBP codes (P + 2 bins) in an roi window."""
    grey = (rgb2gray(img) * 255).astype(np.uint8)
    codes = local_binary_pattern(grey, points, radius, method="uniform").astype(int)
    n_bins = points + 2
    return np.dstack([_local_mean((codes == b).astype(float), roi) for b in range(n_bins)])


def gabor_energy(img: np.ndarray, roi: int = 15,
                 frequencies: Sequence[float] = (0.1, 0.25),
                 n_orient: int = 4) -> np.ndarray:
    """Local mean of Gabor magnitude responses."""
    grey = rgb2gray(img)
    maps = []
    for f in frequencies:
        for k in range(n_orient):
            real, imag = gabor(grey, frequency=f, theta=k * np.pi / n_orient)
            maps.append(_local_mean(np.hypot(real, imag), roi))
    return np.dstack(maps)


EXTRACTORS: dict[str, Callable[..., np.ndarray]] = {
    "thesis": thesis,
    "lab": lab,
    "hsv": hsv,
    "lbp": lbp,
    "gabor": gabor_energy,
}


def minmax_columns(F: np.ndarray) -> np.ndarray:
    lo, hi = F.min(axis=0), F.max(axis=0)
    return (F - lo) / np.where(hi > lo, hi - lo, 1.0)


def build_features(img: np.ndarray, kinds: Sequence[str] = ("thesis",),
                   extra: np.ndarray | None = None) -> np.ndarray:
    """Stack the requested feature maps into an (H*W, d) scaled matrix.

    `extra` is an optional precomputed (H, W, d') map, e.g. DINOv2 features.
    """
    maps = []
    for k in kinds:
        if k not in EXTRACTORS:
            raise ValueError(f"Unknown feature kind {k!r}; choose from {list(EXTRACTORS)}")
        maps.append(EXTRACTORS[k](img))
    if extra is not None:
        maps.append(extra)
    if not maps:
        raise ValueError("No features requested")
    F = np.dstack(maps).reshape(-1, sum(m.shape[-1] for m in maps))
    return minmax_columns(F.astype(float))
