"""Cluster SLIC superpixels instead of single pixels.

Averaging features inside each superpixel removes the salt-and-pepper
noise seen in pixel-wise results and cuts the number of samples by
orders of magnitude, which matters for an evolutionary method like DESSC.
"""

from __future__ import annotations

import numpy as np
from skimage.segmentation import slic


def compute_superpixels(img: np.ndarray, n_segments: int = 800,
                        compactness: float = 10.0) -> np.ndarray:
    """Return an (H, W) map of consecutive superpixel ids starting at 0."""
    seg = slic(img, n_segments=n_segments, compactness=compactness,
               start_label=0, channel_axis=-1)
    _, seg = np.unique(seg, return_inverse=True)
    return seg.reshape(img.shape[:2])


def aggregate(features: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Mean feature vector of each superpixel. `features` is (H*W, d)."""
    ids = seg.ravel()
    n = ids.max() + 1
    counts = np.bincount(ids, minlength=n).astype(float)
    sums = np.zeros((n, features.shape[1]))
    np.add.at(sums, ids, features)
    return sums / counts[:, None]


def to_pixels(sp_labels: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Broadcast one label per superpixel back to an (H, W) label map."""
    return sp_labels[seg]
