"""Image segmentation by clustering pixel features, plus evaluation tools.

Pipeline (as in the Master's thesis, Fig. 3.9):
    image -> per-pixel features -> random pixel sample -> fit clustering
          -> label every pixel -> segmentation map
Evaluation fixes what the thesis left implicit: cluster ids are matched to
ground-truth classes with the Hungarian algorithm before scoring.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import linear_sum_assignment

# DeepGlobe Land Cover colour code (RGB) -> class index
DEEPGLOBE_CLASSES = {
    (0, 255, 255): 0,    # urban
    (255, 255, 0): 1,    # agriculture
    (255, 0, 255): 2,    # rangeland
    (0, 255, 0): 3,      # forest
    (0, 0, 255): 4,      # water
    (255, 255, 255): 5,  # barren
    (0, 0, 0): 6,        # unknown
}
DEEPGLOBE_NAMES = ["urban", "agriculture", "rangeland", "forest",
                   "water", "barren", "unknown"]


# ----------------------------------------------------------------- features
def thesis_features(img: np.ndarray, roi: int = 5) -> np.ndarray:
    """The five features of the thesis: R, G, B, grey level, and the mean
    grey level in an (roi x roi) neighbourhood. Returns (H*W, 5) in [0, 1]."""
    img = img.astype(float) / 255.0
    grey = img @ np.array([0.299, 0.587, 0.114])
    local_mean = uniform_filter(grey, size=roi, mode="reflect")
    feats = np.dstack([img, grey, local_mean])
    return feats.reshape(-1, feats.shape[-1])


def decode_deepglobe_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert a DeepGlobe RGB mask to an (H, W) array of class indices.
    Masks are thresholded at 128 because JPEG/PNG edges are not pure."""
    binary = (mask_rgb >= 128).astype(int)
    labels = np.full(mask_rgb.shape[:2], 6, dtype=int)
    for colour, idx in DEEPGLOBE_CLASSES.items():
        hit = np.all(binary == (np.array(colour) // 255), axis=-1)
        labels[hit] = idx
    return labels


# ------------------------------------------------------------- segmentation
def segment(img: np.ndarray, model, roi: int = 5, n_samples: int = 5000,
            random_state: int = 0, features: np.ndarray | None = None,
            superpixels: int | None = None) -> np.ndarray:
    """Cluster an image and return an (H, W) map of cluster ids.

    model        anything with fit / predict (DESSC, KMeans, ...)
    features     (H*W, d) matrix from `features.build_features`; defaults to
                 the five thesis features with the given roi
    superpixels  if set, cluster this many SLIC superpixels (mean feature per
                 superpixel) instead of a random pixel sample
    """
    if features is None:
        features = thesis_features(img, roi)
    H, W = img.shape[:2]

    if superpixels:
        from .superpixels import aggregate, compute_superpixels, to_pixels
        seg = compute_superpixels(img, n_segments=superpixels)
        sp_feats = aggregate(features, seg)
        model.fit(sp_feats)
        return to_pixels(model.predict(sp_feats), seg)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(features), min(n_samples, len(features)), replace=False)
    model.fit(features[idx])
    return model.predict(features).reshape(H, W)


# --------------------------------------------------------------- evaluation
def match_clusters(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Relabel cluster ids so they best agree with ground-truth classes
    (Hungarian algorithm on the contingency table). Unmatched clusters,
    when there are more clusters than classes, get label -1."""
    clusters, classes = np.unique(pred), np.unique(gt)
    table = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for a, c in enumerate(clusters):
        sel = gt[pred == c]
        for b, k in enumerate(classes):
            table[a, b] = np.count_nonzero(sel == k)
    rows, cols = linear_sum_assignment(-table)
    mapping = {clusters[r]: classes[c] for r, c in zip(rows, cols)}
    return np.vectorize(lambda v: mapping.get(v, -1))(pred)


def majority_match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Many-to-one matching: each cluster takes the most frequent ground-truth
    class among its pixels. Used with over-clustering (more clusters than
    classes), the standard protocol when one class dominates the image."""
    out = np.empty_like(gt)
    for c in np.unique(pred):
        sel = pred == c
        out[sel] = np.bincount(gt[sel]).argmax()
    return out


def evaluate(pred: np.ndarray, gt: np.ndarray, matching: str = "hungarian",
             ignore: int | None = None) -> dict:
    """Pixel accuracy, mean IoU and per-class IoU.

    matching  "hungarian": one cluster per class (requires K == C to be fair)
              "majority" : many clusters per class, each to its majority class
    ignore    class index left out of matching and scoring (DeepGlobe's
              official metric ignores "unknown", index 6)
    """
    if ignore is not None:
        valid = gt != ignore
        if not valid.any():
            return {"pixel_acc": float("nan"), "miou": float("nan"), "iou": {}}
        pred, gt = pred[valid], gt[valid]
    if matching == "hungarian":
        matched = match_clusters(pred, gt)
    elif matching == "majority":
        matched = majority_match(pred, gt)
    else:
        raise ValueError("matching must be 'hungarian' or 'majority'")
    ious = {}
    for k in np.unique(gt):
        inter = np.count_nonzero((matched == k) & (gt == k))
        union = np.count_nonzero((matched == k) | (gt == k))
        ious[int(k)] = inter / union if union else float("nan")
    return {
        "pixel_acc": float(np.mean(matched == gt)),
        "miou": float(np.nanmean(list(ious.values()))),
        "iou": ious,
    }
