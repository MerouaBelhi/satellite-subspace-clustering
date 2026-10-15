import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sscluster import DESSC  # noqa: E402
from sscluster.deep import pca_reduce, tokens_to_pixel_features  # noqa: E402
from sscluster.features import EXTRACTORS, build_features  # noqa: E402
from sscluster.segmentation import evaluate, segment  # noqa: E402
from sscluster.superpixels import aggregate, compute_superpixels, to_pixels  # noqa: E402
from tests.test_pipeline import synthetic_scene  # noqa: E402


def textured_scene(seed=0):
    """Two regions with the SAME mean colour but different texture:
    only texture features can separate them."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((96, 96), dtype=int)
    gt[:, 48:] = 1
    img = np.full((96, 96), 120.0)
    stripes = 60 * np.sign(np.sin(np.arange(96) * np.pi / 2))[None, :].repeat(96, 0)
    img[gt == 1] += stripes[gt == 1]
    img += rng.normal(0, 3, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.dstack([img] * 3), gt


@pytest.mark.parametrize("kind", list(EXTRACTORS))
def test_each_extractor_shape(kind):
    img, _ = synthetic_scene()
    out = EXTRACTORS[kind](img)
    assert out.shape[:2] == img.shape[:2] and out.ndim == 3
    assert np.isfinite(out).all()


def test_build_features_is_scaled():
    img, _ = synthetic_scene()
    F = build_features(img, ["thesis", "lab", "lbp"])
    assert F.shape[0] == img.shape[0] * img.shape[1]
    assert F.min() >= 0 and F.max() <= 1


def test_unknown_feature_kind_raises():
    img, _ = synthetic_scene()
    with pytest.raises(ValueError):
        build_features(img, ["nope"])


def test_texture_features_separate_same_colour_regions():
    img, gt = textured_scene()
    km = lambda: KMeans(2, n_init=10, random_state=0)  # noqa: E731
    colour = evaluate(segment(img, km(), features=build_features(img, ["thesis"])), gt)
    texture = evaluate(segment(img, km(), features=build_features(img, ["lbp"])), gt)
    assert texture["miou"] > 0.9
    assert texture["miou"] > colour["miou"]


def test_superpixel_roundtrip():
    img, _ = synthetic_scene()
    seg = compute_superpixels(img, n_segments=100)
    F = build_features(img, ["thesis"])
    sp = aggregate(F, seg)
    assert sp.shape == (seg.max() + 1, F.shape[1])
    labels = to_pixels(np.arange(len(sp)), seg)
    assert (labels == seg).all()


def test_superpixel_segmentation_with_dessc():
    """DESSC should get close to the best mIoU any superpixel labelling can
    reach (majority ground-truth label per superpixel)."""
    img, gt = synthetic_scene()
    seg = compute_superpixels(img, n_segments=800)
    oracle = np.array([np.bincount(gt[seg == s]).argmax() for s in range(seg.max() + 1)])[seg]
    ceiling = evaluate(oracle, gt)["miou"]
    pred = segment(img, DESSC(3, max_iter=20, random_state=0), superpixels=800)
    assert evaluate(pred, gt)["miou"] > ceiling - 0.03


def test_tokens_to_pixel_features():
    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(4 * 6, 384))
    fmap = tokens_to_pixel_features(tokens, (4, 6), (56, 84), n_components=16)
    assert fmap.shape == (56, 84, 16)
    with pytest.raises(ValueError):
        tokens_to_pixel_features(tokens, (5, 5), (70, 70))


def test_pca_keeps_dominant_direction():
    rng = np.random.default_rng(0)
    X = np.outer(rng.normal(size=200), [3, 4, 0]) + 0.01 * rng.normal(size=(200, 3))
    Z = pca_reduce(X, 1)
    assert abs(np.corrcoef(Z[:, 0], X[:, 0])[0, 1]) > 0.99
