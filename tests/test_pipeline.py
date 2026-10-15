import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import rand_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sscluster import DESSC  # noqa: E402
from sscluster.segmentation import (decode_deepglobe_mask, evaluate,  # noqa: E402
                                    match_clusters, segment)


def synthetic_scene(seed=0):
    """A 128x128 'satellite' image: water, field and barren regions + noise."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((128, 128), dtype=int)
    gt[:, 64:] = 1
    gt[80:, 30:100] = 2
    colours = np.array([[20, 40, 70], [70, 110, 50], [200, 190, 160]], float)
    img = colours[gt] + rng.normal(0, 12, (128, 128, 3))
    return np.clip(img, 0, 255).astype(np.uint8), gt


def test_dessc_reproduces_paper_on_iris():
    X, y = load_iris(return_X_y=True)
    labels = DESSC(3, random_state=0).fit_predict(MinMaxScaler().fit_transform(X))
    assert abs(rand_score(y, labels) - 0.9495) < 1e-3


def test_weights_are_valid():
    X = np.random.default_rng(0).random((60, 5))
    m = DESSC(3, max_iter=5, random_state=0).fit(X)
    assert np.allclose(m.weights_.sum(axis=1), 1)
    assert (m.weights_ >= 0).all()


def test_kmeanspp_init():
    X, y = load_iris(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)
    labels = DESSC(3, init="k-means++", random_state=0).fit_predict(X)
    assert rand_score(y, labels) > 0.94
    import pytest
    with pytest.raises(ValueError):
        DESSC(3, init="nope")


def test_hungarian_matching_is_permutation_invariant():
    gt = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])   # perfect but relabelled
    assert (match_clusters(pred, gt) == gt).all()
    assert evaluate(pred, gt)["miou"] == 1.0


def test_deepglobe_mask_decoding():
    mask = np.array([[[0, 255, 255], [0, 0, 250]], [[255, 255, 255], [3, 2, 1]]])
    assert decode_deepglobe_mask(mask).tolist() == [[0, 4], [5, 6]]


def test_segmentation_of_synthetic_scene():
    img, gt = synthetic_scene()
    pred = segment(img, DESSC(3, max_iter=20, random_state=0), n_samples=2000)
    assert evaluate(pred, gt)["miou"] > 0.9


def test_majority_matching_rewards_overclustering():
    """One dominant class split into several clusters: Hungarian matching
    punishes it, majority matching does not."""
    gt = np.zeros((10, 10), dtype=int)
    gt[:, 8:] = 1                      # 80% class 0, 20% class 1
    pred = np.zeros_like(gt)
    pred[:, 3:6] = 1
    pred[:, 6:8] = 2                   # class 0 split into clusters 0, 1, 2
    pred[:, 8:] = 3
    assert evaluate(pred, gt, matching="majority")["miou"] == 1.0
    assert evaluate(pred, gt, matching="hungarian")["iou"][0] < 0.4   # 30 of 80 pixels


def test_ignore_class_is_left_out():
    gt = np.array([0, 0, 1, 1, 6, 6])
    pred = np.array([0, 0, 1, 1, 0, 1])
    r = evaluate(pred, gt, ignore=6)
    assert r["miou"] == 1.0 and 6 not in r["iou"]
