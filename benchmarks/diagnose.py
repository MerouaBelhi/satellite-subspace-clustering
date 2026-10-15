"""Why does DESSC trail k-means on rich features? Test three hypotheses.

H1 unfair comparison : k-means keeps the best of 10 restarts, DESSC runs once.
H2 weight collapse   : the fitness (Eq. 16) has no entropy term, so individuals
                       whose weights pile onto a few low-variance dimensions
                       look "better" and win selection.
H3 search budget     : 20 individuals x 50 generations is too small.

    python benchmarks/diagnose.py DATA
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.deepglobe import load_pair  # noqa: E402
from sscluster import DESSC  # noqa: E402
from sscluster.features import build_features  # noqa: E402
from sscluster.segmentation import evaluate  # noqa: E402
from sscluster.superpixels import aggregate, compute_superpixels, to_pixels  # noqa: E402

KINDS = ["thesis", "lab", "lbp"]


def prepared(folder):
    for f in sorted(Path(folder).glob("*_sat.jpg")):
        img, gt = load_pair(f, 512)
        seg = compute_superpixels(img, 800)
        X = aggregate(build_features(img, KINDS), seg)
        yield f.name, X, seg, gt, len(np.unique(gt))


def miou(model, X, seg, gt):
    labels = model.fit_predict(X)
    return evaluate(to_pixels(labels, seg), gt)["miou"], model


def weight_stats(W):
    """Effective number of dimensions per cluster: exp(entropy of the weights)."""
    H = -(W * np.log(np.clip(W, 1e-12, None))).sum(axis=1)
    return np.exp(H).mean()


def main(folder):
    data = list(prepared(folder))
    D = data[0][1].shape[1]
    rows = {}

    def record(name, fn):
        rows[name] = [fn(X, seg, gt, C) for _, X, seg, gt, C in data]
        print(f"  {name:38} mIoU={np.mean(rows[name]):.3f}", flush=True)

    print(f"{len(data)} scenes, {D} features per superpixel\n")
    print("Baselines")
    record("k-means, best of 10 restarts", lambda X, s, g, C: miou(KMeans(C, n_init=10, random_state=0), X, s, g)[0])
    record("k-means, 1 run (avg of 5 seeds)", lambda X, s, g, C: np.mean(
        [miou(KMeans(C, n_init=1, random_state=r), X, s, g)[0] for r in range(5)]))
    record("DESSC default (avg of 5 seeds)", lambda X, s, g, C: np.mean(
        [miou(DESSC(C, random_state=r), X, s, g)[0] for r in range(5)]))

    print("\nH2: are the weights collapsing?")
    eff = []
    for _, X, seg, gt, C in data:
        m = DESSC(C, random_state=0).fit(X)
        eff.append(weight_stats(m.weights_))
    print(f"  effective dimensions used per cluster: {np.mean(eff):.1f} of {D}")

    print("\nH3: larger search budget")
    record("DESSC pop 40, 150 generations", lambda X, s, g, C: np.mean(
        [miou(DESSC(C, pop_size=40, max_iter=150, random_state=r), X, s, g)[0] for r in range(3)]))

    print("\nH1: same restart budget as k-means")
    def best_of(X, s, g, C, n=10):
        fits = [DESSC(C, random_state=r).fit(X) for r in range(n)]
        best = min(fits, key=lambda m: m.objective_)        # chosen by its own objective
        return evaluate(to_pixels(best.labels_, s), g)["miou"]
    record("DESSC best of 10 (by its own objective)", best_of)


if __name__ == "__main__":
    main(sys.argv[1])
