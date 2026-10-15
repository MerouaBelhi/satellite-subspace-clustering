"""Reproduce the UCI comparison of Li et al. (2016) for DESSC vs k-means.

Runs each algorithm several times on Iris and Wine (min-max normalised)
and reports mean and std of RI, ARI and NMI.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sscluster import DESSC  # noqa: E402

RUNS = 10
PAPER = {  # Li et al. 2016, Tables 7-9 (DESSC means)
    "Iris": (0.9495, 0.8857, 0.8642),
    "Wine": (0.9179, 0.8519, 0.8087),
}


def scores(y, labels):
    return (rand_score(y, labels), adjusted_rand_score(y, labels),
            normalized_mutual_info_score(y, labels))


def main():
    for name, loader in [("Iris", load_iris), ("Wine", load_wine)]:
        X, y = loader(return_X_y=True)
        X = MinMaxScaler().fit_transform(X)
        C = len(np.unique(y))
        res = {"DESSC": [], "k-means": []}
        for seed in range(RUNS):
            res["DESSC"].append(scores(y, DESSC(C, random_state=seed).fit_predict(X)))
            res["k-means"].append(scores(
                y, KMeans(C, n_init=1, random_state=seed).fit_predict(X)))
        print(f"\n{name}  (N={len(X)}, D={X.shape[1]}, C={C}, {RUNS} runs)")
        print(f"{'':10}{'RI':>16}{'ARI':>16}{'NMI':>16}")
        for algo, vals in res.items():
            v = np.array(vals)
            cells = "".join(f"{m:>9.4f}±{s:.4f}" for m, s in zip(v.mean(0), v.std(0)))
            print(f"{algo:10}{cells}")
        print(f"{'paper':10}" + "".join(f"{p:>16.4f}" for p in PAPER[name]))


if __name__ == "__main__":
    main()
