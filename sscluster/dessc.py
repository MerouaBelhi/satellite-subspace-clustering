"""DESSC: Soft Subspace Clustering using a Differential Evolutionary algorithm.

Python re-implementation of:
    Y. Li, Y. Lu, L. Jiao, "Soft subspace clustering using differential
    evolutionary algorithm", IEEE CEC 2016, pp. 545-552.

Each individual of the population holds a dimension-weight matrix W (C x D)
and a cluster-centre matrix Z (C x D). The hard partition U is derived from
(W, Z) by nearest weighted distance (Eq. 21), so it is not stored.
"""

from __future__ import annotations

import numpy as np


def _weighted_sq_dist(X: np.ndarray, Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted squared distance between every point and every centre.

    Returns an (N, C) matrix: d[j, i] = sum_k W[i, k] * (x_jk - z_ik)^2.
    """
    diff = X[:, None, :] - Z[None, :, :]          # (N, C, D)
    return np.einsum("ncd,cd->nc", diff ** 2, W)


def _normalize_rows(W: np.ndarray) -> np.ndarray:
    W = np.clip(W, 1e-12, None)
    return W / W.sum(axis=1, keepdims=True)


class DESSC:
    """Soft subspace clustering optimised by differential evolution.

    Parameters
    ----------
    n_clusters : int
        Number of clusters C.
    pop_size : int
        Number of individuals NP in the population.
    max_iter : int
        Number of DE generations.
    F : float
        Mutation factor, in [0, 2].
    CR : float
        Crossover rate, in [0, 1].
    alpha : float
        Scales eta = alpha * mean_k(sigma_ik) in the local search (Eq. 19).
    init : {"random", "k-means++"}
        How each individual's initial centres are chosen. "random" picks C
        data points uniformly, as in the paper. "k-means++" spreads them out
        (Arthur & Vassilvitskii, 2007), which helps on high-dimensional data.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(self, n_clusters: int, pop_size: int = 20, max_iter: int = 50,
                 F: float = 0.5, CR: float = 0.9, alpha: float = 1.0,
                 init: str = "random", random_state: int | None = None):
        if init not in ("random", "k-means++"):
            raise ValueError("init must be 'random' or 'k-means++'")
        self.n_clusters = n_clusters
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.alpha = alpha
        self.init = init
        self.random_state = random_state

    def _init_centres(self, X):
        N = len(X)
        if self.init == "random":
            return X[self._rng.choice(N, self.n_clusters, replace=False)].copy()
        from sklearn.cluster import kmeans_plusplus
        seed = int(self._rng.integers(2 ** 31))
        centres, _ = kmeans_plusplus(X, self.n_clusters, random_state=seed)
        return centres.copy()

    # ------------------------------------------------------------------ core
    def _assign(self, X, Z, W):
        """Eq. 21: hard membership by minimum weighted distance."""
        return _weighted_sq_dist(X, Z, W).argmin(axis=1)

    def _update_centres(self, X, labels, Z):
        """Eq. 22, re-seeding empty clusters with a random data point."""
        Z = Z.copy()
        for i in range(self.n_clusters):
            members = labels == i
            if members.any():
                Z[i] = X[members].mean(axis=0)
            else:
                Z[i] = X[self._rng.integers(len(X))]
        return Z

    def _local_search(self, X, labels, Z):
        """Eqs. 19-20: entropy-style weight update with adaptive eta."""
        C, D = Z.shape
        sigma = np.empty((C, D))
        for i in range(C):
            members = X[labels == i]
            if len(members) == 0:
                sigma[i] = 1.0
            else:
                sigma[i] = ((members - Z[i]) ** 2).mean(axis=0)
        eta = self.alpha * sigma.mean(axis=1, keepdims=True)   # alpha * sum_k sigma / D
        eta = np.maximum(eta, 1e-12)
        logits = -sigma / eta
        logits -= logits.max(axis=1, keepdims=True)            # numerical stability
        W = np.exp(logits)
        return W / W.sum(axis=1, keepdims=True)

    def _fitness(self, X, Z, W):
        """Eq. 16: sum of within-cluster weighted squared distances."""
        d = _weighted_sq_dist(X, Z, W)
        return d.min(axis=1).sum()

    def _refine(self, X, W, Z):
        """Steps 3-5 of Table 1: partition, centres, local search, fitness."""
        labels = self._assign(X, Z, W)
        Z = self._update_centres(X, labels, Z)
        labels = self._assign(X, Z, W)
        W = self._local_search(X, labels, Z)
        return W, Z, self._fitness(X, Z, W)

    def _matched_index(self, Z_ref_row, W_ref_row, Z_other):
        """Eq. 18: centre of another individual closest to a reference centre."""
        d = ((Z_other - Z_ref_row) ** 2 * W_ref_row).sum(axis=1)
        return int(d.argmin())

    # ------------------------------------------------------------------- api
    def fit(self, X: np.ndarray) -> "DESSC":
        X = np.asarray(X, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        N, D = X.shape
        C, NP = self.n_clusters, self.pop_size

        # Initialisation (Table 1, steps 1-3)
        pop_W = [_normalize_rows(self._rng.random((C, D))) for _ in range(NP)]
        pop_Z = [self._init_centres(X) for _ in range(NP)]
        fit = np.array([self._fitness(X, Z, W) for W, Z in zip(pop_W, pop_Z)])
        self.history_ = [fit.min()]

        for _ in range(self.max_iter):
            for p in range(NP):
                r1, r2, r3 = self._rng.choice([q for q in range(NP) if q != p], 3,
                                              replace=False)
                W1, Z1 = pop_W[r1], pop_Z[r1]

                # Mutation with cluster matching (Eq. 18)
                V = np.empty((C, D))
                for i in range(C):
                    t1 = self._matched_index(Z1[i], W1[i], pop_Z[r2])
                    t2 = self._matched_index(Z1[i], W1[i], pop_Z[r3])
                    V[i] = W1[i] + self.F * (pop_W[r2][t1] - pop_W[r3][t2])

                # Binomial crossover (Eq. 15) against the target individual
                mask = self._rng.random((C, D)) < self.CR
                forced = self._rng.integers(D, size=C)
                mask[np.arange(C), forced] = True
                child_W = _normalize_rows(np.where(mask, V, pop_W[p]))
                child_Z = Z1.copy()                       # Z inherited from P_r1

                # Partition / centre update, local search, fitness
                child_W, child_Z, child_fit = self._refine(X, child_W, child_Z)

                # Greedy selection
                if child_fit <= fit[p]:
                    pop_W[p], pop_Z[p], fit[p] = child_W, child_Z, child_fit
            self.history_.append(fit.min())

        best = int(fit.argmin())
        self.weights_ = pop_W[best]
        self.cluster_centers_ = pop_Z[best]
        self.labels_ = self._assign(X, self.cluster_centers_, self.weights_)
        self.objective_ = fit[best]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(np.asarray(X, dtype=float), self.cluster_centers_,
                            self.weights_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
